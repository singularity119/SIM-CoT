"""EMA teacher model management for host-agnostic LSP-JEPA core code."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


class EMATeacher(nn.Module):
    """Frozen exponential moving average copy of a student model.

    The teacher is created from a structural copy of the student, is always kept
    in eval mode, and never receives gradients. Call :meth:`update` only after
    ``optimizer.step()`` has updated the student parameters and before the next
    teacher forward/target-building pass.

    Args:
        model: Teacher model instance, usually produced by
            :meth:`from_student`.
        decay: EMA decay in ``[0, 1]``. New teacher parameters are computed as
            ``decay * teacher + (1 - decay) * student``.
        trainable_only: When ``True``, update only parameters whose matching
            student parameter has ``requires_grad=True``.
        device: Optional device for the teacher copy.
        dtype: Optional floating point dtype for the teacher copy.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        decay: float = 0.995,
        trainable_only: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.decay = _validate_decay(decay)
        self.trainable_only = bool(trainable_only)

        to_kwargs: dict[str, Any] = {}
        if device is not None:
            to_kwargs["device"] = device
        if dtype is not None:
            to_kwargs["dtype"] = dtype
        if to_kwargs:
            self.model.to(**to_kwargs)

        self.requires_grad_(False)
        self.eval()

    @classmethod
    def from_student(
        cls,
        student_model: nn.Module,
        *,
        decay: float = 0.995,
        trainable_only: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "EMATeacher":
        """Build a frozen EMA teacher from a student model snapshot."""

        teacher_model = copy.deepcopy(student_model)
        return cls(
            teacher_model,
            decay=decay,
            trainable_only=trainable_only,
            device=device,
            dtype=dtype,
        )

    @property
    def teacher_model(self) -> nn.Module:
        """Return the wrapped teacher model."""

        return self.model

    def requires_grad_(self, requires_grad: bool = False) -> "EMATeacher":
        """Keep all teacher parameters frozen.

        Passing ``True`` is rejected because EMA teachers are stop-gradient
        target networks and must not receive gradients.
        """

        if requires_grad:
            raise ValueError("EMA teacher parameters must remain frozen")
        super().requires_grad_(False)
        return self

    def train(self, mode: bool = True) -> "EMATeacher":
        """Keep the EMA teacher in eval mode even if train mode is requested."""

        del mode
        super().train(False)
        return self

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the wrapped teacher model under ``torch.no_grad()``."""

        self.eval()
        with torch.no_grad():
            output = self.model(*args, **kwargs)
        return _detach_output(output)

    @torch.no_grad()
    def update(self, student_model: nn.Module) -> "EMATeacher":
        """Update teacher parameters from the student EMA source.

        This method must be called only after ``optimizer.step()`` has updated
        the student parameters, and before the next teacher forward pass.
        Registered buffers are copied from the student so generic ``nn.Module``
        state such as BatchNorm running statistics stays consistent.
        """

        student_params = dict(student_model.named_parameters())
        teacher_params = dict(self.model.named_parameters())

        if student_params.keys() != teacher_params.keys():
            missing = sorted(teacher_params.keys() - student_params.keys())
            extra = sorted(student_params.keys() - teacher_params.keys())
            details = []
            if missing:
                details.append(f"missing student params: {missing}")
            if extra:
                details.append(f"unexpected student params: {extra}")
            raise ValueError(
                "student and teacher parameters must match by name; "
                + "; ".join(details)
            )

        for name, teacher_param in teacher_params.items():
            student_param = student_params[name]
            if self.trainable_only and not student_param.requires_grad:
                continue
            if teacher_param.shape != student_param.shape:
                raise ValueError(
                    f"parameter {name!r} shape mismatch: "
                    f"teacher {tuple(teacher_param.shape)} vs student {tuple(student_param.shape)}"
                )
            _ema_update_parameter(teacher_param, student_param, self.decay)

        student_buffers = dict(student_model.named_buffers())
        teacher_buffers = dict(self.model.named_buffers())
        if student_buffers.keys() != teacher_buffers.keys():
            missing = sorted(teacher_buffers.keys() - student_buffers.keys())
            extra = sorted(student_buffers.keys() - teacher_buffers.keys())
            details = []
            if missing:
                details.append(f"missing student buffers: {missing}")
            if extra:
                details.append(f"unexpected student buffers: {extra}")
            raise ValueError(
                "student and teacher buffers must match by name; "
                + "; ".join(details)
            )

        for name, teacher_buffer in teacher_buffers.items():
            _copy_buffer(name, teacher_buffer, student_buffers[name])

        self.requires_grad_(False)
        self.eval()
        return self


def _validate_decay(decay: float) -> float:
    decay = float(decay)
    if not 0.0 <= decay <= 1.0:
        raise ValueError("EMA decay must be in [0, 1]")
    return decay


def _ema_update_parameter(
    teacher_param: nn.Parameter,
    student_param: nn.Parameter,
    decay: float,
) -> None:
    student_value = student_param.detach().to(device=teacher_param.device)
    if teacher_param.is_floating_point() or teacher_param.is_complex():
        student_value = student_value.to(dtype=teacher_param.dtype)
        teacher_param.mul_(decay).add_(student_value, alpha=1.0 - decay)
    else:
        teacher_param.copy_(student_value)


def _copy_buffer(
    name: str,
    teacher_buffer: torch.Tensor,
    student_buffer: torch.Tensor,
) -> None:
    if teacher_buffer.shape != student_buffer.shape:
        raise ValueError(
            f"buffer {name!r} shape mismatch: "
            f"teacher {tuple(teacher_buffer.shape)} vs student {tuple(student_buffer.shape)}"
        )
    student_value = student_buffer.detach().to(
        device=teacher_buffer.device,
        dtype=teacher_buffer.dtype,
    )
    teacher_buffer.copy_(student_value)


def _detach_output(output: Any) -> Any:
    if isinstance(output, torch.Tensor):
        return output.detach()
    if isinstance(output, tuple) and hasattr(output, "_fields"):
        return type(output)(*(_detach_output(item) for item in output))
    if isinstance(output, tuple):
        return tuple(_detach_output(item) for item in output)
    if isinstance(output, list):
        return [_detach_output(item) for item in output]
    if isinstance(output, Mapping):
        return type(output)(
            (key, _detach_output(value)) for key, value in output.items()
        )
    return output
