from __future__ import annotations

import torch
from torch import nn

from lsp_jepa.core.ema_teacher import EMATeacher


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.frozen = nn.Linear(2, 2, bias=False)
        self.frozen.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.frozen(x)


def test_ema_teacher_parameters_do_not_require_grad_and_stay_eval() -> None:
    student = TinyModel()
    student.train()

    teacher = EMATeacher.from_student(student)

    assert not teacher.training
    assert not teacher.model.training
    assert all(not parameter.requires_grad for parameter in teacher.parameters())

    teacher.train()
    assert not teacher.training
    assert not teacher.model.training


def test_ema_teacher_is_excluded_from_optimizer_example() -> None:
    student = TinyModel()
    teacher = EMATeacher.from_student(student)

    optimizer = torch.optim.SGD(
        (parameter for parameter in student.parameters() if parameter.requires_grad),
        lr=0.1,
    )
    optimizer_param_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    assert all(id(parameter) not in optimizer_param_ids for parameter in teacher.parameters())
    assert all(not parameter.requires_grad for parameter in teacher.parameters())


def test_update_after_optimizer_step_applies_decay_formula() -> None:
    student = nn.Linear(2, 1, bias=False)
    decay = 0.25
    with torch.no_grad():
        student.weight.copy_(torch.tensor([[1.0, -2.0]]))

    teacher = EMATeacher.from_student(student, decay=decay)
    teacher_before = teacher.model.weight.detach().clone()

    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    loss = student(torch.tensor([[3.0, -4.0]])).sum()
    loss.backward()
    optimizer.step()

    student_after_step = student.weight.detach().clone()
    teacher.update(student)

    expected = decay * teacher_before + (1.0 - decay) * student_after_step
    assert torch.allclose(teacher.model.weight, expected)
    assert all(not parameter.requires_grad for parameter in teacher.parameters())


def test_update_skips_frozen_student_parameters_when_trainable_only() -> None:
    student = TinyModel()
    teacher = EMATeacher.from_student(student, decay=0.5, trainable_only=True)
    linear_before = teacher.model.linear.weight.detach().clone()
    frozen_before = teacher.model.frozen.weight.detach().clone()

    with torch.no_grad():
        student.frozen.weight.add_(10.0)
        student.linear.weight.add_(2.0)

    teacher.update(student)

    expected_linear = 0.5 * linear_before + 0.5 * student.linear.weight.detach()
    assert torch.allclose(teacher.model.frozen.weight, frozen_before)
    assert torch.allclose(teacher.model.linear.weight, expected_linear)


def test_update_copies_registered_buffers_from_student() -> None:
    student = nn.BatchNorm1d(2)
    teacher = EMATeacher.from_student(student, decay=0.5)

    with torch.no_grad():
        student.running_mean.copy_(torch.tensor([10.0, 20.0]))
        student.running_var.copy_(torch.tensor([3.0, 5.0]))
        student.num_batches_tracked.fill_(7)

    teacher.update(student)

    assert torch.allclose(teacher.model.running_mean, student.running_mean)
    assert torch.allclose(teacher.model.running_var, student.running_var)
    assert torch.equal(
        teacher.model.num_batches_tracked,
        student.num_batches_tracked,
    )


def test_from_student_accepts_device_and_dtype_options() -> None:
    student = TinyModel()

    teacher = EMATeacher.from_student(
        student,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    assert next(teacher.parameters()).device.type == "cpu"
    assert next(teacher.parameters()).dtype is torch.float64
    assert all(not parameter.requires_grad for parameter in teacher.parameters())


def test_teacher_forward_under_no_grad_does_not_generate_grad() -> None:
    student = TinyModel()
    teacher = EMATeacher.from_student(student)
    x = torch.randn(3, 2, requires_grad=True)

    with torch.enable_grad():
        output = teacher(x)

    assert not output.requires_grad
    assert x.grad is None
    assert all(parameter.grad is None for parameter in teacher.parameters())
