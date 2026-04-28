"""Host adapter protocol for LSP-JEPA.

Adapters isolate LSP-JEPA core code from Coconut/CODI/SIM-CoT implementation
details. Implementations may live next to host integrations, but the protocol
itself must remain host-agnostic.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch

from lsp_jepa.core.latent_interface import HostModelOutput


@runtime_checkable
class HostAdapter(Protocol):
    """Protocol every LSP-JEPA host adapter must satisfy.

    Shape contract:
        ``forward_student`` returns ``HostModelOutput`` whose ``latent_states``
        has shape ``[batch, T_student, dim]`` and whose ``latent_mask`` has
        shape ``[batch, T_student]``.

    Core-mode safety:
        ``prepare_batch`` and ``forward_student`` must keep student inputs free
        of ground-truth CoT tokens when experiment mode is ``core``. This
        protocol does not enforce that policy; concrete adapters and tests must.
    """

    def prepare_batch(self, raw_batch: Any) -> dict[str, Any]:
        """Convert a raw host batch into adapter-owned structured inputs."""
        ...

    def forward_student(
        self,
        model: torch.nn.Module,
        batch: dict[str, Any],
        *,
        output_latent_states: bool,
    ) -> HostModelOutput:
        """Run the student and expose host output through ``HostModelOutput``."""
        ...

    def build_teacher_input(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Build teacher inputs for target construction without host internals."""
        ...

    def extract_answer_loss(
        self,
        host_output: HostModelOutput,
        batch: dict[str, Any],
    ) -> torch.Tensor | None:
        """Return optional host answer loss without computing LSP alignment."""
        ...
