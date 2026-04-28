from __future__ import annotations

import torch

from lsp_jepa.core.anti_collapse import (
    variance_regularizer,
    vicreg_variance_covariance_loss,
)
from lsp_jepa.core.losses import (
    compute_lsp_state_loss,
    masked_cosine_loss,
    masked_mse,
    masked_normalized_mse,
    masked_smooth_l1,
)
from lsp_jepa.core.metrics import (
    effective_rank,
    pairwise_cosine,
    pairwise_l2,
    per_dim_variance,
)


def test_all_zero_mask_returns_finite_zero_losses():
    pred = torch.randn(2, 3, 4, requires_grad=True)
    target = torch.randn(2, 3, 4)
    mask = torch.zeros(2, 3, dtype=torch.bool)

    losses = [
        masked_mse(pred, target, mask),
        masked_normalized_mse(pred, target, mask),
        masked_cosine_loss(pred, target, mask),
        masked_smooth_l1(pred, target, mask),
        compute_lsp_state_loss(pred, target, mask, alignment="normalized_mse"),
        variance_regularizer(pred, mask),
        vicreg_variance_covariance_loss(pred, mask),
    ]

    for loss in losses:
        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert loss.detach().item() == 0.0

    sum(losses).backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum().item() == 0.0


def test_padding_positions_do_not_contribute_to_loss_or_gradients():
    pred = torch.tensor(
        [
            [[1.0, 2.0], [1000.0, 1000.0]],
            [[-999.0, -999.0], [3.0, 4.0]],
        ],
        requires_grad=True,
    )
    target = torch.tensor(
        [
            [[2.0, 0.0], [-1000.0, -1000.0]],
            [[999.0, 999.0], [1.0, 1.0]],
        ]
    )
    mask = torch.tensor([[True, False], [False, True]])

    loss = masked_mse(pred, target, mask)
    expected = torch.stack(
        [
            (pred[0, 0] - target[0, 0]).square(),
            (pred[1, 1] - target[1, 1]).square(),
        ]
    ).mean()

    assert torch.allclose(loss, expected)

    loss.backward()
    assert pred.grad is not None
    assert pred.grad[0, 1].abs().sum().item() == 0.0
    assert pred.grad[1, 0].abs().sum().item() == 0.0


def test_losses_accept_3d_element_masks():
    pred = torch.tensor([[[1.0, 2.0, 3.0]]])
    target = torch.zeros_like(pred)
    mask = torch.tensor([[[True, False, True]]])

    assert torch.allclose(masked_mse(pred, target, mask), torch.tensor(5.0))
    assert torch.isfinite(masked_smooth_l1(pred, target, mask))


def test_normalized_mse_zero_vectors_are_numerically_stable():
    pred = torch.zeros(1, 2, 3, requires_grad=True)
    target = torch.zeros(1, 2, 3)
    mask = torch.ones(1, 2, dtype=torch.bool)

    loss = masked_normalized_mse(pred, target, mask)

    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert loss.detach().item() == 0.0

    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_lsp_state_loss_keeps_student_autograd_path():
    pred = torch.randn(2, 2, 3, requires_grad=True)
    target = torch.randn(2, 2, 3, requires_grad=True)
    mask = torch.ones(2, 2, dtype=torch.bool)

    loss = compute_lsp_state_loss(pred, target, mask, alignment="mse")
    loss.backward()

    assert pred.grad is not None
    assert pred.grad.abs().sum().item() > 0.0
    assert target.grad is not None
    assert target.grad.abs().sum().item() > 0.0


def test_metrics_return_scalars_or_dicts_and_preserve_autograd():
    states = torch.randn(2, 3, 4, requires_grad=True)
    mask = torch.tensor(
        [
            [True, True, False],
            [True, False, True],
        ]
    )

    metrics = [
        pairwise_cosine(states, mask),
        pairwise_l2(states, mask),
        per_dim_variance(states, mask),
        effective_rank(states, mask),
    ]

    total = states.sum() * 0.0
    for metric in metrics:
        if isinstance(metric, dict):
            assert metric
            for value in metric.values():
                assert isinstance(value, torch.Tensor)
                assert value.shape == ()
                assert torch.isfinite(value)
                total = total + value
        else:
            assert metric.shape == ()
            assert torch.isfinite(metric)
            total = total + metric

    total.backward()
    assert states.grad is not None
    assert torch.isfinite(states.grad).all()
