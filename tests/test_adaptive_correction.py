import torch

from experiments.adaptive_correction import PICorrection


def test_repeated_error_grows_correction_linearly() -> None:
    controller = PICorrection(
        2,
        proportional_gain=0.1,
        integral_gain=0.01,
        integral_decay=1.0,
    )

    first = controller.update(torch.tensor([1.0, -1.0]))
    second = controller.update(torch.tensor([1.0, -1.0]))

    assert torch.allclose(first, torch.tensor([0.11, -0.11]))
    assert torch.allclose(second, torch.tensor([0.12, -0.12]))


def test_opposite_error_unwinds_integral_and_zero_does_not_update() -> None:
    controller = PICorrection(
        1,
        proportional_gain=0.1,
        integral_gain=0.01,
        integral_decay=1.0,
    )
    controller.update(torch.tensor([1.0]))
    controller.update(torch.tensor([1.0]))

    opposite = controller.update(torch.tensor([-1.0]))
    integral_before_zero = controller.integral.clone()
    zero = controller.update(torch.tensor([0.0]))

    assert torch.allclose(opposite, torch.tensor([-0.09]))
    assert torch.equal(zero, torch.tensor([0.0]))
    assert torch.equal(controller.integral, integral_before_zero)


def test_integral_is_clamped_to_prevent_windup() -> None:
    controller = PICorrection(
        1,
        proportional_gain=0.0,
        integral_gain=1.0,
        integral_decay=1.0,
        integral_limit=2.0,
    )

    for _ in range(10):
        correction = controller.update(torch.tensor([1.0]))

    assert torch.equal(correction, torch.tensor([2.0]))
