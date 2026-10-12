"""Test consensus correction with two heads connected to one backbone.

The model is:

                      /--(w1, b1)--> head 1
    x --(wb, bb)--> h
                      \\--(w2, b2)--> head 2

Each head updates from its own error. The backbone receives the mean of both
head corrections, so equal and opposite corrections cancel by design.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ForkRecord:
    input_value: Tensor
    hidden_value: Tensor
    head_1_prediction: Tensor
    head_2_prediction: Tensor
    hidden_active: Tensor
    head_1_active: Tensor
    head_2_active: Tensor


class TwoHeadFork:
    """One hidden feature shared by two independently adapting heads."""

    def __init__(
        self,
        *,
        weight_learning_rate: float = 2e-3,
        bias_learning_rate: float = 5e-4,
    ) -> None:
        self.backbone_weight = torch.tensor(0.4)
        self.backbone_bias = torch.tensor(0.05)

        self.head_1_weight = torch.tensor(0.4)
        self.head_1_bias = torch.tensor(0.05)
        self.head_2_weight = torch.tensor(0.4)
        self.head_2_bias = torch.tensor(0.05)

        self.weight_learning_rate = weight_learning_rate
        self.bias_learning_rate = bias_learning_rate

    def forward(self, x: Tensor) -> ForkRecord:
        hidden_pre_activation = (
            x * self.backbone_weight - self.backbone_bias
        )
        hidden = torch.relu(hidden_pre_activation)

        head_1_pre_activation = (
            hidden * self.head_1_weight - self.head_1_bias
        )
        head_2_pre_activation = (
            hidden * self.head_2_weight - self.head_2_bias
        )

        return ForkRecord(
            input_value=x,
            hidden_value=hidden,
            head_1_prediction=torch.relu(head_1_pre_activation),
            head_2_prediction=torch.relu(head_2_pre_activation),
            hidden_active=hidden_pre_activation > 0,
            head_1_active=head_1_pre_activation > 0,
            head_2_active=head_2_pre_activation > 0,
        )

    def correct(
        self,
        record: ForkRecord,
        target_1: Tensor,
        target_2: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Update both heads, then update the backbone from their consensus."""
        correction_1 = (
            target_1 - record.head_1_prediction
        ) * record.head_1_active
        correction_2 = (
            target_2 - record.head_2_prediction
        ) * record.head_2_active

        # Each head adapts independently.
        self.head_1_bias -= self.bias_learning_rate * correction_1
        self.head_1_weight += (
            self.weight_learning_rate
            * correction_1
            * record.hidden_value
        )
        self.head_2_bias -= self.bias_learning_rate * correction_2
        self.head_2_weight += (
            self.weight_learning_rate
            * correction_2
            * record.hidden_value
        )

        # Equal and opposite corrections cancel at the shared backbone.
        aggregate_correction = (correction_1 + correction_2) / 2
        aggregate_correction *= record.hidden_active
        self.backbone_bias -= (
            self.bias_learning_rate * aggregate_correction
        )
        self.backbone_weight += (
            self.weight_learning_rate
            * aggregate_correction
            * record.input_value
        )

        # Bias remains a nonnegative suppressive threshold.
        self.backbone_bias.clamp_(min=0.0)
        self.head_1_bias.clamp_(min=0.0)
        self.head_2_bias.clamp_(min=0.0)

        return correction_1, correction_2, aggregate_correction


def evaluation_loss(model: TwoHeadFork) -> tuple[float, float, float]:
    inputs = torch.linspace(0.5, 1.5, 101)
    predictions = [model.forward(x) for x in inputs]
    prediction_1 = torch.stack(
        [record.head_1_prediction for record in predictions]
    )
    prediction_2 = torch.stack(
        [record.head_2_prediction for record in predictions]
    )

    loss_1 = ((0.8 * inputs - prediction_1) ** 2).mean()
    loss_2 = ((0.2 * inputs - prediction_2) ** 2).mean()
    mean_loss = (loss_1 + loss_2) / 2
    return loss_1.item(), loss_2.item(), mean_loss.item()


def run_online_experiment(
    seed: int = 7,
    observations: int = 20_000,
) -> tuple[TwoHeadFork, list[float]]:
    """Simulate two targets arriving after each shared-backbone prediction."""
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)

    model = TwoHeadFork()
    losses = [evaluation_loss(model)[2]]

    for observation in range(observations):
        x = 0.5 + torch.rand(())
        record = model.forward(x)
        model.correct(record, target_1=0.8 * x, target_2=0.2 * x)

        if (observation + 1) % 2_000 == 0:
            losses.append(evaluation_loss(model)[2])

    return model, losses


def main() -> None:
    model, losses = run_online_experiment()
    loss_1, loss_2, mean_loss = evaluation_loss(model)

    print("Autograd used: no")
    print(f"Initial mean MSE: {losses[0]:.6f}")
    print(f"Final mean MSE:   {mean_loss:.6f}")
    print(f"Head 1 MSE:       {loss_1:.6f}")
    print(f"Head 2 MSE:       {loss_2:.6f}")
    print(
        "Final weights: "
        f"backbone={model.backbone_weight.item():.4f}, "
        f"head_1={model.head_1_weight.item():.4f}, "
        f"head_2={model.head_2_weight.item():.4f}"
    )
    print("Mean loss every 2,000 updates:")
    print("  " + " -> ".join(f"{loss:.6f}" for loss in losses))


if __name__ == "__main__":
    main()
