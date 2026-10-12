"""Smallest useful test of the proposed local correction rule.

The model has one input, one hidden neuron, and one output neuron:

    x --(w1, b1)--> h --(w2, b2)--> prediction

It deliberately does not use torch.autograd, loss.backward(), or an optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ForwardRecord:
    """Values a neuron must remember until correction arrives."""

    input_value: Tensor
    hidden_value: Tensor
    prediction: Tensor
    hidden_active: Tensor
    output_active: Tensor


class ScalarChain:
    """Two connection-specific ReLU gates trained without backpropagation."""

    def __init__(
        self,
        *,
        weight_learning_rate: float = 2e-3,
        bias_learning_rate: float = 5e-4,
    ) -> None:
        # These are plain tensors, not autograd Parameters.
        self.w1 = torch.tensor(0.4)
        self.b1 = torch.tensor(0.05)
        self.w2 = torch.tensor(0.4)
        self.b2 = torch.tensor(0.05)

        self.weight_learning_rate = weight_learning_rate
        self.bias_learning_rate = bias_learning_rate

    def forward(self, x: Tensor) -> ForwardRecord:
        """Run one prediction and retain the active path."""
        hidden_pre_activation = x * self.w1 - self.b1
        hidden = torch.relu(hidden_pre_activation)

        output_pre_activation = hidden * self.w2 - self.b2
        prediction = torch.relu(output_pre_activation)

        return ForwardRecord(
            input_value=x,
            hidden_value=hidden,
            prediction=prediction,
            hidden_active=hidden_pre_activation > 0,
            output_active=output_pre_activation > 0,
        )

    def correct(self, record: ForwardRecord, target: Tensor) -> Tensor:
        """Propagate one signed correction over the recorded active path."""
        correction = target - record.prediction

        # The output neuron receives the correction directly.
        output_correction = correction * record.output_active
        self.b2 -= self.bias_learning_rate * output_correction
        self.w2 += (
            self.weight_learning_rate
            * output_correction
            * record.hidden_value
        )

        # With one downstream connection, aggregation is just identity.
        hidden_correction = output_correction * record.hidden_active
        self.b1 -= self.bias_learning_rate * hidden_correction
        self.w1 += (
            self.weight_learning_rate
            * hidden_correction
            * record.input_value
        )

        # Bias is a suppressive threshold in this first experiment.
        self.b1.clamp_(min=0.0)
        self.b2.clamp_(min=0.0)

        return correction.square()

    def predict(self, x: Tensor) -> Tensor:
        return self.forward(x).prediction


def evaluation_loss(model: ScalarChain) -> float:
    """Mean squared error on a fixed grid that training does not iterate over."""
    inputs = torch.linspace(0.5, 1.5, 101)
    targets = 0.8 * inputs
    errors = targets - torch.stack([model.predict(x) for x in inputs])
    return errors.square().mean().item()


def run_online_experiment(
    seed: int = 7,
    observations: int = 10_000,
) -> tuple[ScalarChain, list[float]]:
    """Simulate observations arriving one at a time during normal operation."""
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)

    model = ScalarChain()
    losses = [evaluation_loss(model)]

    for observation in range(observations):
        x = 0.5 + torch.rand(())
        target = 0.8 * x
        record = model.forward(x)
        model.correct(record, target)

        if (observation + 1) % 1_000 == 0:
            losses.append(evaluation_loss(model))

    return model, losses


def main() -> None:
    model, losses = run_online_experiment()

    print("Autograd used: no")
    print(f"Initial evaluation MSE: {losses[0]:.6f}")
    print(f"Final evaluation MSE:   {losses[-1]:.6f}")
    print(
        "Final parameters: "
        f"w1={model.w1.item():.4f}, b1={model.b1.item():.4f}, "
        f"w2={model.w2.item():.4f}, b2={model.b2.item():.4f}"
    )
    print("Loss every 1,000 updates:")
    print("  " + " -> ".join(f"{loss:.6f}" for loss in losses))


if __name__ == "__main__":
    main()
