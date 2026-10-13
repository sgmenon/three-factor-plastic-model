"""Connection-owned proportional-integral correction."""

from __future__ import annotations

import torch
from torch import Tensor


class PICorrection:
    """Bounded PI correction with one integral trace per connection."""

    def __init__(
        self,
        connection_count: int,
        *,
        proportional_gain: float,
        integral_gain: float,
        integral_decay: float = 0.99,
        integral_limit: float = 20.0,
    ) -> None:
        if proportional_gain < 0 or integral_gain < 0:
            raise ValueError("PI gains must be nonnegative")
        if not 0 <= integral_decay <= 1:
            raise ValueError("Integral decay must be between zero and one")
        if integral_limit <= 0:
            raise ValueError("Integral limit must be positive")

        self.proportional_gain = proportional_gain
        self.integral_gain = integral_gain
        self.integral_decay = integral_decay
        self.integral_limit = integral_limit
        self.integral = torch.zeros(connection_count)

    def update(self, correction: Tensor) -> Tensor:
        active = correction != 0
        accumulated = (
            self.integral_decay * self.integral + correction
        ).clamp(
            min=-self.integral_limit,
            max=self.integral_limit,
        )
        self.integral = torch.where(active, accumulated, self.integral)
        pi_correction = (
            self.proportional_gain * correction
            + self.integral_gain * self.integral
        )
        return torch.where(active, pi_correction, 0.0)

    def filter_connections(self, keep: Tensor) -> None:
        """Keep integral state aligned with physically retained edges."""
        self.integral = self.integral[keep]

    def append_connections(self, count: int) -> None:
        """Add zero-history state for newly created edges."""
        if count:
            self.integral = torch.cat(
                [self.integral, self.integral.new_zeros(count)]
            )
