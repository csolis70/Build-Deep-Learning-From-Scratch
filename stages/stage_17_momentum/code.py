"""Stage 17: Momentum.

SGD with a velocity buffer (heavy-ball momentum) and an optional Nesterov
variant, built by subclassing the plain ``SGD`` from stage_14.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

# Tensor (stage_08 engine via stage_12); SGD (stage_14).
from dlfs import stage_import

Stage12_Tensor = stage_import("stage_12", "Tensor")
Stage14_SGD = stage_import("stage_14", "SGD")

# Re-export the autodiff Tensor under its canonical public name.
Tensor = Stage12_Tensor


class SGDMomentum(Stage14_SGD):
    """SGD with momentum (heavy-ball) and an optional Nesterov variant.

    Update with ``g = p.grad``:
        v <- beta * v + g
        p <- p - lr * v                  # heavy-ball (nesterov=False)
        p <- p - lr * (g + beta * v)     # nesterov=True
    Extends stage_14 ``SGD`` (same params/lr contract, inherited zero_grad).
    ``beta=0`` reduces exactly to plain SGD.
    """

    def __init__(
        self,
        params: Iterable["Tensor"],
        lr: float,
        beta: float = 0.9,
        nesterov: bool = False,
    ) -> None:
        # Defer params/lr setup to the stage_14 SGD ctor (single source of truth).
        super().__init__(params, lr)
        # TODO: validate 0 <= beta < 1, store beta/nesterov, allocate one
        # zeros velocity buffer per param (aligned with self.params).
        if beta >= 1 or beta < 0:
            raise ValueError
        self.velocities = [np.zeros_like(param.data) for param in self.params]
        self.beta = beta
        self.nesterov = nesterov

    def step(self) -> None:
        """Apply one momentum update to every parameter in place.

        Velocity buffers persist across calls (write updated v back into
        self.velocities). Does NOT zero grads (that's the inherited zero_grad).
        """
        # TODO: implement the heavy-ball / Nesterov update.
        velocities = []

        for p, v in zip(self.params, self.velocities):

            if p.grad is None:
                velocities.appen(0)
                continue

            g = p.grad
            v = self.beta * v + g
            velocities.append(v)

            if self.nesterov:
                p.data -= self.lr * (g + self.beta * v)
            else:
                p.data -= self.lr * v

        self.velocities = velocities

    def reset(self) -> None:
        """Zero all velocity buffers; leaves p.data and p.grad untouched."""
        # TODO: reset every velocity buffer to zeros.
        self.velocities = [np.zeros_like(param.data) for param in self.params]

    def __repr__(self) -> str:
        # TODO: summarize lr, beta, nesterov.
        return f'SGDMomentum(lr={self.lr}, beta={self.beta}, nesterov={self.nesterov})'


# Backwards-compatible public alias used by the tests.
Momentum = SGDMomentum
