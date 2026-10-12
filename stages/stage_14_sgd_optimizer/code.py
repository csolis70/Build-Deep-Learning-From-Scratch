"""Stage 14: SGD optimizer.

Factors the manual parameter-update ritual into a reusable ``Optimizer`` object
matching the ``torch.optim`` interface. No new gradient to derive here.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np

# Tensor (stage_08 engine, imported via its latest extension in stage_12); the
# optimizer only reads .grad / writes .data.
from dlfs import stage_import

Stage12_Tensor = stage_import("stage_12", "Tensor")

Tensor = Stage12_Tensor


class Optimizer:
    """Base class for optimizers: owns a list of parameter Tensors and updates them in place."""

    def __init__(self, params: Iterable["Stage12_Tensor"]) -> None:
        # TODO: materialize `params` into a concrete list (not a one-shot iterable).
        self.params = list(params)

    def zero_grad(self) -> None:
        """Reset every parameter's .grad to zeros (call before each backward pass)."""
        # TODO: implement gradient reset.
        for param in self.params:
            param.zero_grad()

    def step(self) -> None:
        """Apply one optimization update to ``self.params`` (subclass-defined)."""
        raise NotImplementedError("Optimizer.step must be implemented by a subclass")


class SGD(Optimizer):
    """Plain stochastic gradient descent: p.data -= lr * p.grad."""

    def __init__(self, params: Iterable["Stage12_Tensor"], lr: float = 0.01) -> None:
        # TODO: validate lr > 0, call super().__init__, store self.lr.
        assert lr > 0

        super().__init__(params)
        self.lr = lr

    def step(self) -> None:
        """One SGD step: in-place ``p.data -= self.lr * p.grad`` (skip params with grad None)."""
        # TODO: implement the in-place SGD update.
        
        for p in self.params:
            if p.grad is None:
                continue

            p.data -= self.lr * p.grad

    def __repr__(self) -> str:
        # TODO: e.g. f"SGD(lr={self.lr}, n_params={len(self.params)})"
        
        return f'SGD(lr={self.lr}, n_params={len(self.params)})'