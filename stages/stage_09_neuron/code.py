"""Stage 09: Neuron.

A single neuron ``y = phi(x @ w + b)`` built on stage_08's autodiff ``Tensor``.
Wire up the forward expression only; gradients flow through ``Tensor.backward()``.
"""

from __future__ import annotations

import numpy as np

from dlfs import stage_import

# Tensor engine from stage_08, re-exported as the public ``Tensor``.
Stage8_Tensor = stage_import("stage_08", "Tensor")
Tensor = Stage8_Tensor


class Neuron:
    """A single neuron: ``y = phi(x @ w + b)``, built on stage_08's ``Tensor``."""

    def __init__(self, n_in: int, activation: str = "tanh", seed: int | None = None):
        """Construct leaf params: w shape (n_in,), scalar bias b=0; activation in {tanh, relu, none}."""
        # TODO: validate activation, build RNG, init w/b as leaf Tensors, store config
        rng = np.random.default_rng(seed= seed)

        self.w = Tensor(rng.uniform(low= -1, high= 1, size= (n_in,)))

        if activation not in {'tanh', 'relu', 'none'}:
            raise TypeError('activation parameter must be tanh, relu, or None')

        self.activation = activation
        self.b = Tensor(0)

    def __call__(self, x) -> "Stage8_Tensor":
        """Forward pass: z = x @ w + b then phi(z). x shape (n_in,) or (batch, n_in)."""
        # TODO: coerce x to Tensor, compute affine map, apply activation
        if not isinstance(x, Tensor):
            x = Tensor(x)

        z = x @ self.w + self.b
        if self.activation == 'tanh':
            out = z.tanh()

            return out

        elif self.activation == 'relu':
            out = z.relu()

            return out
        
        else:

            return z

    def parameters(self) -> list:
        """Return the learnable parameters as ``[self.w, self.b]``."""
        return [self.w, self.b]

    def zero_grad(self) -> None:
        """Reset the gradient of every parameter to zeros."""
        self.w.zero_grad()
        self.b.zero_grad()

    def __repr__(self) -> str:
        """e.g. ``Neuron(n_in=3, activation='tanh')``."""
        return f'Neuron(n_in={self.w.shape[0]}, activation=\'{self.activation}\')'
