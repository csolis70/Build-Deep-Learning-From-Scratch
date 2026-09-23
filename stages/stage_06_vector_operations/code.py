"""Stage 06: Vector operations.

A thin ``Vec`` container built ON TOP of the stage_05 scalar ``Value`` engine;
every vector op composes scalar ``Value`` ops so autodiff flows through stage_05.
NumPy may be used for forward array creation only, never for gradients.
"""

from __future__ import annotations

import os
import sys
from typing import Iterable, List, Union

# Plumbing: make the curriculum root importable so `dlfs` resolves.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dlfs import stage_import  # noqa: E402

# Reuse stage_05's completed scalar engine; re-export `Value` for later stages.
Stage5_Value = stage_import("stage_05", "Value")
Value = Stage5_Value


Scalar = Union[int, float, Stage5_Value]


class Vec:
    """A 1-D vector of stage_05 ``Value`` scalars; ops compose ``Value`` ops so
    gradients flow through stage_05. No ``backward()`` (reduce to a scalar first)."""

    def __init__(self, data: Iterable[Scalar]) -> None:
        """Store ``data`` as a list of ``Value``, wrapping non-``Value`` entries."""
        self.data: List[Stage5_Value] = [v if isinstance(v, Stage5_Value) else Stage5_Value(v) for v in data]

    # --- container protocol ---
    def __len__(self) -> int:
        """Number of elements in the vector."""
        return len(self.data)

    def __getitem__(self, i: int) -> Stage5_Value:
        """Return the i-th ``Value`` (no copy)."""
        return self.data[i]

    def __iter__(self):
        """Iterate over the underlying ``Value`` scalars."""
        return iter(self.data)

    def __repr__(self) -> str:
        return f"Vec({[v.data for v in self.data]})"

    # --- elementwise ops (Vec op Vec, or Vec op scalar broadcast) ---
    def __add__(self, other: "Vec | Scalar") -> "Vec":
        """Elementwise add; ``other`` is an equal-length ``Vec`` or a broadcast scalar."""
        if not isinstance(other, Vec):
            other = Vec([other for _ in range(len(self))])

        assert len(self) == len(other), 'Both vectors must be the same length'

        out = Vec(self[i] + other[i] for i in range(len(self)))
        return out

    def __sub__(self, other: "Vec | Scalar") -> "Vec":
        """Elementwise subtract (see ``__add__`` for broadcast/length rules)."""
        if not isinstance(other, Vec):
            other = Vec([other for _ in range(len(self))])

        assert len(self) == len(other), 'Both vectors must be the same length'

        out = Vec([self[i] - other[i] for i in range(len(self))])
        return out

    def __mul__(self, other: "Vec | Scalar") -> "Vec":
        """Elementwise (Hadamard) multiply (see ``__add__`` for the rules)."""
        if not isinstance(other, Vec):
            other = Vec([other for _ in range(len(self))])

        assert len(self) == len(other), 'Both vectors must be the same length'

        out = Vec([self[i] * other[i] for i in range(len(self))])
        return out

    # reflected scalar ops: scalar OP vec
    def __radd__(self, other: Scalar) -> "Vec":
        """Return ``other + self`` (commutative)."""
        other = Vec([other for _ in range(len(self))])

        return other + self

    def __rmul__(self, other: Scalar) -> "Vec":
        """Return ``other * self`` (commutative)."""
        other = Vec([other for _ in range(len(self))])

        return other * self

    def __rsub__(self, other: Scalar) -> "Vec":
        """Return ``other - self`` (e.g. ``10 - vec``)."""
        other = Vec([other for _ in range(len(self))])

        return other - self

    # --- reductions (return a scalar Value) ---
    def dot(self, other: "Vec") -> Stage5_Value:
        """Dot product ``sum_i self[i] * other[i]`` as one ``Value``; ValueError if lengths differ."""
        
        assert len(self) == len(other), 'Both vectors must be the same length'

        dot_product = sum([self[i] * other[i] for i in range(len(self))])
        return dot_product

    def sum(self) -> Stage5_Value:
        """Return ``sum_i self[i]`` as a single ``Value``."""
        return sum(self.data)

    # --- elementwise activation ---
    def relu(self) -> "Vec":
        """Elementwise ReLU; return a new ``Vec`` using stage_05 ``Value.relu()``."""
        out = Vec([v.relu() for v in self])
        return out
