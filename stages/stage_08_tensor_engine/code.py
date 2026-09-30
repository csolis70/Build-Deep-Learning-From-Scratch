"""Stage 08: Tensor engine -- one N-dimensional reverse-mode autodiff class.

`Tensor` collapses the scalar/Vec/Mat graphs (stages 06-08) onto a single
NumPy-backed node (one data array + one grad array). Every later stage imports
this `Tensor`. Binary ops are restricted to EQUAL-SHAPED operands (or a numeric
constant); general broadcasting-gradient reduction arrives in stage_11.
Allowed tools: Python stdlib + NumPy (forward math / storage only).
"""

from __future__ import annotations

from typing import Tuple, Union, List

import numpy as np

from dlfs import stage_import


def _load_value():
    """Return the scalar `Value` class from stage_05 (alias Stage5_Value)."""
    Stage5_Value = stage_import("stage_05", "Value")
    return Stage5_Value


# An operand we accept: a Tensor, or a raw number/array we wrap.
Operand = Union["Tensor", float, int, np.ndarray, list]


class Tensor:
    """An N-dimensional value node in a reverse-mode autodiff graph.

    Mirrors stage_05's scalar `Value` API but on whole arrays (one ndarray of
    data and one of grad per node). This is the engine every later stage uses.
    """

    def __init__(
        self,
        data: Operand,
        _prev: Tuple["Tensor", ...] = (),
        _op: str = "",
    ) -> None:
        """Wrap `data` as a float64 ndarray and init an autodiff node
        (data, grad zeros, _prev, _op, _backward no-op leaf)."""

        self.data = data.data if isinstance(data, Tensor) else np.array(data, dtype=float)
        self.grad = np.zeros(self.data.shape)
        self._prev = _prev
        self._op = _op
        self._backward = lambda: None

    def _make_tensor(self, *args, **kwargs) -> "Tensor":
        """Build a result node of THIS instance's runtime class.

        Every node-building op below routes its child construction through here
        (``self._make_tensor(data, prev, op)``) instead of calling ``Tensor(...)``
        directly. Because the class is ``type(self)``, a chained graph keeps the
        most-derived subclass at every node: a later stage that subclasses
        ``Tensor`` (e.g. stage_11's broadcasting ``Tensor``) inherits every op
        unchanged and its results are still ITS class -- no op needs re-overriding
        just to swap the constructor. Signature mirrors ``__init__``."""
        return type(self)(*args, **kwargs)

    def _coerce(self, other: Operand) -> "Tensor":
        """Return `other` as a Tensor (wrap raw numbers/arrays; pass Tensors through).

        Wrap via ``self._make_tensor(...)`` (== ``type(self)(...)``) so a coerced raw
        operand becomes THIS instance's runtime class, keeping a subclass alive across
        the chain (same reason the node-building ops route through ``_make_tensor``)."""
        if isinstance(other, Tensor):
            return other
        
        return self._make_tensor(other)

    @classmethod
    def from_value(cls, v) -> "Tensor":
        """Bridge a stage_05 scalar `Value` into a 0-d `Tensor` leaf (lifts v.data only)."""
        return cls(v.data)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the underlying data array."""
        return self.data.shape

    def reshape(self, *shape: int) -> "Tensor":
        """Return a Tensor viewing this data under a new shape. z = self.reshape(shape).

        Pure rearrangement -- no entry is created, destroyed, or combined, so the
        chain rule is just the inverse rearrangement: each upstream grad entry
        belongs to exactly one input entry. Forward `self.data.reshape(shape)`;
        backward reshape `out.grad` back to `self.data.shape` and accumulate.
        Accepts dims as varargs (`t.reshape(2, 3)`) or one tuple (`t.reshape((2, 3))`),
        and a `-1` placeholder NumPy infers (`t.reshape(-1)` flattens).
        Build the child via ``self._make_tensor(...)`` so a subclass survives the chain."""

        out = self._make_tensor(self.data.reshape(*shape), (self,))

        def _backward():
        
            self.grad += out.grad.reshape(self.shape)

        out._backward = _backward
        
        return out

    def transpose(self) -> "Tensor":
        """Return a Tensor viewing this data with its axes reversed (np `.T`).

        Like reshape, a pure rearrangement: no entry is created, destroyed, or
        combined, so the backward is the inverse rearrangement — transpose the
        upstream grad back (`out.grad.T`) and accumulate into `self.grad`.
        Needed from stage_28 on: self-attention computes scores as `Q @ K.T`.
        Build the child via ``self._make_tensor(...)`` so a subclass survives."""

        out = self._make_tensor(self.data.T, (self,))

        def _backward():
            self.grad += out.grad.T

        out._backward = _backward

        return out

    @property
    def T(self) -> "Tensor":
        """Transpose property: `t.T` == `t.transpose()` (mirrors NumPy)."""
        return self.transpose()

    @staticmethod
    def _accumulate(grad_into: "Tensor", incoming: np.ndarray) -> None:
        """Add `incoming` into grad_into.grad; sum-to-scalar if grad_into is 0-d
        (the only broadcast case this stage allows; full reduction is stage_11)."""

        if grad_into.grad.shape == ():
            grad_into.grad += np.sum(incoming)

        else:
            grad_into.grad += incoming 

    # elementwise ops (equal-shaped operands only this stage)
    # Every op below builds its result via ``self._make_tensor(...)`` (NOT a bare
    # ``Tensor(...)``) so the node carries the caller's runtime class through the graph.
    def __add__(self, other: Operand) -> "Tensor":
        """Elementwise add (build via self._make_tensor; route both grads through Tensor._accumulate)."""
        if not isinstance(other, Tensor):
            other = self._coerce(other)

        out = self._make_tensor(self.data + other.data, (self, other), '+')

        def _backward():
            Tensor._accumulate(self, out.grad)
            Tensor._accumulate(other, out.grad)

        out._backward = _backward

        return out


    def __mul__(self, other: Operand) -> "Tensor":
        """Elementwise multiply (build via self._make_tensor; route both grads through Tensor._accumulate)."""
        if not isinstance(other, Tensor):
            other = self._coerce(other)

        out = self._make_tensor(self.data * other.data, (self, other), '*')

        def _backward():
            Tensor._accumulate(self, out.grad * other.data)
            Tensor._accumulate(other, out.grad * self.data)

        out._backward = _backward

        return out

    def __pow__(self, c: Union[int, float]) -> "Tensor":
        """Raise to a CONSTANT power. z = self ** c (build via self._make_tensor)."""
        out = self._make_tensor(self.data ** c, (self,), '**')

        def _backward():
            Tensor._accumulate(self, out.grad * c * self.data ** (c - 1))

        out._backward = _backward

        return out

    def relu(self) -> "Tensor":
        """Elementwise ReLU. z = max(0, self) (build via self._make_tensor)."""
        out = self._make_tensor(np.maximum(0, self.data), (self,), 'relu')

        def _backward():
            mask = (out.data != 0).astype(float)
            Tensor._accumulate(self, out.grad * mask)

        out._backward = _backward

        return out

    def tanh(self) -> "Tensor":
        """Elementwise tanh. z = tanh(self); local grad g * (1 - z**2) (build via self._make_tensor)."""
        out = self._make_tensor(np.tanh(self.data), (self,), 'tanh')

        def _backward():
            Tensor._accumulate(self, out.grad * (1 - out.data ** 2))

        out._backward = _backward

        return out

    def exp(self) -> "Tensor":
        """Elementwise exp. z = exp(self); local grad g * z (build via self._make_tensor)."""
        out = self._make_tensor(np.exp(self.data), (self,), 'exp')

        def _backward():
            Tensor._accumulate(self, out.grad * out.data)

        out._backward = _backward

        return out

    def log(self) -> "Tensor":
        """Elementwise natural log. z = log(self); local grad g / self (build via self._make_tensor)."""
        out = self._make_tensor(np.log(self.data), (self,), 'log')

        def _backward():
            Tensor._accumulate(self, out.grad / self.data)

        out._backward = _backward

        return out

    def __matmul__(self, other: Operand) -> "Tensor":
        """Matrix product z = self @ other (the ``@`` operator).

        For 2-D z = A @ B with upstream grad G: dL/dA = G @ B.T, dL/dB = A.T @ G.

        Must also cover the 1-D operand forms the neuron / dense layers use:
        (n,)@(n,) -> scalar, (n,)@(n,m) -> (m,), and (b,n)@(n,) -> (b,) batched.
        A 1-D operand makes the bare G@B.T / A.T@G rule wrong (``.T`` is a no-op
        on 1-D, and the right grad is an outer product). The clean fix: promote
        each 1-D operand to 2-D (left -> (1,n) row, right -> (n,1) column),
        apply the 2-D rule above, then squeeze the inserted axis back out so each
        grad matches its operand's original shape. (No general broadcasting
        beyond this 1-D<->2-D promotion; that arrives in stage_11.)
        Build the result via ``self._make_tensor(...)`` so a subclass survives the chain."""

        if not isinstance(other, Tensor):
            other = self._coerce(other)

        if self.data.ndim == 1 and other.data.ndim == 1:

            left = self.data.reshape((1, self.shape[0])) # (1,n)
            right = other.data.reshape((other.data.shape[0], 1)) # (n,1)

            result = left @ right # scalar result

            out = self._make_tensor(result.reshape(()), (self, other), '@') # shape ()

            def _backward():
                Tensor._accumulate(self, 
                                   (out.grad.reshape((1,1)) @ right.T).reshape(self.grad.shape)
                                   )
                
                Tensor._accumulate(other, 
                                   (left.T @ out.grad.reshape((1,1))).reshape(other.grad.shape)
                                   )


        elif self.data.ndim == 1:

            left = self.data.reshape((1, self.shape[0])) # (1,n)
            right = other.data # (n,m)

            result = left @ right # (1,m) result

            out = self._make_tensor(result.reshape(other.shape[1],), (self, other), '@') # shape (m,)

            def _backward():

                Tensor._accumulate(self, 
                                   (out.grad.reshape((1, out.grad.shape[0])) @ right.T).reshape(self.grad.shape)
                                   ) # reshaping so that matrix Mult is possible and reshaping back to self.grad shape.

                Tensor._accumulate(other, 
                                   left.T @ out.grad.reshape((1, out.grad.shape[0]))
                                   )
            
        elif other.data.ndim == 1:
            left = self.data # (b,n)
            right = other.data.reshape((other.data.shape[0], 1)) # (n, 1)

            result = left @ right # (b,1) result

            out = self._make_tensor(result.reshape(self.data.shape[0],), (self, other), '@') # shape (b,)

            def _backward():
                Tensor._accumulate(self, 
                                   (out.grad.reshape((out.grad.shape[0], 1)) @ right.T)
                                   )

                Tensor._accumulate(other, 
                                   (left.T @ out.grad.reshape((out.grad.shape[0], 1))).reshape(other.grad.shape)
                                   )

        else:

            left = self.data # (n,k)
            right = other.data # (k,m)

            result = left @ right # (n,m) result

            out = self._make_tensor(result, (self, other), '@') # shape (n,m)

            def _backward():
                Tensor._accumulate(self, out.grad @ right.T)
                Tensor._accumulate(other, left.T @ out.grad)

        out._backward = _backward

        return out

    # ops derived from the primitives above (no new _backward)
    def __neg__(self) -> "Tensor":
        """-self, implemented as self * -1."""
        return self * -1

    def __sub__(self, other: Operand) -> "Tensor":
        """self - other, implemented as self + (-other)."""
        return self + (-self._coerce(other))

    def __rsub__(self, other: Operand) -> "Tensor":
        """other - self (for `number - tensor`)."""
        return -self + self._coerce(other)

    def __truediv__(self, other: Operand) -> "Tensor":
        """self / other, implemented as self * other ** -1."""

        return self * (self._coerce(other) ** -1)

    def __rtruediv__(self, other: Operand) -> "Tensor":
        """other / self (for `number / tensor`)."""
        return  (self ** -1) * other

    def __radd__(self, other: Operand) -> "Tensor":
        """Reflected add so `number + tensor` works."""
        return self + other

    def __rmul__(self, other: Operand) -> "Tensor":
        """Reflected mul so `number * tensor` works."""
        return self * other

    # autodiff
    def backward(self) -> None:
        """Reverse-mode autodiff: topo-sort, seed grad=ones, reverse-walk _backward
        (grads accumulate with `+=` so reused tensors sum their contributions)."""
        
        topo = topo_sort(self)

        self.grad = np.ones(self.data.shape)

        for i in range(len(topo)):
            t = topo[-(1 + i)]
            t._backward()
    

    def zero_grad(self) -> None:
        """Reset this tensor's gradient to zeros (same shape as data)."""
        self.grad = np.zeros(self.data.shape)

    def __repr__(self) -> str:
        """Return 'Tensor(data=..., grad=...)'."""
        return f'Tensor(data={self.data}, grad={self.grad})'
    

def topo_sort(root: "Tensor") -> List["Tensor"]:
    """ Returns Tensor nodes reachable from ``root`` in topological order."""

    order = []
    visited_set = set()

    def build(t: 'Tensor') -> None:

        if t not in visited_set:

            if not t._prev:
                order.append(t)
                visited_set.add(t)

            else:

                for parent in t._prev:
                    build(parent)

                order.append(t)
                visited_set.add(t)

    build(root)

    return order
                