"""Stage 11: MLP + broadcasting-aware Tensor.

Two things ship here:

* ``MLP`` -- a multilayer perceptron: a chain of pure-linear ``Dense`` layers
  (stage_10) with a nonlinearity applied between them, built on the autodiff
  ``Tensor`` (stage_08).
* ``Tensor`` -- a thin subclass of the stage_08 ``Tensor`` that finally adds the
  **broadcasting-correct backward** stage_08 deferred to here.  The stage_08
  engine restricts elementwise binary ops to EQUAL-SHAPED operands; this stage
  overrides ``__add__``/``__mul__`` so differently-shaped-but-broadcastable
  operands forward via NumPy broadcasting AND each parent's gradient is reduced
  (the "unbroadcast" rule) back to that parent's original shape.  stage_12's
  stable softmax (``logits - logsumexp(..., keepdims=True)``, i.e. ``(B,C)-(B,1)``)
  relies on this.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

# Building blocks from earlier stages (re-exported for tests / later stages).
from dlfs import stage_import

Stage8_Tensor = stage_import("stage_08", "Tensor")
Stage10_Dense = stage_import("stage_10", "Dense")

# An operand the broadcasting binary ops accept: a Tensor, or a raw
# number/array we wrap (mirrors stage_08's ``Operand`` alias).
Operand = Union["Tensor", float, int, np.ndarray, list]


class Tensor(Stage8_Tensor):
    """The stage_08 autodiff ``Tensor`` extended with broadcasting backward.

    stage_08 keeps elementwise binary ops to equal-shaped operands and defers
    "general broadcasting gradient reduction to stage_11" -- that is this class.

    Only the ops that genuinely CHANGE behaviour are overridden here:
    ``__add__`` and ``__mul__`` (broadcasting forward + ``_unbroadcast`` backward).
    Every other node-building op (``__pow__``, ``relu``, ``tanh``, ``exp``,
    ``log``, ``reshape``, ``@``) is INHERITED from stage_08 unchanged: each builds
    its child via ``self._make_tensor(...)``, which is ``type(self)(...)``, so a
    chained graph like ``(x @ W).relu()`` keeps producing stage_11 ``Tensor``
    nodes and this class's broadcasting ``__add__`` stays reachable for any later
    op keyed on the node's type. Derived ops (``__sub__``, ``__truediv__``,
    ``__neg__`` and the reflected forms) compose the primitives, so they too stay
    stage_11 with no new ``_backward``.
    """
    def _coerce(self, other: Operand) -> "Tensor":
        """Return `other` as a Tensor (wrap raw numbers/arrays; pass Tensors through).

        Wrap via ``self._make_tensor(...)`` (== ``type(self)(...)``) so a coerced raw
        operand becomes THIS instance's runtime class, keeping a subclass alive across
        the chain (same reason the node-building ops route through ``_make_tensor``)."""
        if isinstance(other, Tensor):
            return other
        
        return self._make_tensor(other)

    @staticmethod
    def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        """Sum a broadcasted ``grad`` back to an operand's original ``shape``.

        Forward broadcast *copies* an operand to fill the output; chain rule says
        a copied value's grad is the SUM over its copies. So undo a broadcast by
        summing the copied axes away (stage_08's ``+=`` would shape-mismatch /
        be wrong without this). Two cases compose:
          1. RANK PROMOTION (e.g. bias ``(N,)`` -> ``(B,N)``): sum extra leading
             axes -- ``while grad.ndim > len(shape): grad = grad.sum(axis=0)``.
          2. SIZE-1 STRETCH (e.g. softmax ``(B,1)`` -> ``(B,C)``): sum each
             stretched size-1 axis with ``keepdims=True``.
        Then ``reshape(shape)`` (no-op after 1&2). Equal shapes pass through.
        E.g. ``(2,3)+(3,)``: ``a`` gets ``grad`` as-is, ``b`` gets ``grad.sum(axis=0)``.
        """
        # TODO: implement the unbroadcast reduction described above:
        #   - while grad.ndim > len(shape): grad = grad.sum(axis=0)
        #   - for each axis i where shape[i] == 1 and grad.shape[i] > 1:
        #         grad = grad.sum(axis=i, keepdims=True)
        #   - return grad.reshape(shape)

        while grad.ndim > len(shape):
            grad = grad.sum(axis= 0)

        for i in range(len(shape)):
            if shape[i] == 1 and grad.shape[i] > 1:
                grad = grad.sum(axis= i, keepdims= True)

        return grad.reshape(shape)

    def __add__(self, other: "Operand") -> "Tensor":
        """Broadcasting elementwise add: ``z = self + other``.

        Forward via NumPy broadcasting (``self.data + other.data``); in
        ``_backward`` push ``self._unbroadcast(out.grad, self.shape)`` to ``self``
        and ``self._unbroadcast(out.grad, other.shape)`` to ``other`` so each
        parent's grad is reduced back to its own shape.  Equal-shaped operands
        reduce to the stage_08 behaviour (unbroadcast is then a no-op)."""
        # TODO: coerce other; out = self._make_tensor(self.data + other.data, (self, other), "+");
        #       _backward: each parent.grad += _unbroadcast(out.grad, parent.shape).
        other = self._coerce(other)

        out = self._make_tensor(self.data + other.data, (self, other), '+')

        def _backward():
            Tensor._accumulate(self, Tensor._unbroadcast(out.grad, self.shape))
            Tensor._accumulate(other, Tensor._unbroadcast(out.grad, other.shape))

        out._backward = _backward

        return out

    def __mul__(self, other: "Operand") -> "Tensor":
        """Broadcasting elementwise multiply: ``z = self * other``.

        Forward via NumPy broadcasting (``self.data * other.data``); local grads
        are ``g * other`` for ``self`` and ``g * self`` for ``other`` (each
        evaluated at the BROADCAST shape), then unbroadcast back to each parent's
        original shape before accumulating."""
        # TODO: coerce other; out = self._make_tensor(self.data * other.data, (self, other), "*");
        #       _backward: self.grad  += _unbroadcast(out.grad * other.data, self.shape)
        #                  other.grad += _unbroadcast(out.grad * self.data,  other.shape).
        other = self._coerce(other)

        out = self._make_tensor(self.data * other.data, (self, other), '*')

        def _backward():
            Tensor._accumulate(self, Tensor._unbroadcast(out.grad * other.data, self.shape))
            Tensor._accumulate(other, Tensor._unbroadcast(out.grad * self.data, other.shape))

        out._backward = _backward

        return out

    # NOTE: ``__pow__``, ``relu``, ``tanh``, ``exp``, ``log``, ``reshape`` and
    # ``__matmul__`` are NOT overridden here. stage_08's versions build their
    # child via ``self._make_tensor(...)`` (== ``type(self)(...)``), so when
    # called on a stage_11 instance they already return stage_11 ``Tensor`` nodes
    # -- the subclass survives a chained graph (e.g. ``(x @ W).relu()``) with no
    # re-implementation needed. Only the broadcasting ``__add__``/``__mul__``
    # above genuinely change behaviour, so only they are overridden.


# ``Tensor`` (above) is this stage's public broadcasting-capable autodiff node.
# stage_12's stable softmax needs ``(B,C) - (B,1)`` to backprop correctly; that
# broadcasting backward is delivered HERE, satisfying stage_12's reliance even
# though stage_12 keeps importing the engine via ``stage_import("stage_08",
# "Tensor")`` (its construction sites just feed differently-shaped operands).


class Dense(Stage10_Dense):
    """stage_10 ``Dense`` re-expressed on this stage's broadcasting ``Tensor``.

    stage_10 had no broadcasting backward, so it faked the bias add with the
    matmul trick ``z += ones((B,1)) @ b.reshape(1, n_out)``.  This stage's
    ``Tensor.__add__`` unbroadcasts, so the bias add is just ``z + b`` over a
    ``(B, n_out) + (n_out,)`` broadcast -- the grad reduces back to ``(n_out,)``.

    Params (``W``, ``b``) are rebuilt as this stage's ``Tensor`` so ``z + b``
    routes through the broadcasting ``__add__``.  ``z = x @ W`` is built by the
    inherited ``__matmul__``, which constructs its child via ``self._make_tensor``
    (== ``type(self)``); since ``W`` is a stage_11 ``Tensor``, ``z`` is one too,
    so plain ``z + b`` already keys on this stage's broadcasting ``__add__`` --
    no unbound-call trick needed.  ``parameters``/``zero_grad``/``n_in``/``n_out``
    are inherited as-is.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        bias: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        # TODO: build W (n_in, n_out) and, if bias, b (n_out,) as THIS stage's
        #       broadcasting Tensor (stage_10 builds them as the stage_08 engine,
        #       whose add can't broadcast the bias row).
        rng = np.random.default_rng(seed= seed)

        self.W = Tensor(rng.uniform(low= -1, high= 1, size= (n_in, n_out)))
        self.b = Tensor(np.zeros((n_out,))) if bias else None
        self.n_in = n_in
        self.n_out = n_out
        self.bias = bias

    def __call__(self, x) -> "Tensor":
        """Forward affine pass; ``(n_in,) -> (n_out,)`` or ``(B, n_in) -> (B, n_out)``.

        Bias add is now a plain broadcast: ``(B, n_out) + (n_out,)`` for a batch,
        ``(n_out,) + (n_out,)`` for a single input -- both handled by this stage's
        unbroadcasting ``Tensor.__add__``."""
        # TODO: z = x @ self.W; if bias, add it with plain ``z + self.b``. The
        #       inherited __matmul__ builds z via self._make_tensor, so z is a
        #       stage_11 Tensor and ``z + b`` keys on this stage's broadcasting
        #       __add__ (no unbound-call trick needed).
        z = x @ self.W

        if self.bias:
            return z + self.b

        else:
            return z

class MLP:
    """A multilayer perceptron: ``Dense`` layers + activations. sizes
    ``[n_in, ..., n_out]`` builds len(sizes)-1 Dense layers; activation follows
    each hidden layer, out_activation the last (each in {"tanh","relu","none"})."""
    def __init__(
        self,
        sizes: Sequence[int],
        activation: str = "tanh",
        out_activation: str = "none",
        seed: Optional[int] = None,
    ) -> None:
        # TODO: validate args; build the Dense layers (per-layer derived seeds).
        assert activation in {'tanh', 'relu', 'none'}
        assert out_activation in {'tanh', 'relu', 'none'}
        assert len(sizes) >= 2
        self.sizes = sizes
        self.layers = [Dense(sizes[i], sizes[i + 1], seed= seed + i) for i in range(len(sizes) -1)]
        self.activation = activation
        self.out_activation = out_activation
        

    @staticmethod
    def _apply_activation(z: "Stage8_Tensor", name: str) -> "Stage8_Tensor":
        """Apply named pointwise activation via the Tensor's own methods; raise on unknown name."""
        # TODO: dispatch "none"/"tanh"/"relu" to z / z.tanh() / z.relu().
        if name not in {'tanh', 'relu', 'none'}:
            raise ValueError

        if name == 'none':
            return z
        elif name == 'tanh':
            return z.tanh()
        else:
            return z.relu()

    def forward(self, x: "Stage8_Tensor") -> "Stage8_Tensor":
        """Chain layers, applying activation after each (out_activation after the
        last). x ``(n_in,)`` or ``(batch, n_in)`` -> ``(n_out,)`` / ``(batch, n_out)``."""
        # TODO: chain layers with the right activation per layer.
        if not isinstance(x, Tensor):
            raise TypeError
        z = self._apply_activation(self.layers[0](x), self.activation)

        for i in range(1,len(self.layers)):
            if i == len(self.layers) - 1:
                z = self._apply_activation(self.layers[i](z), self.out_activation)
            else:
                z = self._apply_activation(self.layers[i](z), self.activation)

        return z

    def __call__(self, x: "Stage8_Tensor") -> "Stage8_Tensor":
        """Alias for :meth:`forward`."""
        # TODO: delegate to forward.
        if not isinstance(x, Tensor):
            raise TypeError
        return self.forward(x)

    def parameters(self) -> List["Stage8_Tensor"]:
        """Return every learnable parameter from every layer, flattened in layer order."""
        # TODO: flatten each layer's parameters().
        params = []
        
        for layer in self.layers:
            params += layer.parameters()

        return params

    def zero_grad(self) -> None:
        """Reset the gradient of every parameter to zeros."""
        # TODO: zero each parameter's grad.
        for layer in self.layers:
            layer.zero_grad()

    def __repr__(self) -> str:
        # TODO: summarize sizes and activations.
        return f'MLP({self.sizes}, activation={self.activation}, out_activation={self.out_activation})'
