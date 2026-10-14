"""Stage 16: Weight initialization.

Turns the variance-propagation rule Var(z) = n_in * Var(W) * Var(x) into the
Xavier/Glorot (tanh/linear) and He/Kaiming (relu) schemes, plus `init_dense`
(apply an init to a Dense in place) and the `forward_activation_stats` harness
that measures activation statistics across depth. Imports Dense (stage_10) and
Tensor (stage_08) as-is; nothing is redefined.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

# Dense (stage_10) and Tensor (stage_08) via the shared dlfs shim, used as-is.
from dlfs import stage_import

Stage10_Dense = stage_import("stage_10", "Dense")
Stage8_Tensor = stage_import("stage_08", "Tensor")

Dense = Stage10_Dense
Tensor = Stage8_Tensor


def xavier_uniform(
    n_in: int, n_out: int, *, gain: float = 1.0, seed: Optional[int] = None
) -> np.ndarray:
    """Xavier/Glorot uniform init: U[-a, a], a = gain*sqrt(6/(n_in+n_out)). Returns (n_in, n_out)."""
    # TODO: sample U[-a, a] so Var(W) = gain**2 * 2 / (n_in + n_out).
    rng = np.random.default_rng(seed)
    a = gain * np.sqrt(6 / (n_in + n_out))
    W = rng.uniform(-a, a, (n_in, n_out))

    return W


def xavier_normal(
    n_in: int, n_out: int, *, gain: float = 1.0, seed: Optional[int] = None
) -> np.ndarray:
    """Xavier/Glorot normal init: N(0, std**2), std = gain*sqrt(2/(n_in+n_out)). Returns (n_in, n_out)."""
    # TODO: sample N(0, std**2) with std = gain*sqrt(2/(n_in+n_out)).
    rng = np.random.default_rng(seed)
    std = gain * np.sqrt(2/(n_in + n_out))
    W = rng.normal(0, std, (n_in, n_out))

    return W


def he_normal(n_in: int, n_out: int, *, seed: Optional[int] = None) -> np.ndarray:
    """He/Kaiming normal init for ReLU nets: N(0, 2/n_in). Returns (n_in, n_out)."""
    # TODO: sample N(0, 2/n_in) -- numerator 2 accounts for ReLU halving variance.
    
    rng = np.random.default_rng(seed)

    W = rng.normal(0, np.sqrt(2/n_in), (n_in, n_out))

    return W


def he_uniform(n_in: int, n_out: int, *, seed: Optional[int] = None) -> np.ndarray:
    """He/Kaiming uniform init for ReLU nets: U[-a, a], a = sqrt(6/n_in). Returns (n_in, n_out)."""
    # TODO: sample U[-a, a] so Var(W) = 2 / n_in.
    rng = np.random.default_rng(seed)
    a = np.sqrt(6/n_in)
    W = rng.uniform(-a, a, (n_in, n_out))

    return W

def init_dense(layer, W: np.ndarray, b: Optional[np.ndarray] = None) -> None:
    """Overwrite a stage_10 ``Dense`` layer's weights IN PLACE with `W` (and `b`).

    Keep ``layer.W`` (and ``layer.b``) the SAME leaf ``Tensor`` objects — only
    their ``.data`` is replaced — so any optimizer/graph reference keeps
    accumulating into the same ``.grad`` buffers. Reset those ``.grad``s to
    zeros of the new shape. Raise ``ValueError`` on any shape mismatch.
    """
    # TODO: validate shapes; overwrite layer.W.data (and layer.b.data if given);
    #       reset the matching .grad buffers to zeros. Never rebind layer.W/layer.b.
    if W.shape != (layer.n_in, layer.n_out):
        raise ValueError('shape mismatch')

    layer.W.data = W
    layer.W.grad = np.zeros(W.shape)

    if b is not None:
    
        if b.shape != (layer.n_out,):
            raise ValueError('shape mismatch')

        layer.b.data = b
        layer.b.grad = np.zeros(b.shape)

    

def forward_activation_stats(
    sizes: Sequence[int],
    init_fn: Callable[[int, int], np.ndarray],
    activation: str,
    *,
    n_samples: int = 512,
    seed: Optional[int] = None,
) -> List[Dict[str, float]]:
    """Push N(0,1) inputs through a freshly initialized Dense stack; record stats.

    Build ``len(sizes)-1`` stage_10 ``Dense`` layers (layer k maps
    ``sizes[k] -> sizes[k+1]``), init each weight with ``init_fn(n_in, n_out)``
    and zero bias (via ``init_dense``), then push ``n_samples`` standard-normal
    rows through the stack, applying ``activation`` ("tanh" / "relu" / "none")
    after every layer. Return one dict per layer with keys:

      - "mean":      mean of that layer's post-activation output
      - "std":       std of that layer's post-activation output
      - "saturated": fraction with |value| > 0.98 (tanh saturation measure)
      - "dead":      fraction exactly 0.0 (dead-ReLU measure)

    This is the measurement harness that shows WHY the variance-matched inits
    matter: matched init keeps "std" stable across depth; a tiny init collapses
    it toward 0; a large one saturates tanh / kills ReLUs.
    """
    # TODO: build + init the stack, run the forward pass layer by layer
    #       (activation applied on the Tensor output), and collect the stats
    #       from each layer's post-activation .data.
    rng = np.random.default_rng(seed)
    stats = []

    for i in range(len(sizes) - 1):

        if i == 0:
            X = Tensor(rng.standard_normal((n_samples, sizes[i])))

        W = init_fn(sizes[i], sizes[i + 1])

        layer = Dense(sizes[i], sizes[i + 1], seed= seed)
        init_dense(layer, W, np.zeros((sizes[i + 1],)))

        if activation == 'tanh':

            Z_act = layer(X).tanh()

        elif activation == 'relu':

            Z_act = layer(X).relu()

        else:

            Z_act = layer(X)

        X = Z_act

        mean = np.mean(Z_act.data)
        std = np.std(Z_act.data)
        saturated = np.mean((np.abs(Z_act.data) > 0.98).astype(int))
        dead = np.mean((Z_act.data == 0.0).astype(int))

        stats.append({'mean': mean, 'std': std, 'saturated': saturated, 'dead': dead})

    return stats