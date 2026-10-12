"""Tests for Stage 13: Batched loss functions ((B, C) logits).

softmax / log_softmax / cross_entropy here operate on a whole batch: a (B, C) logit
matrix and B labels. The single-example ((C,)) versions are tested in stage_12; this
suite checks the per-row generalization plus the batch-mean of cross-entropy.

Forward values are checked against plain-NumPy references and each loss is
gradient-checked w.r.t. its logits with central differences, compared to the analytic
gradient from ``Tensor.backward()``. mse_loss / mae_loss are re-exported from stage_12
and not re-tested here. If stage_13 (or the stage_12 it builds on) is not implemented
yet, the suite skips cleanly instead of erroring.

Run with:  pytest stage_13_loss_functions_batched/test.py
"""
import os as _os
import sys as _sys

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
# Make the shared `dlfs` shim importable (it lives at the curriculum root).
sys.path.insert(0, _ROOT)

# --- Import the things under test, skipping cleanly if not ready yet. --------
try:
    # --- resolve sibling code.py (avoid stdlib `code` collision) ---
    import importlib.util as _ilu
    _THIS_DIR = _os.path.dirname(_os.path.abspath(__file__))
    _ROOT = _os.path.dirname(_THIS_DIR)
    if _ROOT not in _sys.path:
        _sys.path.insert(0, _ROOT)
    _spec = _ilu.spec_from_file_location(
        "code", _os.path.join(_THIS_DIR, "code.py")
    )
    _mod = _ilu.module_from_spec(_spec)
    _sys.modules["code"] = _mod
    _spec.loader.exec_module(_mod)
    from code import (
        Tensor,
        cross_entropy_loss,
        log_softmax,
        softmax,
    )
except (ImportError, NotImplementedError) as exc:  # pragma: no cover
    pytest.skip(
        f"stage_13 batched losses / stage_12 not importable yet: {exc}",
        allow_module_level=True,
    )

RNG = np.random.default_rng(13)
EPS = 1e-6
ATOL = 1e-6
RTOL = 1e-4


# --- Small helpers -----------------------------------------------------------
def as_array(t):
    """Underlying ndarray of a Tensor (or pass arrays through)."""
    return np.asarray(t.data if hasattr(t, "data") else t, dtype=float)


def loss_value(t):
    """Scalar python float held by a (0-d) loss Tensor."""
    return float(as_array(t).reshape(-1)[0]) if as_array(t).size == 1 else float(np.sum(as_array(t)))


def _onehot(targets, num_classes):
    """Test-local one-hot: (B,) int -> (B, C)."""
    t = np.asarray(targets, dtype=int)
    oh = np.zeros((t.shape[0], num_classes), dtype=float)
    oh[np.arange(t.shape[0]), t] = 1.0
    return oh


def central_diff(f, x, eps=EPS):
    """Numerical gradient of scalar-valued f at numpy point x (any shape)."""
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        fp = f(x)
        x[idx] = orig - eps
        fm = f(x)
        x[idx] = orig
        grad[idx] = (fp - fm) / (2 * eps)
        it.iternext()
    return grad


def analytic_grad(loss_fn, pred_np, *rest):
    """Build the loss from a fresh prediction Tensor, backprop, return its grad."""
    p = Tensor(pred_np.copy())
    if hasattr(p, "zero_grad"):
        p.zero_grad()
    L = loss_fn(p, *rest)
    L.backward()
    return as_array(p.grad)


# =========================================================================== #
# softmax / log-softmax  (batched: (B, C) -> (B, C), per row)
# =========================================================================== #
def test_softmax_rows_sum_to_one():
    z = RNG.normal(size=(5, 7))
    p = as_array(softmax(Tensor(z)))
    assert np.allclose(p.sum(axis=1), 1.0, atol=ATOL), f"rows sum: {p.sum(axis=1)}"
    assert np.all(p >= 0.0), "softmax must be non-negative"


def test_softmax_matches_numpy_reference():
    z = RNG.normal(size=(4, 6))
    p = as_array(softmax(Tensor(z)))
    e = np.exp(z - z.max(axis=1, keepdims=True))
    ref = e / e.sum(axis=1, keepdims=True)
    assert np.allclose(p, ref, atol=ATOL), f"softmax mismatch:\n{p}\n{ref}"


def test_log_softmax_matches_log_of_softmax():
    z = RNG.normal(size=(4, 5))
    lp = as_array(log_softmax(Tensor(z)))
    ref = np.log(as_array(softmax(Tensor(z))))
    assert np.allclose(lp, ref, atol=ATOL), f"log_softmax mismatch:\n{lp}\n{ref}"


def test_softmax_numerically_stable_large_logits():
    z = np.array([[1000.0, 1001.0, 1002.0]])
    p = as_array(softmax(Tensor(z)))
    assert np.all(np.isfinite(p)), "softmax overflowed on large logits"
    e = np.exp(z - z.max(axis=1, keepdims=True))
    ref = e / e.sum(axis=1, keepdims=True)
    assert np.allclose(p, ref, atol=ATOL)


def test_log_softmax_stable_large_logits():
    z = np.array([[1000.0, 1001.0, 1002.0]])
    lp = as_array(log_softmax(Tensor(z)))
    assert np.all(np.isfinite(lp)), "log_softmax produced inf/nan on large logits"


# =========================================================================== #
# cross-entropy  (batched: (B, C) logits + B labels -> scalar mean)
# =========================================================================== #
def test_cross_entropy_forward_matches_numpy():
    z = RNG.normal(size=(4, 5))
    t = np.array([0, 2, 4, 1])
    got = loss_value(cross_entropy_loss(Tensor(z), t))
    e = np.exp(z - z.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    ref = float(np.mean(-np.log(p[np.arange(4), t])))
    assert np.isclose(got, ref, atol=ATOL), f"CE forward: got {got}, want {ref}"


def test_cross_entropy_accepts_onehot_targets():
    z = RNG.normal(size=(3, 4))
    t_idx = np.array([0, 3, 1])
    t_oh = _onehot(t_idx, 4)
    a = loss_value(cross_entropy_loss(Tensor(z), t_idx))
    b = loss_value(cross_entropy_loss(Tensor(z), t_oh))
    assert np.isclose(a, b, atol=ATOL), f"index vs one-hot CE differ: {a} vs {b}"


def test_cross_entropy_stable_large_logits():
    z = np.array([[1000.0, 1001.0, 1002.0]])
    t = np.array([2])
    L = loss_value(cross_entropy_loss(Tensor(z), t))
    assert np.isfinite(L), "cross-entropy overflowed on large logits"


def test_cross_entropy_gradcheck_wrt_logits():
    z_np = RNG.normal(size=(4, 5))
    t = np.array([0, 2, 4, 1])

    g_analytic = analytic_grad(cross_entropy_loss, z_np, t)

    def f(z):
        return loss_value(cross_entropy_loss(Tensor(z), t))

    g_num = central_diff(f, z_np.copy())
    assert np.allclose(g_analytic, g_num, atol=ATOL, rtol=RTOL), (
        f"CE gradcheck mismatch:\n analytic={g_analytic}\n numeric ={g_num}"
    )


def test_cross_entropy_grad_is_p_minus_y_over_b():
    """The fused batched gradient must equal (softmax(logits) - onehot) / B."""
    z_np = RNG.normal(size=(4, 5))
    t = np.array([0, 2, 4, 1])
    B = z_np.shape[0]

    g_analytic = analytic_grad(cross_entropy_loss, z_np, t)

    p = as_array(softmax(Tensor(z_np)))
    y = _onehot(t, z_np.shape[1])
    expected = (p - y) / B

    assert np.allclose(g_analytic, expected, atol=ATOL), (
        f"CE grad != (p - y)/B:\n analytic={g_analytic}\n expected={expected}"
    )

if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))