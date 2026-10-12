"""Tests for Stage 12: Loss functions (single example -- no batch axis).

softmax / log_softmax / cross_entropy here take ONE example: a (C,) logit vector and
a single label. The batched (B, C) versions are tested in stage_13.

Checks forward values against plain-NumPy reference losses and gradient-checks
each loss with respect to its (differentiable) prediction / logits using central
differences:

    df/dp ~= (f(p + eps) - f(p - eps)) / (2 * eps)

compared against the analytic gradient produced by ``Tensor.backward()`` from
stage_08. The losses live in this stage's ``code.py`` and are built on the
``Tensor`` imported from stage_11 through ``dlfs.stage_import``. If stage_11's
``Tensor`` or this stage's losses are not implemented yet, the suite skips
cleanly instead of erroring.

Run with:  pytest stage_12_loss_functions/test.py
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
# `code.py` re-exports `Tensor` (from stage_11 via dlfs.stage_import) and defines
# this stage's loss functions on top of it.
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
        mae_loss,
        mse_loss,
        softmax,
    )
except (ImportError, NotImplementedError) as exc:  # pragma: no cover
    pytest.skip(
        f"stage_12 losses / stage_11 Tensor not importable yet: {exc}",
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


def _onehot_vec(t, num_classes):
    """Test-local one-hot for ONE label: int t -> (C,) vector with 1.0 at t."""
    y = np.zeros(num_classes, dtype=float)
    y[int(t)] = 1.0
    return y


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
# sum / mean reductions (added to the Tensor in this stage)
# =========================================================================== #
def test_sum_all_forward_and_backward():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    s = x.sum()
    assert np.isclose(loss_value(s), 10.0), f"sum() forward: {loss_value(s)}"
    s.backward()
    g = as_array(x.grad)
    assert g.shape == (2, 2), f"grad shape: {g.shape}"
    assert np.allclose(g, np.ones((2, 2))), f"sum() grad should be all ones, got\n{g}"


def test_sum_axis0_forward_and_backward():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    s = x.sum(axis=0)
    assert as_array(s).shape == (2,), f"sum(axis=0) shape: {as_array(s).shape}"
    assert np.allclose(as_array(s), [4.0, 6.0]), f"sum(axis=0) values: {as_array(s)}"
    # Sum the column-sums to a scalar so backward seeds cleanly.
    s.sum().backward()
    g = as_array(x.grad)
    assert g.shape == (2, 2), f"grad shape: {g.shape}"
    assert np.allclose(g, np.ones((2, 2))), f"sum(axis=0) grad should be ones, got\n{g}"


def test_mean_all_forward_and_backward():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    m = x.mean()
    assert np.isclose(loss_value(m), 2.5), f"mean() forward: {loss_value(m)}"
    m.backward()
    g = as_array(x.grad)
    assert g.shape == (2, 2), f"grad shape: {g.shape}"
    assert np.allclose(g, np.full((2, 2), 0.25)), f"mean() grad should be 1/N, got\n{g}"


def test_mean_axis_keepdims_forward():
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    m = x.mean(axis=1, keepdims=True)
    assert as_array(m).shape == (2, 1), f"mean(axis=1, keepdims) shape: {as_array(m).shape}"
    assert np.allclose(as_array(m), [[2.0], [5.0]]), f"values: {as_array(m)}"


def test_sum_axis1_no_keepdims_backward():
    # keepdims=False drops the reduced axis: (B, C) -> (B,). The backward must
    # restore that axis before broadcasting the grad back — `grad += out.grad`
    # without re-expanding either crashes or adds along the WRONG axis. This is
    # the exact path stage_13's batched cross-entropy takes: .sum(axis=1).mean().
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3): axis sizes differ on purpose
    s = x.sum(axis=1)
    assert as_array(s).shape == (2,), f"sum(axis=1) shape: {as_array(s).shape}"
    assert np.allclose(as_array(s), [6.0, 15.0]), f"sum(axis=1) values: {as_array(s)}"
    s.mean().backward()  # d/dx of mean over rows of row-sums = 1/2 everywhere
    g = as_array(x.grad)
    assert g.shape == (2, 3), f"grad shape: {g.shape}"
    assert np.allclose(g, np.full((2, 3), 0.5)), (
        f"sum(axis=1) grad must re-expand the dropped axis, got\n{g}"
    )


def test_mean_axis1_no_keepdims_backward():
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
    m = x.mean(axis=1)
    assert as_array(m).shape == (2,), f"mean(axis=1) shape: {as_array(m).shape}"
    m.sum().backward()  # each entry contributes 1/3 to its row mean
    g = as_array(x.grad)
    assert g.shape == (2, 3), f"grad shape: {g.shape}"
    assert np.allclose(g, np.full((2, 3), 1.0 / 3.0)), (
        f"mean(axis=1) grad should be 1/3 everywhere, got\n{g}"
    )


def test_sum_gradcheck_against_central_diff():
    """f = (x*x).sum() has analytic grad 2x; check it matches central differences."""
    x_np = RNG.normal(size=(3, 4))

    x = Tensor(x_np.copy())
    if hasattr(x, "zero_grad"):
        x.zero_grad()
    L = (x * x).sum()
    L.backward()
    g_analytic = as_array(x.grad)

    def f(p):
        return loss_value((Tensor(p) * Tensor(p)).sum())

    g_num = central_diff(f, x_np.copy())
    assert np.allclose(g_analytic, 2.0 * x_np, atol=ATOL), (
        f"analytic grad != 2x:\n analytic={g_analytic}\n 2x={2.0 * x_np}"
    )
    assert np.allclose(g_analytic, g_num, atol=ATOL, rtol=RTOL), (
        f"sum gradcheck mismatch:\n analytic={g_analytic}\n numeric ={g_num}"
    )


def test_mean_gradcheck_against_central_diff():
    """f = (x*x).mean() has analytic grad 2x/N; check vs central differences."""
    x_np = RNG.normal(size=(2, 5))
    N = x_np.size

    x = Tensor(x_np.copy())
    if hasattr(x, "zero_grad"):
        x.zero_grad()
    L = (x * x).mean()
    L.backward()
    g_analytic = as_array(x.grad)

    def f(p):
        return loss_value((Tensor(p) * Tensor(p)).mean())

    g_num = central_diff(f, x_np.copy())
    assert np.allclose(g_analytic, 2.0 * x_np / N, atol=ATOL), (
        f"analytic grad != 2x/N:\n analytic={g_analytic}\n 2x/N={2.0 * x_np / N}"
    )
    assert np.allclose(g_analytic, g_num, atol=ATOL, rtol=RTOL), (
        f"mean gradcheck mismatch:\n analytic={g_analytic}\n numeric ={g_num}"
    )


# =========================================================================== #
# MSE
# =========================================================================== #
def test_mse_forward_matches_numpy():
    pred = np.array([1.0, 2.0, 3.0, 4.0])
    targ = np.array([1.5, 1.0, 3.5, 2.0])
    got = loss_value(mse_loss(Tensor(pred), Tensor(targ)))
    ref = float(np.mean((pred - targ) ** 2))
    assert np.isclose(got, ref, atol=ATOL), f"MSE forward: got {got}, want {ref}"


def test_mse_zero_when_equal():
    a = np.array([[1.0, -2.0], [3.0, 4.0]])
    assert np.isclose(loss_value(mse_loss(Tensor(a), Tensor(a))), 0.0, atol=ATOL)


def test_mse_gradcheck():
    pred_np = RNG.normal(size=(3, 4))
    targ_np = RNG.normal(size=(3, 4))

    g_analytic = analytic_grad(mse_loss, pred_np, Tensor(targ_np))

    def f(p):
        return loss_value(mse_loss(Tensor(p), Tensor(targ_np)))

    g_num = central_diff(f, pred_np.copy())
    assert np.allclose(g_analytic, g_num, atol=ATOL, rtol=RTOL), (
        f"MSE gradcheck mismatch:\n analytic={g_analytic}\n numeric ={g_num}"
    )


def test_mse_grad_formula():
    """dL/dpred should equal 2*(pred-target)/N."""
    pred_np = np.array([0.5, -1.0, 2.0, 0.0])
    targ_np = np.array([1.0, 0.0, 1.0, -1.0])
    g = analytic_grad(mse_loss, pred_np, Tensor(targ_np))
    expected = 2.0 * (pred_np - targ_np) / pred_np.size
    assert np.allclose(g, expected, atol=ATOL), f"got {g}, want {expected}"


# =========================================================================== #
# MAE
# =========================================================================== #
def test_mae_forward_matches_numpy():
    pred = np.array([1.0, 2.0, 3.0, 4.0])
    targ = np.array([1.5, 1.0, 3.5, 2.0])
    got = loss_value(mae_loss(Tensor(pred), Tensor(targ)))
    ref = float(np.mean(np.abs(pred - targ)))
    assert np.isclose(got, ref, atol=ATOL), f"MAE forward: got {got}, want {ref}"


def test_mae_gradcheck_away_from_kink():
    # Keep pred - target well away from 0 so |.| is differentiable everywhere.
    targ_np = RNG.normal(size=(3, 4))
    pred_np = targ_np + RNG.choice([-1.0, 1.0], size=(3, 4)) * (
        1.0 + np.abs(RNG.normal(size=(3, 4)))
    )

    g_analytic = analytic_grad(mae_loss, pred_np, Tensor(targ_np))

    def f(p):
        return loss_value(mae_loss(Tensor(p), Tensor(targ_np)))

    g_num = central_diff(f, pred_np.copy())
    assert np.allclose(g_analytic, g_num, atol=1e-5, rtol=RTOL), (
        f"MAE gradcheck mismatch:\n analytic={g_analytic}\n numeric ={g_num}"
    )


def test_mae_grad_is_sign_over_n():
    pred_np = np.array([2.0, -3.0, 0.5, 5.0])
    targ_np = np.array([0.0, 0.0, 0.0, 0.0])
    g = analytic_grad(mae_loss, pred_np, Tensor(targ_np))
    expected = np.sign(pred_np - targ_np) / pred_np.size
    assert np.allclose(g, expected, atol=ATOL), f"got {g}, want {expected}"


# =========================================================================== #
# softmax / log-softmax  (single example: (C,) -> (C,))
# =========================================================================== #
def test_softmax_sums_to_one():
    z = RNG.normal(size=7)
    p = as_array(softmax(Tensor(z)))
    assert np.isclose(p.sum(), 1.0, atol=ATOL), f"sum: {p.sum()}"
    assert np.all(p >= 0.0), "softmax must be non-negative"


def test_softmax_matches_numpy_reference():
    z = RNG.normal(size=6)
    p = as_array(softmax(Tensor(z)))
    e = np.exp(z - z.max())
    ref = e / e.sum()
    assert np.allclose(p, ref, atol=ATOL), f"softmax mismatch:\n{p}\n{ref}"


def test_log_softmax_matches_log_of_softmax():
    z = RNG.normal(size=5)
    lp = as_array(log_softmax(Tensor(z)))
    ref = np.log(as_array(softmax(Tensor(z))))
    assert np.allclose(lp, ref, atol=ATOL), f"log_softmax mismatch:\n{lp}\n{ref}"


def test_softmax_numerically_stable_large_logits():
    z = np.array([1000.0, 1001.0, 1002.0])
    p = as_array(softmax(Tensor(z)))
    assert np.all(np.isfinite(p)), "softmax overflowed on large logits"
    e = np.exp(z - z.max())
    ref = e / e.sum()
    assert np.allclose(p, ref, atol=ATOL)


def test_log_softmax_stable_large_logits():
    z = np.array([1000.0, 1001.0, 1002.0])
    lp = as_array(log_softmax(Tensor(z)))
    assert np.all(np.isfinite(lp)), "log_softmax produced inf/nan on large logits"


# =========================================================================== #
# cross-entropy  (single example: (C,) logits + one label -> scalar)
# =========================================================================== #
def test_cross_entropy_forward_matches_numpy():
    z = RNG.normal(size=5)
    t = 2
    got = loss_value(cross_entropy_loss(Tensor(z), t))
    e = np.exp(z - z.max())
    p = e / e.sum()
    ref = float(-np.log(p[t]))
    assert np.isclose(got, ref, atol=ATOL), f"CE forward: got {got}, want {ref}"


def test_cross_entropy_accepts_onehot_target():
    z = RNG.normal(size=4)
    t_idx = 3
    t_oh = _onehot_vec(t_idx, 4)
    a = loss_value(cross_entropy_loss(Tensor(z), t_idx))
    b = loss_value(cross_entropy_loss(Tensor(z), t_oh))
    assert np.isclose(a, b, atol=ATOL), f"index vs one-hot CE differ: {a} vs {b}"


def test_cross_entropy_stable_large_logits():
    z = np.array([1000.0, 1001.0, 1002.0])
    t = 2
    L = loss_value(cross_entropy_loss(Tensor(z), t))
    assert np.isfinite(L), "cross-entropy overflowed on large logits"


def test_cross_entropy_gradcheck_wrt_logits():
    z_np = RNG.normal(size=5)
    t = 2

    g_analytic = analytic_grad(cross_entropy_loss, z_np, t)

    def f(z):
        return loss_value(cross_entropy_loss(Tensor(z), t))

    g_num = central_diff(f, z_np.copy())
    assert np.allclose(g_analytic, g_num, atol=ATOL, rtol=RTOL), (
        f"CE gradcheck mismatch:\n analytic={g_analytic}\n numeric ={g_num}"
    )


def test_cross_entropy_grad_is_p_minus_y():
    """For one example the gradient w.r.t. logits is exactly softmax(logits) - onehot."""
    z_np = RNG.normal(size=5)
    t = 2

    g_analytic = analytic_grad(cross_entropy_loss, z_np, t)

    p = as_array(softmax(Tensor(z_np)))
    y = _onehot_vec(t, z_np.shape[0])
    expected = p - y

    assert np.allclose(g_analytic, expected, atol=ATOL), (
        f"CE grad != (p - y):\n analytic={g_analytic}\n expected={expected}"
    )

if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))