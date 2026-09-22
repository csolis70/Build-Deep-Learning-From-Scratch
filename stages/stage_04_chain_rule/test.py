"""Tests for Stage 4: The backward pass.

``backward()`` runs reverse-mode autodiff: topo-sort, seed the output grad to 1,
walk in reverse calling each node's ``_backward``. After ``out.backward()`` every
node's ``.grad`` is ``d(out)/d(node)``. We check the canonical micrograd cases
(accumulation over reused nodes, shared subexpressions) and gradcheck against
central differences.

Run: pytest stage_04_chain_rule/test.py
"""
import os as _os
import sys as _sys

import math

import pytest

# --- resolve sibling code.py (avoid stdlib `code` collision) ---
import importlib.util as _ilu
_THIS_DIR = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_THIS_DIR)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_spec = _ilu.spec_from_file_location("code", _os.path.join(_THIS_DIR, "code.py"))
_mod = _ilu.module_from_spec(_spec)
_sys.modules["code"] = _mod
_spec.loader.exec_module(_mod)
from code import Value, topo_sort

TOL = 1e-5
EPS = 1e-6


def central_diff(f, x, eps=EPS):
    """Numerical derivative of scalar f at scalar x via central differences."""
    return (f(x + eps) - f(x - eps)) / (2.0 * eps)


# ---------------------------------------------------------------------------
# topo_sort: dependencies before dependents, each node once
# ---------------------------------------------------------------------------
def test_topo_sort_orders_parents_before_children():
    a, b = Value(2.0), Value(3.0)
    c = a * b
    d = c + a
    order = topo_sort(d)
    assert set(order) == {a, b, c, d}, "topo must include every reachable node"
    assert len(order) == len(set(order)), "each node exactly once"
    pos = {v: i for i, v in enumerate(order)}
    assert pos[a] < pos[c] and pos[b] < pos[c], "parents of c before c"
    assert pos[c] < pos[d] and pos[a] < pos[d], "parents of d before d"


def test_topo_sort_terminates_on_reuse():
    a = Value(3.0)
    out = a * a
    order = topo_sort(out)
    assert set(order) == {a, out}
    assert len(order) == 2


# ---------------------------------------------------------------------------
# backward: seeds output grad = 1
# ---------------------------------------------------------------------------
def test_backward_seeds_output_with_one():
    a, b = Value(2.0), Value(3.0)
    out = a * b
    out.backward()
    assert out.grad == pytest.approx(1.0)


def test_backward_simple_mul():
    a, b = Value(2.0), Value(3.0)
    out = a * b
    out.backward()
    assert a.grad == pytest.approx(3.0)   # d(ab)/da = b
    assert b.grad == pytest.approx(2.0)   # d(ab)/db = a


# ---------------------------------------------------------------------------
# the canonical micrograd accumulation case
#   a=2, b=3, c=a*b, d=c+a, d.backward()  ->  a.grad==4, b.grad==2
# ---------------------------------------------------------------------------
def test_backward_accumulates_over_two_paths():
    a, b = Value(2.0), Value(3.0)
    c = a * b
    d = c + a
    d.backward()
    assert a.grad == pytest.approx(4.0), "a reaches d via c (b=3) and directly (1) -> 4"
    assert b.grad == pytest.approx(2.0)


def test_backward_self_mul():
    a = Value(3.0)
    out = a * a            # d/da (a^2) = 2a = 6
    out.backward()
    assert a.grad == pytest.approx(6.0)


def test_backward_shared_subexpression():
    a = Value(3.0)
    b = a + 1.0            # 4
    c = a * 2.0            # 6
    out = b * c            # 24 ; d/da = (a+1)*2 + 2a = 4a+2 = 14
    out.backward()
    assert a.grad == pytest.approx(14.0)


# ---------------------------------------------------------------------------
# gradcheck a composite expression against central differences
#   f(x) = (x*x + 1) / (x - 4)     (uses + * ** / and a constant)
# ---------------------------------------------------------------------------
def _f(x):
    return (x * x + 1.0) / (x - 4.0)


@pytest.mark.parametrize("x0", [-2.0, -0.5, 0.0, 1.3, 2.7])
def test_backward_gradcheck_composite(x0):
    x = Value(x0)
    out = (x * x + 1.0) / (x - 4.0)
    out.backward()
    num = central_diff(_f, x0)
    assert x.grad == pytest.approx(num, abs=TOL, rel=1e-4), (
        f"backward dy/dx at x={x0}: analytic={x.grad}, numeric={num}"
    )


@pytest.mark.parametrize("x0", [0.5, 1.0, 2.0, 3.5])
def test_backward_gradcheck_with_pow(x0):
    # f(x) = x**3 - 2*x   ->  f'(x) = 3x^2 - 2
    x = Value(x0)
    out = x ** 3 - 2.0 * x
    out.backward()
    assert x.grad == pytest.approx(3.0 * x0 ** 2 - 2.0, abs=TOL, rel=1e-4)


# ---------------------------------------------------------------------------
# does not zero grads first (accumulates across calls if not reset)
# ---------------------------------------------------------------------------
def test_backward_does_not_zero_existing_grad():
    a, b = Value(2.0), Value(3.0)
    out = a * b
    out.backward()
    first = a.grad
    out.backward()  # second pass, no zeroing
    assert a.grad == pytest.approx(2.0 * first), "grad accumulates across backward calls"

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
