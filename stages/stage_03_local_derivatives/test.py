"""Tests for Stage 3: Local derivatives as `_backward` closures.

Each op installs a ``_backward`` closure on the result node. There is no global
``backward()`` yet (that is stage_04), so we test a closure in isolation: build a
result node, seed its ``.grad`` by hand, call ``node._backward()`` once, and
check the local derivative landed on each operand's ``.grad``.

Run: pytest stage_03_local_derivatives/test.py
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
from code import Value

TOL = 1e-6


# ---------------------------------------------------------------------------
# add: a.grad += g ; b.grad += g
# ---------------------------------------------------------------------------
def test_add_backward_local():
    a, b = Value(2.0), Value(3.0)
    out = a + b
    assert out.data == 5.0
    out.grad = 1.0
    out._backward()
    assert a.grad == pytest.approx(1.0)
    assert b.grad == pytest.approx(1.0)


def test_add_backward_scales_with_output_grad():
    a, b = Value(2.0), Value(3.0)
    out = a + b
    out.grad = 7.0          # upstream gradient
    out._backward()
    assert a.grad == pytest.approx(7.0)
    assert b.grad == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# mul: a.grad += b*g ; b.grad += a*g
# ---------------------------------------------------------------------------
def test_mul_backward_local():
    a, b = Value(2.0), Value(3.0)
    out = a * b
    assert out.data == 6.0
    out.grad = 1.0
    out._backward()
    assert a.grad == pytest.approx(3.0)   # dz/da = b
    assert b.grad == pytest.approx(2.0)   # dz/db = a


def test_mul_backward_scales_with_output_grad():
    a, b = Value(4.0), Value(-5.0)
    out = a * b
    out.grad = 2.0
    out._backward()
    assert a.grad == pytest.approx(-5.0 * 2.0)
    assert b.grad == pytest.approx(4.0 * 2.0)


# ---------------------------------------------------------------------------
# pow: a.grad += c*a**(c-1)*g  (c constant)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("a,c", [(3.0, 2), (2.0, 3), (4.0, 0.5), (5.0, -1)])
def test_pow_backward_local(a, c):
    x = Value(a)
    out = x ** c
    assert out.data == pytest.approx(a ** c)
    out.grad = 1.0
    out._backward()
    assert x.grad == pytest.approx(c * a ** (c - 1))


def test_pow_rejects_value_exponent():
    a = Value(3.0)
    with pytest.raises((AssertionError, TypeError)):
        _ = a ** Value(2.0)


# ---------------------------------------------------------------------------
# derived ops compose from + * ** -> their closures still give right locals
# ---------------------------------------------------------------------------
def test_sub_backward_via_composition():
    # a - b = a + (-b); seed and run each intermediate node's _backward by hand
    a, b = Value(7.0), Value(2.0)
    out = a - b
    assert out.data == pytest.approx(5.0)
    # full propagation is stage_04; here just check the top node is an add whose
    # _backward pushes grad to its two parents (a and the (-b) node).
    out.grad = 1.0
    out._backward()
    parents = list(out._prev)
    assert all(p.grad == pytest.approx(1.0) for p in parents), (
        "a - b is a + (-b); the add closure pushes grad 1 to both parents"
    )


def test_div_backward_is_not_symmetric():
    # a / b = a * b**-1. Asymmetry lives in the closures, not in stored order.
    a, b = Value(6.0), Value(3.0)
    out = a / b
    assert out.data == pytest.approx(2.0)
    out.grad = 1.0
    out._backward()
    # top node is a multiply: a*(b**-1). Its parents are `a` and the `b**-1` node.
    # da gets (b**-1)=1/3 ; the other parent gets a=6. They differ -> not symmetric.
    grads = sorted(p.grad for p in out._prev)
    assert grads[0] != pytest.approx(grads[1])


# ---------------------------------------------------------------------------
# accumulation: a reused operand += from each consumer
# ---------------------------------------------------------------------------
def test_self_mul_accumulates():
    # out = a * a ; both factors are the SAME node, so its _backward adds a+a = 2a
    a = Value(3.0)
    out = a * a
    assert out.data == 9.0
    out.grad = 1.0
    out._backward()
    assert a.grad == pytest.approx(6.0), "a*a: a.grad must be a + a = 2a = 6"


# ---------------------------------------------------------------------------
# leaf closure is a no-op (inherited from stage_02)
# ---------------------------------------------------------------------------
def test_leaf_backward_noop():
    a = Value(4.0)
    a._backward()
    assert a.grad == 0.0


def test_repr_has_data_and_grad():
    a = Value(2.0)
    r = repr(a)
    assert "data=" in r and "grad=" in r


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
