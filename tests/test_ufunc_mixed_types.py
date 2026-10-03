# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Comparison and min/max ufuncs on unsigned, boolean and mixed-type inputs, called on
arrays with an output array, on scalars with an output array, and on two arrays."""

import numpy as np
import pytest
from numba_cuda_mlir import cuda

COMPARISONS = [
    np.greater,
    np.greater_equal,
    np.less,
    np.less_equal,
    np.equal,
    np.not_equal,
]
MIN_MAX = [np.maximum, np.minimum, np.fmax, np.fmin]

# Unsigned values above the signed range, which compare wrongly as signed
UINT8S = np.array([0, 1, 127, 128, 200, 255], dtype=np.uint8)
OTHER_UINT8S = np.array([255, 128, 128, 127, 1, 0], dtype=np.uint8)
UINT32S = np.array([0, 1, 2**31 - 1, 2**31, 3_000_000_000, 2**32 - 1], dtype=np.uint32)
OTHER_UINT32S = np.array([2**32 - 1, 2**31, 2**31, 1, 3_000_000_000, 0], dtype=np.uint32)
UINT64S = np.array([0, 1, 2**63 - 1, 2**63, 2**64 - 1, 2**63], dtype=np.uint64)
OTHER_UINT64S = np.array([2**64 - 1, 2**63, 2**63, 1, 2**63 + 1, 0], dtype=np.uint64)
# Negative values must stay negative when converted to a wider type
INT8S = np.array([-128, -1, 0, 1, 127, -2], dtype=np.int8)
INT16S = np.array([-129, 0, 255, -1, 128, -2], dtype=np.int16)
INT32S = np.array([-3, 0, 2**31 - 1, -(2**31), 7, -1], dtype=np.int32)
OTHER_INT32S = np.array([7, -(2**31), 1, 2**31 - 1, -3, 0], dtype=np.int32)
# NumPy compares a signed integer and a uint64 exactly
INT64S = np.array([-1, 2**63 - 1, -1, 2**63 - 1, -(2**63), 0], dtype=np.int64)
# Paired with UINT32S, so that converting those as signed changes every result
FLOATS = np.array([-1.5, 0.0, 2.5e9, 1.0, 3e9, -2.0])
# float32 values compare with float64 ones exactly, e.g. float32(0.1) > 0.1
FLOAT32S = np.array([1.5, 0.1, -0.0, 3e38, -1.0, 16777217], dtype=np.float32)
FLOAT64S = np.array([1.5, 0.1, 0.0, 3e38, -1.0000001, 16777217.0])

SCALAR_PAIRS = {
    "uint8": (UINT8S, OTHER_UINT8S),
    "uint32": (UINT32S, OTHER_UINT32S),
    "uint64": (UINT64S, OTHER_UINT64S),
    "int32": (INT32S, OTHER_INT32S),
    "uint32-int32": (UINT32S, INT32S),
    "int8-int16": (INT8S, INT16S),
    "uint8-int8": (UINT8S, INT8S),
    "int64-uint64": (INT64S, UINT64S),
    "uint32-float64": (UINT32S, FLOATS),
    "float32-float64": (FLOAT32S, FLOAT64S),
}
# Booleans are unsigned too
ARRAY_PAIRS = {
    **SCALAR_PAIRS,
    "bool": (np.array([False, False, True, True]), np.array([False, True, False, True])),
}


def run_array_to_array(ufunc, x, y, out):
    @cuda.jit
    def kernel(x, y, out):
        ufunc(x, y, out)

    kernel[1, 1](x, y, out)


def run_scalar_to_array(ufunc, x, y, out):
    @cuda.jit
    def kernel(x, y, out):
        for i in range(out.size):
            ufunc(x[i], y[i], out[i:])

    kernel[1, 1](x, y, out)


def run_array(ufunc, x, y, out):
    @cuda.jit
    def kernel(x, y, out):
        result = ufunc(x, y)
        for i in range(out.size):
            out[i] = result[i]

    kernel[1, 1](x, y, out)


def check(run, ufunc, x, y, dtype=None):
    expected = np.zeros(x.shape, dtype=dtype or ufunc(x, y).dtype)
    ufunc(x, y, out=expected, casting="unsafe")
    out = np.zeros_like(expected)
    run(ufunc, x, y, out)
    if expected.dtype.kind == "f" and ufunc not in MIN_MAX:
        np.testing.assert_allclose(out, expected, rtol=1e-12)
    else:
        np.testing.assert_array_equal(out, expected)


CASES = [
    pytest.param(run, x, y, id=f"{form}-{name}")
    for form, run, pairs in (
        ("arrays", run_array_to_array, ARRAY_PAIRS),
        ("scalars", run_scalar_to_array, SCALAR_PAIRS),
    )
    for name, (x, y) in pairs.items()
]


@pytest.mark.parametrize("ufunc", COMPARISONS, ids=lambda u: u.__name__)
@pytest.mark.parametrize("run, x, y", CASES)
def test_comparison_with_output(run, ufunc, x, y):
    check(run, ufunc, x, y)


@pytest.mark.parametrize("ufunc", MIN_MAX, ids=lambda u: u.__name__)
@pytest.mark.parametrize(
    "run, x, y",
    CASES + [pytest.param(run_array, x, y, id=f"new-{n}") for n, (x, y) in ARRAY_PAIRS.items()],
)
def test_min_max(run, ufunc, x, y):
    check(run, ufunc, x, y)


# An output array of another type gets the result NumPy computes in the inputs' type
@pytest.mark.parametrize("ufunc", MIN_MAX, ids=lambda u: u.__name__)
@pytest.mark.parametrize(
    "run", [run_array_to_array, run_scalar_to_array], ids=["arrays", "scalars"]
)
@pytest.mark.parametrize(
    "x, y, dtype",
    [
        (INT32S, OTHER_INT32S, np.int64),
        (UINT32S, OTHER_UINT32S, np.int32),
        (INT8S, INT16S, np.float32),
    ],
    ids=["int32-into-int64", "uint32-into-int32", "int8-int16-into-float32"],
)
def test_min_max_into_other_type(run, ufunc, x, y, dtype):
    check(run, ufunc, x, y, dtype)


# The other binary ufuncs convert each input to the output type with its own signedness
@pytest.mark.parametrize(
    "ufunc, x, y, dtype",
    [
        (np.bitwise_and, INT8S, INT16S, np.int16),
        (np.bitwise_or, INT8S, INT16S, np.int16),
        (np.bitwise_xor, UINT8S, INT8S, np.int16),
        (np.arctan2, UINT32S, OTHER_UINT32S, np.float64),
        (np.hypot, UINT32S, OTHER_UINT32S, np.float64),
    ],
    ids=["bitwise_and", "bitwise_or", "bitwise_xor", "arctan2", "hypot"],
)
def test_other_binary_ufuncs_with_output(ufunc, x, y, dtype):
    check(run_array_to_array, ufunc, x, y, dtype)
