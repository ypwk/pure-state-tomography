"""
This file contains general utility code for mathematical operations and file printing.

Functions:
- fast_pow: Computes fast exponentiation using squaring.
- fast_log2: Computes the logarithm base 2 of a number using bit length.
- hamming: Computes the Hamming distance between two integers.
- fprint: Wrapper for the print function that prints to console and writes to a file
    simultaneously.

See each function's respective docstring for detailed usage and parameter information.
"""

from datetime import datetime as dt
import numpy as np
import os


def fast_pow(base, exp) -> int:
    """Fast exponent using squaring"""
    res = 1
    while True:
        if exp & 1:
            res *= base
        exp >>= 1
        if not exp:
            break
        base *= base
    return res


def fast_log2(num: int) -> int:
    """Fast log2 using bit length"""
    return num.bit_length() - 1


def hadamard(vec: np.ndarray) -> np.ndarray:
    """Applies a Hadamard transform to the vector state."""
    nq = fast_log2(len(vec))
    coeff = np.sqrt(len(vec))
    ret = np.zeros(len(vec), dtype="complex128")
    for a in range(len(vec)):
        had = np.zeros(len(vec))
        for i in range(nq):
            had += np.array([((a & b) >> i) & 1 for b in range(len(vec))])
        had %= 2
        had = np.array([-1 if v else 1 for v in had], dtype="complex128")
        had /= coeff
        ret[a] = vec.dot(had.T)
    return ret


def fwht_inplace(vec: np.ndarray) -> None:
    """In-place FWHT on a length-2^k vector (complex OK). No normalization."""
    n = vec.shape[0]
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x = vec[j]
                y = vec[j + h]
                vec[j] = x + y
                vec[j + h] = x - y
        h *= 2


def unmix_hadamard_block(mixed_block: np.ndarray) -> np.ndarray:
    """Return the inverse-Hadamard of 'mixed_block' (length 2^k), normalized."""
    v = mixed_block.astype(np.complex128).copy()
    fwht_inplace(v)
    v /= np.sqrt(v.shape[0])  # H^{-1} = H / sqrt(2^k)
    return v


def hamming(i1: int, i2: int) -> int:
    """Computes the Hamming distance between two integers"""
    x, ones = i1 ^ i2, 0
    while x > 0:
        ones += x & 1
        x >>= 1
    return ones


FILENAME = f"log_{dt.now().strftime('%Y_%m_%dT_%H_%M_%S')}.txt"


def fprint(*args, filename=FILENAME, mode="a", **kwargs):
    """
    Print to console and to a log file.
    """
    file_path = os.path.join("logs", filename)
    os.makedirs("logs", exist_ok=True)
    with open(file_path, mode, encoding="utf-8") as file:
        print(*args, **kwargs)
        print(*args, file=file, **kwargs)


def make_fprint(filename=None):
    """
    Factory for creating a file-specific fprint.
    """
    if filename is None:
        filename = f"log_{dt.now().strftime('%Y_%m_%dT_%H_%M_%S')}.txt"

    def fprint_inner(*args, mode="a", **kwargs):
        file_path = os.path.join("logs", filename)
        os.makedirs("logs", exist_ok=True)
        with open(file_path, mode, encoding="utf-8") as file:
            # print(*args, **kwargs)  # console disabled for cleaner logging
            print(*args, file=file, **kwargs)

    return fprint_inner


def print_header(fprint, experiment, execution_type, mm):
    """Prints experiment header info."""
    fprint(f"Index: {int(experiment)}")
    fprint(f"Hadamard: {int(experiment) % 2 == 1}")
    fprint(f"Experiment: {int(experiment) // 2}")
    fprint(f"Executing on: {execution_type}")
    fprint(f"Running inference at {mm.n_shots} shots\n")


__author__ = "Kevin Wu"
__credits__ = ["Kevin Wu"]
