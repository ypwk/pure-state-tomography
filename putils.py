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

from numpy import ndarray, array, zeros, sqrt

import os


def fast_pow(base, exp) -> int:
    """Fast exponent using squaring

    Args:
        base (int): base of exponent
        exp (int): power of exponent

    Returns:
        result: result of exponent
    """
    res = 1
    while True:
        if exp & 1:
            res *= base
        exp = exp >> 1
        if not exp:
            break
        base *= base
    return res


def fast_log2(num: int) -> int:
    """Fast log2 using bit length

    Args:
        num (int): the number to log

    Returns:
        int: the result
    """
    return num.bit_length() - 1


def hadamard(vec: ndarray) -> ndarray:
    """Applies a hadamard to the vec state.

    Args:
        vec (ndarray): A vector.

    Returns:
        ndarray: The vector with Hadamard applied.
    """
    nq = fast_log2(len(vec))
    coeff = sqrt(len(vec))
    ret = zeros(len(vec), dtype="complex_")
    for a in range(len(vec)):
        hadamard = zeros(len(vec))
        for i in range(nq):
            hadamard += array([((a & b) >> i) & 1 for b in range(len(vec))])
        hadamard %= 2
        hadamard = array([-1 if v else 1 for v in hadamard], dtype="complex_")
        hadamard /= coeff
        ret[a] = vec.dot(hadamard.T)
    return ret


def hamming(i1: int, i2: int) -> int:
    """Computes the Hamming distance between two integers

    Args:
        i1 (int): integer 1
        i2 (int): integer 2

    Returns:
        int: The Hamming distance between the two integers
    """
    x = i1 ^ i2
    ones = 0

    while x > 0:
        ones += x & 1
        x >>= 1

    return ones


FILENAME = "log_{}.txt".format(dt.now().strftime("%Y_%m_%dT_%H_%M_%S"))


def fprint(*args, filename=FILENAME, mode="a", **kwargs):
    """
    Wrapper for the print function that prints to console and writes to a file
    simultaneously.

    Args:
        *args: Positional arguments for the print function.
        filename (str): The name of the file to write to.
        mode (str): File mode for writing (default is 'a' for append).
        **kwargs: Keyword arguments for the print function.
    """
    file_path = os.path.join("logs", filename)

    if not os.path.isdir("logs"):
        os.mkdir("logs")

    with open(file_path, mode, encoding="utf-8") as file:
        print(*args, **kwargs)
        print(*args, file=file, **kwargs)


def make_fprint(filename=None):
    """
    Function factory to create a version of fprint with a specific filename.

    Args:
        filename (str): Optional. The base name of the file to write to. If not specified,
                        it uses a default pattern based on the current datetime.

    Returns:
        A customized fprint function that writes to the specified filename.
    """
    if filename is None:
        # If no filename is given, use a default name with the current datetime
        filename = "log_{}.txt".format(dt.now().strftime("%Y_%m_%dT_%H_%M_%S"))

    def fprint(*args, mode="a", **kwargs):
        """
        Wrapper for the print function that prints to console and writes to a file
        simultaneously, using the specified filename from the outer function.

        Args:
            *args: Positional arguments for the print function.
            mode (str): File mode for writing (default is 'a' for append).
            **kwargs: Keyword arguments for the print function.
        """
        file_path = os.path.join("logs", filename)

        if not os.path.isdir("logs"):
            os.mkdir("logs")

        with open(file_path, mode, encoding="utf-8") as file:
            # print(*args, **kwargs)
            print(*args, file=file, **kwargs)

    return fprint


__author__ = "Kevin Wu"
__credits__ = ["Kevin Wu"]
