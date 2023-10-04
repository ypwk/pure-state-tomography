"""
This file contains general utility code for mathematical operations and file printing.

Functions:
- fast_pow: Computes fast exponentiation using squaring.
- fast_log2: Computes the logarithm base 2 of a number using bit length.
- hamming: Computes the Hamming distance between two integers.
- fprint: Wrapper for the print function that prints to console and writes to a file simultaneously.

See each function's respective docstring for detailed usage and parameter information.
"""


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


def fprint(*args, filename="output.txt", mode="a", **kwargs):
    """
    Wrapper for the print function that prints to console and writes to a file simultaneously.

    Args:
        *args: Positional arguments for the print function.
        filename (str): The name of the file to write to.
        mode (str): File mode for writing (default is 'a' for append).
        **kwargs: Keyword arguments for the print function.
    """
    with open(filename, mode) as file:
        print(*args, **kwargs)
        print(*args, file=file, **kwargs)


__author__ = "Kevin Wu"
__credits__ = ["Kevin Wu"]
