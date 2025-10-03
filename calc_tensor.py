import numpy as np

np.set_printoptions(edgeitems=30, linewidth=100000,  formatter=dict(float=lambda x: "%.3g" % x))

# Define some basic gates
I = np.array([[1, 0],
              [0, 1]], dtype=complex)

H = (1/np.sqrt(2)) * np.array([[1,  1],
                               [1, -1]], dtype=complex)

D = np.array([[1, 0],
              [0, 1j]], dtype=complex)

V = H @ D  # = (1/sqrt(2)) * [[1, i], [1, -i]]

# Mapping from characters to matrices
gate_map = {
    "I": I,
    "H": H,
    "V": V,
}


def tensor_from_string(op_string):
    """
    Given a string like "HIV", return the tensor product H ⊗ I ⊗ V.
    """
    if not op_string:
        raise ValueError("Operator string cannot be empty.")

    # Start with the first gate
    if op_string[0] not in gate_map:
        raise ValueError(f"Unknown operator: {op_string[0]}")
    result = gate_map[op_string[0]]

    # Iteratively apply kron product
    for char in op_string[1:]:
        if char not in gate_map:
            raise ValueError(f"Unknown operator: {char}")
        result = np.kron(result, gate_map[char])

    return result


# Example usage:
if __name__ == "__main__":
    op = "IVH"
    mat = tensor_from_string(op)
    print(f"{op} =\n{mat}\n")
