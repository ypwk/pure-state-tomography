def embed_support_state(
    amplitudes,
    *,
    support_indices: list[int],
    num_qubits: int,
    normalize: bool = True,
):
    """
    Embed support amplitudes into a full 2^n complex statevector.

    Args:
        amplitudes: Iterable of complex amplitudes for each support basis index.
        support_indices: Computational-basis indices where amplitudes are nonzero.
        num_qubits: Number of qubits, full dimension is 2**num_qubits.
        normalize: Whether to normalize the resulting vector to unit norm.

    Returns:
        np.ndarray of shape (2**num_qubits,) with complex dtype.
    """
    import numpy as np

    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive")
    if len(support_indices) == 0:
        raise ValueError("support_indices must be non-empty")

    amps = np.asarray(list(amplitudes), dtype=complex)
    if amps.shape[0] != len(support_indices):
        raise ValueError("len(amplitudes) must match len(support_indices)")
    if len(set(support_indices)) != len(support_indices):
        raise ValueError("support_indices must be unique")

    dim = 1 << int(num_qubits)
    full = np.zeros(dim, dtype=complex)

    for amp, idx in zip(amps, support_indices):
        idx = int(idx)
        if idx < 0 or idx >= dim:
            raise ValueError(f"support index {idx} out of range [0, {dim})")
        full[idx] = amp

    if normalize:
        nrm = float(np.linalg.norm(full))
        if nrm <= 0.0:
            raise ValueError("cannot normalize zero vector")
        full /= nrm

    return full
