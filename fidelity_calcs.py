from math import sqrt

num_shots = 16384
num_CNOT = 3
num_SQ = 1

p_CNOT = 2.5e-3
p_SQ = 2.5e-4

prob_no_error = (1 - p_CNOT) ** num_CNOT * (1 - p_SQ) ** num_SQ
infidelity_gates = 1 - prob_no_error - (1 - prob_no_error) / (2 ** (num_CNOT + 1))

print(f"Estimated gate-induced infidelity per measurement: {infidelity_gates:.6f}")
print(f"Number of CNOTs: {num_CNOT}, Number of single-qubit gates: {num_SQ}")
print(f"CNOT error rate: {p_CNOT}, Single-qubit gate error rate: {p_SQ}")

walsh_hadamard_dimension = 2 ** (num_CNOT + 1)
relative_error = sqrt((1 / (walsh_hadamard_dimension) * (1 - 1 / (walsh_hadamard_dimension))) / num_shots) * walsh_hadamard_dimension
infidelity = 2 * (2 * relative_error) ** 2

print(f"Estimated statistical infidelity per measurement: {infidelity:.6f}")
print(f"Relative error due to finite shots: {relative_error:.6f}")