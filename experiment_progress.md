# Pure State Tomography Paper Experiments

## States with Two Nonzero-Entries

### 000, 001

- Normal Job File: job_2023_11_11T_02_41_28.txt (DONE)
- Hadamard Job File: job_2023_11_13T_17_06_44.txt (DONE)
- Circuit:

```
state = qiskit.QuantumCircuit(3)
state.h(0)
```

### 000, 110

- Normal Job File: job_2023_11_11T_02_42_01.txt (DONE)
- Hadamard Job File: job_2023_11_12T_15_21_26.txt (DONE)
- Circuit:

```
state = qiskit.QuantumCircuit(3)
state.h(2)
state.x(2)
state.cnot(2, 1)
```

### 000, 111

- Normal Job File: job_2023_11_11T_02_43_16.txt (DONE)
- Hadamard Job File: job_2023_11_14T_08_12_18.txt
- Circuit:

```
state = qiskit.QuantumCircuit(3)
state.h(2)
state.x(2)
state.cnot(2, 1)
state.cnot(2, 0)
```

## States with Three Nonzero-Entries

### 000, 010, 100

- Normal Job File: job_2023_11_14T_08_14_09.txt
- Hadamard Job File: job_2023_11_15T_02_23_34.txt (DONE)
- Circuit:

```
state = qiskit.QuantumCircuit(3)
state.u(pi / 4, 0, 0, 1)
state.u(pi / 4, 0, 0, 2)
state.cx(2, 1)
state.h(2)
```

### 000, 010, 101

- Normal Job File: job_2023_11_17T_22_13_12.txt (DONE)
- Hadamard Job File: job_2023_11_17T_22_20_16.txt (DONE)
- Circuit:

```
state = qiskit.QuantumCircuit(3)
state.u(pi / 4, 0, 0, 1)
state.u(pi / 4, 0, 0, 2)
state.cx(2, 1)
state.h(2)
state.cx(2, 0)
```

### 000, 011, 110

- Normal Job File: job_2023_11_17T_19_33_27.txt (DONE)
- Hadamard Job File: job_2023_11_15T_02_38_50.txt (DONE)
- Circuit:

```
state = qiskit.QuantumCircuit(3)
state.u(pi / 4, 0, 0, 1)
state.u(pi / 4, 0, 0, 2)
state.cx(2, 1)
state.h(2)
state.cx(1, 0)
state.cx(2, 1)
```

## General Unitary

- Normal Job File: job_2023_11_15T_02_39_53.txt (DONE)
- Circuit:

```
state = qiskit.QuantumCircuit(3)
state.h(0)
state.h(1)
state.h(2)
state.x(0)
state.u(pi / 4, 0, 0, 2)
```
