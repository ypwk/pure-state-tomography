# Pure State Tomography Paper Experiments - New

## States with Two Nonzero-Entries

### 000, 001

- Normal Job File: job_2024_02_05T_01_34_03.txt
- Hadamard Job File: job_2024_02_05T_01_34_32.txt
- Circuit:

```
state = qiskit.QuantumCircuit(3)
state.h(0)
```

### 000, 110

- Normal Job File: job_2024_02_05T_01_37_23.txt
- Hadamard Job File:
- Circuit:

```
state = qiskit.QuantumCircuit(3)
state.h(2)
state.x(2)
state.cnot(2, 1)
```

### 000, 111

- Normal Job File:
- Hadamard Job File:
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

- Normal Job File:
- Hadamard Job File:
- Circuit:

```
state = qiskit.QuantumCircuit(3)
state.u(pi / 4, 0, 0, 1)
state.u(pi / 4, 0, 0, 2)
state.cx(2, 1)
state.h(2)
```

### 000, 011, 100

- Normal Job File:
- Hadamard Job File:
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

- Normal Job File:
- Hadamard Job File:
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

- Normal Job File:
- Circuit:

```
state = qiskit.QuantumCircuit(3)
state.h(0)
state.h(1)
state.h(2)
state.x(0)
state.u(pi / 4, 0, 0, 2)
```
