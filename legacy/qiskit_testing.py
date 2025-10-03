import json

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService


key = json.load(open("apikey.json"))  # Load your IBM Quantum API key from a local file
print(key)

# --- 1) Build an example circuit (Bell state as a demo) ---
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# --- 2) Pull an IBM backend and build a noise model from it ---
# Make sure you have saved your IBM Quantum account first:
# QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_API_TOKEN", overwrite=True)
# service = QiskitRuntimeService(channel="ibm_quantum_platform", token="y1hVsWedGu8FfoHLF5FBZoZuCRz2cXUmxRwrDLVshz_j")

service = QiskitRuntimeService(channel="ibm_quantum_platform", token=key["apikey"])

# Choose a small real device to copy noise from. Change this to a backend you have access to.
# Examples: "ibm_sherbrooke", "ibm_kyiv", "ibm_osaka", "ibm_brisbane", or any available "ibm_..." device.
backend_name = "ibm_brisbane"   # <- set me to a real device you can see in your account
backend = service.backend(backend_name)

# Build the noise model directly from device calibration data
noise_model = NoiseModel.from_backend(backend)

# Grab the device's coupling map and basis gates (helps produce realistic routing for the noise model)
coupling_map = backend.configuration().coupling_map
basis_gates = noise_model.basis_gates

# --- 3) Create a GPU-enabled Aer simulator and run with noise ---
# Method "density_matrix" (or "automatic") is recommended for noisy sims. "stabilizer" won't support general noise.
sim = AerSimulator(method="density_matrix", device="GPU")

# (Optional but handy) set the seed for reproducibility
sim.set_options(seed_simulator=1234)

# Transpile the circuit to match the device's layout & basis (so the noise model is applied correctly)
qc_trans = transpile(
    qc,
    basis_gates=basis_gates,
    coupling_map=coupling_map,
    optimization_level=3,
)

# timing code
import time
start = time.time()

# Execute with the noise model on the GPU simulator
shots = 20_000
result = sim.run(qc_trans, shots=shots, noise_model=noise_model).result()
counts = result.get_counts()

end = time.time()
print(f"Simulation time: {end - start:.2f} seconds")

print(f"Backend noise source: {backend_name}")
print(f"Shots: {shots}")
print("Counts:", counts)
