# Custom QAOA-based MIMO Beamforming Optimization Using Graph Representation
# This code performs MIMO beamforming optimization using a custom Quantum Approximate Optimization Algorithm (QAOA).
# It uses graph representations to handle matrix multiplications, which are utilized to design and construct quantum circuits.
# The objective is to optimize the beamforming vectors `f` and `g` to minimize |g^* H f|^2, where H is the complex channel matrix.
# The optimization alternates between two sub-problems using the QAOA method.

# Import necessary libraries and packages
from qiskit_algorithms.utils import algorithm_globals 
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit_optimization.algorithms import (
    MinimumEigenOptimizer,
    RecursiveMinimumEigenOptimizer,
    SolutionSample,
    OptimizationResultStatus,
)
from qiskit_optimization import QuadraticProgram
from qiskit.visualization import plot_histogram
from typing import List, Tuple
import numpy as np
import networkx as nx
from scipy.optimize import minimize
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator

# ####### Utility functions for QAOA result processing and graph manipulation #######

# Filter the solution samples based on probability and result status
def get_filtered_samples(
    samples: List[SolutionSample],
    threshold: float = 0,
    allowed_status: Tuple[OptimizationResultStatus] = (OptimizationResultStatus.SUCCESS,),
):
    """Returns filtered solution samples based on a probability threshold and status."""
    res = []
    for s in samples:
        if s.status in allowed_status and s.probability > threshold:
            res.append(s)
    return res

# Create a directed graph from a matrix for representing matrix multiplications
def create_graph_from_matrix(matrix: np.ndarray) -> nx.DiGraph:
    """Converts a matrix into a directed bipartite graph for matrix multiplication."""
    G = nx.DiGraph()
    for dim, axis, layer in zip(matrix.shape, ["i", "j"], [0, 1]):
        G.add_nodes_from([f"{axis}_{dim_entry+1}" for dim_entry in range(dim)], layer=layer)

    for i, j in np.ndindex(matrix.shape):
        if matrix[i, j] != 0:
            G.add_edge(f"i_{i+1}", f"j_{j+1}", weight=matrix[i, j])

    return G

# Get the maximum layer index 'k' in the graph nodes
def get_max_k(nodes: list):
    """Returns the maximum 'k' layer index in graph nodes."""
    return max(map(lambda x: int(x.split('k')[1].split('_')[0]), filter(lambda x: x.startswith('k'), nodes)), default=-1)

# Relabel nodes in a graph based on current step and old node name
def new_node_name(old_name: str, step: int) -> str:
    """Relabels graph nodes during matrix multiplication."""
    axis, index = old_name.split("_")
    if axis == "i":
        return f"j_{index}"
    if axis == "j":
        return f"final_i_{index}"
    if axis.startswith('k'):
        old_index = int(old_name.split("k")[1].split("_")[0])
        return f"k{old_index + step}_{old_name.split('_')[1]}"

# Get the layer index for a node in a graph for plotting
def get_layer_index(node: str, max_k: int):
    """Returns the layer index for a node in the graph."""
    if node.startswith('i'):
        return 0
    if node.startswith('k'):
        return int(node.split('k')[1].split('_')[0]) + 1
    if node.startswith('j'):
        return max_k + 2

# Function to multiply two graphs, representing matrix multiplication in quantum circuits
def multiply_graphs(G1: nx.DiGraph, G2: nx.DiGraph) -> nx.DiGraph:
    """Multiplies two graphs representing matrix multiplications."""
    max_index = get_max_k(G1.nodes)
    G_2_relabeled = nx.relabel_nodes(G2, {node: new_node_name(node, max_index + 2) for node in G2.nodes})
    G_composed = nx.compose(G1, G_2_relabeled)
    G_composed_relabeled = nx.relabel_nodes(G_composed, {node: node.replace("j", f"k{max_index+1}").replace("final_i", "j") for node in G_composed.nodes})
    max_relabeled_index = get_max_k(G_composed_relabeled.nodes)
    attrs = {node: {"layer": get_layer_index(node, max_relabeled_index)} for node in G_composed_relabeled.nodes}
    nx.set_node_attributes(G_composed_relabeled, attrs)
    return G_composed_relabeled

# Function to plot the matrix represented by the graph
def plot_matrix(G: nx.DiGraph):
    """Visualizes the matrix by plotting the corresponding graph."""
    pos = nx.multipartite_layout(G, subset_key='layer')
    nx.draw(G, with_labels=True, pos=pos)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels, label_pos=0.6)

# RZ rotation matrix used in the quantum circuit
def RZ(fixed_theta):
    """Returns the RZ rotation matrix for a given angle fixed_theta."""
    I = np.eye(2, 2)
    return np.cos(fixed_theta / 2) * I - 1j * np.sin(fixed_theta / 2) * Z

# Construct Kronecker product for quantum gates based on node indices and theta
def kron_identity(N_t, G, node1, node2, Z, fixed_theta):
    """Computes the Kronecker product identity for quantum gates."""
    if (int(node2[3]) == 1 or int(node1[3]) == 1):
        I_1 = np.eye(1, 1)
    else:
        I_1 = np.eye(2 ** (int(min(node1[3], node2[3])) - 1), 2 ** (int(min(node1[3], node2[3])) - 1))

    if (int(node2[3]) == N_t or int(node1[3]) == N_t):
        I_3 = np.eye(1, 1)
    else:
        I_3 = np.eye(2 ** (N_t - int(max(node1[3], node2[3]))), 2 ** (N_t - int(max(node1[3], node2[3]))))

    I_2 = np.eye(1, 1)
    for i in range(N_t):
        j = i + 1
        if int(node1[3]) == j or int(node2[3]) == j:
            I_2 = np.kron(I_2, RZ(fixed_theta))
        elif (min(int(node2[3]), int(node1[3])) < j < max(int(node2[3]), int(node1[3]))):
            I_2 = np.kron(I_2, np.eye(2, 2))

    I_fin = np.kron(I_1, np.kron(I_2, I_3))
    return I_fin

# Returns a custom quantum gate from the graph and the theta values
def return_gate(G, N_t, fixed_theta, g, Z, all_paths):
    """Returns a custom unitary gate for the quantum circuit based on the graph and theta."""
    a = np.zeros((2 ** N_t, 2 ** N_t))
    for path in all_paths:
        node1, node2 = path[1], path[2]
        if int(node1[3]) != int(node2[3]):
            a += G[node1][node2]['weight'] * kron_identity(N_t, G, node1, node2, Z, fixed_theta)
    exp_A = expm(-1j * g * a)
    return UnitaryGate(exp_A)

# Function to create the QAOA quantum circuit based on the graph and parameters
def create_qaoa_circ(G, N_t, fixed_theta, params, Z, all_paths):
    """Creates a parameterized QAOA circuit based on the graph.
    Args:
        G: Graph representation of the problem
        N_t: Number of transmit antennas (also number of qubits)
        fixed_theta: Fixed angle for the rotation gates
        params: Parameters (gamma, beta) for QAOA optimization
        Z: Pauli-Z matrix
        all_paths: All paths in the graph representing matrix multiplications
    Returns:
        A Qiskit quantum circuit for QAOA
    """
    nqubits = N_t
    n_layers = 1  # Currently using one layer for the alternating unitaries
    gamma = params[0]
    beta = params[1]
    qc = QuantumCircuit(nqubits)

    # Apply Hadamard gate to initialize the qubits in superposition
    qc.h(range(nqubits))

    for layer_index in range(n_layers):
        # Apply the problem unitary
        gate = return_gate(G, N_t, fixed_theta, gamma, Z, all_paths)
        qc.append(gate, range(nqubits))

        # Apply the mixer unitary (RX rotations)
        for qubit in range(nqubits):
            qc.rx(2 * beta, qubit)

    # Measure all qubits
    qc.measure_all()
    return qc

# Function to calculate the expectation value from QAOA results
def get_expectation(G, N_t, fixed_theta, params, Z, all_paths, qc, Q, shots=2 * 1024):
    """Calculates the expectation value of the QAOA circuit.
    Args:
        G: Graph representation of the problem
        N_t: Number of transmit antennas (also number of qubits)
        fixed_theta: Fixed angle for the rotation gates
        params: Parameters (gamma, beta) for QAOA optimization
        Z: Pauli-Z matrix
        all_paths: All paths in the graph representing matrix multiplications
        qc: Quantum circuit for the current QAOA iteration
        Q: Objective matrix for optimization
        shots: Number of measurement shots
    Returns:
        A function that executes the circuit and computes the expectation value.
    """
    backend = AerSimulator()

    def execute_circ(params):
        qc = create_qaoa_circ(G, N_t, fixed_theta, params, Z, all_paths)
        counts = backend.run(qc, nshots=shots).result().get_counts()
        return compute_expectation(counts, Q, fixed_theta)

    return execute_circ

# Compute the expectation value from measurement counts and the objective matrix Q
def compute_expectation(counts, Q, fixed_theta):
    """Computes the expectation value based on the measurement counts.
    Args:
        counts: Measurement results from the quantum circuit
        Q: Objective matrix for optimization
        fixed_theta: Fixed angle for the rotation gates
    Returns:
        The real part of the computed expectation value.
    """
    avg = 0
    sum_count = 0
    for bit_string, count in counts.items():
        bit_list = [int(bit) for bit in bit_string]
        bit_array = np.array(bit_list)
        bit_array = np.exp(1j * fixed_theta * bit_array)
        obj = np.dot(np.dot(np.conjugate(bit_array).T, Q), bit_array)
        avg += obj * count
        sum_count += count
    return np.real(avg / sum_count)

############################################# MAIN CODE BEGINNING #############################################

# Identity matrix for 2x2 identity gate
I = np.array([[1, 0], [0, 1]], dtype=complex)

# Define the Pauli-X, Pauli-Y, and Pauli-Z matrices
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Number of received and transmit antennas
N_r = 3
N_t = 3

# Number of alternating optimization iterations
num_iter = 3

# Create the complex channel matrix H (real and imaginary parts)
real_part = np.random.normal(0, 1, size=(N_r, N_t))
imaginary_part = np.random.normal(0, 1, size=(N_r, N_t))
H = real_part + 1j * imaginary_part

# Random initial solution for f and g in complex form
f = np.random.randint(0, 2, size=(N_t, 1)) + 1j * np.random.randint(0, 2, size=(N_t, 1))
g = np.random.choice([-1, 1], size=(N_r, 1)) + np.random.choice([-1, 1], size=(N_r, 1)) * 1j

# Initialize QAOA parameters
gamma_1 = 0
beta_1 = 0
theta = np.pi
params = [gamma_1, beta_1]

# Goal is to minimize |g^*Hf|^2 via alternating optimization and QAOA
for i in range(num_iter):
    print("Iteration:", i)

    # Step 1: Optimize f given g
    A = np.dot(np.conjugate(g).T, H)
    Q = np.conjugate(A).T
    Q = -np.dot(Q, A) / (N_r * N_t)

    # Create graphs based on the matrix multiplication f^H Q f
    G1_f = create_graph_from_matrix(np.conjugate(f).T)
    G2_f = create_graph_from_matrix(Q)
    G3_f = create_graph_from_matrix(f)
    G = multiply_graphs(multiply_graphs(G1_f, G2_f), G3_f)
    all_paths = list(nx.all_simple_paths(G, source='i_1', target='j_1'))

    # Create QAOA circuit and calculate expectation
    circuit_qaoa = create_qaoa_circ(G, N_t, theta, params, Z, all_paths)
    expectation = get_expectation(G, N_t, theta, params, Z, all_paths, circuit_qaoa, Q, shots=1024)
    res = minimize(expectation, params, method='COBYLA', options={'maxiter': 1000, 'disp': True})

    # Step 2: Extract top 4 bitstrings
    alpha = res.x
    backend = AerSimulator()
    circuit_qaoa = create_qaoa_circ(G, N_t, theta, alpha, Z, all_paths)
    counts = backend.run(circuit_qaoa, nshots=1024).result().get_counts()
    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    top_4_bitstrings = sorted_counts[:4]

    # Update f based on the optimal bitstring
    objj = 100
    for bit_string, count in top_4_bitstrings:
        bit_list = [int(bit) for bit in bit_string]
        bit_array = np.array(bit_list)
        bit_array = np.exp(1j * theta * bit_array)
        obj = np.dot(np.dot(np.conjugate(bit_array).T, Q), bit_array)
        if obj.real < objj.real:
            objj = obj
            bit_opt = bit_array

    f = bit_opt.reshape((N_t, 1))

    # Step 3: Optimize g given f
    A = np.dot(H, f)
    Q = np.conjugate(A).T
    Q = -np.dot(A, Q) / (N_r * N_t)

    # Create graphs based on the matrix multiplication g^H Q g
    G1_f = create_graph_from_matrix(np.conjugate(g).T)
    G2_f = create_graph_from_matrix(Q)
    G3_f = create_graph_from_matrix(g)
    G = multiply_graphs(multiply_graphs(G1_f, G2_f), G3_f)
    all_paths = list(nx.all_simple_paths(G, source='i_1', target='j_1'))

    # Create QAOA circuit and calculate expectation for g
    circuit_qaoa = create_qaoa_circ(G, N_r, theta, params, Z, all_paths)
    expectation = get_expectation(G, N_r, theta, params, Z, all_paths, circuit_qaoa, Q, shots=1024)
    res = minimize(expectation, params, method='COBYLA', options={'maxiter': 1000, 'disp': True})

    # Step 4: Extract top 4 bitstrings for g
    alpha = res.x
    circuit_qaoa = create_qaoa_circ(G, N_r, theta, alpha, Z, all_paths)
    counts_g = backend.run(circuit_qaoa, nshots=1024).result().get_counts()
    sorted_counts_g = sorted(counts_g.items(), key=lambda item: item[1], reverse=True)
    top_4_bitstrings = sorted_counts_g[:4]

    objj = 100
    for bit_string, count in top_4_bitstrings:
        bit_list = [int(bit) for bit in bit_string]
        bit_array = np.array(bit_list)
        bit_array = np.exp(1j * theta * bit_array)
        obj = np.dot(np.dot(np.conjugate(bit_array).T, Q), bit_array)
        if obj.real < objj.real:
            objj = obj
            bit_opt = bit_array

    g = bit_opt.reshape((N_r, 1))

print("Optimization Completed")
