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
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate


# ####### packages for simulating quantum circuits #######
# from qiskit_aer import AerSimulator
from qiskit_aer import AerSimulator

### get the filtered samples of the QAOA results, based on their probability
def get_filtered_samples(
    samples: List[SolutionSample],
    threshold: float = 0,
    allowed_status: Tuple[OptimizationResultStatus] = (OptimizationResultStatus.SUCCESS,),
):
    res = []
    for s in samples:
        if s.status in allowed_status and s.probability > threshold:
            res.append(s)

    return res

### functions for translating matrix multiplications to bipartite graph passing

def create_graph_from_matrix(matrix: np.ndarray) -> nx.DiGraph:
    G = nx.DiGraph()

    for dim, axis, layer in zip(matrix.shape, ["i", "j"], [0, 1]):
        G.add_nodes_from([f"{axis}_{dim_entry+1}" for dim_entry in range(dim)], layer=layer)

    for i, j in np.ndindex(matrix.shape):
        if matrix[i, j] != 0:
            G.add_edge(f"i_{i+1}", f"j_{j+1}", weight=matrix[i, j])

    return G

def get_max_k(nodes: list):
    return max(map(lambda x: int(x.split('k')[1].split('_')[0]), filter(lambda x: x.startswith('k'), nodes)), default=-1)
    
def new_node_name(old_name: str, step: int) -> str:
    axis, index = old_name.split("_")
    
    if axis == "i":
        return f"j_{index}"

    if axis == "j":
        return f"final_i_{index}"

    if axis.startswith('k'):
        old_index = int(old_name.split("k")[1].split("_")[0])

        return f"k{old_index + step}_{old_name.split('_')[1]}"

def get_layer_index(node: str, max_k: int):
    if node.startswith('i'):
        return 0

    if node.startswith('k'):
        return int(node.split('k')[1].split('_')[0]) + 1

    if node.startswith('j'):
        return max_k + 2

def multiply_graphs(G1: nx.DiGraph, G2: nx.DiGraph) -> nx.DiGraph:
    max_index = get_max_k(G1.nodes)

    G_2_relabeled = nx.relabel_nodes(G2, {node: new_node_name(node, max_index + 2) for node in G2.nodes})
    G_composed = nx.compose(G1, G_2_relabeled)
    G_composed_relabeled = nx.relabel_nodes(G_composed, {node: node.replace("j", f"k{max_index+1}").replace("final_i", "j") for node in G_composed.nodes})

    max_relabeled_index = get_max_k(G_composed_relabeled.nodes)

    attrs = {node: {"layer": get_layer_index(node, max_relabeled_index)} for node in G_composed_relabeled.nodes}

    nx.set_node_attributes(G_composed_relabeled, attrs)

    return G_composed_relabeled

def plot_matrix(G: nx.DiGraph):
    pos = nx.multipartite_layout(G, subset_key='layer')
    
    nx.draw(G, with_labels=True, pos=pos)

    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels, label_pos=0.6)

def RZ(fixed_theta):
    """
    Returns the RZ rotation matrix for a given angle fixed_theta.
    """
    I = np.eye(2,2)
    return np.cos(fixed_theta / 2) * I - 1j * np.sin(fixed_theta / 2) * Z

def kron_identity(N_t, G, node1, node2, Z, fixed_theta):
    # number of qubits equals b*N_t or b*N_r where b is the bit complexity.
    # Create the base identity matrix of given size
    # G[path[0]][path[1]]['weight']*G[path[1]][path[2]]['weight']*G[path[2]][path[3]]['weight']
    if (int(node2[3]) == 1 or int(node1[3]) == 1):
        I_1 = np.eye(1,1)
    else:
        if int(node1[3]) < int(node2[3]):
            I_1 = np.eye(2**(int(node1[3])-1), 2**(int(node1[3])-1))
        else:
            I_1 = np.eye(2**(int(node2[3])-1), 2**(int(node2[3])-1))

    if (int(node2[3]) == N_t or int(node1[3]) == N_t):
        I_3 = np.eye(1,1)
    else:
        if int(node1[3]) > int(node2[3]):
            I_3 = np.eye(2**(N_t-int(node1[3])), 2**(N_t-int(node1[3])))
        else: 
            I_3 = np.eye(2**(N_t-int(node2[3])), 2**(N_t-int(node2[3])))

        # (int(node2[3]) || int(node2[3])) == N_t:
        # I3 = np.eye(1,1)
        
    # Initialize the result as I
    I = np.eye(2,2)
    I_2 = np.eye(1,1)
    reps = np.abs(int(node2[3]) - int(node1[3])) 
    # Perform Kronecker product n-1 times
    ind = 0
    for i in range(N_t):
        j = i + 1
        if int(node1[3]) == j:
            I_2 = np.kron(I_2, RZ(fixed_theta))
            ind = ind +1
        elif int(node2[3]) == j: 
            I_2 = np.kron(I_2, RZ(fixed_theta))
            ind = ind +1
        elif (np.minimum(int(node2[3]), int(node1[3])) < j < np.maximum(int(node2[3]), int(node1[3]))):
            I_2 = np.kron(I_2, I)
        if ind == 2: 
            break
    I_fin = np.kron(I_1, I_2)
    I_fin = np.kron(I_fin, I_3)
    # print(I_1.shape, I_2.shape, I_3.shape, I_fin.shape)
    # print(I_fin.shape)
    return I_fin

def return_gate(G, N_t, fixed_theta, g, Z, all_paths):
    # all_paths = list(nx.all_simple_paths(G, source = 'i_1', target = 'j_1'))
    a = np.zeros((2**N_t, 2**N_t))
    # gamma = complex(1,2) 
    for path in all_paths:
        node1 = path[1]
        node2 = path[2]
        if int(node1[3]) != int(node2[3]):
            a = a + G[path[1]][path[2]]['weight']*kron_identity(N_t, G, node1, node2, Z, fixed_theta)
    
    exp_A = expm(-1j*g*a)
    gate = UnitaryGate(exp_A)
    return gate

def create_qaoa_circ(G, N_t, fixed_theta, params, Z, all_paths):
    """Creates a parametrized qaoa circuit
    Args:
        graph: networkx graph
        fixed_theta: (list) unitary parameters
    Returns:
        (QuantumCircuit) qiskit circuit
    """
    nqubits = N_t
    n_layers = 1#len(fixed_theta)//2  # number of alternating unitaries
    # beta = fixed_theta[:n_layers]
    # gamma = fixed_theta[n_layers:]
    gamma = params[0]
    beta = params[1]
    qc = QuantumCircuit(nqubits)

    # initial_state
    qc.h(range(nqubits))

    for layer_index in range(n_layers):
        # problem unitary
        # for pair in list(graph.edges()):
        #     qc.rzz(2 * gamma[layer_index], pair[0], pair[1])
        gate = return_gate(G, N_t, fixed_theta, gamma, Z, all_paths)
        # mixer unitary
        qc.append(gate, range(nqubits))
        for qubit in range(nqubits):
            qc.rx(2 * beta, qubit)

    qc.measure_all()
    return qc

def get_expectation(G, N_t, fixed_theta, params, Z, all_paths, qc, Q, shots=2*1024):
    """Runs parametrized circuit
    Args:
        graph: networkx graph
    """, 
    backend = AerSimulator()#Aer.get_backend('qasm_simulator')
    # backend.shots = shots
    # params = [gamma, beta]
    def execute_circ(params):
        # params = [gamma, beta]
        # def execute_circ(G, N_t, fixed_theta, gamma, beta, Z, all_paths):
        qc = create_qaoa_circ(G, N_t, fixed_theta, params, Z, all_paths)
        counts = backend.run(qc, nshots = shots).result().get_counts() #seed_simulator=10,
        return compute_expectation(counts, Q, fixed_theta) #compute_expectation(counts, graph)
        
    return execute_circ

def compute_expectation(counts, Q, fixed_theta):
#     """Computes expectation value based on measurement results
    # Args:
    #     counts: (dict) key as bit string, val as count
    #     graph: networkx graph
    # Returns:
    #     avg: float
    #          expectation value
    avg = 0
    sum_count = 0
    for bit_string, count in counts.items(): 
        bit_list = [int(bit) for bit in bit_string]
        bit_array = np.array(bit_list)
        bit_array = np.exp(1j*fixed_theta*bit_array)
        obj = np.dot(np.dot(np.conjugate(bit_array).T,Q),bit_array)
        avg += obj * count
        sum_count += count
    # print("obj is:", obj)
    aa = np.real(avg/sum_count)
    return aa
 
############################################# MAIN CODE BEGINNING #############################################

############################################# MAIN CODE BEGINNING #############################################
I = np.array([[1, 0],
              [0, 1]], dtype=complex)

# Define the Pauli-X matrix
X = np.array([[0, 1],
              [1, 0]], dtype=complex)

# Define the Pauli-Y matrix
Y = np.array([[0, -1j],
              [1j, 0]], dtype=complex)

# Define the Pauli-Z matrix
Z = np.array([[1, 0],
              [0, -1]], dtype=complex)

# np.random.seed(51)
# Number of received and transmit antennas 
N_r = 3
N_t = 3

# Number of alternating optimization iterations
num_iter = 3

# Create the complex channel matrix H
real_part = np.random.normal(0, 1, size=(N_r, N_t))#np.random.rand(N_r, N_t)
imaginary_part = np.random.normal(0, 1, size=(N_r, N_t))#np.random.rand(N_r, N_t)
H = real_part + 1j * imaginary_part

# random initial solution to f,g in complex form
f = np.random.randint(0, 2, size = (N_t,1)) + 1j * np.random.randint(0, 2, size = (N_t,1))
g = np.random.choice([-1, 1], size = (N_r,1)) + np.random.choice([-1, 1], size = (N_r,1)) * 1j
# g = np.random.choice([0, 1], size = (N_r,1)) # + np.random.choice([-1, 1], size = (N_r,1)) * 1j


gamma_1 = 0
# gamma = Parameter("$\\gamma$")
beta_1 = 0
# beta = Parameter("$\\beta$")
theta = np.pi
params = [gamma_1, beta_1]
# Goal is to minimize |g^*Hf|^2 via alternating optimization and QAOA
for i in range(num_iter):
    print("iteration is:", i)
    # Create 1st subQUB0 --> maximize |Af|^2 where A = g^*H
    # these are the matrices that will be used to build the qcircuit and etc.
    # reminder: the problem is in +-1 form, thus we transformed it into 0,1 form.
    
    A = np.dot(np.conjugate(g).T, H)
    Q = np.conjugate(A).T
    Q = -np.dot(Q,A)/(N_r*N_t) 
    # linear = -4*np.dot(np.ones((1, N_t)),Q)
    # Q = Q/4           
    
    ## Create the graphs based on the multiplication of f^H Q F
    G1_f = create_graph_from_matrix(np.conjugate(f).T)
    G2_f = create_graph_from_matrix(Q)
    G3_f = create_graph_from_matrix(f)
    
    # total graph
    G = multiply_graphs(multiply_graphs(G1_f, G2_f), G3_f)
    # all paths of the graph --> all underlying multiplications
    all_paths = list(nx.all_simple_paths(G, source = 'i_1', target = 'j_1'))

    # create the circuit for the QAOA using the two hamiltonians
    circuit_qaoa = create_qaoa_circ(G, N_t, theta, params, Z, all_paths)
    expectation = get_expectation(G, N_t, theta, params, Z, all_paths, circuit_qaoa, Q, shots=1024)
    res = minimize(expectation, params, method='COBYLA', options={'maxiter': 1000, 'disp': True})
    # circuit_qaoa.decompose().draw() 
    # run the cobyla optimizer to optimize beta_1, gamma_1
    
    # Step 3: Extract the top 4 bitstrings

    alpha = res.x
    # a = [-0.3, -0.2]
    backend  = AerSimulator() 
    shots = 1024
    fixed_theta = theta
    circuit_qaoa = create_qaoa_circ(G, N_t, fixed_theta, alpha, Z, all_paths)
    counts = backend.run(circuit_qaoa, nshots = shots).result().get_counts() 

    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)  
    top_4_bitstrings = sorted_counts[:4]
    objj = 100
    for bit_string, count in top_4_bitstrings: 
        bit_list = [int(bit) for bit in bit_string]
        bit_array = np.array(bit_list)
        bit_array = np.exp(1j*fixed_theta*bit_array)
        obj = np.dot(np.dot(np.conjugate(bit_array).T,Q),bit_array)
        if obj.real < objj.real:
            objj = obj 
            bit_opt = bit_array
    
# # exact_result = exact.solve(qubo)
# # print(exact_result)#exact_result.prettyprint(),

#     qaoa_result = qaoa.solve(qubo)
    
#     ### Create the 2nd subQUBO
    f = bit_opt
    f = np.reshape(f, (N_t,1))
    A = np.dot(H,f)#np.dot(np.conjugate(g).T, H)
    Q = np.conjugate(A).T
    Q = -np.dot(A,Q)/(N_r*N_t) 

    # f = np.random.randint(0, 2, size = (N_t,1)) + 1j * np.random.randint(0, 2, size = (N_t,1))
    g = np.random.randint(0, 2, size = (N_r,1)) + 1j * np.random.randint(0, 2, size = (N_r,1))

    G1_f = create_graph_from_matrix(np.conjugate(g).T)
    G2_f = create_graph_from_matrix(Q)
    G3_f = create_graph_from_matrix(g)

    G = multiply_graphs(multiply_graphs(G1_f, G2_f), G3_f)
    # all paths of the graph --> all underlying multiplications
    all_paths = list(nx.all_simple_paths(G, source = 'i_1', target = 'j_1'))

    # create the circuit for the QAOA using the two hamiltonians
    circuit_qaoa = create_qaoa_circ(G, N_r, theta, params, Z, all_paths)
    expectation = get_expectation(G, N_r, theta, params, Z, all_paths, circuit_qaoa, Q, shots=1024)
    res = minimize(expectation, params, method='COBYLA', options={'maxiter': 1000, 'disp': True})
    # circuit_qaoa.decompose().draw() 
    # run the cobyla optimizer to optimize beta_1, gamma_1
    
    # Step 3: Extract the top 4 bitstrings

    alpha = res.x
    # a = [-0.3, -0.2]
    backend  = AerSimulator() 
    shots = 1024
    fixed_theta = theta
    circuit_qaoa = create_qaoa_circ(G, N_r, fixed_theta, alpha, Z, all_paths)
    counts_g = backend.run(circuit_qaoa, nshots = shots).result().get_counts() 

    sorted_counts_g = sorted(counts_g.items(), key=lambda item: item[1], reverse=True)  
    top_4_bitstrings = sorted_counts_g[:4]
    objj = 100
    for bit_string, count in top_4_bitstrings: 
        bit_list = [int(bit) for bit in bit_string]
        bit_array = np.array(bit_list)
        bit_array = np.exp(1j*fixed_theta*bit_array)
        obj = np.dot(np.dot(np.conjugate(bit_array).T,Q),bit_array)
        if obj.real < objj.real:
            objj = obj 
            bit_opt = bit_array
    g = bit_opt
    g = np.reshape(g, (N_r,1))
    
print("ENDED")
