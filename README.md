# QAOA-based-1-bit-MIMO-beamforming (under construction)

## Overview
This repository implements two approaches to optimizing beamforming in **Multiple-Input Multiple-Output (MIMO)** systems using the **Quantum Approximate Optimization Algorithm (QAOA)** with **Successive Convex Approximation (SCA)**. The goal of both implementations is to optimize the phases of the beamforming vectors to maximize the received signal power, where the phases are constrained to be either 0 or π.

The key difference between the two approaches is:
1. **Classical QAOA with Ising/QUBO Formulation**: This implementation leverages classical QAOA libraries to solve the problem in its Ising/QUBO form.
2. **Custom Graph-based QAOA Circuit Construction**: This implementation translates matrix multiplications into bipartite graph representations, using them to construct quantum circuits for QAOA from scratch.

Both implementations aim to solve the beamforming problem by alternating between two subproblems that iteratively optimize the beamforming and receive vectors.

## Problem Definition

In MIMO systems, the goal is to optimize the beamforming vectors at the transmitter and receiver to maximize the strength of the received signal. Mathematically, this can be expressed as maximizing \( |g^* H f|^2 \), where \(H\) is the channel matrix, \(g\) is the receive vector, and \(f\) is the transmit beamforming vector. Both \(g\) and \(f\) are constrained to have phases of 0 or π, corresponding to binary variables (\( +1 \) or \( -1 \)).

This optimization problem is challenging due to the non-convex nature of the objective function and the binary constraints, but it can be mapped into a **Quadratic Unconstrained Binary Optimization (QUBO)** form, which QAOA can solve efficiently.

### Approaches

### 1. Classical QAOA with Ising/QUBO Formulation

In this approach, we:
1. **Formulate the problem as a QUBO**: The beamforming optimization problem is expressed as a QUBO problem by breaking it into real and imaginary parts of the quadratic form.
2. **Iterate with two subQUBOs**:
   - **SubQUBO 1**: Optimize the beamforming vector \(f\) using QAOA by maximizing \( |Af|^2 \), where \(A = g^* H\).
   - **SubQUBO 2**: Using the optimized \(f\), optimize the receive vector \(g\) by solving a second QUBO problem.
3. **QAOA Implementation**: The QAOA algorithm is used to approximate the optimal solution for both subproblems. The algorithm alternates between optimizing \(f\) and \(g\), refining the solutions iteratively.

### 2. Custom Graph-based QAOA Circuit Construction

This implementation performs the same optimization but constructs QAOA circuits from scratch using a graph-based approach:
1. **Graph Representation of Matrix Multiplication**: The matrix multiplications involved in calculating \( |g^* H f|^2 \) are represented as bipartite graphs. Each multiplication step corresponds to traversing edges between nodes representing matrix dimensions.
2. **Circuit Construction**: Quantum circuits are built using these graphs, where each edge represents a matrix element. The resulting quantum circuit is used to perform the QAOA optimization.
3. **QAOA Execution**: The quantum circuits are executed on a quantum simulator (using Qiskit) to optimize the parameters of the QAOA, minimizing the objective function.
4. **Iterations**: Similar to the classical approach, this method iteratively refines the beamforming vector \(f\) and receive vector \(g\) by alternating between subproblems.

## Procedure

1. **Initialization**: 
   - A random receive vector \(g\) is generated with binary phase values (either 0 or π), and a random initial beamforming vector \(f\) is also generated.
   
2. **SubQUBO 1 - Optimize Beamforming Vector \(f\)**:
   - In both approaches, the first subQUBO optimizes the beamforming vector \(f\) by maximizing the quadratic form \( |g^* H f|^2 \).
   - The graph-based approach translates this quadratic form into a quantum circuit, while the classical QAOA approach directly solves the QUBO using a classical optimizer.

3. **SubQUBO 2 - Optimize Receive Vector \(g\)**:
   - Using the optimized beamforming vector \(f\), the second subQUBO optimizes the receive vector \(g\), similarly maximizing the received signal power.
   - Again, the graph-based approach constructs circuits, while the classical approach relies on QUBO formulations.

4. **Iterations**:
   - The optimization alternates between the two subproblems, refining the vectors \(g\) and \(f\) iteratively until convergence.

5. **Post-Processing**:
   - The results are filtered based on the probability of success, and a histogram is generated to visualize the distribution of solutions.
   - In the graph-based approach, the quantum circuit outputs are evaluated based on the most probable bitstrings.

## Key Techniques

### 1. Quantum Approximate Optimization Algorithm (QAOA)
QAOA is a hybrid quantum-classical algorithm used to solve optimization problems by leveraging quantum circuits. It approximates solutions by alternating between classical optimization of quantum parameters and quantum execution.

### 2. Quadratic Unconstrained Binary Optimization (QUBO)
QUBO is a mathematical model that represents the beamforming problem in terms of binary variables and quadratic cost functions, which is the standard formulation for QAOA optimization.

### 3. Graph-based Circuit Construction
In the second approach, matrix multiplications are transformed into graphs, where nodes represent matrix elements and edges represent the connections between them. These graphs are then used to construct quantum circuits.

### Dependencies

Both approaches use similar core dependencies, including:
- **Qiskit**: For quantum circuit creation, execution, and QAOA solvers.
- **NumPy**: For linear algebra operations and handling complex matrices.
- **NetworkX**: In the graph-based approach, this is used for graph construction and visualization.
- **Matplotlib**: For visualizing the results, such as histograms of the QAOA solutions.
- **SciPy**: For numerical optimization in the graph-based approach.

### How to Use

1. **Input**:
   - The main input is the channel matrix \(H\), which characterizes the MIMO system. Parameters for the number of transmit (\(N_t\)) and receive (\(N_r\)) antennas are also required.

2. **Customization**:
   - The number of iterations, QAOA parameters, and optimizer settings (e.g., maximum iterations in COBYLA) can be adjusted to refine the results.

3. **Output**:
   - Both approaches output the optimized beamforming and receive vectors.
   - Probability distributions of the results can be visualized with histograms to assess solution quality.

## Conclusion

This repository provides two implementations of QAOA for beamforming optimization in MIMO systems:
1. A classical QAOA approach leveraging standard QUBO solvers.
2. A custom graph-based QAOA approach constructing quantum circuits from scratch using matrix-graph representations.

Both methods aim to maximize the received signal power by iteratively refining beamforming vectors in a constrained phase system (0 or π).
