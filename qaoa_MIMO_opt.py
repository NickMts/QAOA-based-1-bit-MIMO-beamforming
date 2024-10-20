
"""
QAOA-based Beamforming Optimization in MIMO Systems
This script implements a Quantum Approximate Optimization Algorithm (QAOA) 
with Successive Convex Approximation (SCA) for optimizing beamforming in 
Multiple-Input Multiple-Output (MIMO) systems. The goal is to optimize the 
phases of beamforming vectors to maximize the received signal power, where 
the phases are constrained to be either 0 or π.

Author: NickMts
Date: October 2024
"""

import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_algorithms import QAOA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit.visualization import plot_histogram

# Set a random seed for reproducibility
np.random.seed(51)

# Generate a random complex vector 'g' with phases constrained to 0 or π
g = np.random.choice([-1, 1], size=(N_r, 1)) + np.random.choice([-1, 1], size=(N_r, 1)) * 1j

# Iterate through the optimization process 3 times
for i in range(3):
    print(i)
    
    # Create 1st subQUBO --> Maximize |Af|^2 where A = g^*H
    A = np.dot(np.conjugate(g).T, H)  # Compute A = g^*H
    Q = np.conjugate(A).T  # Conjugate transpose of A
    Q = np.dot(Q, A) / 1  # Multiply A with its conjugate transpose
    Q_R = Q.real  # Real part of Q
    Q_I = Q.imag  # Imaginary part of Q

    # Formulate the quadratic form in binary +-1 variables (transformed to 0,1)
    quadratic_1 = (np.dot(Q_R, Q_R.T) + np.dot(Q_I, Q_I.T)) / (N_r * N_t)
    linear = 4 * np.dot(np.ones((1, N_t)), quadratic_1)  # Linear term
    quadratic = -4 * quadratic_1  # Quadratic term

    # Define the QUBO problem using the Qiskit QuadraticProgram
    qubo = QuadraticProgram()
    qubo.binary_var_list(keys=N_t, name="x", key_format='_{}')
    qubo.minimize(linear=linear.flatten(), quadratic=quadratic)

    # Define QAOA parameters and optimizer
    qaoa_mes = QAOA(sampler=Sampler(), optimizer=COBYLA(maxiter=1000), reps=1, initial_point=[1, 1])
    qaoa = MinimumEigenOptimizer(qaoa_mes)  # Use QAOA to solve the QUBO

    # Solve the QUBO using QAOA
    qaoa_result = qaoa.solve(qubo)

    #### Create the 2nd subQUBO for optimizing the receive vector g

    # Update the beamforming vector f from the QAOA result (convert back to +-1)
    f = 1 - 2 * qaoa_result.x
    f = np.reshape(f, (N_t, 1))  # Reshape f into a column vector

    # Compute A = Hf and recalculate the Q matrix
    A = np.dot(H, f)
    Q = np.conjugate(A).T
    Q = np.dot(A, Q) / 1  # Multiply A by its conjugate transpose
    Q_R = Q.real  # Real part of Q
    Q_I = Q.imag  # Imaginary part of Q

    # Recalculate the quadratic and linear terms for the 2nd subQUBO
    quadratic_1 = (np.dot(Q_R, Q_R.T) + np.dot(Q_I, Q_I.T)) / (N_r * N_t)
    linear = 4 * np.dot(np.ones((1, N_r)), quadratic_1)
    quadratic = -4 * quadratic_1

    # Define the second QUBO problem
    qubo_1 = QuadraticProgram()
    qubo_1.binary_var_list(keys=N_r, name="x", key_format='_{}')
    qubo_1.minimize(linear=linear.flatten(), quadratic=quadratic)

    # Solve the second QUBO using QAOA
    qaoa_mes_1 = QAOA(sampler=Sampler(), optimizer=COBYLA(maxiter=1000), reps=1, initial_point=[0.5, 0.5])
    qaoa_1 = MinimumEigenOptimizer(qaoa_mes_1)
    qaoa_result_1 = qaoa_1.solve(qubo_1)

    # Update the receive vector g from the QAOA result
    g = 1 - 2 * qaoa_result_1.x
    g = np.reshape(g, (N_r, 1))

# Filter samples based on the optimization result and their probabilities
filtered_samples = get_filtered_samples(
    qaoa_result.samples, threshold=0.0001, allowed_status=(OptimizationResultStatus.SUCCESS,)
)

# Display the filtered samples
for s in filtered_samples:
    print(s)

# Calculate the mean and standard deviation of the objective function values
fvals = [s.fval for s in qaoa_result.samples]
probabilities = [s.probability for s in qaoa_result.samples]
print("Mean Objective Value:", np.mean(fvals))
print("Standard Deviation of Objective Value:", np.std(fvals))

# Prepare data for plotting the histogram of results
samples_for_plot = {
    " ".join(f"{qaoa_result.variables[i].name}={int(v)}" for i, v in enumerate(s.x)): s.probability
    for s in filtered_samples
}

# Plot the result histogram
plot_histogram(samples_for_plot)
