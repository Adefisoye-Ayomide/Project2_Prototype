import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define the parameters
box_length = 10  # Length of the 1D box
temperature = 1.0  # Temperature for the simulation
num_steps = 100000  # Number of Metropolis steps
num_bins = 50  # Number of quantum energy levels
mass = 1.0  # Mass of the particle
# initial_position_mm = 5.0  # Initial position of the particle in mm

# Sample initial position and velocity randomly
initial_velocity = random.uniform(-1.0, 1.0)  # Initial velocity

# Define a non-uniform probability distribution function for rejection sampling
def non_uniform_probability_distribution(x):
    return 2.0 * np.exp(-2.0 * x)

# Initialize the particle's position using rejection sampling
while True:
    proposed_position = random.uniform(0, box_length)
    acceptance_probability = non_uniform_probability_distribution(proposed_position)
    if random.random() < acceptance_probability:
        initial_position = proposed_position
        break

# Convert the initial position to meters
initial_position_meters = proposed_position / 1000.0

# Define the potential energy landscape (harmonic potential)
def potential_energy(x, k):
    return 0.5 * k * (x - box_length / 2)**2

# Define the ODE system for particle motion 
def particle_motion(y, t, k):
    position, velocity = y
    force = -k * (position - box_length / 2)
    acceleration = force / mass
    return [velocity, acceleration]


# Simulation parameters
k = 1.0  # Spring constant
time_step = 0.01  # Time step for ODE integration
total_time = num_steps * time_step  # Total simulation time

# Additional parameters for ODE integration
current_position = initial_position_meters
current_velocity = initial_velocity

positions = [current_position]
velocities = [current_velocity]

# Define the probability of surface interaction
surface_interaction_probability = 0.1

# Initialize lists to store results
positions = []
energies = []

# Define the quantum energy levels and associated probabilities
def quantum_energy_levels(num_bins, box_length):
    levels = []
    probabilities = []
    hbar = 1.0545718e-34  # Planck's constant divided by 2π
    total_energy = 0.0

    for n in range(1, num_bins + 1):
        level_energy = (n**2 * np.pi**2 * hbar**2) / (2 * mass * box_length**2)
        levels.append(level_energy)
        total_energy += level_energy

    for energy in levels:
        probability = energy / total_energy
        probabilities.append(probability)

    return levels, probabilities

quantum_levels, quantum_probabilities = quantum_energy_levels(num_bins, box_length)

# Generate random positions based on the inverse CDF
def sample_position(quantum_probabilities):
    U = random.random()
    n = 1

    # Find the corresponding quantum state using the inverse CDF
    while n < num_bins and U > sum(quantum_probabilities[:n]):
        n += 1

    return random.uniform(0, box_length), n

# Metropolis algorithm with ODE integration
for step in range(num_steps):
    position, n = sample_position(quantum_probabilities)
    current_energy = quantum_levels[n - 1]

    # Update position and velocity based on the ODE integration
    time_points = np.linspace(0, time_step, 2)
    ode_solution = odeint(particle_motion, [current_position, current_velocity], time_points, args=(k,))
    current_position, current_velocity = ode_solution[-1]  # Use the last values from the ODE solution

    # Probabilistic surface interaction
    if random.random() < surface_interaction_probability:
        # Interaction occurred, decide between specular or diffuse reflection
        if random.random() < 0.5:
            # Specular reflection (reverse velocity)
            current_velocity = -current_velocity
        else:
            # Diffuse reflection (randomize velocity)
            current_velocity = random.uniform(-1.0, 1.0)

    # Calculate the energy based on the new position
    proposed_energy = quantum_levels[n - 1]

    # Calculate the energy difference
    energy_difference = proposed_energy - current_energy

    # Metropolis acceptance/rejection step
    if energy_difference <= 0 or random.random() < np.exp(-energy_difference / temperature):
        # Accept the move
        position = current_position

    # Store the accepted position and velocity
    positions.append(position)
    velocities.append(current_velocity)
    energies.append(current_energy)

# Calculate the probability distribution
hist, bins = np.histogram(positions, bins=num_bins, range=(0, box_length), density=True)
bin_centers = 0.5 * (bins[1:] + bins[:-1])

# Plot the probability distribution
plt.figure(figsize=(10, 6))
plt.hist(positions, bins=num_bins, range=(0, box_length), density=True, alpha=0.7, color='b')
plt.plot(bin_centers, hist, 'ro-', label='Probability Distribution')
# plt.axvline(x=initial_position_meters, color='g', linestyle='--', label='Initial Position')
plt.xlabel('Position (m)')
plt.ylabel('Probability')
plt.title('Monte Carlo Simulation: Particle in a 1D Box with Quantum Energy Levels, Harmonic Potential, and Probabilistic Surface Interaction')
plt.legend()
plt.grid()

# Show the plot
plt.show()
