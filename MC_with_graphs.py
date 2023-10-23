import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Defining parameters
box_length = 10  # Length of the 1D box
temperature = 1.0  # Temperature for the simulation
num_steps = 100000  # Number of Metropolis steps
num_bins = 50  # Number of bins
mass = 1.0  # Mass of the particle
surface_interaction_probability = 0

# Sample initial velocity randomly
initial_velocity = random.uniform(0, 1.0)

# Define particle's initial position using a non-uniform probability distribution function (rejection sampling)
def non_uniform_probability_distribution(x):
    return 2.0 * np.exp(-2.0 * x)

# Initialize the particle's position using rejection sampling
while True:
    proposed_position = random.uniform(0, box_length)
    acceptance_probability = non_uniform_probability_distribution(proposed_position)
    if random.random() < acceptance_probability:
        initial_position = proposed_position
        break

# Define an harmonic potential acting on the particle (source of second order ODE in which the particle is evolving)
def potential_energy(x, k):
    return 0.5 * k * (x - box_length / 2)**2

# Define the ODE system influencing the particle's motion
def particle_motion(y, t, k):
    position, velocity = y
    force = -k * (position - box_length / 2)
    acceleration = force / mass

    if random.random() < surface_interaction_probability:
        if random.random() < 0.5:
            # Specular reflection (reverse velocity)
            velocity = -velocity
        else:
            # Diffuse reflection (randomize velocity)
            velocity = random.uniform(-1.0, 1.0)

    return [velocity, acceleration]

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


# Define a function for Metropolis simulation
def metropolis_simulation(num_steps, quantum_probabilities, time_step, k, surface_interaction_probability, potential_energy=False):

    # Initialize lists to store results
    positions = []

    # Additional parameters for ODE integration
    current_position = initial_position
    current_velocity = initial_velocity

    for step in range(num_steps):
        # Sample a quantum state
        n = random.choices(range(1, num_bins + 1), quantum_probabilities)[0]

        # Update position and velocity based on the ODE integration
        time_points = np.linspace(0, time_step, 2)
        ode_solution = odeint(particle_motion, [current_position, current_velocity], time_points, args=(k,))
        current_position, current_velocity = ode_solution[-1]

        # Probabilistic surface interaction
        if random.random() < surface_interaction_probability:
            if random.random() < 0.5:
                # Specular reflection (reverse velocity)
                current_velocity = -current_velocity
            else:
                # Diffuse reflection (randomize velocity)
                current_velocity = random.uniform(-1.0, 1.0)

        # Calculate the energy based on the quantum state
        current_energy = quantum_levels[n - 1]

        # Calculate the energy difference
        if potential_energy:
            proposed_energy = current_energy + potential_energy(current_position, k)
        else:
            proposed_energy = current_energy

        energy_difference = proposed_energy - current_energy

        # Metropolis acceptance/rejection step
        if energy_difference <= 0 or random.random() < np.exp(-energy_difference / temperature):
            # Accept the move
            current_position = current_position

        # Store the accepted position
        positions.append(current_position)

    return positions

# Define the quantum energy levels and associated probabilities
quantum_levels, quantum_probabilities = quantum_energy_levels(num_bins, box_length)

# Simulate different scenarios
positions_potential_and_surface = metropolis_simulation(num_steps, quantum_probabilities, 0.01, 1.0, 0.1, potential_energy)
positions_surface_interaction = metropolis_simulation(num_steps, quantum_probabilities, 0.01, 0, 0.1)
positions_harmonic_potential = metropolis_simulation(num_steps, quantum_probabilities, 0.01, 1.0, 0)

# Create histograms for each scenario
hist_potential_and_surface, bins_potential_and_surface = np.histogram(positions_potential_and_surface, bins=num_bins, range=(0, box_length), density=True)
hist_surface_interaction, bins_surface_interaction = np.histogram(positions_surface_interaction, bins=num_bins, range=(0, box_length), density=True)
hist_harmonic_potential, bins_harmonic_potential = np.histogram(positions_harmonic_potential, bins=num_bins, range=(0, box_length), density=True)

# Calculate bin centers for plotting
bin_centers_potential_and_surface = 0.5 * (bins_potential_and_surface[1:] + bins_potential_and_surface[:-1])
bin_centers_surface_interaction = 0.5 * (bins_surface_interaction[1:] + bins_surface_interaction[:-1])
bin_centers_harmonic_potential = 0.5 * (bins_harmonic_potential[1:] + bins_harmonic_potential[:-1])

# Create plots in separate figures
plt.figure(figsize=(8, 6))
plt.hist(positions_potential_and_surface, bins=num_bins, range=(0, box_length), density=True, alpha=0.7, color='g')
plt.plot(bin_centers_potential_and_surface, hist_potential_and_surface, 'ro-')
plt.xlabel('Position (m)')
plt.ylabel('Probability')
plt.title('Harmonic Potential and Surface Interaction')

# Create plots
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.hist(positions_potential_and_surface, bins=num_bins, range=(0, box_length), density=True, alpha=0.7, color='g')
plt.plot(bin_centers_potential_and_surface, hist_potential_and_surface, 'ro-')
plt.xlabel('Position (m)')
plt.ylabel('Probability')
plt.title('Harmonic Potential and Surface Interaction')

plt.subplot(2, 2, 2)
plt.hist(positions_surface_interaction, bins=num_bins, range=(0, box_length), density=True, alpha=0.7, color='m')
plt.plot(bin_centers_surface_interaction, hist_surface_interaction, 'ro-')
plt.xlabel('Position (m)')
plt.ylabel('Probability')
plt.title('Surface Interaction Only')

plt.subplot(2, 2, 3)
plt.hist(positions_harmonic_potential, bins=num_bins, range=(0, box_length), density=True, alpha=0.7, color='c')
plt.plot(bin_centers_harmonic_potential, hist_harmonic_potential, 'ro-')
plt.xlabel('Position (m)')
plt.ylabel('Probability')
plt.title('Harmonic Potential Only')

plt.tight_layout()
plt.show()