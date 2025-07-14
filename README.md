# Picobot Genetic Algorithm

## Introduction
This project implements an evolutionary algorithm to automatically generate control policies for Picobot, a simple agent navigating a grid world. The goal is to evolve a set of rules (a policy) that enables Picobot to maximize its coverage of the environment, analogous to maximizing reward in reinforcement learning.

## Approach
We use a **Genetic Algorithm (GA)**, a population-based optimization technique inspired by natural selection. Each individual in the population encodes a policy (a mapping from internal state and local observation to action and next state). Over successive generations, the population evolves through selection, crossover, and mutation, guided by a fitness function that rewards policies which visit more unique cells in the environment.

### Key Machine Learning Concepts
- **Population**: A set of candidate policies (genomes) evaluated each generation.
- **Fitness**: The reward signal, defined as the fraction of unique cells visited by Picobot when following a policy.
- **Selection**: The process of choosing the top-performing policies to serve as parents for the next generation.
- **Crossover (Recombination)**: Combining parts of two parent policies to create a new child policy, promoting exploration of the search space.
- **Mutation**: Randomly altering a policy to introduce genetic diversity and avoid local optima.

## Code Structure
- `picobot.py`: Main implementation of the genetic algorithm, agent (Program), and environment (World).
  - **Program**: Encodes a policy genome, supports random initialization, mutation, and crossover.
  - **World**: Simulates the grid environment and evaluates the agent's performance (fitness).
  - **GA**: Main genetic algorithm loop, including population initialization, fitness evaluation, selection, crossover, mutation, and generational updates.
- `gen*.txt`: Snapshots of the best policy from each generation, useful for analysis and reproducibility.

## Usage
1. **Run the Genetic Algorithm**
   - Open `picobot.py` and call the `GA(popsize, numgens)` function with your desired population size and number of generations. For example:
     ```python
     GA(100, 20)
     ```
   - The script will print fitness statistics for each generation and save the best policy to `genX.txt` files.

2. **Analyze Results**
   - Inspect the `gen*.txt` files to see the evolved policies.
   - You can visualize the agent's behavior by instantiating a `World` with a saved policy and calling its `run()` and `__repr__()` methods.

## Example
```python
from picobot import Program, World
# Load a policy from a file or create a new one
p = Program()
# ...initialize or load rules...
w = World(p)
w.run(500)
print(w)
```

## Extending the Project
- Experiment with different fitness functions, mutation rates, or selection strategies.
- Try alternative representations for the policy genome.
- Integrate more advanced evolutionary or reinforcement learning techniques.

## References
- [Genetic Algorithms](https://en.wikipedia.org/wiki/Genetic_algorithm)
- [Evolutionary Computation](https://en.wikipedia.org/wiki/Evolutionary_computation)
- [Picobot Assignment (HMC CS5)](https://www.cs.hmc.edu/picobot/) 