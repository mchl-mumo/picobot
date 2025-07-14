import random

# --- Hyperparameters and Environment Setup ---
# Dimensions of the environment grid (analogous to the state space in RL)
HEIGHT = 25
WIDTH = 25

# Number of discrete internal states for the agent (Picobot)
NUMSTATES = 5

# All possible local observations (surroundings) for the agent
POSSIBLE_SURROUNDINGS = ['xxxx', 'Nxxx', 'NExx', 'NxWx', 'xxxS', 'xExS', 'xxWS', 'xExx', 'xxWx']

# Proportion of top-performing individuals (parents) to retain for the next generation
PARENT_SIZE = 0.1

# --- Genetic Program Representation ---
class Program:
    """
    Represents a policy (chromosome) for Picobot, mapping (state, observation) pairs to (action, next_state).
    This is the individual in the genetic algorithm population.
    """
    def __init__(self):
        """
        Initializes an empty rule set (policy genome) for the agent.
        """
        self.rules = {}  # (state, surroundings) -> (action, next_state)

    def __repr__(self):
        """
        Returns a human-readable string of the policy (genome).
        Useful for logging and analysis of evolved solutions.
        """
        unsorted_conditions = list(self.rules.keys())
        sorted_conditions = sorted(unsorted_conditions)
        formated_rules = ''
        
        current_state = 0
        for condition in sorted_conditions:
            formated_rule = f'{condition[0]} {condition[1]} -> {self.rules[condition][0]} {self.rules[condition][1]} \n'
            
            # Adds empty line transitioning before new state
            if current_state != condition[0]:
                current_state = condition[0]
                formated_rule = '\n' + formated_rule

            formated_rules += formated_rule

        return formated_rules

    def randomize(self):
        """
        Initializes the policy genome with random legal actions for each (state, observation) pair.
        This is analogous to random initialization of weights in ML models.
        """
        for state in range(NUMSTATES):
            for surroundings in POSSIBLE_SURROUNDINGS:
                possible_steps = []
                for step in 'NEWS':
                    if step not in surroundings:
                        possible_steps.append(step)
                condition = (state, surroundings)
                outcome = (random.choice(possible_steps), random.randint(0, NUMSTATES - 1))
                self.rules[condition] = outcome

    def get_move(self, state, surroundings):
        """
        Given the current state and observation, returns the action and next state as dictated by the policy genome.
        """

        outcome = self.rules[(state, surroundings)]
        return outcome
    
    def mutate(self):
        """
        Applies a random mutation to the policy genome, altering one rule.
        This introduces genetic diversity and enables exploration of the search space.
        """

        random_rule = random.choice(list(self.rules.keys()))
        current_outcome = self.rules[random_rule]

        possible_steps = []
        for step in 'NEWS':
            if step not in random_rule[1]:
                possible_steps += [step]

        new_outcome = (random.choice(possible_steps), random.randint(0, NUMSTATES - 1))
        if new_outcome == current_outcome:
            new_outcome = (random.choice(possible_steps), random.randint(0, NUMSTATES - 1))
        
        self.rules[random_rule] = new_outcome

    def crossover(self, other):
        """
        Performs crossover (recombination) between two parent genomes to produce a child genome.
        This simulates sexual reproduction in evolutionary algorithms.
        """

        child = Program()

        self_states = []
        other_states = []
        division = random.randint(0, NUMSTATES - 2)
        for state in range(0, 5):
            if state <= division:
                self_states.append(state)
            else:
                other_states.append(state)

        for condition in list(self.rules.keys()):
            if condition[0] in self_states:
                child.rules[condition] = self.rules[condition]
            else:
                break

        for condition in list(other.rules.keys()):
            if condition[0] in other_states:
                child.rules[condition] = other.rules[condition]

        return child 

    def __gt__(self, other):
        """
        Placeholder for fitness-based comparison (not used in selection).
        """
        return random.choice([True, False])

    def __lt__(self, other):
        """
        Placeholder for fitness-based comparison (not used in selection).
        """
        return random.choice([True, False])

# --- Environment Simulation ---
class World:
    """
    Simulates the environment (grid world) in which Picobot operates.
    Used to evaluate the fitness of a policy genome by running the agent and measuring coverage.
    """
    def __init__(self, program, initial_row=random.randint(1, WIDTH - 2), initial_column=random.randint(1, HEIGHT - 2)):
        """
        Initializes the environment and places the agent at a random starting position.
        """
        self.row = initial_row
        self.column = initial_column
        self.state = 0
        self.program = program
        self.room = [[' ']*WIDTH for row in range(HEIGHT)]

        for col in range(WIDTH):
            self.room[0][col] = '-'
            self.room[HEIGHT - 1][col] = '-'

        for row in range(HEIGHT):
            self.room[row][0] = '|'
            self.room[row][WIDTH - 1] = '|'

        self.room[0][0] = '+'
        self.room[0][WIDTH - 1] = '+'
        self.room[HEIGHT - 1][0] = '+'
        self.room[HEIGHT - 1][WIDTH - 1] = '+'
        
        self.room[self.row][self.column] = 'P'

    def __repr__(self):
        """
        Returns a string representation of the environment grid.
        Useful for visualization and debugging.
        """
        w = ''

        for row in range(HEIGHT):
            for col in range(WIDTH):
                w += self.room[row][col]
            w += '\n'

        return w
        

    def get_current_surroundings(self):
        """
        Returns the agent's local observation (used as input to the policy genome).
        """  
        if self.room[self.row - 1][self.column] != '-':
            north = 'x'
        else:
            north = 'N'

        if self.room[self.row + 1][self.column] != '-':
            south = 'x'
        else:
            south = 'S'

        if self.room[self.row][self.column - 1] != '|':
            west = 'x'
        else:
            west = 'W'

        if self.room[self.row][self.column + 1] != '|':
            east = 'x'
        else:
            east = 'E'

        current_surroundings = f'{north}{east}{west}{south}'
        return current_surroundings

    def step(self):
        """
        Executes one action in the environment according to the agent's policy genome.
        Updates the agent's state and position.
        """
        surroundings = self.get_current_surroundings()
        outcomes = self.program.get_move(self.state, surroundings)
        #update previous position to o
        self.room[self.row][self.column] = 'o'
        self.state = outcomes[1]

        # make new position P
        if outcomes[0] == "N":
            self.row = self.row - 1
            self.room[self.row][self.column] = 'P'
        if outcomes[0] == "S":
            self.row = self.row + 1
            self.room[self.row][self.column] = 'P'
        if outcomes[0] == "W":
            self.column = self.column - 1
            self.room[self.row][self.column] = 'P'
        if outcomes[0] == "E":
            self.column = self.column + 1
            self.room[self.row][self.column] = 'P'


    def run(self, steps):
        """
        Runs the agent for a fixed number of steps (episode length).
        """
        for i in range(steps):
            self.step()

    def fraction_visited_cells(self):
        """
        Computes the fitness of the agent as the fraction of unique cells visited.
        This is the reward signal for the genetic algorithm.
        """

        visited = 1 # For the current location of picobot
        total = 1
        for row in self.room:
            for cell in row:
                if cell == 'o':
                    visited += 1
                    total += 1
                if cell == ' ':
                    total += 1
        
        return visited / total
    


# --- Genetic Algorithm Core Functions ---
def population(size):
    """
    Initializes a population of random policy genomes (agents).
    """

    programs = []
    for x in range(0, size):
        #How do I name different programs?

        prog = Program()
        prog.randomize()

        programs.append(prog)

    return programs

def save_to_file(filename, p):
    """
    Saves the policy genome to a file for later analysis or reproduction.
    """
    f = open(filename, "w")
    print(p, file = f)        
    f.close()

def evaluate_fitness(program, trials, steps):
    """
    Evaluates the fitness of a policy genome by running it in multiple randomized environments.
    Returns the average coverage (reward) over all trials.
    """
    all_fitness = []

    for x in range(trials):
        w = World(program)
        w.run(steps)
        fitness = w.fraction_visited_cells()
        all_fitness.append(fitness)

    avg_fitness = sum(all_fitness) / trials
    return avg_fitness

def GA(popsize, numgens):
    """
    Main loop for the genetic algorithm:
    - Initializes a population of random agents (policy genomes)
    - Evaluates fitness (coverage) for each agent
    - Selects top-performing parents (selection)
    - Generates new population via crossover and mutation (variation)
    - Repeats for a fixed number of generations
    Returns the best evolved policy genome.
    """

    #Initialize the population and list of fitness
    programs = population(popsize)
    L = []
    gen = 1

    while gen <= numgens:
        #Evaluate fitness of programs in the population
        total_fitness = 0
        for program in programs:
            fitness = evaluate_fitness(program, 20, 800)
            L.append((fitness, program))
            total_fitness += fitness

        SL = sorted(L, reverse=True)

        #identify the fittest parents
        keep = int(PARENT_SIZE * popsize)
        fittest_programs = SL[0: keep]

        # Print fitnesses
        avg_fitness = total_fitness/popsize
        best_fitness = SL[0][0]
        best_program = SL[0][1]
        save_to_file(f"gen{gen}.txt", best_program)
        print(
            f'Generation {gen}'
            f'  Average fitness: {avg_fitness}'
            f'  Best fitness: {best_fitness}'
        )

        # Print final generation
        if gen == numgens:
            print("Best picobot program")
            print(best_program)
            return best_program

        #make children for next full generation
        programs.clear()
        L.clear()
        for x in range(popsize):
            parent1 = random.choice(fittest_programs)[1]
            parent2 = random.choice(fittest_programs)[1]
            while parent1 == parent2:
                parent2 = random.choice(fittest_programs)[1]
            child = parent1.crossover(parent2)
            if gen % 3 == 0:
                child.mutate()
            programs.append(child)

        #go to next generation
        gen += 1

        