import random
from deap import base, tools, algorithms, creator
from functools import partial
import matplotlib.pyplot as plt

# Constants
USER_INVESTMENT_AMOUNT = 10000  # Example user investment amount
EPSILON = 1e-6  # Small value for numerical stability

# Define problem dimensions
NUM_OBJECTIVES = 2
NUM_DECISION_VARIABLES = 2

# Create a Multi-Objective Fitness class
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))

# Create an individual class with decision variables and fitness attributes
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

# Decision variable bounds (normalized)
toolbox.register("attr_float", random.uniform, 0.0, 1.0)

# Structure the individual and population
toolbox.register(
    "individual",
    tools.initRepeat,
    creator.Individual,
    toolbox.attr_float,
    n=NUM_DECISION_VARIABLES,
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def eval_risk_fitness(expense_ratio, volatility):
    return 1 / (volatility + EPSILON)


def eval_returns_fitness(expense_ratio, volatility, sigma):
    return (1 - expense_ratio) + sigma * volatility


def evaluate(individual, sigma):
    # Extract decision variables
    expense_ratio, volatility = individual
    expense_ratio = abs(expense_ratio)
    volatility = abs(volatility)

    # Ensure very small values are treated as zero
    epsilon = 1e-10
    expense_ratio = max(epsilon, expense_ratio)
    volatility = max(epsilon, volatility)

    # Fitness for returns
    returns_fitness = eval_returns_fitness(expense_ratio, volatility, sigma)

    # Fitness for risk
    risk_fitness = eval_risk_fitness(expense_ratio, volatility)

    # Return as a tuple (minimize risk, maximize returns)
    return risk_fitness, returns_fitness


def mutate(individual, mu, sigma, indpb):
    mutated = individual.copy()  # Create a copy to avoid modifying the original

    # Apply mutation to each element of the individual
    for i in range(len(mutated)):
        if random.random() < indpb:
            mutation_range = 0.2  # Adjust mutation range as needed
            mutated[i] += random.uniform(-mutation_range, mutation_range)
            mutated[i] = max(
                0, min(1, mutated[i])
            )  # Ensure values are within [0, bounds

    for i in range(len(mutated)):
        if mutated[i] < 0:
            mutated[i] = abs(mutated[i])

    return (creator.Individual(mutated),)


sigma = 0.5  # Risk aversion parameter, higher values lead to less risky solutions


evaluate_with_sigma = partial(evaluate, sigma=sigma)
toolbox.register("evaluate", evaluate_with_sigma)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", mutate, mu=0, sigma=0.2, indpb=0.2)
toolbox.register("select", tools.selNSGA2)

population = toolbox.population(n=500)  # Adjust population size as needed

# Number of generations
n_gen = 10
original_population = list(population)
# Run the algorithm
algorithms.eaMuPlusLambda(
    population,
    toolbox,
    mu=100,
    lambda_=200,
    cxpb=0.7,
    mutpb=0.2,
    ngen=n_gen,
    stats=None,
    halloffame=None,
    verbose=True,
)

pareto_front = tools.sortNondominated(
    population, len(population), first_front_only=True
)[0]

# Extract solutions from the Pareto front
pareto_solutions = [list(pareto_front[i]) for i in range(len(pareto_front))]

# Choose the best solution based on user's investment amount
best_solution = max(
    pareto_solutions,
    key=lambda x: eval_returns_fitness(x[0], x[1], sigma) * USER_INVESTMENT_AMOUNT,
)

# Print the best solution
print(
    f"Best solution for user investment amount of RM {USER_INVESTMENT_AMOUNT} and sigma of {sigma}"
)
print(f"Generated {n_gen} generations")
print(f"Expense ratio: {round(best_solution[0], 2)}")
print(f"Volatility: {round(best_solution[1], 2)}")
print(
    f"Risk fitness: {round(eval_risk_fitness(best_solution[0], best_solution[1]), 2)}"
)
print(
    f"Returns fitness: {round(eval_returns_fitness(best_solution[0], best_solution[1], sigma), 2)}"
)
print(
    f"Returns: {round(eval_returns_fitness(best_solution[0], best_solution[1], sigma) * USER_INVESTMENT_AMOUNT, 2)}"
)

pareto_solutions = [list(pareto_front[i]) for i in range(len(pareto_front))]

# Plot the Volatility vs Expense Ratio of the solutions

plt.scatter(
    [ind[0] for ind in original_population],
    [ind[1] for ind in original_population],
    c="blue",
    label="Initial Population",
)
plt.scatter(
    [ind[0] for ind in pareto_solutions],
    [ind[1] for ind in pareto_solutions],
    c="red",
    label="Pareto Front",
)
plt.scatter(
    best_solution[0],
    best_solution[1],
    c="green",
    label="Best Solution",
)

# Set the scale to log to better visualize the spread of solutions
plt.yscale("log")
plt.xscale("log")

plt.xlabel("Volatility")
plt.ylabel("Expense Ratio")

plt.title("Volatility vs Expense Ratio")
plt.legend()
plt.show()


# Show another plot with the returns fitness and risk fitness of the solutions
plt.scatter(
    [eval_risk_fitness(ind[0], ind[1]) for ind in original_population],
    [eval_returns_fitness(ind[0], ind[1], sigma) for ind in original_population],
    c="blue",
    label="Initial Population",
)

plt.scatter(
    [eval_risk_fitness(ind[0], ind[1]) for ind in pareto_solutions],
    [eval_returns_fitness(ind[0], ind[1], sigma) for ind in pareto_solutions],
    c="red",
    label="Pareto Front",
)

plt.scatter(
    eval_risk_fitness(best_solution[0], best_solution[1]),
    eval_returns_fitness(best_solution[0], best_solution[1], sigma),
    c="green",
    label="Best Solution",
)

plt.xlabel("Risk Fitness")
plt.ylabel("Returns Fitness")

plt.xscale("log")
plt.yscale("log")

plt.title("Risk Fitness vs Returns Fitness")
plt.legend()
plt.show()
