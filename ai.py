import array
import math
import random
from deap import base, tools, algorithms, creator
from functools import partial
import matplotlib.pyplot as plt

# Constants
USER_INVESTMENT_AMOUNT = 2000  # Example user investment amount
EPSILON = 1e-6  # Small value for numerical stability

# Define problem dimensions
NUM_OBJECTIVES = 2
NUM_DECISION_VARIABLES = 2

# Define bounds for decision variables
BOUND_LOW, BOUND_UP = 0.7, 1.0


def calculate(user_investment_amount, sigma):
    """
    Calculate the optimal expense ratio and portfolio turnover ratio for a unit trust fund.

    Parameters:
    - user_investment_amount (float): The amount the user wants to invest in the fund.
    - expense_ratio (float): The expense ratio of the fund.
    - portfolio_turnover_ratio (float): The portfolio turnover ratio of the fund.
    - sigma (float): The risk aversion parameter.

    Returns:
    - result (dict): A dictionary containing the optimal expense ratio, portfolio turnover ratio, risk fitness, returns fitness, and returns.
    """
    # Create a Multi-Objective Fitness class
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))

    # Create an individual class with decision variables and fitness attributes
    creator.create(
        "Individual", array.array, typecode="d", fitness=creator.FitnessMulti
    )

    toolbox = base.Toolbox()

    # Decision variable bounds
    toolbox.register("attr_float", random.uniform, BOUND_LOW, BOUND_UP)

    # Structure the individual and population
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_float,
        n=NUM_DECISION_VARIABLES,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate_risk(expense_ratio, portfolio_turnover_ratio, sigma):
        """
        Evaluate the risk of a unit trust fund based on expense ratio and portfolio turnover ratio.

        Parameters:
        - expense_ratio (float): The expense ratio of the fund.
        - portfolio_turnover_ratio (float): The portfolio turnover ratio of the fund.

        Returns:
        - risk (float): The evaluated risk of the unit trust fund.
        """
        risk = (
            (sigma * abs(portfolio_turnover_ratio)) / (abs(expense_ratio) + EPSILON)
        ) / 10
        return risk

    def evaluate_return(expense_ratio, portfolio_turnover_ratio, sigma):
        """
        Evaluate the return fitness of a unit trust fund based on expense ratio and portfolio turnover ratio.

        Parameters:
        - expense_ratio (float): The expense ratio of the fund.
        - portfolio_turnover_ratio (float): The portfolio turnover ratio of the fund.
        - sigma (float): The risk aversion parameter.

        Returns:
        - return_fitness (float): The evaluated return fitness of the unit trust fund.
        """
        return_fitness = max(0, (1 - abs(expense_ratio))) + sigma * abs(
            portfolio_turnover_ratio
        )

        return return_fitness

    # TEST_EXPENSE_RATIO = 0.1
    # TEST_PORTFOLIO_TURNOVER_RATIO = 0.008
    # TEST_SIGMA = 0.8

    # print(
    #     f"Risk: {round(evaluate_risk(TEST_EXPENSE_RATIO, TEST_PORTFOLIO_TURNOVER_RATIO), 2)}"
    # )
    # print(
    #     f"Returns: {round(evaluate_return(TEST_EXPENSE_RATIO, TEST_PORTFOLIO_TURNOVER_RATIO, TEST_SIGMA), 2)}"
    # )
    # exit()

    def evaluate(individual, sigma):
        # Extract decision variables
        expense_ratio, portfolio_turnover_ratio = individual

        # Fitness for returns
        returns_fitness = evaluate_return(
            expense_ratio, portfolio_turnover_ratio, sigma
        )

        # Fitness for risk
        risk_fitness = evaluate_risk(expense_ratio, portfolio_turnover_ratio, sigma)

        # Ensure solutions are within bounds
        expense_ratio = max(BOUND_LOW, min(BOUND_UP, expense_ratio))
        portfolio_turnover_ratio = max(
            BOUND_LOW, min(BOUND_UP, portfolio_turnover_ratio)
        )

        individual[0] = expense_ratio
        individual[1] = portfolio_turnover_ratio

        # Return as a tuple (minimize risk, maximize returns)
        return (risk_fitness, returns_fitness)

    # def mutate(individual, mu, sigma, indpb):
    #     mutated = individual.copy()  # Create a copy to avoid modifying the original

    #     # Apply mutation to each element of the individual
    #     for i in range(len(mutated)):
    #         if random.random() < indpb:
    #             mutation_range = 0.2  # Adjust mutation range as needed
    #             mutated[i] += random.uniform(-mutation_range, mutation_range)
    #             mutated[i] = max(
    #                 0, min(1, mutated[i])
    #             )  # Ensure values are within [0, bounds

    #     for i in range(len(mutated)):
    #         if mutated[i] < 0:
    #             mutated[i] = abs(mutated[i])

    #     return (creator.Individual(mutated),)

    # sigma = 0.9  # Risk aversion parameter, higher values lead to less risky solutions

    evaluate_with_sigma = partial(evaluate, sigma=sigma)
    toolbox.register("evaluate", evaluate_with_sigma)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register(
        "mutate", tools.mutPolynomialBounded, eta=20, low=0, up=1, indpb=0.2
    )
    toolbox.register("select", tools.selNSGA2)

    POPULATION_SIZE = 300
    N_GEN = 100
    MUTATION_PROBABILITY = 0.2

    population = toolbox.population(n=POPULATION_SIZE)

    # Run the algorithm
    algorithms.eaMuPlusLambda(
        population,
        toolbox,
        mu=math.floor(POPULATION_SIZE / 3),
        lambda_=POPULATION_SIZE,
        cxpb=1.0 - MUTATION_PROBABILITY,
        mutpb=MUTATION_PROBABILITY,
        ngen=N_GEN,
        stats=None,
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
        key=lambda x: evaluate_return(x[0], x[1], sigma) * user_investment_amount,
    )

    # Print the best solution
    print(
        f"Best solution for user investment amount of RM {USER_INVESTMENT_AMOUNT} and sigma of {sigma}"
    )
    print(f"Generated {N_GEN} generations")
    print(f"Expense ratio: {round(best_solution[0], 2)}")
    print(f"Portfolio turnover ratio: {round(best_solution[1], 2)}")
    print(
        f"Risk fitness: {round(evaluate_risk(best_solution[0], best_solution[1], sigma), 2)}"
    )
    print(
        f"Returns fitness: {round(evaluate_return(best_solution[0], best_solution[1], sigma), 2)}"
    )
    print(
        f"Returns: {round(evaluate_return(best_solution[0], best_solution[1], sigma) * USER_INVESTMENT_AMOUNT, 2)}"
    )

    pareto_solutions = [list(pareto_front[i]) for i in range(len(pareto_front))]

    # Plot the Portfolio turnover ratio vs Expense Ratio of the solutions

    plt.figure(1)

    plt.scatter(
        [ind[0] for ind in population],
        [ind[1] for ind in population],
        c="blue",
        label="Last Population",
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

    plt.xlabel("Expense Ratio")
    plt.ylabel("Portfolio turnover ratio")

    plt.title("Expense Ratio vs Portfolio Turnover Ratio")
    plt.legend()

    # Show another plot with the returns fitness and risk fitness of the solutions

    plt.figure(2)

    plt.scatter(
        [evaluate_risk(ind[0], ind[1], sigma) for ind in population],
        [evaluate_return(ind[0], ind[1], sigma) for ind in population],
        c="blue",
        label="Last Population",
    )

    plt.scatter(
        [evaluate_risk(ind[0], ind[1], sigma) for ind in pareto_solutions],
        [evaluate_return(ind[0], ind[1], sigma) for ind in pareto_solutions],
        c="red",
        label="Pareto Front",
    )

    # plt.scatter(
    #     evaluate_risk(best_solution[0], best_solution[1], sigma),
    #     evaluate_return(best_solution[0], best_solution[1], sigma),
    #     c="green",
    #     label="Best Solution",
    # )

    plt.xlabel("Risk Fitness")
    plt.ylabel("Returns Fitness")

    # plt.xscale("log")
    # plt.yscale("log")

    plt.title("Risk Fitness vs Returns Fitness")
    plt.legend()
    plt.show()

    return {
        "expense_ratio": 1 + round(best_solution[0], 2),
        "portfolio_turnover_ratio": round(best_solution[1], 2),
        "risk_fitness": round(
            evaluate_risk(best_solution[0], best_solution[1], sigma), 2
        ),
        "returns_fitness": round(
            evaluate_return(best_solution[0], best_solution[1], sigma), 2
        ),
        "returns": round(
            evaluate_return(best_solution[0], best_solution[1], sigma)
            * user_investment_amount,
            2,
        ),
    }


def main():
    calculate(USER_INVESTMENT_AMOUNT, 0.5)


if __name__ == "__main__":
    main()
