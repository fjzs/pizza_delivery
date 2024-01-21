from problem.cvrp import CVRP

from .initial_solution import get_initial_solution


class SolverColumnGeneration:
    """This class has the responsibility of solving the CVRP
    with a column generation approach
    """

    def __init__(self, instance: CVRP):
        self.instance = instance
        self.initial_solution = get_initial_solution(instance)
        instance.draw(self.initial_solution)
