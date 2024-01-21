from initial_solution import get_initial_solution

from cvrp.cvrp import CVRP


class SolverColumnGeneration:
    """This class has the responsibility of solving the CVRP
    with a column generation approach
    """

    def __init__(self, data: CVRP):
        self.data = data
        self.initial_solution = get_initial_solution(data)
