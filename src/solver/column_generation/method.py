from problem.cvrp import CVRP
from problem.solution import Solution

from .initial_solution import get_initial_solution
from .master_problem import MasterProblem


class SolverColumnGeneration:
    """This class has the responsibility of solving the CVRP
    with a column generation approach
    """

    def __init__(self, instance: CVRP):
        self.instance: CVRP = instance
        self.initial_solution: Solution = get_initial_solution(instance)
        # instance.draw(self.initial_solution)

        # Create the master problem and fill it with the initial solution
        self.master = MasterProblem(self.instance.N)
        self.master.build_model()
