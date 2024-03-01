from problem.cvrp import CVRP, Route
from problem.solution import Solution
from utils.drawer import Drawer

from . import heuristic_2_opt_inter_route, heuristic_2_opt_intra_route

MIN_IMPROVEMENT = 0.01


class Improver:
    """This class has the responsibility of improving a CVRP solution
    with heuristics
    """

    def __init__(self, initial_solution: Solution, max_iterations: int, drawer: Drawer):
        """Improves an initial solution with heuristics

        Args:
            initial_solution (Solution):
            max_iterations (int):
            drawer (Drawer):
        """
        self.initial_solution = initial_solution
        self.max_iterations = max_iterations
        self.instance = initial_solution.instance
        self.drawer = drawer

    def apply(self) -> Solution:
        """Generates a neighborhood of solutions and moves to the best one iteratively

        Returns:
            Solution:
        """

        best_solution = self.initial_solution
        best_cost = best_solution.get_cost()
        finish = False
        for _ in range(self.max_iterations):

            # Store the list of (cost, solution)
            solutions = []
            solutions.extend(
                heuristic_2_opt_intra_route.get_neighborhood(best_solution)
            )
            solutions.extend(
                heuristic_2_opt_inter_route.get_neighborhood(best_solution)
            )
            cost_solutions = []
            for s in solutions:
                cost_solutions.append((s.get_cost(), s))

            # Move to the best greedy
            cost_solutions.sort(key=lambda tup: tup[0])  # sort by cost
            if cost_solutions[0][0] + MIN_IMPROVEMENT < best_cost:
                best_cost = cost_solutions[0][0]
                best_solution = cost_solutions[0][1]
            else:
                finish = True

            # Plot the current solution
            self.drawer.draw_solution(best_solution, filename=None, save_iteration=True)

            if finish:
                break

        return best_solution
