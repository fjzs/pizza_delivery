from problem.cvrp import CVRP, Route
from problem.solution import Solution
from solver.improver.method import Improver
from utils.drawer import Drawer
from utils.logger import Log

from . import initial_solution
from .master_problem import MasterProblem
from .pricing_problem import PricingProblem


class SolverColumnGeneration:
    """This class has the responsibility of solving the CVRP
    with a column generation approach
    """

    def __init__(
        self,
        instance: CVRP,
        folder: str,
        cg_max_iterations: int,
        improvement_iterations: int,
        heuristic: str = "cw",
    ):
        """Initializes the solver

        Args:
            instance (CVRP): data for the problem
            folder (str): folder to save results
            cg_max_iterations (int, optional): Max iterations to run column generation
            improvement_iterations (int): Iterations to improve the solution.
            heuristic (str): what heuristic to use to construct initial solution, it can
            be 'cw' (Clarke & Wright), 'closest' (Closest client iteratively) or 'single'
            (one route per client)
        """
        assert heuristic in ["cw", "closest", "single"]
        assert cg_max_iterations >= 0
        assert improvement_iterations >= 0

        self.instance: CVRP = instance
        self.solution: Solution = None
        self.log = Log()
        self.folder = folder
        self.cg_max_iterations = cg_max_iterations
        self.improver_max_iterations = improvement_iterations
        self.heuristic = heuristic
        self.drawer = Drawer(self.folder, self.instance)

        # Apply the heuristic for solution construction
        self._apply_heuristic()

        # Apply column generation
        self._apply_column_generation()

        # Applying improvement
        self._apply_improver()

        # Now save the log
        self.log.save(folder=folder)

    def _apply_heuristic(self):
        """Applies the heuristic to generate an initial solution"""

        # Save the heuristic as the first iteration
        if self.heuristic == "cw":
            self.solution = initial_solution.clarke_and_wright(self.instance)
        elif self.heuristic == "closest":
            self.solution = initial_solution.closest_client(self.instance)
        elif self.heuristic == "single":
            self.solution = initial_solution.one_route_per_client(self.instance)
        else:
            raise ValueError(f"Heuristic {self.heuristic} not recognized")

        # Log this info
        self.log.add(
            of_linear_lower_bound=None,
            of_integer_optimal_value=self.solution.get_cost(),
            number_routes=len(self.solution.routes),
            min_reduced_cost=None,
        )

        # Draw the first iteration of the algorithm
        self.drawer.draw_of_and_solution(self.solution, self.log)

    def _apply_column_generation(self):
        """Runs the column generation algorithm for a fixed amount of iterations
        or until there are no more reduced-cost columns (routes)
        """

        if self.cg_max_iterations == 0:
            return

        # Create the master problem and fill it with the initial solution
        self.master = MasterProblem(self.solution)

        # Create the pricing problem
        self.pricing = PricingProblem(
            distances=self.instance.distance,
            capacity=self.instance.q,
            demand=self.instance.demand,
        )

        for i in range(self.cg_max_iterations):
            print(f"\n\n========== MASTER ITERATION: {i+1} ==========")

            # Solve the Linear MP to get the duals
            print("\n\nSOLVING LINEAR MP")
            print("----------------------------------------------")
            obj_value_linear, client_duals = self._solve_MP(is_linear=True)
            self.pricing.set_duals(client_duals)
            # print(f"\nClient duals: {client_duals}")

            # Solve the pricing problem to find reduced-cost columns
            print("\n\nSOLVING PRICING PROBLEM")
            print("----------------------------------------------")
            min_reduced_cost_entered = self._solve_pricing()

            # Solve the Integer MP to **see** the current solution (this can be skipped until the final iteration)
            print("\nSOLVING INTEGER MP")
            print("----------------------------------------------")
            obj_value_integer, _ = self._solve_MP(is_linear=False)
            self.solution = self.master.get_solution()

            # Record the log
            self.log.add(
                of_linear_lower_bound=obj_value_linear,
                of_integer_optimal_value=obj_value_integer,
                number_routes=len(self.master.routes),
                min_reduced_cost=min_reduced_cost_entered,
            )

            # Draw the current integer solution
            self.drawer.draw_of_and_solution(self.solution, self.log)

            # If there are no negative reduced-cost columns, stop the process
            if min_reduced_cost_entered is None:
                print(f"No reduced-cost < 0 columns found...")
                break

        print(f"\n\nCOLUMN GENERATION ENDED!!!")

    def _solve_MP(self, is_linear: bool):
        """Solves the Restricted Master Problem (RMP), either in its linear or integer form

        Args:
            is_linear (bool): True to solve the linear version

        Returns:
            * obj_value (float)
            * client_duals (Dict[int, float] or None if MP is integer)
        """
        self.master.build_model(is_linear=is_linear)
        self.master.solve()
        obj_value = self.master.get_Obj_Value()
        client_duals = None
        if is_linear:
            client_duals = self.master.get_duals()
        return obj_value, client_duals

    def _solve_pricing(self):
        """Solves the pricing problem to find reduced-cost columns. If there are
        those are added to the Master Problem.

        Returns:
        * min_reduced_cost_entered (float): the minimum reduced-cost column found
        """
        cost_path_solutions = self.pricing.solve()
        min_reduced_cost_entered = None
        if len(cost_path_solutions) > 0:
            min_reduced_cost_entered = min([c for c, p in cost_path_solutions])
            print("\nResults of Pricing Problem:")
            for i, (reduced_cost, path) in enumerate(cost_path_solutions):
                # add route to the master problem
                path[-1] = 0  # replace the last virtual node with the actual depot
                route = Route(path)
                if self.instance.is_valid_route(route):
                    cost = self.instance.get_route_cost(route)
                    self.master.add_route(route, cost)
                    print(
                        f"\t {i+1}: red-cost: {reduced_cost} path: {path}, cost: {cost}"
                    )
                else:
                    raise ValueError()
        return min_reduced_cost_entered

    def _apply_improver(self):
        improver = Improver(self.solution, self.improver_max_iterations, self.drawer)
        self.solution = improver.apply(self.log)
