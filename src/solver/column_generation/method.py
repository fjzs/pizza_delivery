from problem.cvrp import CVRP, Route
from problem.solution import Solution
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
        max_iterations: int,
    ):
        """Initializes the solver

        Args:
            instance (CVRP): data for the problem
            folder (str): folder to save results
            max_iterations (int, optional): Max iterations to run column generation
        """
        self.instance: CVRP = instance
        self.initial_solution: Solution = None
        self.log = Log()
        self.folder = folder

        # heuristic solution
        self.initial_solution = initial_solution.closest_client(instance)
        # self.initial_solution = initial_solution.one_route_per_client(instance)
        instance.draw(
            self.initial_solution.routes,
            title=f"Heuristic Cost = {round(self.initial_solution.total_cost, 1)}",
            filename="heuristic",
            folder_to_save=folder,
        )
        instance.draw(
            self.initial_solution.routes,
            title=f"Heuristic Cost = {round(self.initial_solution.total_cost, 1)}",
            filename="current",
            folder_to_save=folder,
        )

        # Create the master problem and fill it with the initial solution
        self.master = MasterProblem(self.initial_solution)

        # Create the pricing problem
        self.pricing = PricingProblem(
            distances=self.instance.distance,
            capacity=self.instance.q,
            demand=self.instance.demand,
        )

        for i in range(max_iterations):
            print(
                f"\nMASTER ITERATION: {i+1} ---------------------------------------------\n"
            )

            # Get the current state to show performance evolution
            # self.master.build_model(is_linear=False)
            # self.master.solve()
            # obj_bound, obj_value = self.master.get_Obj_Values()
            # self.log.add(
            #     iteration=i,
            #     of_linear_lower_bound=obj_bound,
            #     of_integer_optimal_value=obj_value,
            #     number_routes=len(self.master.routes),
            #     min_reduced_cost=last_min_reduced_cost_entered,
            # )

            # Solve the Integer MP and draw solution
            obj_value_integer, _ = self._solve_MP(is_linear=False)
            self._draw_current_solution(i + 1)

            # Solve the Linear MP to get the duals
            obj_value_linear, client_duals = self._solve_MP(is_linear=True)
            self.pricing.set_duals(client_duals)
            # print(f"\nClient duals: {client_duals}")

            # Solve the pricing problem to find reduced-cost columns
            print("\nSolving the pricing problem...")
            min_reduced_cost_entered = self._solve_pricing()

            # Record the log
            self.log.add(
                iteration=i + 1,
                of_linear_lower_bound=obj_value_linear,
                of_integer_optimal_value=obj_value_integer,
                number_routes=len(self.master.routes),
                min_reduced_cost=min_reduced_cost_entered,
            )

            # If there are no negative reduced-cost columns, stop the process
            if min_reduced_cost_entered is None:
                print(f"No reduced-cost < 0 columns found...")
                break

        print(f"\n\nCOLUMN GENERATION ENDED!!!")

        # Now save the log
        self.log.save(folder=folder)
        self.log.plot(folder=folder)

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
                self.instance.is_valid_route(route)
                cost = self.instance.get_route_cost(route)
                self.master.add_route(route, cost)
                print(f"\t {i+1}: red-cost: {reduced_cost} path: {path}, cost: {cost}")
        return min_reduced_cost_entered

    def _draw_current_solution(self, iteration: int, characters_length=3):
        """Saves the current drawing of the MIP solution

        Args:
            iteration (int):
            characters_length (int, optional):
        """
        solution = self.master.get_solution()
        name = str(iteration).zfill(characters_length)
        self.instance.draw(
            routes=solution.routes,
            title=f"Iteration #{name}, Z = {round(solution.total_cost, 1)}",
            filename=name,
            folder_to_save=self.folder,
        )

        # This is to create a video effect in visual studio, I'll have a window only
        # displaying this solution so I can see in real time how the MIP changes
        self.instance.draw(
            routes=solution.routes,
            title=f"Iteration #{name}, Z = {round(solution.total_cost, 1)}",
            filename="current",
            folder_to_save=self.folder,
        )
