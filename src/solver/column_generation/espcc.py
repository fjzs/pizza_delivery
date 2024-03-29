from itertools import combinations, permutations
from typing import List, Set

import cspy
import networkx as nx
import numpy as np

REDUCED_COST_MAX_VALUE = -0.1
TIME_LIMIT_S = 10
REPETITIONS_PER_HEURISTIC = 3


class ESPCC:
    """This class has the responsibility of solving the
    Elementary Shortest Path Problem with Capacity Constraints
    Source: Column Generation Book (2005)
    """

    def __init__(
        self, costs: np.ndarray, times: np.ndarray, T: float, source: int, target: int
    ):
        """Initialization of the . There are N nodes, where the index 0 and N-1 are the
        source and sink respectively

        Args:
            - costs (np.ndarray): this is the matrix cost to minimize, it has shape (N, N)
            - times (np.ndarray): this is the resource, it has shape (N, N)
            - T (float): this is the total amount of resource available
            - source (int): starting node index
            - target (int): ending node index
        """
        # Check costs
        assert costs is not None
        assert isinstance(costs, np.ndarray)
        assert len(costs.shape) == 2
        assert costs.shape[0] == costs.shape[1]  # square matrix
        N = costs.shape[0]

        # Check times
        assert times is not None
        assert isinstance(times, np.ndarray)
        assert len(times.shape) == 2
        assert times.shape == costs.shape
        assert np.all(times >= 0)
        assert T >= 0

        # Check source and target
        assert source is not None
        assert target is not None
        assert 0 <= source < N
        assert 0 <= target < N
        assert source != target

        self.N = N  # number of nodes
        self.costs = costs
        self.times = times
        self.T = T
        self.source = source
        self.target = target

    def _solve_exact_enumeration(
        self, max_num_solutions: int, max_clients_per_path: int
    ) -> List[List[int]]:
        """The enumeration procedure finds solutions of the type:
        source -> M -> target, where M is a set of unique nodes different than {s, t},
        such that the solution is feasible and the cost is < 0.

        Args:
            max_num_solutions (int)
            max_clients_per_path (int)

        Returns:
            paths (List[List[int]]): list of paths
        """
        assert max_num_solutions > 0

        # Add all the feasible paths here with its cost and set of node
        paths: List[List[int]] = []
        M = list(range(self.N))
        M.remove(self.source)
        M.remove(self.target)

        # Get all the routes of size 1, 2, ..., max_clients_in_route
        paths_explored = 0
        for i in range(1, max_clients_per_path + 1):
            print(f"\tAnalyzing combinations C({len(M)},{i})")
            clients_combinations = combinations(M, i)
            for clients in clients_combinations:
                sequences = permutations(clients)
                sequence_path = [
                    [self.source] + list(s) + [self.target] for s in sequences
                ]
                paths.extend(sequence_path)
                paths_explored += len(sequence_path)
        print(f"\tExplored {paths_explored} paths")
        return paths

    def _evaluate_path(self, p: List[int]) -> tuple[bool, float]:
        """Evaluates a path and returns if its feasible and its cost

        Args:
            p (List[int]): path

        Returns:
            tuple[bool, float]: feasible, cost
        """
        feasible = True
        time = 0
        cost = 0
        i = p[0]
        for j in p[1:]:
            cost += self.costs[i, j]
            time += self.times[i, j]
            i = j
            if time > self.T:
                feasible = False
                break
        return feasible, cost

    def _prepare_graph(self):
        """
        Prepares the graph to be used in the cspy library
        """
        G = nx.DiGraph(directed=True, n_res=2)

        # Add the edges to this graph
        # Found that the algorithms require the nodes to be str
        for i in range(self.N):
            name_i = "Source" if i == 0 else str(i)
            for j in range(self.N):
                name_j = "Sink" if j == self.N - 1 else str(j)
                dist_ij = self.costs[i, j]
                time_ij = self.times[i, j]
                if dist_ij != np.inf:
                    G.add_edge(
                        name_i,
                        name_j,
                        weight=dist_ij,
                        res_cost=np.asarray([0, time_ij]),
                    )
        self.graph = G
        print(f"\tGraph has {len(G.edges)} edges")

    def _solve_with_cspy(
        self, method: str, threshold: float = REDUCED_COST_MAX_VALUE
    ) -> List[int]:
        """
        Uses the Bidirectional labeling algorithm with dynamic halfway point from Tilk et al (2017), implemented
        in the package cspy. https://cspy.readthedocs.io/en/latest/python_api/cspy.BiDirectional.html

        Args:
        * method (str)
        * threshold (float)

        Returns:
        * path (List[int]): an elementary path in the graph: ['Source', ..., 'Sink']
        * reduced_cost (float)
        """

        # print(f"\nEdges in the ESPCC:")
        # for edge in G.edges(data=True):
        #     print(edge)

        algorithm = None
        if method == "bidirectional":
            algorithm = cspy.BiDirectional(
                G=self.graph,
                max_res=[self.T, self.T],
                min_res=[0, 0],
                elementary=True,
                time_limit=TIME_LIMIT_S,
                threshold=threshold,
            )
        elif method == "tabu":
            algorithm = cspy.Tabu(
                G=self.graph,
                max_res=[self.T, self.T],
                min_res=[0, 0],
                threshold=threshold,
                time_limit=TIME_LIMIT_S,
            )
        elif method == "greedy":
            algorithm = cspy.GreedyElim(
                G=self.graph,
                max_res=[self.T, self.T],
                min_res=[0, 0],
                threshold=threshold,
                time_limit=TIME_LIMIT_S,
            )
        elif method == "grasp":  # not working
            algorithm = cspy.GRASP(
                G=self.graph,
                max_res=[self.T, self.T],
                min_res=[0, 0],
                threshold=threshold,
                time_limit=TIME_LIMIT_S,
            )
        elif method == "psolgent":
            algorithm = cspy.PSOLGENT(
                G=self.graph,
                max_res=[self.T, self.T],
                min_res=[0, 0],
                threshold=threshold,
                time_limit=TIME_LIMIT_S,
            )
        else:
            raise ValueError(f"method: {method} not recognized")

        path = None
        red_cost = None

        try:
            print(f"\t\tthreshold: {threshold}")
            algorithm.run()
            path = algorithm.path
            red_cost = algorithm.total_cost
            print(f"\t\tpath: {path}")
            print(f"\t\treduced-cost: {red_cost}")
            print(f"\t\tdemand captured: {algorithm.consumed_resources[1]} / {self.T}")
            if path[0] == "Source" and path[-1] == "Sink":
                pass
            else:
                raise ValueError(f"path does not match: {path}")
        except:
            print(f"\t\t[!] did not return a solution")
            pass

        return path, red_cost

    def solve(self) -> List[tuple[float, List[int]]]:
        """Solves the ESPCC by multiple methods to get various columns

        Returns:
            solutions (List[float, List[int]]): list of (cost, path)
        """

        # First prepare the graph
        self._prepare_graph()

        # Solve with heuristics, only use the exact method if the heuristics fail
        # psolgent works with time limit
        # tabu does not work with time limit
        # greedy does not work with time limit
        paths = []
        heuristics = ["psolgent"]
        for h in heuristics:
            print(f"\n\tRunning heuristic: {h}")
            path, reduced_cost_original = self._solve_with_cspy(method=h)
            if path is not None:
                paths.append(path)

                # Do a binary search to find multiple good paths
                right = reduced_cost_original  # this is negative
                left = reduced_cost_original * 10  # this is more negative
                for i in range(REPETITIONS_PER_HEURISTIC):
                    print(f"\n\t\titeration {i+1} / {REPETITIONS_PER_HEURISTIC}")
                    mid = (right + left) / 2.0
                    path, red_cost = self._solve_with_cspy(method=h, threshold=mid)

                    # there was a solution in the time limit
                    if path is not None:
                        paths.append(path)
                        right = red_cost * 0.75
                        left = min(left, red_cost * 1.50)

                    # there was not a solution in the time limit
                    else:
                        left = mid

        if len(paths) == 0:
            print(f"\n\tRunning exact: bidirectional")
            path, reduced_cost_original = self._solve_with_cspy(method="bidirectional")
            if path is not None:
                paths.append(path)

        # Check the paths and compute their cost and feasibility
        solutions: List[float, List[int]] = []
        for p in paths:
            if p is not None:  # p is like ['Source', '2', '5', 'Sink']
                path_ints = []  # need to transform to ints
                for i, node in enumerate(p):
                    if i == 0:  # This is 'Source'
                        path_ints.append(0)
                    elif i == len(p) - 1:  # This is 'Sink'
                        path_ints.append(self.N - 1)
                    else:
                        path_ints.append(int(node))

                feasible, cost = self._evaluate_path(path_ints)
                if feasible and cost < REDUCED_COST_MAX_VALUE:
                    solutions.append((cost, path_ints))

        # Return some of the solutions found
        return solutions
