from itertools import combinations, permutations
from typing import List, Set

import numpy as np


class ESPCC:
    """This class has the responsibility of solving the
    Elementary Shortest Path Problem with Capacity Constraints
    Source: Column Generation Book (2005)
    """

    def __init__(
        self, costs: np.ndarray, times: np.ndarray, T: float, source: int, target: int
    ):
        """Initialization of the problem

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
        
        self.N = N # number of nodes
        self.costs = costs
        self.times = times
        self.T = T
        self.source = source
        self.target = target

    
    def solve_enumeration(self, max_num_solutions: int, max_clients_per_path: int) -> List[tuple[float, List[int]]]:
        """The enumeration procedure finds solutions of the type:
        source -> M -> target, where M is a set of unique nodes different than {s, t},
        such that the solution is feasible and the cost is < 0.

        Args:
            max_num_solutions (int)
            max_clients_per_path (int)

        Returns:
            solutions (List[float, List[int]]): list of (cost, path)
        """
        assert max_num_solutions > 0

        # Add all the feasible paths here with its cost and set of node
        solutions: List[float, List[int]] = []
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
                paths = [[self.source] + list(s) + [self.target] for s in sequences]
                for p in paths:
                    paths_explored += 1
                    feasible, cost = self.evaluate_path(p)
                    if feasible and cost < 0:
                        solutions.append((cost, p))
        print(f"\tExplored {paths_explored} paths")
        
        solutions.sort()
        return solutions[0 : min(len(solutions), max_num_solutions)]

    def evaluate_path(self, p: List[int]) -> tuple[bool, float]:
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
