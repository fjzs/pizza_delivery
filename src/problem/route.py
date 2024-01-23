from typing import List, Set


class Route:
    """This class manages a Route"""

    def __init__(self, nodes: List[int]):
        """Create a route by giving the list of nodes to visit

        Args:
            nodes (List[int]): [0, c1, c2, ..., 0]
        """
        self.nodes: List[int] = nodes
        """List of ordered nodes to visit, for instance:
        [0, 4, 3, 0]
        """
        self.clients: Set[int] = set(self.nodes[1:-1])
        """Set of clients to visit, for instance:
        {3, 4}
        """
