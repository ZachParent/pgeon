import abc
from typing import (
    Collection,
    Optional,
    Tuple,
    Any,
    Dict,
    Iterator,
    TypeVar,
    cast,
)

import networkx as nx

from pgeon.discretizer import Discretizer, StateRepresentation, Action

# Define generic types for better type hinting
S = TypeVar("S", bound=StateRepresentation)
A = TypeVar("A", bound=Action)


class IntentionMixin: ...


class ProbabilityQuery: ...

class PolicyRepresentation(abc.ABC):
    def __init__(self):
        self._discretizer: Discretizer

    @staticmethod
    @abc.abstractmethod
    def load(path: str) -> "PolicyRepresentation": ...

    @abc.abstractmethod
    def save(self, ext: str, path: str): ...

    @abc.abstractmethod
    def get_possible_actions(
        self, state: StateRepresentation
    ) -> Collection[Action]: ...

    @abc.abstractmethod
    def get_possible_next_states(
        self, state: StateRepresentation, action: Optional[Action] = None
    ) -> Collection[StateRepresentation]: ...

    @abc.abstractmethod
    def has_node(self, node: StateRepresentation) -> bool:
        """Check if a node exists in the graph"""
        ...

    @abc.abstractmethod
    def add_node(self, node: StateRepresentation, **kwargs) -> None:
        """Add a node to the graph with attributes"""
        ...

    @abc.abstractmethod
    def add_nodes_from(self, nodes: Collection[StateRepresentation], **kwargs) -> None:
        """Add multiple nodes to the graph with attributes"""
        ...

    @abc.abstractmethod
    def add_edge(
        self, node_from: StateRepresentation, node_to: StateRepresentation, **kwargs
    ) -> None:
        """Add an edge to the graph with attributes"""
        ...

    @abc.abstractmethod
    def add_edges_from(
        self,
        edges: Collection[Tuple[StateRepresentation, StateRepresentation, Action]],
        **kwargs,
    ) -> None:
        """Add multiple edges to the graph with attributes"""
        ...

    @abc.abstractmethod
    def get_edge_data(
        self, node_from: StateRepresentation, node_to: StateRepresentation, key: Any
    ) -> Dict:
        """Get edge data for a specific edge"""
        ...

    @abc.abstractmethod
    def has_edge(
        self,
        node_from: StateRepresentation,
        node_to: StateRepresentation,
        key: Any = None,
    ) -> bool:
        """Check if an edge exists in the graph"""
        ...

    @abc.abstractmethod
    def get_node_attributes(self, name: str) -> Dict[StateRepresentation, Any]:
        """Get node attributes"""
        ...

    @abc.abstractmethod
    def set_node_attributes(
        self, attributes: Dict[StateRepresentation, Any], name: str
    ) -> None:
        """Set node attributes"""
        ...

    @abc.abstractmethod
    def nodes(self) -> Iterator[StateRepresentation]:
        """Get all nodes in the graph"""
        ...

    @abc.abstractmethod
    def edges(self, data: bool = False) -> Iterator:
        """Get all edges in the graph"""
        ...

    @abc.abstractmethod
    def out_edges(self, node: StateRepresentation, data: bool = False) -> Iterator:
        """Get outgoing edges from a node"""
        ...

    @abc.abstractmethod
    def clear(self) -> None:
        """Clear the graph"""
        ...

    def __getitem__(self, node: StateRepresentation) -> Any:
        """Get the successors of a node"""
        ...


class GraphRepresentation(PolicyRepresentation):

    # Package-agnostic
    class Graph(abc.ABC):
        @abc.abstractmethod
        def add_node(self, node: StateRepresentation, **kwargs) -> None: ...

        @abc.abstractmethod
        def add_nodes_from(
            self, nodes: Collection[StateRepresentation], **kwargs
        ) -> None: ...

        @abc.abstractmethod
        def add_edge(
            self, node_from: StateRepresentation, node_to: StateRepresentation, **kwargs
        ) -> None: ...

        @abc.abstractmethod
        def add_edges_from(
            self,
            edges: Collection[Tuple[StateRepresentation, StateRepresentation, Action]],
            **kwargs,
        ) -> None: ...

        @abc.abstractmethod
        def get_edge_data(
            self, node_from: StateRepresentation, node_to: StateRepresentation, key: Any
        ) -> Dict: ...

        @abc.abstractmethod
        def has_node(self, node: StateRepresentation) -> bool: ...

        @abc.abstractmethod
        def has_edge(
            self,
            node_from: StateRepresentation,
            node_to: StateRepresentation,
            key: Any = None,
        ) -> bool: ...

        @abc.abstractmethod
        def nodes(self, data: bool = False) -> Iterator: ...

        @abc.abstractmethod
        def edges(self, data: bool = False) -> Iterator: ...

        @abc.abstractmethod
        def out_edges(
            self, node: StateRepresentation, data: bool = False
        ) -> Iterator: ...

        @abc.abstractmethod
        def clear(self) -> None: ...

        @abc.abstractmethod
        def __getitem__(self, node: StateRepresentation) -> Any: ...

        @property
        @abc.abstractmethod
        def nx_graph(self) -> nx.MultiDiGraph:
            """Return the underlying networkx graph if available"""
            ...

    class NetworkXGraph(Graph):
        def __init__(self):
            # Not calling super().__init__() since Graph is an ABC
            self._nx_graph = nx.MultiDiGraph()

        def __getitem__(self, node: StateRepresentation) -> Any:
            return cast(
                Dict[StateRepresentation, Dict[Any, Dict[str, Any]]],
                self._nx_graph[node],
            )

        def add_node(self, node: StateRepresentation, **kwargs) -> None:
            self._nx_graph.add_node(node, **kwargs)

        def add_nodes_from(
            self, nodes: Collection[StateRepresentation], **kwargs
        ) -> None:
            self._nx_graph.add_nodes_from(nodes, **kwargs)

        def add_edge(
            self, node_from: StateRepresentation, node_to: StateRepresentation, **kwargs
        ) -> None:
            self._nx_graph.add_edge(node_from, node_to, **kwargs)

        def add_edges_from(
            self,
            edges: Collection[Tuple[StateRepresentation, StateRepresentation, Action]],
            **kwargs,
        ) -> None:
            self._nx_graph.add_edges_from(edges, **kwargs)

        def get_edge_data(
            self, node_from: StateRepresentation, node_to: StateRepresentation, key: Any
        ) -> Dict:
            data = self._nx_graph.get_edge_data(node_from, node_to, key)
            return cast(Dict, data)

        def has_node(self, node: StateRepresentation) -> bool:
            return self._nx_graph.has_node(node)

        def has_edge(
            self,
            node_from: StateRepresentation,
            node_to: StateRepresentation,
            key: Any = None,
        ) -> bool:
            return self._nx_graph.has_edge(node_from, node_to, key)

        def nodes(self, data: bool = False) -> Iterator:
            return self._nx_graph.nodes(data=data)

        def edges(self, data: bool = False) -> Iterator:
            return self._nx_graph.edges(data=data)

        def out_edges(self, node: StateRepresentation, data: bool = False) -> Iterator:
            return self._nx_graph.out_edges(node, data=data)

        def clear(self) -> None:
            self._nx_graph.clear()

        @property
        def nx_graph(self) -> nx.MultiDiGraph:
            return self._nx_graph

    def __init__(self, graph_backend: str = "networkx"):
        super().__init__()
        # p(s) and p(s',a | s)
        self.graph: GraphRepresentation.Graph
        match graph_backend:
            case "networkx":
                self.graph = GraphRepresentation.NetworkXGraph()
            case _:
                raise NotImplementedError

    def prob(self, query: ProbabilityQuery) -> float: ...

    # This refers to getting all states present in graph. Some representations may not be able to iterate over
    #   all states.
    def get_states_in_graph(self) -> Collection[StateRepresentation]:
        return list(self.graph.nodes())

    def get_possible_actions(self, state: StateRepresentation) -> Collection[Action]:
        if not self.graph.has_node(state):
            return []
        actions = set()
        for _, _, key in self.graph.out_edges(state, data=False):
            actions.add(key)
        return list(actions)

    def get_possible_next_states(
        self, state: StateRepresentation, action: Optional[Action] = None
    ) -> Collection[StateRepresentation]:
        if not self.graph.has_node(state):
            return []
        if action is None:
            return [to_node for _, to_node in self.graph.out_edges(state)]
        next_states = []
        for _, to_node, key in self.graph.out_edges(state, data=True):
            if key.get("action") == action:
                next_states.append(to_node)
        return next_states

    # Forwarding methods to the graph
    def has_node(self, node: StateRepresentation) -> bool:
        return self.graph.has_node(node)

    def add_node(self, node: StateRepresentation, **kwargs) -> None:
        self.graph.add_node(node, **kwargs)

    def add_nodes_from(self, nodes: Collection[StateRepresentation], **kwargs) -> None:
        self.graph.add_nodes_from(nodes, **kwargs)

    def add_edge(
        self, node_from: StateRepresentation, node_to: StateRepresentation, **kwargs
    ) -> None:
        self.graph.add_edge(node_from, node_to, **kwargs)

    def add_edges_from(
        self,
        edges: Collection[Tuple[StateRepresentation, StateRepresentation, Action]],
        **kwargs,
    ) -> None:
        self.graph.add_edges_from(edges, **kwargs)

    def get_edge_data(
        self, node_from: StateRepresentation, node_to: StateRepresentation, key: Any
    ) -> Dict:
        return self.graph.get_edge_data(node_from, node_to, key)

    def has_edge(
        self,
        node_from: StateRepresentation,
        node_to: StateRepresentation,
        key: Any = None,
    ) -> bool:
        return self.graph.has_edge(node_from, node_to, key)

    def get_node_attributes(self, name: str) -> Dict[StateRepresentation, Any]:
        return nx.get_node_attributes(self.graph.nx_graph, name)

    def set_node_attributes(
        self, attributes: Dict[StateRepresentation, Any], name: str
    ) -> None:
        nx.set_node_attributes(self.graph.nx_graph, attributes, name)

    def nodes(self) -> Iterator[StateRepresentation]:
        return self.graph.nodes()

    def edges(self, data: bool = False) -> Iterator:
        return self.graph.edges(data=data)

    def out_edges(self, node: StateRepresentation, data: bool = False) -> Iterator:
        return self.graph.out_edges(node, data=data)

    def clear(self) -> None:
        self.graph.clear()

    def __getitem__(self, node: StateRepresentation) -> Any:
        return self.graph[node]

    # minimum P(s',a|p) forall possible probs.
    def get_overall_minimum_state_transition_probability(self) -> float: ...

    @staticmethod
    def load(path: str) -> "PolicyRepresentation": ...

    def save(self, ext: str, path: str):
        pass


class IntentionalPolicyGraphRepresentation(GraphRepresentation, IntentionMixin): ...
