import abc
from typing import Collection, Optional, Tuple, Mapping, Any

import networkx as nx

from pgeon.discretizer import Discretizer, StateRepresentation, Action
from pgeon.policy_approximator import ProbabilityQuery

class IntentionMixin:
    ...


class PolicyRepresentation(abc.ABC):
    def __init__(self):
        self._discretizer: Discretizer

    @staticmethod
    @abc.abstractmethod
    def load(path: str) -> "PolicyRepresentation":
        ...

    @abc.abstractmethod
    def save(self, ext: str, path: str):
        ...

    @abc.abstractmethod
    def get_possible_actions(self, state: StateRepresentation) -> Collection[Action]:
        ...

    @abc.abstractmethod
    def get_possible_next_states(self, state: StateRepresentation, action: Optional[Action] = None) -> Collection[StateRepresentation]:
        ...


class GraphRepresentation(PolicyRepresentation):

    # Package-agnostic
    class Graph(abc.ABC):
        @abc.abstractmethod
        def add_node(self, node: StateRepresentation, **kwargs) -> None:
            ...

        @abc.abstractmethod
        def add_nodes_from(self, nodes: Collection[StateRepresentation]) -> None:
            ...

        @abc.abstractmethod
        def add_edge(self, node_from: StateRepresentation, node_to: StateRepresentation, action: Action) -> None:
            ...

        @abc.abstractmethod
        def add_edges_from(self, edges: Collection[Tuple[StateRepresentation, StateRepresentation, Action]]) -> None:
            ...

        @abc.abstractmethod
        def get_edge_data(self, node_from: StateRepresentation, node_to: StateRepresentation, action: Action) -> dict:
            ...

    class NetworkXGraph(nx.MultiDiGraph, Graph):
        def __init__(self):
            self.nx_graph = nx.MultiDiGraph()
        
        def __getitem__(self, node: StateRepresentation) -> Mapping[Action, Any]:
            return self.nx_graph[node]

        def add_node(self, node: StateRepresentation, **kwargs) -> None:
            self.nx_graph.add_node(node, **kwargs)

        def add_nodes_from(self, nodes: Collection[StateRepresentation], **kwargs) -> None:
            self.nx_graph.add_nodes_from(nodes, **kwargs)

        def add_edge(self, node_from: StateRepresentation, node_to: StateRepresentation, **kwargs) -> None:
            self.nx_graph.add_edge(node_from, node_to, **kwargs)

        def add_edges_from(self, edges: Collection[Tuple[StateRepresentation, StateRepresentation, Action]]) -> None:
            self.nx_graph.add_edges_from(edges)

        def get_edge_data(self, node_from: StateRepresentation, node_to: StateRepresentation, action: Action) -> Mapping[Action, Any]:
            return self.nx_graph.get_edge_data(node_from, node_to, action)


    def __init__(self, graph_backend: str = "networkx"):
        super().__init__()
        # p(s) and p(s',a | s)
        self.graph: GraphRepresentation.Graph
        match graph_backend:
            case "networkx":
                self.graph = GraphRepresentation.NetworkXGraph()
            case _:
                raise NotImplementedError

    def prob(self, query: ProbabilityQuery) -> float:
        ...

    # This refers to getting all states present in graph. Some representations may not be able to iterate over
    #   all states.
    def get_states_in_graph(self) -> Collection[StateRepresentation]:
        ...

    def get_possible_actions(self, state: StateRepresentation) -> Collection[Action]:
        ...

    def get_possible_next_states(self, state: StateRepresentation, action: Optional[Action] = None) -> Collection[StateRepresentation]:
        ...

    # minimum P(s',a|p) forall possible probs.
    def get_overall_minimum_state_transition_probability(self) -> float:
        ...

    @staticmethod
    def load(path: str) -> "PolicyRepresentation":
        ...

    def save(self, ext: str, path: str):
        pass


class IntentionalPolicyGraphRepresentation(GraphRepresentation, IntentionMixin):
    ...