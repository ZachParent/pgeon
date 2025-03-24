import unittest
from typing import cast, Type

from enum import Enum
from pgeon import GraphRepresentation, Predicate
from pgeon.policy_approximator import PolicyApproximatorFromBasicObservation
from test.domain.test_env import TestingEnv, TestingDiscretizer, TestingAgent, State
from pgeon.discretizer import StateRepresentation


class TestCreateGraphFromEnvironment(unittest.TestCase):
    env = TestingEnv()
    discretizer = TestingDiscretizer()
    policy_representation = GraphRepresentation()
    agent = TestingAgent()

    def test_initialize_approximator(self):
        """Test initializing a policy approximator."""
        approximator = PolicyApproximatorFromBasicObservation(
            self.discretizer, self.policy_representation, self.env, self.agent
        )

        self.assertEqual(approximator.discretizer, self.discretizer)
        self.assertEqual(approximator.policy_representation, self.policy_representation)
        self.assertEqual(approximator.environment, self.env)
        self.assertEqual(approximator.agent, self.agent)
        self.assertEqual(approximator._is_fit, False)
        self.assertEqual(approximator._trajectories_of_last_fit, [])

    def test_fit_from_agent_and_env(self):
        """Test fitting a policy approximator from an agent and environment."""
        # Clear the representation
        self.policy_representation.clear()

        # Create a new approximator
        approximator = PolicyApproximatorFromBasicObservation(
            self.discretizer, self.policy_representation, self.env, self.agent
        )

        # Fit the approximator
        approximator.fit(n_episodes=1)

        # Check that the approximator is fit
        self.assertEqual(approximator._is_fit, True)

        # Check that states were added
        states = self.policy_representation.get_all_states()
        self.assertEqual(len(states), 4)

        # Define the expected states
        # State is an Enum type, not a value
        state_type: Type[Enum] = State
        p = [
            cast(StateRepresentation, (Predicate(state_type, [State.ZERO]),)),
            cast(StateRepresentation, (Predicate(state_type, [State.ONE]),)),
            cast(StateRepresentation, (Predicate(state_type, [State.TWO]),)),
            cast(StateRepresentation, (Predicate(state_type, [State.THREE]),)),
        ]

        # Check that all the expected states are in the representation
        for state in p:
            self.assertIn(state, states)

        # Check transitions between states - use include_data=True to get the correct format
        transitions = list(
            self.policy_representation.get_all_transitions(include_data=True)
        )

        # Extract source, target and action from transitions to check
        expected_transitions_types = [
            (p[0], p[1]),
            (p[1], p[2]),
            (p[2], p[3]),
            (p[3], p[0]),
        ]

        # Check that each expected transition exists in some form
        for from_state, to_state in expected_transitions_types:
            transition_exists = False
            for edge in transitions:
                # Unpack based on actual format (source, target, data)
                src, tgt, _ = edge
                if src == from_state and tgt == to_state:
                    transition_exists = True
                    break
            self.assertTrue(
                transition_exists,
                f"Expected transition from {from_state} to {to_state} not found",
            )

        # Check frequency and probability attributes for states
        state_freq_attributes = self.policy_representation.get_state_attributes(
            "frequency"
        )
        state_prob_attributes = self.policy_representation.get_state_attributes(
            "probability"
        )

        for state in p:
            # Get state attributes
            frequency = state_freq_attributes.get(state, 0)
            probability = state_prob_attributes.get(state, 0)

            # States should have been visited at least once
            self.assertGreater(frequency, 0)
            self.assertGreater(probability, 0)

        # Check that we have exactly 4 transitions (one for each state)
        transitions_with_data = list(
            self.policy_representation.get_all_transitions(include_data=True)
        )
        self.assertEqual(len(transitions_with_data), 4)

        # Check that each transition has the expected attributes
        for transition in transitions_with_data:
            # Extract data from transition
            _, _, data = transition

            # Check that data is a dictionary with the expected keys
            self.assertIsInstance(data, dict)
            self.assertIn("frequency", data)
            self.assertIn("probability", data)
            self.assertIn("action", data)

            # Check that action is 0 (the only action in TestingEnv)
            self.assertEqual(data["action"], 0)

            # Check that frequency and probability are reasonable
            self.assertGreater(data["frequency"], 0)
            self.assertGreaterEqual(data["probability"], 0.0)
            self.assertLessEqual(data["probability"], 1.0)


if __name__ == "__main__":
    unittest.main()
