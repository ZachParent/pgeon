import unittest
from typing import List, Any, cast

import numpy as np

from pgeon import GraphRepresentation, Predicate
from pgeon.policy_approximator import PolicyApproximatorFromBasicObservation
from pgeon.discretizer import StateRepresentation
from test.domain.test_env import State, TestingDiscretizer, TestingEnv, TestingAgent


class TestCreateGraphFromTraces(unittest.TestCase):
    """
    Tests for creating graph representations from traces using PolicyApproximatorFromBasicObservation.
    """

    def setUp(self):
        """Set up test environment, discretizer, and policy representation."""
        self.env = TestingEnv()
        self.discretizer = TestingDiscretizer()
        self.representation = GraphRepresentation()
        self.agent = TestingAgent()

        # Initialize the policy approximator
        self.approximator = PolicyApproximatorFromBasicObservation(
            self.discretizer, self.representation, self.env, self.agent
        )

        # Create states for testing
        self.state0 = cast(StateRepresentation, (Predicate(State, [State.ZERO]),))
        self.state1 = cast(StateRepresentation, (Predicate(State, [State.ONE]),))
        self.state2 = cast(StateRepresentation, (Predicate(State, [State.TWO]),))
        self.state3 = cast(StateRepresentation, (Predicate(State, [State.THREE]),))

        # Action for testing
        self.action0 = 0  # TestingEnv only supports action 0

    def test_update_with_trajectory(self):
        """Test updating a representation with a manually created trajectory."""
        # Clear the representation
        self.representation.clear()

        # Create a manual trajectory
        # Format: [state0, action0, state1, action1, state2, ...]
        trajectory = [
            self.state0,  # State 0
            self.action0,  # Action 0
            self.state1,  # State 1
            self.action0,  # Action 0
            self.state2,  # State 2
            self.action0,  # Action 0
            self.state3,  # State 3
            self.action0,  # Action 0
            self.state0,  # Back to State 0
        ]

        # Update the representation with the trajectory
        self.approximator._update_with_trajectory(trajectory)

        # Normalize the frequencies
        self.approximator._normalize()

        # Check that states were added
        states = self.representation.get_all_states()
        self.assertEqual(len(states), 4)
        self.assertIn(self.state0, states)
        self.assertIn(self.state1, states)
        self.assertIn(self.state2, states)
        self.assertIn(self.state3, states)

        # Check transitions were added correctly
        transitions = self.representation.get_all_transitions()
        self.assertEqual(len(transitions), 4)

        # Check that expected transitions exist
        self.assertTrue(
            self.representation.has_transition(self.state0, self.state1, self.action0)
        )
        self.assertTrue(
            self.representation.has_transition(self.state1, self.state2, self.action0)
        )
        self.assertTrue(
            self.representation.has_transition(self.state2, self.state3, self.action0)
        )
        self.assertTrue(
            self.representation.has_transition(self.state3, self.state0, self.action0)
        )

        # Check that frequencies were set
        state_freqs = self.representation.get_state_attributes("frequency")
        self.assertEqual(state_freqs[self.state0], 2)  # Appears twice in trajectory
        self.assertEqual(state_freqs[self.state1], 1)
        self.assertEqual(state_freqs[self.state2], 1)
        self.assertEqual(state_freqs[self.state3], 1)

        # Check that probabilities were set
        state_probs = self.representation.get_state_attributes("probability")
        self.assertAlmostEqual(state_probs[self.state0], 0.4, delta=0.001)  # 2/5 = 0.4
        self.assertAlmostEqual(state_probs[self.state1], 0.2, delta=0.001)  # 1/5 = 0.2
        self.assertAlmostEqual(state_probs[self.state2], 0.2, delta=0.001)  # 1/5 = 0.2
        self.assertAlmostEqual(state_probs[self.state3], 0.2, delta=0.001)  # 1/5 = 0.2

        # Check transition frequencies and probabilities
        for from_state, to_state, action in [
            (self.state0, self.state1, self.action0),
            (self.state1, self.state2, self.action0),
            (self.state2, self.state3, self.action0),
            (self.state3, self.state0, self.action0),
        ]:
            transition_data = self.representation.get_transition_data(
                from_state, to_state, action
            )
            self.assertEqual(transition_data["frequency"], 1)
            self.assertEqual(transition_data["probability"], 1.0)

    def test_fit_multiple_episodes(self):
        """Test fitting the approximator with multiple episodes."""
        # Clear the representation
        self.representation.clear()

        # Fit the approximator with multiple episodes
        self.approximator.fit(n_episodes=3)

        # Check that states were added
        states = self.representation.get_all_states()
        self.assertEqual(len(states), 4)
        self.assertIn(self.state0, states)
        self.assertIn(self.state1, states)
        self.assertIn(self.state2, states)
        self.assertIn(self.state3, states)

        # Check that transitions were added
        self.assertTrue(
            self.representation.has_transition(self.state0, self.state1, self.action0)
        )
        self.assertTrue(
            self.representation.has_transition(self.state1, self.state2, self.action0)
        )
        self.assertTrue(
            self.representation.has_transition(self.state2, self.state3, self.action0)
        )
        self.assertTrue(
            self.representation.has_transition(self.state3, self.state0, self.action0)
        )

        # Check that trajectories were stored
        self.assertEqual(len(self.approximator._trajectories_of_last_fit), 3)

        # Check that the approximator is fit
        self.assertTrue(self.approximator._is_fit)

    def test_update_existing_representation(self):
        """Test updating an existing representation with new traces."""
        # Clear the representation
        self.representation.clear()

        # Fit with one episode
        self.approximator.fit(n_episodes=1)

        # Store the initial frequencies
        initial_freqs = self.representation.get_state_attributes("frequency")

        # Update with another episode
        self.approximator.fit(n_episodes=1, update=True)

        # Get the updated frequencies
        updated_freqs = self.representation.get_state_attributes("frequency")

        # Check that frequencies increased for all states
        for state in self.representation.get_all_states():
            self.assertGreaterEqual(updated_freqs[state], initial_freqs[state])


if __name__ == "__main__":
    unittest.main()
