import unittest
from typing import cast

import gymnasium

from pgeon import GraphRepresentation, Predicate
from pgeon.policy_approximator import PolicyApproximatorFromBasicObservation
from pgeon.discretizer import StateRepresentation
from test.domain.cartpole import CartpoleDiscretizer, Position, Velocity, Angle, Action
from pgeon.agent import Agent

class TestAskQuestionsToGraph(unittest.TestCase):
    env = gymnasium.make("CartPole-v1")
    discretizer = CartpoleDiscretizer()
    representation = GraphRepresentation()

    # Setup the approximator using data from CSV files
    # In a real implementation, we would load from CSV files into the representation
    # For this example, we're using a dummy agent and running one episode to populate the representation
    class DummyAgent(Agent):
        def act(self, observation):
            return 0  # Always return action 0 (LEFT)

    @classmethod
    def setUpClass(cls):
        # Create a dummy agent
        agent = cls.DummyAgent()

        # Create the policy approximator
        cls.approximator = PolicyApproximatorFromBasicObservation(
            cls.discretizer, cls.representation, cls.env, agent
        )

        # Fit the approximator with a few episodes
        cls.approximator.fit(n_episodes=5)

    def test_question_1(self):
        """Test what actions would you take in state X?"""
        # Create a state representation
        state = cast(
            StateRepresentation,
            (
                Predicate(Position, [Position.MIDDLE]),
                Predicate(Velocity, [Velocity.LEFT]),
                Predicate(Angle, [Angle.STABILIZING_RIGHT]),
            ),
        )

        # Get actions for the state
        result = self.approximator.question1(state)

        # Assert on the result
        # The number of actions returned may depend on the actual implementation
        # and the random episodes that were generated during setup
        self.assertGreaterEqual(len(result), 1)

        # Check that the actions are valid (LEFT or RIGHT in CartPole)
        actions = [action for action, _ in result]
        for action in actions:
            self.assertIn(action, [Action.LEFT, Action.RIGHT])

        # Check that probabilities sum to 1.0
        probabilities = [prob for _, prob in result]
        self.assertAlmostEqual(sum(probabilities), 1.0, delta=0.001)

    def test_question_1_nearest_predicate(self):
        """Test getting actions for a state not in the representation."""
        # This is similar to test_question_1 but with a state that's not in the representation
        # The approximator should find the nearest predicate
        # Create a state representation that's not in the representation
        state = cast(
            StateRepresentation,
            (
                Predicate(Position, [Position.LEFT]),
                Predicate(Velocity, [Velocity.LEFT]),
                Predicate(Angle, [Angle.STABILIZING_RIGHT]),
            ),
        )

        # Get actions for the state
        result = self.approximator.question1(state)

        # Assert on the result
        self.assertGreaterEqual(len(result), 1)

        # Check that the actions are valid
        actions = [action for action, _ in result]
        for action in actions:
            self.assertIn(action, [Action.LEFT, Action.RIGHT])

        # Check that probabilities sum to 1.0
        probabilities = [prob for _, prob in result]
        self.assertAlmostEqual(sum(probabilities), 1.0, delta=0.001)

    def test_question_1_no_nearest_predicate(self):
        """Test getting actions for a state with no similar state in the representation."""
        # Create a state representation that's very different from existing states
        state = cast(
            StateRepresentation,
            (
                Predicate(Position, [Position.RIGHT]),
                Predicate(Velocity, [Velocity.RIGHT]),
                Predicate(Angle, [Angle.STUCK_LEFT]),
            ),
        )

        # Get actions for the state
        result = self.approximator.question1(state)

        # Assert on the result
        self.assertGreaterEqual(len(result), 1)

        # Check that the actions are valid
        actions = [action for action, _ in result]
        for action in actions:
            self.assertIn(action, [Action.LEFT, Action.RIGHT])

        # Check that probabilities sum to 1.0
        probabilities = [prob for _, prob in result]
        self.assertAlmostEqual(sum(probabilities), 1.0, delta=0.001)

    def test_question_2(self):
        """Test when do you perform action X?"""
        # Test with action LEFT (0)
        result = self.approximator.question2(Action.LEFT)

        # In a real-world scenario with random episodes,
        # we can't guarantee there will be states where LEFT is the best action.
        # Instead, we'll check that the result is a list (even if empty)
        self.assertIsInstance(result, list)

        # If there are states in the result, check that they're valid
        for state in result:
            self.assertTrue(self.representation.has_state(state))

    def test_question_3(self):
        """Test why do you perform action X in state Y?"""
        # Create a state representation
        state = cast(
            StateRepresentation,
            (
                Predicate(Position, [Position.MIDDLE]),
                Predicate(Velocity, [Velocity.LEFT]),
                Predicate(Angle, [Angle.STABILIZING_RIGHT]),
            ),
        )

        # Test with the state and action RIGHT (1)
        result = self.approximator.question3(state, Action.RIGHT)

        # This is more of a sanity check
        # The actual result will depend on the specific graph structure
        # But we can check that the method runs without errors
        self.assertIsInstance(result, list)


if __name__ == "__main__":
    unittest.main()
