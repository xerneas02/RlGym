import unittest
from unittest.mock import Mock
import numpy as np
import Reward
from Reward import GoalScoredReward

class TestGoalScoredReward(unittest.TestCase):
    def setUp(self):
        self.reward = GoalScoredReward()
        self.player = Mock()
        self.player.team_num = 0
        self.player.car_id = 1

        self.state = Mock()
        self.state.blue_score = 0
        self.state.orange_score = 0
        # Initialise linear_velocity avec un vecteur 3D pour éviter l'erreur
        self.state.ball = Mock()
        self.state.ball.linear_velocity = np.array([0, 0, 0])

    def test_reward_given_on_goal_scored(self):
        Reward.LAST_TOUCH = self.player.car_id
        Reward.TOUCH_VERIF = True

        self.state.blue_score += 1
        self.assertEqual(self.reward.get_reward(self.player, self.state, None), 1.0)

    def test_no_reward_without_goal(self):
        self.assertEqual(self.reward.get_reward(self.player, self.state, None), 0.0)

if __name__ == "__main__":
    unittest.main()
