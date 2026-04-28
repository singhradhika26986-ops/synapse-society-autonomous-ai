import random
from collections import defaultdict


class QLearningBrain:
    """Small tabular Q-learning helper for each agent."""

    def __init__(self, actions, alpha=0.18, gamma=0.85, epsilon=0.12):
        self.actions = list(actions)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q = defaultdict(lambda: {action: 0.0 for action in self.actions})

    def state_key(self, energy, hunger, nearby_count, mood, thirst=0, zone="neutral"):
        energy_bucket = "low_energy" if energy < 30 else "ok_energy" if energy < 70 else "high_energy"
        hunger_bucket = "hungry" if hunger > 60 else "peckish" if hunger > 35 else "fed"
        thirst_bucket = "thirsty" if thirst > 60 else "ok_thirst"
        social_bucket = "social" if nearby_count else "alone"
        return (energy_bucket, hunger_bucket, thirst_bucket, social_bucket, mood, zone)

    def score(self, state, action):
        return self.q[state][action]

    def choose_weighted(self, state, candidate_scores):
        if random.random() < self.epsilon:
            return random.choice(list(candidate_scores.keys()))
        best_score = max(candidate_scores.values())
        best_actions = [action for action, score in candidate_scores.items() if score == best_score]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state):
        current = self.q[state][action]
        future = max(self.q[next_state].values())
        self.q[state][action] = current + self.alpha * (reward + self.gamma * future - current)


class RewardSystem:
    def reward_for(self, action, result):
        if action == "interact":
            if result.get("success"):
                sentiment = result.get("sentiment", 0.0)
                return 8.0 + 6.0 * sentiment
            return -5.0
        if action == "eat":
            return 6.0 if result.get("success") else -3.0
        if action == "gather":
            if result.get("success"):
                return 4.0 + result.get("resource_value", 0.0) * 4.0
            return -2.0
        if action == "rest":
            return 3.0 if result.get("energy_gain", 0) > 0 else -1.0
        if action == "move":
            reward = 1.0 if result.get("moved") else -2.0
            reward += result.get("zone_reward", 0.0)
            return reward
        return 0.0
