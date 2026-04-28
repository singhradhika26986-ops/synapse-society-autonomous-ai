from collections import defaultdict, deque


class SocietyMetrics:
    def __init__(self, max_points=160):
        self.max_points = max_points
        self.reward_history = defaultdict(lambda: deque(maxlen=max_points))
        self.trust_history = defaultdict(lambda: deque(maxlen=max_points))
        self.interaction_counts = defaultdict(int)
        self.action_counts = defaultdict(lambda: defaultdict(int))
        self.survival_duration = defaultdict(int)
        self.zone_visits = defaultdict(lambda: defaultdict(int))

    def update(self, timestep, environment, records):
        for agent in environment.agents:
            self.survival_duration[agent.id] = max(self.survival_duration[agent.id], timestep)
            self.zone_visits[agent.id][environment.zone_at(agent.position)] += 1
            for other_id, relation in agent.memory.relationship_summary().items():
                key = f"{agent.id}->{other_id}"
                self.trust_history[key].append({"t": timestep, "value": relation["trust"]})

        for record in records:
            agent_id = record["agent_id"]
            reward = record.get("reward", 0.0)
            self.reward_history[agent_id].append({"t": timestep, "value": round(reward, 3)})
            action = record.get("decision", {}).get("action") or record.get("decision")
            if action:
                self.action_counts[agent_id][action] += 1
            result = record.get("result", {})
            if action == "interact" and result.get("target_id"):
                key = f"{agent_id}->{result['target_id']}"
                self.interaction_counts[key] += 1

    def snapshot(self):
        return {
            "reward_history": {str(k): list(v) for k, v in self.reward_history.items()},
            "trust_history": {k: list(v) for k, v in self.trust_history.items()},
            "interaction_counts": dict(self.interaction_counts),
            "action_counts": {str(k): dict(v) for k, v in self.action_counts.items()},
            "survival_duration": {str(k): v for k, v in self.survival_duration.items()},
            "zone_visits": {str(k): dict(v) for k, v in self.zone_visits.items()},
        }
