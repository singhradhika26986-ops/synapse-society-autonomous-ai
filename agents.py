import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from config import (
    EAT_HUNGER_REDUCTION,
    HUNGER_INCREASE_PER_STEP,
    INTERACT_ENERGY_COST,
    MEMORY_TOP_K,
    MOVE_ENERGY_COST,
    PERSONALITIES,
    PERCEPTION_RANGE,
    RESOURCE_TYPES,
    REST_ENERGY_GAIN,
    SPECIALIZATIONS,
    THIRST_INCREASE_PER_STEP,
)
from explainability import build_action_explanation
from learning import QLearningBrain, RewardSystem
from memory import AgentMemory


ACTIONS = ["move", "rest", "eat", "gather", "interact"]
MOVES = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


@dataclass
class Agent:
    id: int
    position: Tuple[int, int]
    energy: int
    hunger: int
    thirst: int = 20
    personality: str = "cooperative"
    specialization: str = "scout"
    mood: str = "curious"
    goal: str = "survive, secure resources, and build useful relationships"
    memory: AgentMemory = field(default_factory=AgentMemory)
    last_action: str = "spawn"
    last_message: str = ""
    last_decision_scores: Dict = field(default_factory=dict)
    last_explanation: Dict = field(default_factory=dict)
    inventory: Dict[str, int] = field(default_factory=lambda: {"food": 0, "water": 0, "cache": 0})
    total_reward: float = 0.0
    survival_steps: int = 0
    conversation_history: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        self.brain = QLearningBrain(ACTIONS)
        self.rewards = RewardSystem()
        self.previous_state_key = None
        self.previous_action = None
        self.personality_profile = PERSONALITIES[self.personality]
        self.specialization_profile = SPECIALIZATIONS[self.specialization]

    def perceive(self, environment):
        visible = []
        for other in environment.agents:
            if other.id == self.id:
                continue
            distance = self._distance(self.position, other.position)
            if distance <= self.perception_range():
                visible.append({"agent": other, "distance": distance})
        visible.sort(key=lambda item: item["distance"])
        return visible

    def retrieve_memories(self, perception):
        nearby_ids = ", ".join(str(item["agent"].id) for item in perception) or "none"
        query = (
            f"energy {self.energy} hunger {self.hunger} mood {self.mood} personality {self.personality} "
            f"thirst {self.thirst} specialization {self.specialization} nearby agents {nearby_ids} goal {self.goal}"
        )
        return self.memory.retrieve(query, MEMORY_TOP_K + self.specialization_profile["memory_bonus"])

    def decide(self, perception, memories, environment):
        zone = environment.zone_at(self.position)
        scenario_weights = environment.scenario["weights"]
        state = self.brain.state_key(self.energy, self.hunger, len(perception), self.mood, self.thirst, zone)
        profile = self.personality_profile
        spec = self.specialization_profile
        hunger_pressure = self.hunger / 100.0
        thirst_pressure = self.thirst / 100.0
        energy_pressure = (100 - self.energy) / 100.0
        best_target = self._select_interaction_target(perception)
        trust = self.memory.relationship_score(best_target["agent"].id) if best_target else 0.0
        nearby_bonus = 1.0 if perception else -0.65
        recent_social_repetition = self._recent_action_count("interact", 4) * 0.55
        recent_move_repetition = self._recent_action_count("move", 4) * 0.2
        local_resources = environment.resources_at(self.position)
        wanted_resource = self._wanted_resource_type()
        rivalry_pressure = self._rivalry_pressure(perception)
        alliance_pressure = self._alliance_pressure(perception)
        zone_risk = environment.zone_effect(self.position)["risk"]
        memory_sentiment = sum(memory["sentiment"] for memory in memories[:3]) / max(1, min(3, len(memories)))

        candidate_scores = {
            "rest": self.brain.score(state, "rest") + energy_pressure * 8.0 * scenario_weights["energy"] + zone_risk * 3.0 * scenario_weights["risk"],
            "eat": (
                self.brain.score(state, "eat")
                + hunger_pressure * 7.0 * profile["resource"] * scenario_weights["hunger"]
                + thirst_pressure * 6.5 * scenario_weights["thirst"]
                + (2.0 if self.inventory.get(wanted_resource, 0) else 0.0)
            ),
            "gather": (
                self.brain.score(state, "gather")
                + len(local_resources) * 5.0 * spec["resource_skill"] * scenario_weights["resources"]
                + max(hunger_pressure, thirst_pressure) * 3.0
                + (2.0 if zone == "high_resource" else 0.0)
            ),
            "interact": (
                self.brain.score(state, "interact")
                + nearby_bonus * 2.2 * scenario_weights["social"]
                + trust * 7.0 * profile["trust_bias"] * spec["influence"] * scenario_weights["trust"]
                + profile["social"] * 1.8 * scenario_weights["social"]
                + alliance_pressure * 2.8 * scenario_weights["trust"]
                - rivalry_pressure * (3.2 if self.personality != "aggressive" else -1.0) * scenario_weights["competition"]
                + memory_sentiment * 1.4
                - energy_pressure * 2.5
                - max(hunger_pressure, thirst_pressure) * (1.4 if self.personality != "selfish" else 3.4)
                - recent_social_repetition
            ),
            "move": (
                self.brain.score(state, "move")
                + profile["curiosity"] * 2.0
                + max(hunger_pressure, thirst_pressure) * 3.0 * scenario_weights["resources"]
                + zone_risk * 2.0 * scenario_weights["risk"]
                + rivalry_pressure * (2.2 if self.personality != "aggressive" else 0.4) * scenario_weights["competition"]
                - energy_pressure * 2.0
                - recent_move_repetition
            ),
        }
        if not best_target:
            candidate_scores["interact"] -= 6.5
        elif best_target["distance"] > self.interaction_range():
            candidate_scores["interact"] -= 3.0
        if local_resources:
            candidate_scores["gather"] += 4.0
        else:
            candidate_scores["gather"] -= 6.0
        if self.inventory.get(wanted_resource, 0) or environment.resource_available(self.position, wanted_resource):
            candidate_scores["eat"] += 5.0
        else:
            candidate_scores["eat"] -= 8.0
            candidate_scores["move"] += 2.0 * profile["resource"]
        if max(self.hunger, self.thirst) > 72 and not environment.resource_available(self.position, wanted_resource):
            candidate_scores["move"] += 4.0 * profile["resource"]
            candidate_scores["eat"] -= 4.0
        if max(self.hunger, self.thirst) > 82 and self.energy < 35:
            candidate_scores["rest"] += 2.0
        for action in candidate_scores:
            candidate_scores[action] += random.uniform(-0.8, 0.8) * profile["risk"]

        action = self.brain.choose_weighted(state, candidate_scores)
        target = best_target if action == "interact" else None
        direction = self._choose_movement_direction(perception) if action == "move" else None
        self.previous_state_key = state
        self.previous_action = action
        self.last_decision_scores = {name: round(score, 2) for name, score in candidate_scores.items()}
        factors = {
            "scenario": environment.scenario_name,
            "hunger_pressure": hunger_pressure,
            "thirst_pressure": thirst_pressure,
            "energy_pressure": energy_pressure,
            "target_trust": trust,
            "nearby_bonus": nearby_bonus,
            "alliance_pressure": alliance_pressure,
            "rivalry_pressure": rivalry_pressure,
            "zone": zone,
            "zone_risk": zone_risk,
            "local_resources": len(local_resources),
            "wanted_resource": wanted_resource,
            "personality_social": profile["social"],
            "personality_resource": profile["resource"],
            "specialization": self.specialization,
            "specialization_influence": spec["influence"],
            "memory_sentiment": memory_sentiment,
            "q_values": {candidate: round(self.brain.score(state, candidate), 3) for candidate in ACTIONS},
            "scenario_weights": scenario_weights,
        }
        self.last_explanation = build_action_explanation(self, state, action, candidate_scores, factors, memories)
        return {
            "action": action,
            "target_id": target["agent"].id if target else None,
            "direction": direction,
            "memories_used": memories,
            "state_key": state,
            "scores": self.last_decision_scores,
            "explanation": self.last_explanation,
        }

    def _select_interaction_target(self, perception):
        if not perception:
            return None
        weighted = []
        for item in perception:
            other = item["agent"]
            relationship = self.memory.relationship_score(other.id)
            status = self.memory.relationship_status(other.id)
            hostility_penalty = 0.25 if relationship < -0.35 else 1.0
            weight = max(0.05, (1.0 + relationship * self.personality_profile["trust_bias"]) * hostility_penalty)
            if status == "alliance":
                weight *= 2.2
            if status == "rivalry" and self.personality != "aggressive":
                weight *= 0.25
            if self.personality == "aggressive" and relationship < -0.2:
                weight += 0.45
            weighted.append((item, weight))
        total = sum(weight for _, weight in weighted)
        pick = random.random() * total
        running = 0.0
        for item, weight in weighted:
            running += weight
            if running >= pick:
                return item
        return weighted[-1][0]

    def _interaction_score(self, other_agent):
        return self.memory.relationship_score(other_agent.id)

    def _recent_action_count(self, action, window):
        recent = [item for item in self.memory.items[-window:] if item.metadata.get("action") == action]
        return len(recent)

    def perception_range(self):
        return PERCEPTION_RANGE + (1 if self.specialization in {"scout", "archivist"} else 0)

    def interaction_range(self):
        return 1 + int(self.specialization_profile["influence"] >= 1.4)

    def _distance(self, first, second):
        return abs(first[0] - second[0]) + abs(first[1] - second[1])

    def _wanted_resource_type(self):
        if self.thirst >= self.hunger and self.thirst > 45:
            return "water"
        if self.hunger > 45:
            return "food"
        return "cache"

    def _alliance_pressure(self, perception):
        return sum(1 for item in perception if self.memory.relationship_status(item["agent"].id) == "alliance")

    def _rivalry_pressure(self, perception):
        return sum(1 for item in perception if self.memory.relationship_status(item["agent"].id) == "rivalry")

    def apply_passive_updates(self, environment):
        self.hunger = min(100, self.hunger + HUNGER_INCREASE_PER_STEP)
        self.thirst = min(100, self.thirst + THIRST_INCREASE_PER_STEP)
        zone_effect = environment.zone_effect(self.position)
        self.energy = max(0, min(100, self.energy + zone_effect["energy_delta"]))
        if random.random() < zone_effect["risk"]:
            self.energy = max(0, self.energy - random.randint(3, 9))
        if self.hunger > 80:
            self.energy = max(0, self.energy - 3)
        if self.thirst > 80:
            self.energy = max(0, self.energy - 4)
        self.survival_steps += 1
        self._update_mood()

    def perform(self, decision, environment, communicator, conversation_history, timestep):
        action = decision["action"]
        result = {"success": False}
        if action == "rest":
            before = self.energy
            self.energy = min(100, self.energy + REST_ENERGY_GAIN)
            result = {"success": True, "energy_gain": self.energy - before}
        elif action == "eat":
            result = self._eat(environment)
        elif action == "gather":
            result = self._gather(environment)
        elif action == "move":
            result = self._move(environment, decision["direction"])
        elif action == "interact":
            result = self._interact(environment, decision, communicator, conversation_history, timestep)

        self.last_action = action
        self.apply_passive_updates(environment)
        reward = self.rewards.reward_for(action, result)
        reward += self._global_objective_reward(environment, result)
        self.total_reward += reward
        next_state = self.brain.state_key(
            self.energy,
            self.hunger,
            len(self.perceive(environment)),
            self.mood,
            self.thirst,
            environment.zone_at(self.position),
        )
        if self.previous_state_key and self.previous_action:
            self.brain.update(self.previous_state_key, self.previous_action, reward, next_state)
        self._store_action_memory(action, result, reward, timestep)
        return result, reward

    def _choose_movement_direction(self, perception):
        best_target = None
        if max(self.hunger, self.thirst) > 55:
            best_target = ("resource", self._wanted_resource_type())
        elif perception:
            trusted = [item for item in perception if self.memory.relationship_score(item["agent"].id) > 0.25]
            disliked = [item for item in perception if self.memory.relationship_score(item["agent"].id) < -0.3]
            if trusted and self.personality in {"cooperative", "social", "analytical"}:
                best_target = ("toward_agent", trusted[0]["agent"].position)
            elif disliked and self.personality != "aggressive":
                best_target = ("away_agent", disliked[0]["agent"].position)
        return best_target or ("random", None)

    def _move(self, environment, direction):
        direction_name = direction
        if isinstance(direction, tuple):
            direction_name = self._resolve_direction(environment, direction)
        if direction_name not in MOVES:
            direction_name = random.choice(list(MOVES.keys()))
        dx, dy = MOVES[direction_name]
        x, y = self.position
        steps = self.specialization_profile["speed"]
        new_position = self.position
        for _ in range(steps):
            candidate = (
                max(0, min(environment.size - 1, new_position[0] + dx)),
                max(0, min(environment.size - 1, new_position[1] + dy)),
            )
            if environment.is_blocked(candidate):
                break
            occupied = {agent.position for agent in environment.agents if agent.id != self.id}
            if candidate in occupied:
                break
            new_position = candidate
        occupied = {agent.position for agent in environment.agents if agent.id != self.id}
        if new_position == self.position or new_position in occupied:
            self.energy = max(0, self.energy - 1)
            return {"success": False, "moved": False, "direction": direction_name, "to": self.position}
        self.position = new_position
        cost = max(1, int(MOVE_ENERGY_COST * self.specialization_profile["energy_efficiency"]))
        self.energy = max(0, self.energy - cost)
        zone = environment.zone_at(self.position)
        zone_reward = 1.5 if zone == "high_resource" else -0.8 if zone == "risky" else 0.4 if zone == "safe" else 0.0
        return {"success": True, "moved": True, "direction": direction_name, "to": self.position, "zone": zone, "zone_reward": zone_reward}

    def _resolve_direction(self, environment, movement_intent):
        mode, target_position = movement_intent
        if mode == "resource":
            resource = environment.nearest_resource(self.position, target_position)
            target_position = resource["position"] if resource else None
        if target_position is None:
            return random.choice(list(MOVES.keys()))
        x, y = self.position
        tx, ty = target_position
        options = []
        for name, (dx, dy) in MOVES.items():
            next_pos = (max(0, min(environment.size - 1, x + dx)), max(0, min(environment.size - 1, y + dy)))
            distance = abs(next_pos[0] - tx) + abs(next_pos[1] - ty)
            if mode == "away_agent":
                distance = -distance
            options.append((distance, random.random(), name))
        options.sort()
        return options[0][2]

    def _eat(self, environment):
        wanted = self._wanted_resource_type()
        if self.inventory.get(wanted, 0) > 0:
            return self._consume_inventory(wanted)
        if environment.resource_available(self.position, wanted):
            resource = environment.consume_resource(self.position, wanted)
            if resource:
                self.inventory[wanted] += 1
                return self._consume_inventory(wanted)
        self.energy = max(0, self.energy - 2)
        return {"success": False, "reason": f"no_{wanted}_available"}

    def _gather(self, environment):
        resources = environment.resources_at(self.position)
        if not resources:
            return {"success": False, "reason": "no_resource_here"}
        preferred = self._wanted_resource_type()
        resource = environment.consume_resource(self.position, preferred) or environment.consume_resource(self.position)
        if not resource:
            return {"success": False, "reason": "resource_unavailable"}
        resource_type = resource["type"]
        self.inventory[resource_type] = self.inventory.get(resource_type, 0) + 1
        if self.specialization == "forager" and random.random() < 0.28:
            self.inventory[resource_type] += 1
        return {
            "success": True,
            "resource": resource_type,
            "resource_value": RESOURCE_TYPES[resource_type]["value"],
            "inventory": dict(self.inventory),
        }

    def _consume_inventory(self, resource_type):
        self.inventory[resource_type] -= 1
        spec = RESOURCE_TYPES[resource_type]
        before_hunger = self.hunger
        before_thirst = self.thirst
        self.hunger = max(0, self.hunger + spec.get("hunger", 0))
        self.thirst = max(0, self.thirst + spec.get("thirst", 0))
        self.energy = min(100, self.energy + spec.get("energy", 0))
        return {
            "success": True,
            "resource": resource_type,
            "hunger_reduced": before_hunger - self.hunger,
            "thirst_reduced": before_thirst - self.thirst,
        }

    def _interact(self, environment, decision, communicator, conversation_history, timestep):
        target = environment.get_agent(decision["target_id"])
        if not target:
            return {"success": False, "reason": "target_missing"}
        distance = self._distance(self.position, target.position)
        if distance > self.interaction_range():
            return {"success": False, "reason": "out_of_social_range", "distance": distance, "target_id": target.id}

        memories = self.memory.retrieve(f"talking with Agent {target.id}", MEMORY_TOP_K)
        message = communicator.generate(self, target, memories, environment.summary(), conversation_history.messages)
        sentiment = self._message_sentiment(target)
        interact_cost = max(1, int(INTERACT_ENERGY_COST / self.specialization_profile["influence"]))
        self.energy = max(0, self.energy - interact_cost)
        self.last_message = message
        target.last_message = f"Heard from Agent {self.id}: {message}"
        entry = conversation_history.add(timestep, self.id, target.id, message, sentiment)
        self.conversation_history.append(entry)
        target.conversation_history.append(entry)
        self.memory.add(
            f"Interaction with Agent {target.id}: {message}",
            {"type": "interaction", "other_agent_id": target.id, "speaker_id": self.id, "listener_id": target.id},
            sentiment,
            timestep,
        )
        target.memory.add(
            f"Agent {self.id} said: {message}",
            {"type": "interaction", "other_agent_id": self.id, "speaker_id": self.id, "listener_id": target.id},
            sentiment,
            timestep,
        )
        if sentiment < -0.25 and self.personality == "aggressive":
            target.memory.update_relationship(self.id, -0.15)
        return {"success": True, "message": message, "target_id": target.id, "sentiment": sentiment}

    def _message_sentiment(self, target):
        relationship = self.memory.relationship_score(target.id)
        energy_factor = (self.energy - 50) / 100
        hunger_factor = -(self.hunger - 50) / 120
        thirst_factor = -(self.thirst - 50) / 130
        personality_shift = {
            "cooperative": 0.18,
            "social": 0.12,
            "analytical": 0.03,
            "selfish": -0.08,
            "aggressive": -0.18,
        }[self.personality]
        noise = random.uniform(-0.2, 0.2)
        influence = self.specialization_profile["influence"]
        return max(-1.0, min(1.0, (relationship * 0.55 + energy_factor + hunger_factor + thirst_factor + personality_shift + noise) * influence))

    def _global_objective_reward(self, environment, result):
        survival = 0.08
        resource_balance = (100 - self.hunger + 100 - self.thirst + self.energy) / 300.0
        weights = environment.scenario["weights"]
        social_bonus = sum(
            0.05 for relation in self.memory.relationship_summary().values() if relation["status"] == "alliance"
        )
        rivalry_cost = sum(
            0.03 for relation in self.memory.relationship_summary().values() if relation["status"] == "rivalry"
        )
        zone = environment.zone_at(self.position)
        risk_cost = 0.18 if zone == "risky" else 0.0
        return (
            survival
            + resource_balance * (weights["hunger"] + weights["thirst"] + weights["energy"]) / 3
            + social_bonus * weights["trust"]
            - rivalry_cost * weights["competition"]
            - risk_cost * weights["risk"]
        )

    def _store_action_memory(self, action, result, reward, timestep):
        text = (
            f"At timestep {timestep}, Agent {self.id} chose {action}. "
            f"Result: {result}. Reward: {reward:.2f}. Mood: {self.mood}. "
            f"Personality: {self.personality}. Specialization: {self.specialization}. "
            f"Decision scores: {self.last_decision_scores}."
        )
        sentiment = max(-1.0, min(1.0, reward / 10.0))
        metadata = {"type": "action", "action": action}
        if "target_id" in result:
            metadata["other_agent_id"] = result["target_id"]
        self.memory.add(text, metadata, sentiment, timestep)

    def _update_mood(self):
        if self.energy < 25:
            self.mood = "tired"
        elif self.thirst > 75:
            self.mood = "thirsty"
        elif self.hunger > 75:
            self.mood = "hungry"
        elif self.last_action == "interact":
            self.mood = "social"
        elif self.energy > 70 and self.hunger < 40:
            self.mood = "curious"
        else:
            self.mood = "focused"

    def snapshot(self):
        return {
            "id": self.id,
            "position": {"x": self.position[0], "y": self.position[1]},
            "energy": self.energy,
            "hunger": self.hunger,
            "thirst": self.thirst,
            "personality": self.personality,
            "specialization": self.specialization,
            "specialization_description": self.specialization_profile["description"],
            "mood": self.mood,
            "goal": self.goal,
            "last_action": self.last_action,
            "last_message": self.last_message,
            "decision_scores": self.last_decision_scores,
            "explanation": self.last_explanation,
            "relationships": self.memory.relationship_summary(),
            "inventory": self.inventory,
            "total_reward": round(self.total_reward, 2),
            "survival_steps": self.survival_steps,
        }
