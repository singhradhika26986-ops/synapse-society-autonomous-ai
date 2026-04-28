import random

from agents import Agent
from config import (
    DEFAULT_AGENT_COUNT,
    GRID_SIZE,
    OBSTACLE_COUNT,
    PERSONALITIES,
    RESOURCE_TYPES,
    SPECIALIZATIONS,
    START_ENERGY_RANGE,
    START_HUNGER_RANGE,
    ZONE_TYPES,
)
from scenario import get_scenario


class GridWorld:
    def __init__(self, size=GRID_SIZE, agent_count=DEFAULT_AGENT_COUNT, scenario_name="survival"):
        if agent_count < 3 or agent_count > 5:
            raise ValueError("The simulation requires exactly 3 to 5 agents.")
        self.size = size
        self.timestep = 0
        self.scenario_name = scenario_name
        self.scenario = get_scenario(scenario_name)
        self.agents = []
        self.resources = []
        self.obstacles = set()
        self.zones = {}
        self._create_zones()
        self._spawn_obstacles()
        self._spawn_resources()
        self._spawn_agents(agent_count)

    @property
    def food(self):
        return {resource["position"] for resource in self.resources if resource["type"] == "food"}

    def _create_zones(self):
        for y in range(self.size):
            for x in range(self.size):
                if x <= 2 and y <= 2:
                    zone = "safe"
                elif x >= self.size - 3 and y >= self.size - 3:
                    zone = "high_resource"
                elif (x + y) % 7 == 0 or (x >= 7 and y <= 3):
                    zone = "risky"
                else:
                    zone = "neutral"
                self.zones[(x, y)] = zone

    def _spawn_obstacles(self):
        attempts = 0
        while len(self.obstacles) < OBSTACLE_COUNT and attempts < 300:
            attempts += 1
            position = (random.randrange(self.size), random.randrange(self.size))
            if self.zones[position] == "safe":
                continue
            self.obstacles.add(position)

    def _random_empty_position(self, allow_high_resource=True):
        occupied = {agent.position for agent in self.agents} | self.obstacles
        occupied |= {resource["position"] for resource in self.resources}
        while True:
            position = (random.randrange(self.size), random.randrange(self.size))
            if position in occupied:
                continue
            if not allow_high_resource and self.zone_at(position) == "high_resource":
                continue
            return position

    def _spawn_resources(self):
        for resource_type, spec in RESOURCE_TYPES.items():
            while self.resource_count(resource_type) < spec["count"]:
                position = self._weighted_resource_position(resource_type)
                if self.is_blocked(position):
                    continue
                if any(resource["position"] == position for resource in self.resources):
                    continue
                self.resources.append({"type": resource_type, "position": position, "age": 0})

    def _weighted_resource_position(self, resource_type):
        for _ in range(120):
            position = (random.randrange(self.size), random.randrange(self.size))
            zone = self.zone_at(position)
            bonus = ZONE_TYPES[zone]["resource_bonus"]
            if resource_type == "cache" and zone == "high_resource":
                bonus += 1.2
            if random.random() < min(0.9, 0.22 * bonus):
                return position
        return (random.randrange(self.size), random.randrange(self.size))

    def _spawn_agents(self, agent_count):
        goals = [
            "survive by balancing alliances and food access",
            "secure resources before exhaustion",
            "learn who is trustworthy under scarcity",
            "build influence through information sharing",
            "optimize survival through selective cooperation",
        ]
        personalities = list(PERSONALITIES.keys())
        specializations = list(SPECIALIZATIONS.keys())
        for agent_id in range(1, agent_count + 1):
            self.agents.append(
                Agent(
                    id=agent_id,
                    position=self._random_empty_position(allow_high_resource=False),
                    energy=random.randint(*START_ENERGY_RANGE),
                    hunger=random.randint(*START_HUNGER_RANGE),
                    thirst=random.randint(5, 40),
                    personality=personalities[(agent_id - 1) % len(personalities)],
                    specialization=specializations[(agent_id - 1) % len(specializations)],
                    goal=goals[(agent_id - 1) % len(goals)],
                )
            )

    def step_environment(self):
        for resource in self.resources:
            resource["age"] += 1
        if self.timestep % 12 == 0:
            self._spawn_resources()

    def get_agent(self, agent_id):
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        return None

    def zone_at(self, position):
        return self.zones.get(position, "neutral")

    def zone_effect(self, position):
        return ZONE_TYPES[self.zone_at(position)]

    def is_blocked(self, position):
        x, y = position
        return x < 0 or y < 0 or x >= self.size or y >= self.size or position in self.obstacles

    def resource_count(self, resource_type):
        return sum(1 for resource in self.resources if resource["type"] == resource_type)

    def resources_at(self, position):
        return [resource for resource in self.resources if resource["position"] == position]

    def resource_available(self, position, resource_type=None):
        return any(
            resource["position"] == position and (resource_type is None or resource["type"] == resource_type)
            for resource in self.resources
        )

    def food_available(self, position):
        return self.resource_available(position, "food")

    def consume_resource(self, position, preferred_type=None):
        for index, resource in enumerate(self.resources):
            if resource["position"] != position:
                continue
            if preferred_type and resource["type"] != preferred_type:
                continue
            return self.resources.pop(index)
        if preferred_type:
            return None
        return None

    def consume_food(self, position):
        return self.consume_resource(position, "food")

    def nearest_resource(self, position, resource_type=None):
        candidates = [
            resource
            for resource in self.resources
            if resource_type is None or resource["type"] == resource_type
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda resource: (
                abs(resource["position"][0] - position[0]) + abs(resource["position"][1] - position[1]),
                -RESOURCE_TYPES[resource["type"]]["value"],
            ),
        )

    def nearest_food(self, position):
        resource = self.nearest_resource(position, "food")
        return resource["position"] if resource else None

    def summary(self):
        return {
            "size": self.size,
            "timestep": self.timestep,
            "resources": {name: self.resource_count(name) for name in RESOURCE_TYPES},
            "agent_count": len(self.agents),
            "scenario": self.scenario_name,
            "zones": {name: sum(1 for zone in self.zones.values() if zone == name) for name in ZONE_TYPES},
        }

    def snapshot(self):
        return {
            "timestep": self.timestep,
            "size": self.size,
            "scenario": self.scenario_name,
            "scenario_description": self.scenario["description"],
            "food": [{"x": x, "y": y} for x, y in sorted(self.food)],
            "resources": [
                {"type": resource["type"], "x": resource["position"][0], "y": resource["position"][1]}
                for resource in self.resources
            ],
            "obstacles": [{"x": x, "y": y} for x, y in sorted(self.obstacles)],
            "zones": [
                {"x": x, "y": y, "type": zone}
                for (x, y), zone in sorted(self.zones.items(), key=lambda item: (item[0][1], item[0][0]))
            ],
            "agents": [agent.snapshot() for agent in self.agents],
        }
