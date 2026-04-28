import pygame

from config import CELL_SIZE, FPS, GRID_SIZE, SIDEBAR_WIDTH


class PygameUI:
    COLORS = {
        "background": (20, 24, 30),
        "grid": (58, 64, 74),
        "food": (75, 190, 120),
        "text": (235, 238, 245),
        "muted": (165, 172, 185),
        "panel": (32, 38, 48),
        "safe": (31, 54, 48),
        "neutral": (26, 31, 39),
        "risky": (58, 35, 43),
        "high_resource": (54, 50, 30),
        "obstacle": (10, 12, 16),
        "water": (72, 170, 230),
        "cache": (232, 198, 92),
        "agent_1": (91, 168, 255),
        "agent_2": (255, 177, 66),
        "agent_3": (236, 97, 130),
        "agent_4": (146, 224, 115),
        "agent_5": (189, 137, 255),
    }

    def __init__(self, size=GRID_SIZE):
        pygame.init()
        self.size = size
        self.width = size * CELL_SIZE + SIDEBAR_WIDTH
        self.height = size * CELL_SIZE
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Synapse Society - Autonomous Multi-Agent AI Civilization")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 16)
        self.small_font = pygame.font.SysFont("consolas", 13)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def draw(self, environment, conversations):
        self.screen.fill(self.COLORS["background"])
        self._draw_grid(environment)
        self._draw_sidebar(environment, conversations)
        pygame.display.flip()
        self.clock.tick(FPS)

    def _draw_grid(self, environment):
        for y in range(self.size):
            for x in range(self.size):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                zone = environment.zone_at((x, y))
                pygame.draw.rect(self.screen, self.COLORS.get(zone, self.COLORS["neutral"]), rect)
                pygame.draw.rect(self.screen, self.COLORS["grid"], rect, 1)

        for obstacle in environment.obstacles:
            rect = pygame.Rect(obstacle[0] * CELL_SIZE + 14, obstacle[1] * CELL_SIZE + 14, CELL_SIZE - 28, CELL_SIZE - 28)
            pygame.draw.rect(self.screen, self.COLORS["obstacle"], rect)

        for resource in environment.resources:
            x, y = resource["position"]
            cx = x * CELL_SIZE + CELL_SIZE // 2
            cy = y * CELL_SIZE + CELL_SIZE // 2
            color = self.COLORS.get(resource["type"], self.COLORS["food"])
            pygame.draw.circle(self.screen, color, (cx, cy), 8)

        for agent in environment.agents:
            x, y = agent.position
            cx = x * CELL_SIZE + CELL_SIZE // 2
            cy = y * CELL_SIZE + CELL_SIZE // 2
            color = self.COLORS.get(f"agent_{agent.id}", (240, 240, 240))
            pygame.draw.circle(self.screen, color, (cx, cy), 20)
            label = self.font.render(str(agent.id), True, (10, 12, 16))
            self.screen.blit(label, label.get_rect(center=(cx, cy)))
            self._draw_bar(cx - 22, cy + 25, agent.energy, (91, 168, 255))
            self._draw_bar(cx - 22, cy + 31, 100 - agent.hunger, (75, 190, 120))
            self._draw_bar(cx - 22, cy + 37, 100 - agent.thirst, (72, 170, 230))

    def _draw_bar(self, x, y, value, color):
        pygame.draw.rect(self.screen, (13, 15, 20), (x, y, 44, 4))
        pygame.draw.rect(self.screen, color, (x, y, max(0, min(44, int(value * 0.44))), 4))

    def _draw_sidebar(self, environment, conversations):
        left = self.size * CELL_SIZE
        pygame.draw.rect(self.screen, self.COLORS["panel"], (left, 0, SIDEBAR_WIDTH, self.height))
        self._text(f"Timestep: {environment.timestep}", left + 18, 18, self.font)
        self._text("Agents", left + 18, 50, self.font)
        y = 78
        for agent in environment.agents:
            color = self.COLORS.get(f"agent_{agent.id}", self.COLORS["text"])
            self._text(f"A{agent.id} {agent.personality}/{agent.specialization}", left + 18, y, self.font, color)
            y += 20
            self._text(f"{agent.mood} pos={agent.position} zone={environment.zone_at(agent.position)}", left + 28, y, self.small_font)
            y += 18
            self._text(f"E={agent.energy:3d} H={agent.hunger:3d} T={agent.thirst:3d} R={agent.total_reward:5.1f}", left + 28, y, self.small_font)
            y += 18
            self._text(f"action={agent.last_action} inv={agent.inventory}", left + 28, y, self.small_font)
            y += 18
            score_line = " ".join(f"{key}:{value}" for key, value in agent.last_decision_scores.items())
            for wrapped in self._wrap(score_line, 39):
                self._text(wrapped, left + 28, y, self.small_font, self.COLORS["muted"])
                y += 15
            trust_line = self._trust_line(agent)
            if trust_line:
                for wrapped in self._wrap(f"trust {trust_line}", 39):
                    self._text(wrapped, left + 28, y, self.small_font, self.COLORS["muted"])
                    y += 15
            y += 10

        self._text("Live communication", left + 18, y + 6, self.font)
        y += 34
        for entry in conversations.messages[-8:]:
            line = f"A{entry['speaker_id']} -> A{entry['listener_id']}: {entry['message']}"
            for wrapped in self._wrap(line, 39):
                self._text(wrapped, left + 18, y, self.small_font, self.COLORS["muted"])
                y += 16
                if y > self.height - 20:
                    return

    def _text(self, text, x, y, font, color=None):
        surface = font.render(text, True, color or self.COLORS["text"])
        self.screen.blit(surface, (x, y))

    def _trust_line(self, agent):
        pieces = []
        for other_id, relation in sorted(agent.memory.relationship_summary().items(), key=lambda item: item[0]):
            pieces.append(f"A{other_id}:{relation['trust']:+.2f}/{relation['status']}")
        return " ".join(pieces)

    def _wrap(self, text, width):
        words = text.split()
        lines = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if len(candidate) <= width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word[:width]
        if current:
            lines.append(current)
        return lines

    def close(self):
        pygame.quit()
