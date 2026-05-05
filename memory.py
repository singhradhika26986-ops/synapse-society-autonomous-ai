import hashlib
import math
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Optional

try:
    import numpy as np
except Exception:  # pragma: no cover - pure Python fallback for lightweight local runs
    np = None

try:
    import faiss
except Exception:  # pragma: no cover - local fallback for machines without faiss
    faiss = None

from config import MEMORY_DIMENSION


@dataclass
class MemoryItem:
    text: str
    metadata: Dict
    sentiment: float
    timestep: int


class EmbeddingModel:
    """Deterministic local embedding model.

    The project stores every memory as a vector. This lightweight embedder keeps
    the simulation runnable without downloads while still feeding FAISS vectors.
    """

    def __init__(self, dimension=MEMORY_DIMENSION):
        self.dimension = dimension

    def encode(self, text: str):
        vector = [0.0] * self.dimension
        tokens = [token.strip(".,!?;:()[]{}").lower() for token in text.split()]
        for token in tokens:
            if not token:
                continue
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0:
            vector = [value / norm for value in vector]
        if np is not None:
            return np.array(vector, dtype="float32").reshape(1, -1)
        return [vector]


class SimpleVectorIndex:
    def __init__(self, dimension):
        self.dimension = dimension
        self.vectors = []

    def add(self, vector):
        row = vector[0].tolist() if np is not None else list(vector[0])
        self.vectors.append(row)

    def search(self, vector, top_k):
        if len(self.vectors) == 0:
            return [[]], [[]]
        query = vector[0].tolist() if np is not None else list(vector[0])
        scored = []
        for index, candidate in enumerate(self.vectors):
            distance = math.sqrt(sum((left - right) ** 2 for left, right in zip(candidate, query)))
            scored.append((distance, index))
        scored.sort(key=lambda item: item[0])
        trimmed = scored[:top_k]
        return [[distance for distance, _ in trimmed]], [[index for _, index in trimmed]]


class AgentMemory:
    def __init__(self, dimension=MEMORY_DIMENSION):
        self.dimension = dimension
        self.embedder = EmbeddingModel(dimension)
        self.items: List[MemoryItem] = []
        self.index = faiss.IndexFlatL2(dimension) if faiss and np is not None else SimpleVectorIndex(dimension)
        self.relationships = defaultdict(
            lambda: {"trust": 0.0, "positive": 0, "negative": 0, "interactions": 0, "last_sentiment": 0.0}
        )

    def add(self, text: str, metadata: Optional[Dict] = None, sentiment: float = 0.0, timestep: int = 0):
        metadata = metadata or {}
        vector = self.embedder.encode(text)
        self.index.add(vector)
        self.items.append(MemoryItem(text=text, metadata=metadata, sentiment=sentiment, timestep=timestep))
        other_id = metadata.get("other_agent_id")
        if other_id is not None:
            self.update_relationship(other_id, sentiment)

    def retrieve(self, query: str, top_k: int = 5):
        if not self.items:
            return []
        vector = self.embedder.encode(query)
        distances, indices = self.index.search(vector, min(top_k, len(self.items)))
        memories = []
        for distance, index in zip(distances[0], indices[0]):
            if index < 0 or math.isinf(float(distance)):
                continue
            item = self.items[int(index)]
            memories.append(
                {
                    "text": item.text,
                    "metadata": item.metadata,
                    "sentiment": item.sentiment,
                    "timestep": item.timestep,
                    "distance": float(distance),
                }
            )
        return memories

    def relationship_score(self, other_agent_id: int) -> float:
        return self.relationships[other_agent_id]["trust"]

    def relationship_status(self, other_agent_id: int) -> str:
        relation = self.relationships[other_agent_id]
        trust = relation["trust"]
        interactions = relation["interactions"]
        if trust >= 0.55 and interactions >= 4:
            return "alliance"
        if trust <= -0.45 and interactions >= 3:
            return "rivalry"
        if trust >= 0.18:
            return "friendly"
        if trust <= -0.18:
            return "hostile"
        return "neutral"

    def update_relationship(self, other_agent_id: int, sentiment: float):
        relation = self.relationships[other_agent_id]
        relation["interactions"] += 1
        relation["last_sentiment"] = sentiment
        if sentiment >= 0.15:
            relation["positive"] += 1
        elif sentiment <= -0.15:
            relation["negative"] += 1
        repeated_negative_pressure = max(0, relation["negative"] - relation["positive"]) * 0.06
        delta = sentiment * 0.22 - repeated_negative_pressure
        relation["trust"] = max(-1.0, min(1.0, relation["trust"] * 0.88 + delta))

    def relationship_summary(self):
        return {
            str(agent_id): {
                "trust": round(values["trust"], 3),
                "positive": values["positive"],
                "negative": values["negative"],
                "interactions": values["interactions"],
                "last_sentiment": round(values["last_sentiment"], 3),
                "status": self.relationship_status(agent_id),
            }
            for agent_id, values in self.relationships.items()
        }
