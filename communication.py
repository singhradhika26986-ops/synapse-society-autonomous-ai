import os
import random
import re


class Communicator:
    """Context-aware short message generator with HuggingFace support."""

    def __init__(self):
        self.generator = None
        self.transformer_ready = False
        self._load_transformer_if_available()

    def _load_transformer_if_available(self):
        if os.environ.get("SYNAPSE_DISABLE_TRANSFORMER") == "1":
            return
        model_name = os.environ.get("SYNAPSE_TEXT_MODEL", "sshleifer/tiny-gpt2")
        allow_download = os.environ.get("SYNAPSE_ALLOW_MODEL_DOWNLOAD") == "1"
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=not allow_download)
            model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=not allow_download)
            pad_token_id = tokenizer.eos_token_id

            self.generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=18,
                do_sample=True,
                temperature=0.8,
                pad_token_id=pad_token_id,
            )
            self.transformer_ready = True
        except Exception:
            self.generator = None
            self.transformer_ready = False

    def generate(self, speaker, listener, memories, environment_summary, conversation_history=None):
        conversation_history = conversation_history or []
        memory_hint = memories[0]["text"] if memories else "no shared history"
        recent_lines = [
            f"A{item['speaker_id']} to A{item['listener_id']}: {item['message']}"
            for item in conversation_history[-6:]
        ]
        recent_hint = " | ".join(recent_lines) if recent_lines else "no recent conversations"
        trust = speaker.memory.relationship_score(listener.id)
        prompt = (
            f"Agent {speaker.id} is {speaker.personality}, feels {speaker.mood}, hunger {speaker.hunger}, "
            f"thirst {speaker.thirst}, energy {speaker.energy}, specialization {speaker.specialization}, "
            f"trust toward Agent {listener.id}: {trust:.2f}. "
            f"Goal: {speaker.goal}. World: {environment_summary}. Memory: {memory_hint}. "
            f"Recent dialogue: {recent_hint}. Write one fresh non-repeating short message:"
        )
        if self.generator:
            try:
                output = self.generator(prompt, num_return_sequences=1)[0]["generated_text"]
                message = output.replace(prompt, "").strip().split("\n")[0]
                if message and not self._recently_repeated(message, conversation_history):
                    return message[:120]
            except Exception:
                pass
        return self._fallback_message(speaker, listener, memories, environment_summary, conversation_history)

    def _fallback_message(self, speaker, listener, memories, environment_summary, conversation_history):
        trust = speaker.memory.relationship_score(listener.id)
        strongest_memory = max(memories, key=lambda memory: abs(memory["sentiment"]), default=None)
        memory_fragment = self._memory_fragment(strongest_memory)
        pressure = self._pressure_phrase(speaker, environment_summary)
        intent = random.choice(
            [
                "I need a practical signal",
                "my next move is still open",
                "this could change my route",
                "I am testing whether this alliance holds",
                "the resource map is shifting",
                "I am comparing trust against hunger",
            ]
        )
        style_openers = {
            "cooperative": ["I can coordinate", "Let's split the risk", "I will share a route"],
            "aggressive": ["Keep pace or clear space", "I am taking the direct line", "Prove you are useful"],
            "selfish": ["I will trade information", "I need the resource angle", "Convince me this helps me"],
            "social": ["I want to compare notes", "Your pattern caught my attention", "Let's keep contact"],
            "analytical": ["The trend suggests", "My memory points to", "I calculate a better option"],
        }
        trust_phrases = self._trust_phrases(trust)
        variants = []
        for opener in style_openers[speaker.personality]:
            variants.append(
                f"{opener}, Agent {listener.id}; {pressure}, {trust_phrases}, and {intent}."
            )
            variants.append(
                f"{opener}. {memory_fragment} Energy {speaker.energy}, hunger {speaker.hunger}; {intent}."
            )
        if speaker.hunger > 70:
            variants.append(f"Food pressure is high for me; share a lead or I will search alone, Agent {listener.id}.")
        if trust < -0.35:
            variants.append(f"I remember trouble between us, Agent {listener.id}; keep distance unless you offer value.")
        if trust > 0.35:
            variants.append(
                f"Your past help still matters, Agent {listener.id}; at step {environment_summary['timestep']} I am leaning toward cooperation."
            )
        return self._choose_unique(variants, conversation_history)

    def _pressure_phrase(self, speaker, environment_summary):
        if speaker.hunger > 75:
            return "hunger is forcing a resource-first choice"
        if speaker.energy < 35:
            return "my energy is low, so I need a low-risk plan"
        resources = environment_summary.get("resources", {})
        if resources.get("food", 0) < 4 or resources.get("water", 0) < 3:
            return "basic resources are getting scarce"
        if speaker.thirst > 72:
            return "thirst is pushing me toward water"
        return "the grid still has room to explore"

    def _trust_phrases(self, trust):
        if trust > 0.45:
            return "my trust in you is rising"
        if trust > 0.1:
            return "you have earned a little room"
        if trust < -0.45:
            return "my trust in you is badly damaged"
        if trust < -0.1:
            return "I am watching for another bad exchange"
        return "I have not decided if you are reliable"

    def _memory_fragment(self, memory):
        if not memory:
            return "I have no useful memory of you yet."
        cleaned = re.sub(r"\s+", " ", memory["text"]).strip()
        message_match = re.search(r"message[\"']?: [\"']([^\"']+)", cleaned)
        if message_match:
            return f"I remember you around the phrase '{message_match.group(1)[:48]}'."
        interaction_match = re.search(r"Interaction with Agent \d+: (.+)", cleaned)
        if interaction_match:
            return f"I remember that exchange as '{interaction_match.group(1)[:54]}'."
        action_match = re.search(r"chose ([a-z]+).*Reward: ([\-0-9.]+)", cleaned)
        if action_match:
            return f"I remember a {action_match.group(1)} choice with reward {action_match.group(2)}."
        return f"I remember a relevant signal: {cleaned[:58]}."

    def _choose_unique(self, variants, conversation_history):
        recent = {item["message"] for item in conversation_history[-12:]}
        random.shuffle(variants)
        for variant in variants:
            if variant not in recent and not self._recently_repeated(variant, conversation_history):
                return variant[:140]
        suffix = random.randint(100, 999)
        return f"{variants[0][:112]} Fresh signal {suffix}."

    def _recently_repeated(self, message, conversation_history):
        normalized = self._normalize(message)
        for item in conversation_history[-10:]:
            if self._normalize(item["message"]) == normalized:
                return True
        return False

    def _normalize(self, message):
        return re.sub(r"[^a-z0-9]+", " ", message.lower()).strip()


class ConversationHistory:
    def __init__(self, limit=20):
        self.limit = limit
        self.messages = []

    def add(self, timestep, speaker_id, listener_id, message, sentiment):
        entry = {
            "timestep": timestep,
            "speaker_id": speaker_id,
            "listener_id": listener_id,
            "message": message,
            "sentiment": sentiment,
        }
        self.messages.append(entry)
        self.messages = self.messages[-self.limit :]
        return entry
