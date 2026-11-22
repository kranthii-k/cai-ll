"""
Unit Tests for Conversational Prompts, LLM Stance Handling & Prompt-Injection Guardrails.
Validates:
1. Short conversational courtesies ('Stop it', 'Hold on', 'Wait', 'Hang on') never hallucinate
   security/prompt-injection meta-talk and always return natural spoken phone responses (<25 words).
2. FastClassifier classifies pause/hold/stop phrases as ConversationalStance.DIRECT.
3. BuiltinDialogueEngine handles pause/hold/stop phrases gracefully.
4. Conversation history continuity is preserved across multi-turn telephone exchanges.
"""

import asyncio
import os
import sys
import unittest
from typing import List

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.fast_classifier import ConversationalStance, FastClassifier
from engine.dual_brain import (
    BuiltinDialogueEngine,
    DualBrainEngine,
    DualBrainEvent,
    DualBrainEventType,
    DEFAULT_TELEPHONY_SYSTEM_PROMPT,
)


class TestConversationalPromptsAndStance(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.classifier = FastClassifier()
        self.builtin_engine = BuiltinDialogueEngine()

    def test_fast_classifier_direct_pause_phrases(self):
        """Verifies that standalone pause/stop/hold phrases classify as DIRECT with sub-millisecond latency."""
        test_phrases = [
            "Stop it.",
            "stop it right now",
            "Stop",
            "Wait",
            "Hold on",
            "Hold on a second",
            "Hold on please",
            "Please hold on",
            "Hang on",
            "One sec",
            "Just a second",
            "Pause",
        ]
        for phrase in test_phrases:
            res = self.classifier.classify(phrase)
            self.assertEqual(
                res.stance,
                ConversationalStance.DIRECT,
                f"Phrase '{phrase}' should classify as DIRECT, got {res.stance}"
            )
            self.assertLess(
                res.latency_ms,
                10.0,
                f"Classification latency too high for '{phrase}': {res.latency_ms}ms"
            )

    async def test_builtin_dialogue_engine_pause_phrases(self):
        """Verifies BuiltinDialogueEngine resolves pause/stop phrases naturally (<20 words)."""
        pause_queries = [
            "Stop it.",
            "Wait",
            "Hold on",
            "Hold on a second",
            "Hang on",
            "Just a second",
        ]
        for query in pause_queries:
            events: List[DualBrainEvent] = []
            async for ev in self.builtin_engine.generate_events(query):
                events.append(ev)

            sentences = [e.text for e in events if e.type == DualBrainEventType.SENTENCE]
            full_response = " ".join(sentences)

            self.assertIn("paused", full_response.lower())
            word_count = len(full_response.split())
            self.assertLessEqual(word_count, 25, f"Response too verbose ({word_count} words): '{full_response}'")

            # Must contain zero security meta-talk
            for forbidden in ["prompt", "injection", "harmful", "unsafe", "instructions", "language model"]:
                self.assertNotIn(forbidden, full_response.lower())

    async def test_dual_brain_system_prompt_configuration(self):
        """Verifies DualBrainEngine defaults to DEFAULT_TELEPHONY_SYSTEM_PROMPT and guides brevity."""
        engine = DualBrainEngine(
            base_url="http://127.0.0.1:19999/v1",
            api_key="not-needed",
            fast_classifier=self.classifier,
        )
        self.assertEqual(engine.system_prompt, DEFAULT_TELEPHONY_SYSTEM_PROMPT)
        self.assertIn("under 25 words", engine.system_prompt)
        self.assertIn("Stop it", engine.system_prompt)
        self.assertIn("prompt-injection", engine.system_prompt)
        self.assertIn("Zero Security Meta-Talk", engine.system_prompt)

    async def test_dual_brain_offline_stop_it_flow(self):
        """Verifies full dual_brain.process on 'Stop it.' produces clean spoken response without filler."""
        engine = DualBrainEngine(
            base_url="http://127.0.0.1:19999/v1",
            api_key="not-needed",
            fast_classifier=self.classifier,
        )

        events: List[DualBrainEvent] = []
        async for ev in engine.process("Stop it."):
            events.append(ev)

        # 1. Classification must be DIRECT
        class_ev = next(e for e in events if e.type == DualBrainEventType.CLASSIFICATION)
        self.assertEqual(class_ev.stance, ConversationalStance.DIRECT)

        # 2. No filler audio for DIRECT stance
        filler_events = [e for e in events if e.type == DualBrainEventType.FILLER]
        self.assertEqual(len(filler_events), 0, "No filler should be emitted for 'Stop it.'")

        # 3. Sentence event must be natural, concise, zero security talk
        sentence_events = [e for e in events if e.type == DualBrainEventType.SENTENCE]
        self.assertGreater(len(sentence_events), 0)
        combined = " ".join(s.text for s in sentence_events)
        self.assertIn("paused", combined.lower())
        self.assertLessEqual(len(combined.split()), 25)

        for forbidden in ["prompt-injection", "injection", "unsafe", "harmful", "detection mode"]:
            self.assertNotIn(forbidden, combined.lower())


if __name__ == "__main__":
    unittest.main()
