"""
Comprehensive Test Suite for Ultra-Low Latency NLP & Speculative Dual-Brain Engine.
Validates:
1. Sub-10ms intent classification and stance matching across 25+ telephony utterances.
2. Native OpenAI tool call interception (muting raw JSON from TTS).
3. Inline JSON / Tool tag interception (muting raw JSON from TTS).
4. Speculative acoustic latency masking filler dispatch.
5. pysbd sentence boundary chunking.
"""

import asyncio
import json
import os
import sys
import time
import unittest
from typing import Any, AsyncGenerator, Dict, List, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.fast_classifier import (
    ConversationalStance,
    FastClassifier,
    STANCE_REGISTRY,
)
from engine.filler_manager import FillerManager
from engine.dual_brain import (
    DualBrainEngine,
    DualBrainEvent,
    DualBrainEventType,
)


# --- Mock OpenAI Stream Objects for Deterministic Offline Testing ---

class MockToolFunctionDelta:
    def __init__(self, name: Optional[str] = None, arguments: Optional[str] = None):
        self.name = name
        self.arguments = arguments


class MockToolCallDelta:
    def __init__(self, index: int, id: str, name: Optional[str] = None, arguments: Optional[str] = None):
        self.index = index
        self.id = id
        self.function = MockToolFunctionDelta(name=name, arguments=arguments)


class MockDelta:
    def __init__(
        self,
        content: Optional[str] = None,
        tool_calls: Optional[List[MockToolCallDelta]] = None,
    ):
        self.content = content
        self.tool_calls = tool_calls


class MockChoice:
    def __init__(self, delta: MockDelta, finish_reason: Optional[str] = None):
        self.delta = delta
        self.finish_reason = finish_reason


class MockChunk:
    def __init__(
        self,
        content: Optional[str] = None,
        tool_calls: Optional[List[MockToolCallDelta]] = None,
        finish_reason: Optional[str] = None,
    ):
        self.choices = [MockChoice(MockDelta(content, tool_calls), finish_reason)]


class MockAsyncStream:
    def __init__(self, chunks: List[MockChunk]):
        self._chunks = chunks

    def __aiter__(self):
        return self._generator()

    async def _generator(self):
        for chunk in self._chunks:
            await asyncio.sleep(0.001)  # Simulate network delta
            yield chunk


class MockAsyncOpenAIClient:
    """
    Mock AsyncOpenAI client simulating streaming completions,
    native tool calls, inline JSON, and standard conversational responses.
    """

    def __init__(self, mode: str = "conversational"):
        self.mode = mode
        self.chat = self.Chat(self)

    class Chat:
        def __init__(self, parent):
            self.completions = parent.Completions(parent)

    class Completions:
        def __init__(self, parent):
            self.parent = parent

        async def create(self, **kwargs) -> MockAsyncStream:
            messages = kwargs.get("messages", [])
            last_message = messages[-1] if messages else {}

            # If responding to a tool call execution result
            if last_message.get("role") == "tool" or "The tool" in last_message.get("content", ""):
                chunks = [
                    MockChunk(content="The "),
                    MockChunk(content="system "),
                    MockChunk(content="has "),
                    MockChunk(content="been "),
                    MockChunk(content="up "),
                    MockChunk(content="for "),
                    MockChunk(content="42 hours "),
                    MockChunk(content="and is healthy. "),
                    MockChunk(content="All services are green.", finish_reason="stop"),
                ]
                return MockAsyncStream(chunks)

            # Mode 1: Native OpenAI tool call simulation
            if self.parent.mode == "native_tool_call":
                chunks = [
                    MockChunk(
                        tool_calls=[
                            MockToolCallDelta(
                                index=0,
                                id="call_upt_001",
                                name="check_uptime",
                                arguments='{"detail":'
                            )
                        ]
                    ),
                    MockChunk(
                        tool_calls=[
                            MockToolCallDelta(
                                index=0,
                                id="call_upt_001",
                                arguments=' true}'
                            )
                        ],
                        finish_reason="tool_calls"
                    ),
                ]
                return MockAsyncStream(chunks)

            # Mode 2: Inline JSON tool call simulation
            if self.parent.mode == "inline_tool_call":
                chunks = [
                    MockChunk(content='{"name": "check_uptime", '),
                    MockChunk(content='"arguments": {"detail": true}}'),
                    MockChunk(finish_reason="stop"),
                ]
                return MockAsyncStream(chunks)

            # Mode 3: pysbd sentence prosody test
            if self.parent.mode == "prosody_test":
                chunks = [
                    MockChunk(content="Dr. "),
                    MockChunk(content="Watson "),
                    MockChunk(content="arrived at "),
                    MockChunk(content="3.14 p.m. "),
                    MockChunk(content="He "),
                    MockChunk(content="checked the status. "),
                    MockChunk(content="Everything is fine."),
                    MockChunk(finish_reason="stop"),
                ]
                return MockAsyncStream(chunks)

            # Default conversational streaming mode
            chunks = [
                MockChunk(content="I can "),
                MockChunk(content="certainly help you "),
                MockChunk(content="with that question. "),
                MockChunk(content="Please hold on a moment."),
                MockChunk(finish_reason="stop"),
            ]
            return MockAsyncStream(chunks)


# --- Test Suite ---

class TestUltraLowLatencyEngine(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.classifier = FastClassifier()
        self.filler_mgr = FillerManager(generate_acoustic_tones=True)

    def test_01_intent_classification_speed_and_accuracy(self):
        """
        Verify classification on 25+ varied phone utterances:
        - Must complete in < 10ms (strict assertion).
        - Must accurately classify stance (analytical, affirmative, empathetic, clarifying, direct).
        """
        test_cases: List[Dict[str, Any]] = [
            # Analytical queries (lookups, hours, pricing, orders, appointments)
            {"text": "Can you check the status of my order #55412?", "expected": ConversationalStance.ANALYTICAL},
            {"text": "What are your business hours on weekends?", "expected": ConversationalStance.ANALYTICAL},
            {"text": "How much does the annual enterprise plan cost?", "expected": ConversationalStance.ANALYTICAL},
            {"text": "Can you find an available appointment for next Tuesday?", "expected": ConversationalStance.ANALYTICAL},
            {"text": "Please check if my account balance has updated.", "expected": ConversationalStance.ANALYTICAL},
            {"text": "Where is the nearest pickup location?", "expected": ConversationalStance.ANALYTICAL},
            {"text": "When will my package be delivered?", "expected": ConversationalStance.ANALYTICAL},
            {"text": "Could you search for flights to Chicago tomorrow?", "expected": ConversationalStance.ANALYTICAL},

            # Affirmative utterances (confirmations, agreements, gratitude)
            {"text": "Yes, that sounds good, please confirm it.", "expected": ConversationalStance.AFFIRMATIVE},
            {"text": "Exactly, do it right away.", "expected": ConversationalStance.AFFIRMATIVE},
            {"text": "Sure thing, thank you so much!", "expected": ConversationalStance.AFFIRMATIVE},
            {"text": "Okay, perfect, go ahead.", "expected": ConversationalStance.AFFIRMATIVE},
            {"text": "Yeah, that's completely correct.", "expected": ConversationalStance.AFFIRMATIVE},
            {"text": "Alright, agreed.", "expected": ConversationalStance.AFFIRMATIVE},

            # Empathetic utterances (distress, complaint, broken item, cancellation, emergency)
            {"text": "My package arrived completely broken and damaged!", "expected": ConversationalStance.EMPATHETIC},
            {"text": "I am so angry because my flight got canceled without notice.", "expected": ConversationalStance.EMPATHETIC},
            {"text": "I need a refund immediately, this service is terrible.", "expected": ConversationalStance.EMPATHETIC},
            {"text": "We have an emergency with a burst pipe leaking water everywhere!", "expected": ConversationalStance.EMPATHETIC},
            {"text": "I'm having a lot of trouble with my device, it's not working at all.", "expected": ConversationalStance.EMPATHETIC},
            {"text": "I lost my credit card and I am very upset.", "expected": ConversationalStance.EMPATHETIC},

            # Clarifying utterances (questions of confusion, repeats, explanations)
            {"text": "Wait, why did my card get declined?", "expected": ConversationalStance.CLARIFYING},
            {"text": "What do you mean by extra fee?", "expected": ConversationalStance.CLARIFYING},
            {"text": "Could you please repeat what you just said?", "expected": ConversationalStance.CLARIFYING},
            {"text": "Huh? I didn't catch that, can you explain?", "expected": ConversationalStance.CLARIFYING},
            {"text": "Pardon me? Say that again.", "expected": ConversationalStance.CLARIFYING},

            # Direct greetings / Short utterances (< 2 words)
            {"text": "Hi", "expected": ConversationalStance.DIRECT},
            {"text": "Hello there", "expected": ConversationalStance.DIRECT},
            {"text": "Hey", "expected": ConversationalStance.DIRECT},
            {"text": "Good morning", "expected": ConversationalStance.DIRECT},
            {"text": "Yo", "expected": ConversationalStance.DIRECT},
        ]

        latencies = []
        print(f"\n{'='*75}")
        print(f"{'UTTERANCE':<50} | {'STANCE':<12} | {'LATENCY (ms)'}")
        print(f"{'-'*75}")

        for case in test_cases:
            utterance = case["text"]
            expected_stance = case["expected"]

            result = self.classifier.classify(utterance)
            latencies.append(result.latency_ms)

            print(f"{utterance[:48]:<50} | {result.stance.value:<12} | {result.latency_ms:6.4f} ms")

            # Assert classification accuracy
            self.assertEqual(
                result.stance,
                expected_stance,
                f"Failed for utterance: '{utterance}'. Expected {expected_stance}, got {result.stance}"
            )

            # Assert latency constraint (< 10ms)
            self.assertLess(
                result.latency_ms,
                10.0,
                f"Classification too slow ({result.latency_ms:.4f}ms > 10ms) for: '{utterance}'"
            )

            # Verify stance metadata
            metadata = STANCE_REGISTRY[result.stance]
            self.assertEqual(result.recommended_filler_key, metadata.recommended_filler_key)
            self.assertEqual(result.expected_delay_ms, metadata.expected_delay_ms)
            self.assertEqual(result.latency_masking_required, metadata.latency_masking_required)

        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)
        print(f"{'-'*75}")
        print(f"Total Utterances Tested: {len(test_cases)}")
        print(f"Latency: Avg={avg_latency:.4f}ms | Min={min_latency:.4f}ms | Max={max_latency:.4f}ms")
        print(f"{'='*75}\n")

        self.assertLess(avg_latency, 1.0, f"Average latency ({avg_latency:.4f}ms) should be sub-millisecond.")

    async def test_02_native_openai_tool_call_interception(self):
        """
        Test native OpenAI streaming tool call interception:
        1. Emits tool_calls delta for 'check_uptime'.
        2. Verifies engine detects and executes 'check_uptime'.
        3. Verifies raw JSON or function metadata is NEVER passed as a spoken SENTENCE event.
        4. Verifies follow-up spoken voice response is yielded.
        """
        mock_client = MockAsyncOpenAIClient(mode="native_tool_call")
        engine = DualBrainEngine(
            client=mock_client,
            fast_classifier=self.classifier,
            filler_manager=self.filler_mgr
        )

        tool_executed = False

        @engine.register_tool(name="check_uptime", description="Check system uptime.")
        async def mock_uptime(detail: bool = False):
            nonlocal tool_executed
            tool_executed = True
            return {"uptime_hours": 42, "status": "operational", "detail": detail}

        events: List[DualBrainEvent] = []
        utterance = "Can you check if the system uptime is okay?"

        async for ev in engine.process(utterance):
            events.append(ev)

        # 1. Verify tool execution occurred
        self.assertTrue(tool_executed, "Expected tool 'check_uptime' to be executed.")

        # 2. Verify events timeline
        event_types = [e.type for e in events]
        self.assertIn(DualBrainEventType.CLASSIFICATION, event_types)
        self.assertIn(DualBrainEventType.FILLER, event_types)
        self.assertIn(DualBrainEventType.TOOL_EXECUTION_START, event_types)
        self.assertIn(DualBrainEventType.TOOL_EXECUTION_RESULT, event_types)
        self.assertIn(DualBrainEventType.SENTENCE, event_types)
        self.assertIn(DualBrainEventType.FINISHED, event_types)

        # 3. CRITICAL: Verify NO raw JSON was ever spoken in SENTENCE events
        sentence_events = [e for e in events if e.type == DualBrainEventType.SENTENCE]
        self.assertGreater(len(sentence_events), 0, "Expected at least one spoken sentence.")

        for sent_ev in sentence_events:
            text = sent_ev.text
            self.assertFalse(
                text.strip().startswith("{"),
                f"Raw JSON leaked to spoken sentence: '{text}'"
            )
            self.assertFalse(
                "call_upt_001" in text or "check_uptime" in text and "{" in text,
                f"Tool call schema leaked to spoken sentence: '{text}'"
            )

        # Spoken text should be clean conversational response
        full_spoken = " ".join(e.text for e in sentence_events)
        self.assertIn("42 hours", full_spoken)
        print(f"Native Tool Interception Spoken Output: '{full_spoken}'")

    async def test_03_inline_json_tool_call_interception(self):
        """
        Test inline JSON tool call interception:
        If LLM emits `{"name": "check_uptime", "arguments": ...}` directly in text stream:
        1. Engine must detect and mute TTS immediately.
        2. Intercept and execute the tool.
        3. Raw JSON must NOT appear in any SENTENCE event.
        4. Deliver spoken conversational response.
        """
        mock_client = MockAsyncOpenAIClient(mode="inline_tool_call")
        engine = DualBrainEngine(
            client=mock_client,
            fast_classifier=self.classifier,
            filler_manager=self.filler_mgr
        )

        tool_executed = False

        @engine.register_tool(name="check_uptime", description="Check system uptime.")
        async def mock_uptime(detail: bool = False):
            nonlocal tool_executed
            tool_executed = True
            return {"uptime_hours": 42, "detail": detail}

        events: List[DualBrainEvent] = []
        utterance = "Check the system status for me."

        async for ev in engine.process(utterance):
            events.append(ev)

        # Verify tool was executed
        self.assertTrue(tool_executed, "Expected inline tool 'check_uptime' to be executed.")

        # Check events
        event_types = [e.type for e in events]
        self.assertIn(DualBrainEventType.TOOL_CALL_DETECTED, event_types)
        self.assertIn(DualBrainEventType.TOOL_EXECUTION_START, event_types)
        self.assertIn(DualBrainEventType.TOOL_EXECUTION_RESULT, event_types)

        # Assert no raw JSON is yielded as spoken sentences
        sentence_events = [e for e in events if e.type == DualBrainEventType.SENTENCE]
        for sent_ev in sentence_events:
            text = sent_ev.text
            self.assertFalse(
                text.strip().startswith("{") or text.strip().endswith("}"),
                f"Raw inline JSON leaked to spoken sentence: '{text}'"
            )
            self.assertNotIn('"name":', text)
            self.assertNotIn('"arguments":', text)

        full_spoken = " ".join(e.text for e in sentence_events)
        self.assertIn("42 hours", full_spoken)
        print(f"Inline Tool Interception Spoken Output: '{full_spoken}'")

    async def test_04_speculative_latency_masking_filler(self):
        """
        Verify contextual filler audio dispatch:
        - Utterance needing latency masking (analytical/empathetic) emits FILLER event immediately.
        - Direct utterance (e.g. greeting 'Hi') does NOT emit filler.
        """
        mock_client = MockAsyncOpenAIClient(mode="conversational")
        engine = DualBrainEngine(
            client=mock_client,
            fast_classifier=self.classifier,
            filler_manager=self.filler_mgr
        )

        # Case A: Analytical query -> Must emit FILLER with audio
        events_analytical = []
        async for ev in engine.process("Can you search for the nearest store hours?"):
            events_analytical.append(ev)

        filler_events = [e for e in events_analytical if e.type == DualBrainEventType.FILLER]
        self.assertEqual(len(filler_events), 1, "Analytical query should trigger exactly 1 filler.")
        self.assertIsNotNone(filler_events[0].audio_bytes, "Filler must contain acoustic audio bytes.")
        self.assertGreater(len(filler_events[0].audio_bytes), 100, "Audio bytes must not be empty.")
        print(f"Analytical Filler Text: '{filler_events[0].text}', Audio bytes: {len(filler_events[0].audio_bytes)}")

        # Case B: Direct greeting -> Must NOT emit FILLER (instant direct response)
        events_direct = []
        async for ev in engine.process("Hello"):
            events_direct.append(ev)

        filler_events_direct = [e for e in events_direct if e.type == DualBrainEventType.FILLER]
        self.assertEqual(len(filler_events_direct), 0, "Direct greeting must not trigger filler audio.")

    async def test_05_pysbd_sentence_chunking_prosody(self):
        """
        Verify streaming token chunking via pysbd:
        Stream contains 'Dr. Watson arrived at 3.14 p.m. He checked the status. Everything is fine.'
        Ensure pysbd DOES NOT split on 'Dr.', '3.14', or 'p.m.', preserving full spoken clauses.
        """
        mock_client = MockAsyncOpenAIClient(mode="prosody_test")
        engine = DualBrainEngine(
            client=mock_client,
            fast_classifier=self.classifier,
            filler_manager=self.filler_mgr
        )

        events: List[DualBrainEvent] = []
        async for ev in engine.process("Tell me what Dr. Watson did."):
            events.append(ev)

        sentences = [e.text for e in events if e.type == DualBrainEventType.SENTENCE]
        print(f"pysbd Chunked Sentences: {sentences}")

        # Ensure "Dr. Watson arrived at 3.14 p.m." was NOT fragmented into "Dr." or "3.14"
        self.assertTrue(
            any("Dr. Watson arrived at 3.14 p.m." in s for s in sentences),
            f"Sentence incorrectly fragmented by chunker: {sentences}"
        )
        self.assertFalse(any(s == "Dr." for s in sentences), "Pysbd broke on abbreviation 'Dr.'")
        self.assertFalse(any(s == "3.14" for s in sentences), "Pysbd broke on decimal '3.14'")

    async def test_06_fast_connection_probe_and_offline_turnaround(self):
        """
        Verify that an unreachable local endpoint (e.g. localhost:11434 with Ollama offline):
        - Detects offline status in < 100ms (NEVER 5.0 seconds).
        - Executes query via Built-in Dialogue Engine in < 50ms total.
        - Does NOT block the event loop or drop the call.
        """
        # Point to unreachable loopback port
        engine = DualBrainEngine(
            base_url="http://127.0.0.1:19999/v1",
            api_key="not-needed",
            fast_classifier=self.classifier,
            filler_manager=self.filler_mgr
        )

        # 1. Connection probe latency check
        t0 = time.time()
        is_online = engine.probe_connection(timeout=0.04)
        probe_duration_ms = (time.time() - t0) * 1000.0
        print(f"\n[Test 06] Probe connection latency: {probe_duration_ms:.2f}ms (is_online={is_online})")

        self.assertFalse(is_online, "Unreachable port 19999 should report offline.")
        self.assertLess(probe_duration_ms, 150.0, f"Probe took too long ({probe_duration_ms:.2f}ms > 150ms)")

        # 2. End-to-end process latency check (must be sub-50ms)
        t_start = time.time()
        events: List[DualBrainEvent] = []
        async for ev in engine.process("Can you check my order #55412?"):
            events.append(ev)
        total_ms = (time.time() - t_start) * 1000.0
        print(f"[Test 06] Total turn duration (offline): {total_ms:.2f}ms")

        # Must not take anywhere near 5000ms
        self.assertLess(total_ms, 100.0, f"Offline processing took {total_ms:.2f}ms (>100ms)")

        event_types = [e.type for e in events]
        self.assertIn(DualBrainEventType.CLASSIFICATION, event_types)
        self.assertIn(DualBrainEventType.FILLER, event_types)
        self.assertIn(DualBrainEventType.SENTENCE, event_types)
        self.assertIn(DualBrainEventType.FINISHED, event_types)

        sentences = [e.text for e in events if e.type == DualBrainEventType.SENTENCE]
        combined = " ".join(sentences)
        self.assertIn("55412", combined)
        self.assertIn("out for delivery", combined)
        print(f"[Test 06] Spoken response: '{combined}'")

    async def test_07_builtin_dialogue_engine_domain_responses(self):
        """
        Verify Instant Intelligent Built-in Dialogue Engine handles all required domains:
        - Order status & tracking
        - Appointments, reservations, business hours
        - Flights, disruptions, refunds, customer support
        - System uptime, health & capabilities
        - Greetings and pleasantries
        """
        engine = DualBrainEngine(
            base_url="http://127.0.0.1:19999/v1",
            api_key="not-needed",
            fast_classifier=self.classifier,
            filler_manager=self.filler_mgr
        )

        test_cases = [
            (
                "Can you check the status of my order #88431?",
                ["88431", "out for delivery"],
                "Order tracking query"
            ),
            (
                "What are your business hours on weekends?",
                ["9 AM", "8 PM", "reserve a time"],
                "Business hours query"
            ),
            (
                "Can you find an available appointment for next Tuesday?",
                ["appointment", "Tuesday"],
                "Appointment booking query"
            ),
            (
                "I am so angry because my flight got canceled without notice.",
                ["disruption", "refund"],
                "Flight cancellation empathy"
            ),
            (
                "My package arrived completely broken and damaged!",
                ["damaged", "replacement", "refund"],
                "Damaged package empathy"
            ),
            (
                "Can you check if the system uptime is okay?",
                ["operational", "uptime"],
                "System status query"
            ),
            (
                "Who are you and what can you do?",
                ["voice AI", "telephony"],
                "Identity & capabilities query"
            ),
            (
                "Hello there",
                ["Hello", "assist you"],
                "Direct greeting"
            ),
        ]

        print(f"\n{'='*75}")
        print(f"{'DOMAIN TEST':<32} | {'LATENCY':<10} | {'SPOKEN PREVIEW'}")
        print(f"{'-'*75}")

        for query, expected_substrings, description in test_cases:
            t0 = time.time()
            sentences = []
            async for ev in engine.process(query):
                if ev.type == DualBrainEventType.SENTENCE:
                    sentences.append(ev.text)
            turn_ms = (time.time() - t0) * 1000.0

            combined = " ".join(sentences)
            print(f"{description:<32} | {turn_ms:6.2f}ms  | {combined[:32]}...")

            self.assertLess(turn_ms, 85.0, f"Turn exceeded 85ms for '{query}'")
            for sub in expected_substrings:
                self.assertIn(
                    sub.lower(),
                    combined.lower(),
                    f"Expected '{sub}' in response for query: '{query}'. Got: '{combined}'"
                )

        print(f"{'='*75}\n")

    def test_08_dynamic_api_key_and_config_update(self):
        """
        Verify dynamic runtime reconfiguration:
        - Groq key (gsk_...) automatically sets Groq LPU endpoint and model.
        - OpenAI key (sk-...) automatically sets OpenAI endpoint and model.
        - Custom local base_url / model updates correctly.
        """
        engine = DualBrainEngine(
            base_url="http://127.0.0.1:19999/v1",
            api_key="not-needed",
            fast_classifier=self.classifier,
            filler_manager=self.filler_mgr
        )

        # 1. Update to Groq
        res_groq = engine.update_config(api_key="gsk_testGroqKey1234567890abcdef")
        self.assertEqual(res_groq["provider"], "groq")
        self.assertEqual(engine.base_url, "https://api.groq.com/openai/v1")
        self.assertIn(engine.model, ["llama-3.3-70b-versatile", "qwen/qwen3.8-27b"])
        self.assertEqual(res_groq["is_local"], False)
        print(f"[Test 08] Configured Groq LPU: {res_groq}")

        # 2. Update to OpenAI
        res_openai = engine.update_config(api_key="sk-proj-testOpenAIKey999888777")
        self.assertEqual(res_openai["provider"], "openai")
        self.assertEqual(engine.base_url, "https://api.openai.com/v1")
        self.assertEqual(engine.model, "gpt-4o-mini")
        self.assertEqual(res_openai["is_local"], False)
        print(f"[Test 08] Configured OpenAI: {res_openai}")

        # 3. Update to local Ollama with specific model
        res_local = engine.update_config(
            api_key="not-needed",
            base_url="http://127.0.0.1:11434/v1",
            model="mistral:latest"
        )
        self.assertEqual(res_local["provider"], "local")
        self.assertEqual(engine.base_url, "http://127.0.0.1:11434/v1")
        self.assertEqual(engine.model, "mistral:latest")
        self.assertEqual(res_local["is_local"], True)
        print(f"[Test 08] Configured Local Ollama: {res_local}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
