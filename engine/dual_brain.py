"""
Dual Brain Engine: Speculative Dual-Path Telephony Orchestration.
Integrates sub-10ms fast intent classification, contextual filler dispatch,
concurrent LLM streaming with pysbd sentence chunking, and tool call interception.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterable,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from openai import AsyncOpenAI
import pysbd

from .fast_classifier import (
    ClassificationResult,
    ConversationalStance,
    FastClassifier,
)
from .filler_manager import FillerItem, FillerManager

logger = logging.getLogger("dual_brain")


class DualBrainEventType(str, Enum):
    CLASSIFICATION = "classification"
    FILLER = "filler"
    SENTENCE = "sentence"
    TOOL_CALL_DETECTED = "tool_call_detected"
    TOOL_EXECUTION_START = "tool_execution_start"
    TOOL_EXECUTION_RESULT = "tool_execution_result"
    FINISHED = "finished"
    ERROR = "error"


@dataclass
class DualBrainEvent:
    type: DualBrainEventType
    text: str = ""
    audio_bytes: Optional[bytes] = None
    stance: Optional[ConversationalStance] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "text": self.text,
            "has_audio": self.audio_bytes is not None,
            "stance": self.stance.value if self.stance else None,
            "metadata": self.metadata,
        }


@dataclass
class ToolDefinition:
    name: str
    description: str
    func: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_openai_schema(self) -> dict:
        schema = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
        return schema


class DualBrainEngine:
    """
    Speculative Dual-Brain Engine for Ultra-Low Latency Telephony Agents:
    1. Fast stance classification (<5ms) via FastClassifier.
    2. Contextual acoustic filler dispatch via FillerManager for latency masking.
    3. Concurrent streaming LLM query via AsyncOpenAI.
    4. Sentence boundary chunking via pysbd for zero prosody break.
    5. Tool call interception: captures JSON/tool schemas, executes async,
       and prevents raw JSON from reaching TTS.
    """

    def __init__(
        self,
        client: Optional[AsyncOpenAI] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        filler_manager: Optional[FillerManager] = None,
        fast_classifier: Optional[FastClassifier] = None,
    ):
        groq_key = os.getenv("GROQ_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if client:
            self.client = client
            self.model = model or os.getenv("LLM_MODEL", "qwen2.5:0.5b")
        elif groq_key:
            # Ultra-low latency cloud LPU provider (Groq: ~100ms TTFT)
            self.client = AsyncOpenAI(
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"),
                api_key=groq_key,
            )
            self.model = model or os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        elif openai_key and openai_key != "not-needed":
            # Cloud OpenAI provider
            self.client = AsyncOpenAI(
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                api_key=openai_key,
            )
            self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        else:
            # Local Ollama / vLLM provider
            self.client = AsyncOpenAI(
                base_url=os.getenv("OPENAI_BASE_URL", base_url or "http://localhost:11434/v1"),
                api_key=api_key or "not-needed",
            )
            self.model = model or os.getenv("LLM_MODEL", "qwen2.5:0.5b")

        self.system_prompt = system_prompt or (
            "You are an ultra-fast, helpful voice AI telephony agent. "
            "Keep spoken responses concise, direct, and conversational. "
            "Never output raw code or JSON to the user."
        )
        self.classifier = fast_classifier or FastClassifier()
        self.filler_manager = filler_manager or FillerManager()
        self.segmenter = pysbd.Segmenter(language="en", clean=False)
        self.tools: Dict[str, ToolDefinition] = {}

    def register_tool(
        self,
        name: Optional[str] = None,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Callable:
        """
        Decorator or method to register an async or sync function as an agent tool.
        """
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            tool_desc = description or (func.__doc__ or "").strip()
            self.tools[tool_name] = ToolDefinition(
                name=tool_name,
                description=tool_desc,
                func=func,
                parameters=parameters or {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            )
            return func

        return decorator

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Executes a registered tool asynchronously.
        """
        if tool_name not in self.tools:
            raise KeyError(f"Tool '{tool_name}' not registered in DualBrainEngine.")

        tool = self.tools[tool_name]
        func = tool.func

        try:
            if inspect.iscoroutinefunction(func):
                result = await func(**arguments)
            else:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, lambda: func(**arguments))
            return result
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
            return {"error": str(e)}

    @staticmethod
    def _is_tool_call_start(text: str) -> bool:
        """
        Detects if text begins with a tool call signature (e.g. JSON or tags).
        """
        stripped = text.strip()
        if not stripped:
            return False
        return (
            stripped.startswith("{")
            or stripped.startswith("<tool_call>")
            or stripped.startswith("[TOOL_CALL]")
            or stripped.startswith("```json")
            or stripped.startswith("@function_tool")
            or '{"name":' in stripped
            or '{"tool":' in stripped
        )

    @staticmethod
    def _extract_inline_tool_call(text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Extracts tool name and arguments from inline JSON or tags.
        """
        stripped = text.strip()

        # Remove markdown fences or tags if present
        cleaned = re.sub(r"^```(?:json)?", "", stripped, flags=re.MULTILINE)
        cleaned = re.sub(r"```$", "", cleaned, flags=re.MULTILINE).strip()
        cleaned = re.sub(r"</?tool_call>", "", cleaned).strip()

        # Find outer JSON object {...}
        match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if not match:
            return None

        try:
            data = json.loads(match.group(1))
            if not isinstance(data, dict):
                return None

            # Look for tool name keys
            tool_name = (
                data.get("name")
                or data.get("tool")
                or data.get("function")
                or data.get("action")
            )

            # Look for arguments
            args = (
                data.get("arguments")
                or data.get("parameters")
                or data.get("args")
                or data.get("input")
            )

            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            elif args is None:
                # Use remaining dictionary keys if tool_name is extracted
                args = {k: v for k, v in data.items() if k not in {"name", "tool", "function", "action"}}

            if tool_name and isinstance(args, dict):
                return str(tool_name), args
        except Exception:
            return None

        return None

    async def process(
        self,
        user_utterance: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[DualBrainEvent, None]:
        """
        Primary speculative dual-path pipeline:
        1. Sub-5ms fast intent classification.
        2. Immediate contextual filler audio dispatch if latency masking needed.
        3. Concurrently dispatches prompt to LLM.
        4. Sentence boundary chunking with pysbd.
        5. Intercepts tool calls and muting TTS from raw JSON.
        """
        # Step 1: Fast Intent Classification (<5ms)
        classification: ClassificationResult = self.classifier.classify(user_utterance)
        yield DualBrainEvent(
            type=DualBrainEventType.CLASSIFICATION,
            stance=classification.stance,
            metadata=classification.to_dict()
        )

        # Step 2: Contextual Latency Masking Filler
        if classification.latency_masking_required:
            filler: Optional[FillerItem] = self.filler_manager.get_filler(
                classification.stance,
                user_utterance
            )
            if filler:
                yield DualBrainEvent(
                    type=DualBrainEventType.FILLER,
                    text=filler.text,
                    audio_bytes=filler.audio_bytes,
                    stance=classification.stance,
                    metadata=filler.to_dict()
                )

        # Step 3: Prepare messages for LLM
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_utterance})

        # Step 4: Stream response with pysbd sentence chunking & tool interception
        async for event in self._stream_and_chunk(messages):
            yield event

        yield DualBrainEvent(type=DualBrainEventType.FINISHED)

    async def _stream_and_chunk(
        self,
        messages: List[Dict[str, Any]],
        depth: int = 0
    ) -> AsyncGenerator[DualBrainEvent, None]:
        """
        Streams completions, chunking sentences with pysbd and intercepting tool calls.
        Recursively handles tool call execution and model follow-up.
        """
        if depth > 3:
            logger.warning("Maximum tool call recursion depth reached.")
            return

        openai_tools = [tool.to_openai_schema() for tool in self.tools.values()] or None

        try:
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "stream": True,
            }
            if openai_tools:
                kwargs["tools"] = openai_tools

            response = await self.client.chat.completions.create(**kwargs)
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            err_str = str(e).lower()
            if any(term in err_str for term in ["connection", "connect", "11434", "refused", "offline"]):
                fallback_text = (
                    "I can hear you loud and clear. My acoustic filler reflex, equal-power crossfader, "
                    "and full-duplex audio stream are working, but the LLM server is currently offline. "
                    "Please start Ollama or set an API key to enable full conversational intelligence."
                )
                yield DualBrainEvent(
                    type=DualBrainEventType.SENTENCE,
                    text=fallback_text,
                    metadata={"fallback": True, "error": str(e)}
                )
            else:
                yield DualBrainEvent(
                    type=DualBrainEventType.ERROR,
                    text=str(e),
                    metadata={"error": str(e)}
                )
            return

        text_buffer = ""
        is_inline_tool_mode = False
        inline_tool_buffer = ""

        # Tracking for native OpenAI tool_calls
        has_native_tool_call = False
        native_tool_calls: Dict[int, Dict[str, Any]] = {}

        async for chunk in response:
            choice = chunk.choices[0] if chunk.choices else None
            if not choice:
                continue

            delta = choice.delta

            # 1. Check for native OpenAI tool_calls
            if hasattr(delta, "tool_calls") and delta.tool_calls:
                has_native_tool_call = True
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in native_tool_calls:
                        native_tool_calls[idx] = {
                            "id": tc.id or f"call_{idx}",
                            "name": tc.function.name if tc.function and tc.function.name else "",
                            "arguments": tc.function.arguments if tc.function and tc.function.arguments else ""
                        }
                    else:
                        if tc.id:
                            native_tool_calls[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                native_tool_calls[idx]["name"] += tc.function.name
                            if tc.function.arguments:
                                native_tool_calls[idx]["arguments"] += tc.function.arguments
                continue

            # 2. Check content stream
            content = delta.content or ""
            if not content:
                continue

            # Check if entering inline tool call mode
            if not is_inline_tool_mode and self._is_tool_call_start(text_buffer + content):
                is_inline_tool_mode = True
                inline_tool_buffer = text_buffer + content
                text_buffer = ""  # Mute TTS buffer immediately!
                yield DualBrainEvent(
                    type=DualBrainEventType.TOOL_CALL_DETECTED,
                    metadata={"mode": "inline_json"}
                )
                continue

            if is_inline_tool_mode:
                inline_tool_buffer += content
                continue

            # Normal text generation -> segment sentences with pysbd
            text_buffer += content

            # Run pysbd segmentation on ongoing buffer
            segments = self.segmenter.segment(text_buffer)
            if len(segments) > 1:
                # All except last segment are complete sentences
                for sentence in segments[:-1]:
                    clean_sentence = sentence.strip()
                    if clean_sentence:
                        # Safety check: ensure no JSON leaked
                        if not self._is_tool_call_start(clean_sentence):
                            yield DualBrainEvent(
                                type=DualBrainEventType.SENTENCE,
                                text=clean_sentence
                            )
                text_buffer = segments[-1]

        # Handle Native OpenAI Tool Call Execution
        if has_native_tool_call and native_tool_calls:
            assistant_tool_calls = []
            tool_results = []

            for idx, tc_data in native_tool_calls.items():
                call_id = tc_data["id"]
                tool_name = tc_data["name"]
                args_str = tc_data["arguments"]

                try:
                    args = json.loads(args_str) if args_str else {}
                except Exception:
                    args = {}

                yield DualBrainEvent(
                    type=DualBrainEventType.TOOL_EXECUTION_START,
                    metadata={"tool_name": tool_name, "arguments": args, "id": call_id}
                )

                # Execute asynchronously
                tool_result = await self.execute_tool(tool_name, args)

                yield DualBrainEvent(
                    type=DualBrainEventType.TOOL_EXECUTION_RESULT,
                    metadata={"tool_name": tool_name, "result": tool_result, "id": call_id}
                )

                assistant_tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": args_str
                    }
                })
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(tool_result) if not isinstance(tool_result, str) else tool_result
                })

            # Append to messages and query LLM again for spoken voice response
            new_messages = list(messages)
            new_messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": assistant_tool_calls
            })
            new_messages.extend(tool_results)

            # Recurse to generate final spoken voice turn
            async for ev in self._stream_and_chunk(new_messages, depth=depth + 1):
                yield ev
            return

        # Handle Inline JSON / Tool Tag Execution
        if is_inline_tool_mode and inline_tool_buffer:
            parsed = self._extract_inline_tool_call(inline_tool_buffer)
            if parsed:
                tool_name, args = parsed
                yield DualBrainEvent(
                    type=DualBrainEventType.TOOL_EXECUTION_START,
                    metadata={"tool_name": tool_name, "arguments": args, "mode": "inline"}
                )

                tool_result = await self.execute_tool(tool_name, args)

                yield DualBrainEvent(
                    type=DualBrainEventType.TOOL_EXECUTION_RESULT,
                    metadata={"tool_name": tool_name, "result": tool_result, "mode": "inline"}
                )

                # Feed result back to LLM to speak the answer
                new_messages = list(messages)
                new_messages.append({
                    "role": "assistant",
                    "content": inline_tool_buffer
                })
                new_messages.append({
                    "role": "user",
                    "content": f"The tool '{tool_name}' returned: {json.dumps(tool_result)}. "
                               f"Please speak the natural answer concisely to the caller without mentioning raw code or JSON."
                })

                async for ev in self._stream_and_chunk(new_messages, depth=depth + 1):
                    yield ev
                return

        # Flush any remaining text in text_buffer
        final_sentence = text_buffer.strip()
        if final_sentence and not self._is_tool_call_start(final_sentence):
            yield DualBrainEvent(
                type=DualBrainEventType.SENTENCE,
                text=final_sentence
            )
