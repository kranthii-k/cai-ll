"""
Fast Classifier: Ultra-Low-Latency Stance & Intent Classifier (<10ms).
Specialized for streaming telephony voice agents to trigger contextual acoustic
fillers before heavy LLM reasoning completes.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


class ConversationalStance(str, Enum):
    ANALYTICAL = "analytical"
    AFFIRMATIVE = "affirmative"
    EMPATHETIC = "empathetic"
    CLARIFYING = "clarifying"
    DIRECT = "direct"


@dataclass(frozen=True)
class StanceMetadata:
    stance: ConversationalStance
    recommended_filler_key: str
    expected_delay_ms: int
    latency_masking_required: bool
    description: str


STANCE_REGISTRY: Dict[ConversationalStance, StanceMetadata] = {
    ConversationalStance.ANALYTICAL: StanceMetadata(
        stance=ConversationalStance.ANALYTICAL,
        recommended_filler_key="analytical",
        expected_delay_ms=600,
        latency_masking_required=True,
        description="Lookup, status query, calculation, or knowledge retrieval requiring backend latency masking."
    ),
    ConversationalStance.AFFIRMATIVE: StanceMetadata(
        stance=ConversationalStance.AFFIRMATIVE,
        recommended_filler_key="affirmative",
        expected_delay_ms=250,
        latency_masking_required=True,
        description="Acknowledgement, confirmation, or closure acknowledging user input with minimal delay."
    ),
    ConversationalStance.EMPATHETIC: StanceMetadata(
        stance=ConversationalStance.EMPATHETIC,
        recommended_filler_key="empathetic",
        expected_delay_ms=400,
        latency_masking_required=True,
        description="Caller distress, complaint, cancellation, or emergency requiring empathetic de-escalation."
    ),
    ConversationalStance.CLARIFYING: StanceMetadata(
        stance=ConversationalStance.CLARIFYING,
        recommended_filler_key="clarifying",
        expected_delay_ms=450,
        latency_masking_required=True,
        description="Query clarification, repetition request, or ambiguity resolution."
    ),
    ConversationalStance.DIRECT: StanceMetadata(
        stance=ConversationalStance.DIRECT,
        recommended_filler_key="direct",
        expected_delay_ms=300,
        latency_masking_required=True,
        description="Greetings, one/two-word utterances, or immediate answers requiring zero-lag response."
    ),
}


@dataclass
class ClassificationResult:
    stance: ConversationalStance
    confidence: float
    recommended_filler_key: str
    expected_delay_ms: int
    latency_masking_required: bool
    matched_keywords: List[str] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "stance": self.stance.value,
            "confidence": round(self.confidence, 4),
            "recommended_filler_key": self.recommended_filler_key,
            "expected_delay_ms": self.expected_delay_ms,
            "latency_masking_required": self.latency_masking_required,
            "matched_keywords": self.matched_keywords,
            "latency_ms": round(self.latency_ms, 4)
        }


class FastClassifier:
    """
    Sub-10ms heuristic, regex, and keyword stance classifier.
    Designed for zero-lag telephony pipelines.
    """

    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self):
        # 1. Empathetic (complaint, distress, damage, cancel, refund, emergency)
        empathetic_phrases = [
            r"\b(broken|broke|damag(?:e|ed)|defect(?:ive)?)\b",
            r"\b(cancel(?:ed|led|ling|lation)?)\b",
            r"\b(sad|angry|mad|furious|upset|frustrat(?:ed|ing)|annoy(?:ed|ing))\b",
            r"\b(bad|awful|terrible|horrible|dreadful|disaster|nightmare)\b",
            r"\b(problem|problems|issue|issues|trouble|complaint|fault)\b",
            r"\b(hurt|pain|sick|injur(?:ed|y)|bleed(?:ing)?)\b",
            r"\b(refund|money\s+back|chargeback|overcharg(?:ed|e))\b",
            r"\b(emergency|urgent|urgently|911|danger|danger(?:ous)?)\b",
            r"\b(lost|stolen|missing)\b",
            r"\b(not\s+working|doesn't\s+work|won't\s+work|can't\s+use)\b",
            r"\b(burst\s+pipe|pipe\s+burst|leak|water\s+damage|car\s+accident)\b",
        ]
        self._empathetic_regex = re.compile("|".join(empathetic_phrases), re.IGNORECASE)

        # 2. Clarifying (requests for explanation, repetition, confusion)
        clarifying_phrases = [
            r"\bwhy\b(?:\s+(?:did|is|are|was|were|would|should|do|does|can't))?",
            r"\bwhat\s+do\s+you\s+mean\b",
            r"\b(?:could\s+you\s+)?repeat(?:\s+(?:that|what\s+you\s+said))?\b",
            r"\bexplain(?:\s+(?:that|this|further|again))?\b",
            r"\bhuh\b",
            r"\bpardon(?:\s+me)?\b",
            r"\bwhat\s+did\s+you\s+say\b",
            r"\bsay\s+that\s+again\b",
            r"\bcome\s+again\b",
            r"\bi\s+don't\s+understand\b",
            r"\b(?:could\s+you\s+)?clarify\b",
            r"\bdidn't\s+catch\s+that\b",
            r"\bwhat\s+was\s+that\b",
            r"\bwhat\s+are\s+you\s+talking\s+about\b",
        ]
        self._clarifying_regex = re.compile("|".join(clarifying_phrases), re.IGNORECASE)

        # 3. Analytical (search, check, lookup, query, pricing, status, etc.)
        analytical_phrases = [
            r"\b(?:search|lookup|look\s+up|query|find)\b",
            r"\bcheck(?:\s+(?:if|on|out|status|balance|my))?\b",
            r"\bstatus\b",
            r"\bpric(?:ing|e|es)\b",
            r"\bcost(?:s)?\b",
            r"\brate(?:s)?\b",
            r"\bfee(?:s)?\b",
            r"\bhour(?:s)?\b",
            r"\bappointment(?:s)?\b",
            r"\bschedul(?:e|ing)\b",
            r"\bwhen\b",
            r"\bwhere\b",
            r"\bhow\s+much\b",
            r"\bhow\s+many\b",
            r"\bwhat\s+time\b",
            r"\bwhat\s+date\b",
            r"\border(?:s)?\b",
            r"\baccount(?:s)?\b",
            r"\bcan\s+you\s+find\b",
            r"\bcheck\s+if\b",
            r"\bavailab(?:le|ility)\b",
            r"\bbalance\b",
            r"\binventory\b",
            r"\btrack(?:ing)?\b",
            r"\bbill(?:ing|s)?\b",
            r"\binvoice(?:s)?\b",
        ]
        self._analytical_regex = re.compile("|".join(analytical_phrases), re.IGNORECASE)

        # 4. Affirmative (agreements, acknowledgements, confirmations)
        affirmative_phrases = [
            r"\b(?:yes|yeah|yep|yup|aye)\b",
            r"\bcorrect\b",
            r"\bexactly\b",
            r"\bsounds\s+good\b",
            r"\bsounds\s+great\b",
            r"\bplease(?:\s+confirm)?\b",
            r"\bconfirm(?:\s+it)?\b",
            r"\bdo\s+it\b",
            r"\bgo\s+ahead\b",
            r"\bthat\s+works\b",
            r"\ball\s+right\b",
            r"\balright\b",
            r"\bokay\b",
            r"\bok\b",
            r"\bsure(?:\s+thing)?\b",
            r"\bthanks\b",
            r"\bthank\s+you(?:\s+so\s+much)?\b",
            r"\bperfect\b",
            r"\bgreat\b",
            r"\bdefinitely\b",
            r"\babsolutely\b",
            r"\baffirmative\b",
            r"\bagreed\b",
        ]
        self._affirmative_regex = re.compile("|".join(affirmative_phrases), re.IGNORECASE)

        # 5. Direct (greetings, short contact prompts, telephony pause/hold courtesies)
        direct_phrases = [
            r"^(?:hi|hello|hey|howdy|yo|sup|greetings)(?:\s+(?:there|agent|bot|sir|ma'am))?[\.!\?]*$",
            r"^(?:good\s+(?:morning|afternoon|evening|day))[\.!\?]*$",
            r"^(?:who\s+(?:is\s+this|are\s+you))[\.!\?]*$",
            r"^(?:please\s+)?(?:stop(?:\s+it)?(?:\s+right\s+now)?|wait(?:\s+(?:a\s+sec(?:ond)?|a\s+minute|up))?|hold\s+on(?:\s+(?:a\s+sec(?:ond)?|a\s+minute))?|pause|hang\s+on|one\s+sec(?:ond)?|just\s+a\s+(?:sec(?:ond)?|minute|moment))(?:\s+please)?[\.!\?]*$",
        ]
        self._direct_regex = re.compile("|".join(direct_phrases), re.IGNORECASE)

        # Common word tokenizer regex
        self._word_tokenizer = re.compile(r"\b[\w']+\b")

    def classify(self, text: str) -> ClassificationResult:
        """
        Classify conversational stance of input utterance in sub-10ms.
        """
        start_ns = time.perf_counter_ns()

        if not text or not text.strip():
            metadata = STANCE_REGISTRY[ConversationalStance.DIRECT]
            latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
            return ClassificationResult(
                stance=ConversationalStance.DIRECT,
                confidence=1.0,
                recommended_filler_key=metadata.recommended_filler_key,
                expected_delay_ms=metadata.expected_delay_ms,
                latency_masking_required=metadata.latency_masking_required,
                matched_keywords=[],
                latency_ms=latency_ms
            )

        clean_text = text.strip()
        words = self._word_tokenizer.findall(clean_text)
        word_count = len(words)

        # 1. Direct Greetings Check (e.g. "hi", "hello there", "good morning")
        if self._direct_regex.match(clean_text):
            metadata = STANCE_REGISTRY[ConversationalStance.DIRECT]
            latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
            return ClassificationResult(
                stance=ConversationalStance.DIRECT,
                confidence=0.99,
                recommended_filler_key=metadata.recommended_filler_key,
                expected_delay_ms=metadata.expected_delay_ms,
                latency_masking_required=metadata.latency_masking_required,
                matched_keywords=[clean_text],
                latency_ms=latency_ms
            )

        # 2. Extract matches across categories
        empathetic_matches = [m.group(0) for m in self._empathetic_regex.finditer(clean_text)]
        clarifying_matches = [m.group(0) for m in self._clarifying_regex.finditer(clean_text)]
        analytical_matches = [m.group(0) for m in self._analytical_regex.finditer(clean_text)]
        affirmative_matches = [m.group(0) for m in self._affirmative_regex.finditer(clean_text)]

        # Scoring & Precedence Logic
        # Empathetic signals take top priority for customer sentiment & de-escalation
        if empathetic_matches:
            # If strongly empathetic (e.g. broken, cancel, angry, refund, emergency)
            metadata = STANCE_REGISTRY[ConversationalStance.EMPATHETIC]
            latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
            confidence = min(0.99, 0.75 + len(empathetic_matches) * 0.1)
            return ClassificationResult(
                stance=ConversationalStance.EMPATHETIC,
                confidence=confidence,
                recommended_filler_key=metadata.recommended_filler_key,
                expected_delay_ms=metadata.expected_delay_ms,
                latency_masking_required=metadata.latency_masking_required,
                matched_keywords=empathetic_matches,
                latency_ms=latency_ms
            )

        # Clarifying signals (e.g., "why", "repeat that", "what do you mean")
        if clarifying_matches:
            metadata = STANCE_REGISTRY[ConversationalStance.CLARIFYING]
            latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
            confidence = min(0.98, 0.75 + len(clarifying_matches) * 0.1)
            return ClassificationResult(
                stance=ConversationalStance.CLARIFYING,
                confidence=confidence,
                recommended_filler_key=metadata.recommended_filler_key,
                expected_delay_ms=metadata.expected_delay_ms,
                latency_masking_required=metadata.latency_masking_required,
                matched_keywords=clarifying_matches,
                latency_ms=latency_ms
            )

        # Analytical signals (e.g. "search", "check", "pricing", "status", "hours")
        # Notice: If sentence has both analytical and affirmative (e.g. "yes, check my status")
        # the operational intent is analytical (requires DB/tool lookup)
        if analytical_matches:
            metadata = STANCE_REGISTRY[ConversationalStance.ANALYTICAL]
            latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
            confidence = min(0.99, 0.70 + len(analytical_matches) * 0.1)
            return ClassificationResult(
                stance=ConversationalStance.ANALYTICAL,
                confidence=confidence,
                recommended_filler_key=metadata.recommended_filler_key,
                expected_delay_ms=metadata.expected_delay_ms,
                latency_masking_required=metadata.latency_masking_required,
                matched_keywords=analytical_matches,
                latency_ms=latency_ms
            )

        # Affirmative signals (e.g. "yes", "correct", "sounds good", "please confirm")
        if affirmative_matches:
            metadata = STANCE_REGISTRY[ConversationalStance.AFFIRMATIVE]
            latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
            confidence = min(0.98, 0.80 + len(affirmative_matches) * 0.08)
            return ClassificationResult(
                stance=ConversationalStance.AFFIRMATIVE,
                confidence=confidence,
                recommended_filler_key=metadata.recommended_filler_key,
                expected_delay_ms=metadata.expected_delay_ms,
                latency_masking_required=metadata.latency_masking_required,
                matched_keywords=affirmative_matches,
                latency_ms=latency_ms
            )

        # Utterance < 2 words that need instant direct reply
        if word_count < 2:
            metadata = STANCE_REGISTRY[ConversationalStance.DIRECT]
            latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
            return ClassificationResult(
                stance=ConversationalStance.DIRECT,
                confidence=0.85,
                recommended_filler_key=metadata.recommended_filler_key,
                expected_delay_ms=metadata.expected_delay_ms,
                latency_masking_required=metadata.latency_masking_required,
                matched_keywords=words,
                latency_ms=latency_ms
            )

        # Utterance with 2 words (e.g. greeting or short phrase)
        if word_count == 2 and words[0].lower() in {"hi", "hello", "hey", "yo", "sup"}:
            metadata = STANCE_REGISTRY[ConversationalStance.DIRECT]
            latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
            return ClassificationResult(
                stance=ConversationalStance.DIRECT,
                confidence=0.90,
                recommended_filler_key=metadata.recommended_filler_key,
                expected_delay_ms=metadata.expected_delay_ms,
                latency_masking_required=metadata.latency_masking_required,
                matched_keywords=words,
                latency_ms=latency_ms
            )

        # Default fallback for informational questions or statements
        if "?" in clean_text or clean_text.lower().startswith(("can", "could", "would", "is", "are", "do", "does")):
            # Question likely requiring lookup
            metadata = STANCE_REGISTRY[ConversationalStance.ANALYTICAL]
            latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
            return ClassificationResult(
                stance=ConversationalStance.ANALYTICAL,
                confidence=0.60,
                recommended_filler_key=metadata.recommended_filler_key,
                expected_delay_ms=metadata.expected_delay_ms,
                latency_masking_required=metadata.latency_masking_required,
                matched_keywords=["question_fallback"],
                latency_ms=latency_ms
            )

        # General direct conversational response
        metadata = STANCE_REGISTRY[ConversationalStance.DIRECT]
        latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
        return ClassificationResult(
            stance=ConversationalStance.DIRECT,
            confidence=0.50,
            recommended_filler_key=metadata.recommended_filler_key,
            expected_delay_ms=metadata.expected_delay_ms,
            latency_masking_required=metadata.latency_masking_required,
            matched_keywords=[],
            latency_ms=latency_ms
        )
