"""
Engine package for Ultra-Low-Latency Voice AI Pipeline.
"""

from .fast_classifier import (
    ClassificationResult,
    ConversationalStance,
    FastClassifier,
    STANCE_REGISTRY,
    StanceMetadata,
)
from .filler_manager import (
    FillerItem,
    FillerManager,
)
from .dual_brain import (
    DualBrainEngine,
    DualBrainEvent,
    DualBrainEventType,
    ToolDefinition,
)
from .echo_canceller import (
    EchoReferenceTap,
    InboundAudioProcessor,
    EchoCanceller,
    ProcessingMetrics,
)
from .crossfade_buffer import (
    CrossfadeBuffer,
    equal_power_crossfade,
    apply_crossfade,
    apply_fade_in,
    apply_fade_out,
    float32_to_int16,
    int16_to_float32,
    int16_to_mulaw,
    mulaw_to_int16,
    float32_to_mulaw,
    mulaw_to_float32,
    pcm_to_mulaw_bytes,
    mulaw_bytes_to_pcm,
    resample_audio,
    resample_8k_to_16k,
    resample_16k_to_8k,
    resample_16k_to_24k,
    resample_24k_to_16k,
    resample_24k_to_8k,
    resample_8k_to_24k,
)

__all__ = [
    "ConversationalStance",
    "StanceMetadata",
    "ClassificationResult",
    "FastClassifier",
    "STANCE_REGISTRY",
    "FillerItem",
    "FillerManager",
    "DualBrainEngine",
    "DualBrainEvent",
    "DualBrainEventType",
    "ToolDefinition",
    "EchoReferenceTap",
    "InboundAudioProcessor",
    "EchoCanceller",
    "ProcessingMetrics",
    "CrossfadeBuffer",
    "equal_power_crossfade",
    "apply_crossfade",
    "apply_fade_in",
    "apply_fade_out",
    "float32_to_int16",
    "int16_to_float32",
    "int16_to_mulaw",
    "mulaw_to_int16",
    "float32_to_mulaw",
    "mulaw_to_float32",
    "pcm_to_mulaw_bytes",
    "mulaw_bytes_to_pcm",
    "resample_audio",
    "resample_8k_to_16k",
    "resample_16k_to_8k",
    "resample_16k_to_24k",
    "resample_24k_to_16k",
    "resample_24k_to_8k",
    "resample_8k_to_24k",
]
