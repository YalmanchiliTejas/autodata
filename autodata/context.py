"""Model-budgeted context preparation for generic rollout harnesses.

The controller operates on semantic segments rather than opaque prompt strings.
That lets a caller protect policy/task/environment material while treating source
excerpts, old observations, and feedback as compressible context.
"""
from __future__ import annotations

from dataclasses import dataclass, field


class ContextOverflow(ValueError):
    """The protected task surface cannot fit the selected model deployment."""


@dataclass(slots=True)
class ContextPolicy:
    """Per-model context policy; the server maximum remains a hard ceiling."""

    context_window_tokens: int
    reserve_output_tokens: int = 2048
    soft_limit_ratio: float = 0.75
    chars_per_token: float = 2.0

    @property
    def input_budget(self) -> int:
        if self.context_window_tokens <= self.reserve_output_tokens:
            raise ValueError("context window must exceed the reserved output budget")
        return self.context_window_tokens - self.reserve_output_tokens


@dataclass(slots=True)
class ContextSegment:
    name: str
    content: str
    protected: bool = False
    priority: int = 100
    artifact_ref: str | None = None


@dataclass(slots=True)
class ContextPlan:
    text: str
    estimated_tokens: int
    compacted: list[str] = field(default_factory=list)
    artifact_refs: list[str] = field(default_factory=list)


class ContextController:
    """Preserve essential context and compact lower-priority segments first."""

    def __init__(self, policy: ContextPolicy):
        self.policy = policy

    def estimate_tokens(self, text: str) -> int:
        # Conservative fallback for arbitrary providers/tokenizers, especially
        # code and terminal output. A provider-specific tokenizer may replace it.
        return int((len(text) + self.policy.chars_per_token - 1) // self.policy.chars_per_token)

    @staticmethod
    def _render(segments: list[ContextSegment]) -> str:
        return "\n\n".join(f"## {segment.name}\n{segment.content}" for segment in segments if segment.content)

    @staticmethod
    def _excerpt(text: str, target_chars: int, *, artifact_ref: str | None) -> str:
        if len(text) <= target_chars:
            return text
        suffix = f" Full material: {artifact_ref}." if artifact_ref else ""
        marker_prefix = "[... "
        marker_suffix = f" characters omitted.{suffix} ...]"
        # The notice itself belongs to the requested bound; otherwise a long
        # artifact reference could unexpectedly overflow the context.
        available = target_chars - len(marker_prefix) - len(marker_suffix) - 8
        if available < 64:
            return ("[Context omitted." + suffix + "]")[:target_chars]
        head = max(24, int(available * 0.55))
        tail = max(24, available - head)
        ref = f" Full material: {artifact_ref}." if artifact_ref else ""
        return f"{text[:head]}\n\n[... {len(text) - head - tail} characters omitted.{ref} ...]\n\n{text[-tail:]}"

    def prepare(self, segments: list[ContextSegment]) -> ContextPlan:
        """Return a bounded prompt without dropping protected information."""
        working = [
            ContextSegment(
                name=segment.name,
                content=segment.content,
                protected=segment.protected,
                priority=segment.priority,
                artifact_ref=segment.artifact_ref,
            )
            for segment in segments
        ]
        rendered = self._render(working)
        soft_budget = int(self.policy.input_budget * self.policy.soft_limit_ratio)
        compacted: list[str] = []

        if self.estimate_tokens(rendered) > soft_budget:
            # Low priority represents older/less essential material. Equal
            # priority remains stable in caller order for reproducibility.
            for segment in sorted((item for item in working if not item.protected), key=lambda item: item.priority):
                if self.estimate_tokens(self._render(working)) <= soft_budget:
                    break
                target = max(256, len(segment.content) // 4)
                replacement = self._excerpt(segment.content, target, artifact_ref=segment.artifact_ref)
                if replacement != segment.content:
                    segment.content = replacement
                    compacted.append(segment.name)

        # A second, aggressive pass targets the hard input budget but still
        # preserves the existence and artifact pointer of every segment.
        for segment in sorted((item for item in working if not item.protected), key=lambda item: item.priority):
            if self.estimate_tokens(self._render(working)) <= self.policy.input_budget:
                break
            replacement = self._excerpt(segment.content, 160, artifact_ref=segment.artifact_ref)
            if replacement != segment.content:
                segment.content = replacement
                if segment.name not in compacted:
                    compacted.append(segment.name)

        rendered = self._render(working)
        estimated = self.estimate_tokens(rendered)
        if estimated > self.policy.input_budget:
            raise ContextOverflow(
                f"protected context exceeds input budget ({estimated} estimated tokens > {self.policy.input_budget}); "
                "use a long-context model or split the task"
            )
        return ContextPlan(rendered, estimated, compacted, [item.artifact_ref for item in working if item.artifact_ref])
