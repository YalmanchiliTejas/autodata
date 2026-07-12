"""Persistent evidence-card coverage graph for source-grounded generation."""
import hashlib
import json
from pathlib import Path

from . import agents


SCHEMA_VERSION = "autodata.coverage.v1"


def _sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CoverageGraph:
    def __init__(self, path):
        self.path = Path(path)
        self.data = {"schema_version": SCHEMA_VERSION, "documents": {}, "tag_usage": {}}
        if self.path.exists():
            loaded = json.loads(self.path.read_text())
            if loaded.get("schema_version") == SCHEMA_VERSION:
                self.data = loaded

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n")

    def get_or_create(self, cfg, doc_id, document, log=None):
        source_sha256 = _sha256(document)
        entry = self.data["documents"].get(source_sha256)
        if entry:
            if log:
                log.event("coverage_cache_hit", source_sha256=source_sha256)
            return source_sha256, entry

        mapped, raw = agents.extract_coverage(cfg, document)
        if log:
            log.log("coverage_mapper", 0, document, raw,
                    {"model": cfg.extractor, "system_prompt": agents.COVERAGE_EXTRACTOR_SYS})
        cards = []
        for index, card in enumerate(mapped["cards"]):
            if not isinstance(card, dict):
                continue
            facts = [str(fact).strip() for fact in card.get("facts", []) if str(fact).strip()]
            if not facts:
                continue
            tags = [str(tag).strip() for tag in card.get("tags", []) if str(tag).strip()]
            locations = []
            start_hint = card.get("_source_offset", 0)
            start_hint = start_hint if isinstance(start_hint, int) and start_hint >= 0 else 0
            for quote in card.get("evidence_quotes", []):
                quote = str(quote).strip()
                start = document.find(quote, start_hint)
                if quote and start >= 0:
                    locations.append({"start_char": start, "end_char": start + len(quote)})
            if not locations:
                continue
            cards.append({"id": f"card_{index:03d}",
                          "title": str(card.get("title", "source concept")).strip(),
                          "tags": tags or ["source_grounding"], "facts": facts,
                          "source_locations": locations,
                          "attempts": 0, "accepted": 0})
        if not cards:
            cards = [{"id": "card_000", "title": "source extract", "tags": ["source_grounding"],
                      "facts": [str(mapped.get("summary", ""))],
                      "source_locations": [{"start_char": 0, "end_char": len(document)}],
                      "attempts": 0, "accepted": 0}]
        entry = {"document_id": doc_id, "summary": str(mapped.get("summary", "")).strip(), "cards": cards}
        self.data["documents"][source_sha256] = entry
        self._save()
        return source_sha256, entry

    def _ranked_cards(self, source_sha256):
        entry = self.data["documents"][source_sha256]

        def score(card):
            tag_coverage = sum(self.data["tag_usage"].get(tag, 0) for tag in card["tags"])
            return (card["attempts"], card["accepted"], tag_coverage, card["id"])

        return sorted(entry["cards"], key=score)

    def preview(self, source_sha256, max_cards):
        """Return the next cards without mutating usage counters."""
        return self._ranked_cards(source_sha256)[:max_cards]

    def select(self, source_sha256, max_cards):
        """Reserve cards for a future integration and record their use."""
        selected = self.preview(source_sha256, max_cards)
        for card in selected:
            card["attempts"] += 1
            for tag in card["tags"]:
                self.data["tag_usage"][tag] = self.data["tag_usage"].get(tag, 0) + 1
        self._save()
        return selected

    def accept(self, source_sha256, card_ids):
        cards = {card["id"]: card for card in self.data["documents"][source_sha256]["cards"]}
        for card_id in card_ids:
            if card_id in cards:
                cards[card_id]["accepted"] += 1
        self._save()

    def validate(self, source_sha256, document):
        """Return structural and provenance errors for one cached source map."""
        entry = self.data["documents"].get(source_sha256)
        if not entry:
            return ["source is not in the coverage graph"]
        errors, ids = [], set()
        for card in entry["cards"]:
            card_id = card.get("id")
            if not isinstance(card_id, str) or card_id in ids:
                errors.append(f"duplicate or invalid card id: {card_id}")
            ids.add(card_id)
            if not card.get("title") or not card.get("facts") or not card.get("tags"):
                errors.append(f"{card_id}: title, facts, and tags are required")
            locations = card.get("source_locations")
            if not isinstance(locations, list) or not locations:
                errors.append(f"{card_id}: source_locations are required")
                continue
            for location in locations:
                start, end = location.get("start_char"), location.get("end_char")
                if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(document):
                    errors.append(f"{card_id}: invalid source location")
        return errors

    def report(self):
        """Produce a concise, human-readable cache and coverage summary."""
        lines = [f"Coverage graph: {len(self.data['documents'])} document(s)"]
        for source_sha256, entry in sorted(self.data["documents"].items()):
            lines.append(f"{entry['document_id']} {source_sha256[:12]}: {len(entry['cards'])} cards")
            for card in entry["cards"]:
                lines.append(f"  {card['id']} attempts={card['attempts']} accepted={card['accepted']} "
                             f"tags={','.join(card['tags'])} title={card['title']}")
            next_ids = ", ".join(card["id"] for card in self.preview(source_sha256, 4))
            lines.append(f"  next_cards={next_ids}")
        if self.data["tag_usage"]:
            lines.append("Tag usage: " + ", ".join(
                f"{tag}={count}" for tag, count in sorted(self.data["tag_usage"].items())))
        return "\n".join(lines) + "\n"

    @staticmethod
    def render(entry, cards):
        sections = [f"SOURCE SUMMARY:\n{entry['summary']}", "SELECTED EVIDENCE CARDS:"]
        for card in cards:
            sections.append(f"[{card['id']}] {card['title']}\nTAGS: {', '.join(card['tags'])}\nFACTS:\n- "
                            + "\n- ".join(card["facts"]))
        return "\n\n".join(sections)
