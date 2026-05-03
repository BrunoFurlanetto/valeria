import json
from datetime import datetime, timezone
from pathlib import Path


class EventLogger:
    def __init__(self, path=Path("tmp/valeria-logs/events.jsonl")):
        self.path = Path(path)

    def log(self, event):
        try:
            payload = self._sanitize_event(event)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as event_file:
                event_file.write(json.dumps(payload, ensure_ascii=True, sort_keys=True))
                event_file.write("\n")
        except Exception:
            return

    def _sanitize_event(self, event):
        safe = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": str(event.get("event", "assistant_event"))[:80],
        }
        for key in (
            "status",
            "model",
            "fallback",
            "error_type",
            "latency_ms",
            "input_length",
            "response_length",
            "history_size",
        ):
            if key in event:
                safe[key] = self._safe_value(event[key])
        return safe

    def _safe_value(self, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return round(value, 3)
        if value is None:
            return None
        return str(value)[:120]
