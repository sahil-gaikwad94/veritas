"""Phase 2: CRANE dual-mode controller."""

from __future__ import annotations


class CraneModeController:
    def __init__(self, think_open: str = "<think>", think_close: str = "</think>",
                 emit_open: str = "<emit>", emit_close: str = "</emit>"):
        self.think_open = think_open
        self.think_close = think_close
        self.emit_open = emit_open
        self.emit_close = emit_close
        self.mode = "reasoning"  # reasoning | emit
        self.buffer = ""
        self.mined_equations: list[str] = []
        self._cursor = 0

    def in_reasoning_scratchpad(self) -> bool:
        return self.mode == "reasoning"

    def feed(self, text: str) -> list[str]:
        self.buffer += text
        events: list[str] = []
        while True:
            candidates = []
            for tag, kind in ((self.think_open, "open"), (self.think_close, "close"),
                             (self.emit_open, "emit_open"), (self.emit_close, "emit_close")):
                idx = self.buffer.find(tag, self._cursor)
                if idx != -1:
                    candidates.append((idx, tag, kind))
            if not candidates:
                break
            idx, tag, kind = min(candidates, key=lambda x: x[0])
            if kind == "close" and self.mode == "reasoning":
                self.mode = "emit"
                events.append("switch_to_emit")
                for line in self.buffer.splitlines():
                    if "=" in line:
                        self.mined_equations.append(line.strip())
            elif kind == "open" and self.mode == "emit":
                self.mode = "reasoning"
                events.append("switch_to_reasoning")
            elif kind == "emit_open" and self.mode == "reasoning":
                self.mode = "emit"
                events.append("switch_to_emit")
            elif kind == "emit_close" and self.mode == "emit":
                self.mode = "reasoning"
                events.append("switch_to_reasoning")
            self._cursor = idx + len(tag)
        return events

    def passive_mine(self, text: str) -> list[str]:
        found = []
        for line in text.splitlines():
            s = line.strip()
            if "=" in s:
                found.append(s)
                self.mined_equations.append(s)
        return found

    def reset(self) -> None:
        self.mode = "reasoning"
        self.buffer = ""
        self.mined_equations = []
        self._cursor = 0


# aliases
CraneController = CraneModeController
CraneMode = CraneModeController
DualModeController = CraneModeController
