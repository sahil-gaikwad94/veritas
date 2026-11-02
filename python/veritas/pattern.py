"""Tier-0 regular-expression prefix recognition for string schemas.

JSON Schema's ``pattern`` keyword (and Pydantic's ``Field(pattern=...)``) constrains a
string field.  The decoder cannot wait for the whole string and then fail: it has to know,
*after every character*, whether some completion can still satisfy the pattern.  This
module builds a Thompson NFA from the pattern's parse tree and answers three questions in
time linear in the prefix length:

``satisfied(text)``
    Does ``text`` contain/equal a match (``re.search`` vs anchored ``fullmatch``
    semantics, exactly as JSON Schema defines them)?
``prefix_feasible(prefix)``
    Does at least one continuation of ``prefix`` satisfy the pattern?  This is the mask.
``allowed_next(prefix, alphabet)``
    Which single characters keep the prefix feasible.  This is the character-level mask.

Only a regular subset is supported.  Anything outside it raises
:class:`UnsupportedPattern` so callers can fall back to length bounds plus the concrete
verifier instead of silently weakening the mask.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

try:  # Python 3.11+: the parser lives in ``re._parser`` (``sre_parse`` is deprecated)
    from re import _parser as _sre_parse
except ImportError:  # pragma: no cover - older interpreters
    import sre_parse as _sre_parse  # type: ignore[no-redef]


class UnsupportedPattern(ValueError):
    """The pattern uses a construct this recognizer cannot mask exactly."""


# ---------------------------------------------------------------------------
# character predicates
# ---------------------------------------------------------------------------

_CATEGORIES: dict[Any, Callable[[str], bool]] = {
    _sre_parse.CATEGORY_DIGIT: lambda ch: ch.isdecimal(),
    _sre_parse.CATEGORY_NOT_DIGIT: lambda ch: not ch.isdecimal(),
    _sre_parse.CATEGORY_SPACE: lambda ch: ch.isspace(),
    _sre_parse.CATEGORY_NOT_SPACE: lambda ch: not ch.isspace(),
    _sre_parse.CATEGORY_WORD: lambda ch: ch.isalnum() or ch == "_",
    _sre_parse.CATEGORY_NOT_WORD: lambda ch: not (ch.isalnum() or ch == "_"),
}


@dataclass(frozen=True)
class _Class:
    """A character class as a list of (negated, members) where members are predicates."""

    predicates: tuple[Callable[[str], bool], ...] = ()
    negated: bool = False

    def __call__(self, ch: str) -> bool:
        hit = any(predicate(ch) for predicate in self.predicates)
        return not hit if self.negated else hit


def _predicate(ch: str) -> bool:  # pragma: no cover - trivial
    return True


class _Nfa:
    """Thompson NFA.  States are indices; edges are ``None`` (epsilon) or a predicate."""

    def __init__(self) -> None:
        self.edges: list[list[tuple[Callable[[str], bool] | None, int]]] = []
        self.accept = -1

    # -- construction ----------------------------------------------------
    def new_state(self) -> int:
        self.edges.append([])
        return len(self.edges) - 1

    def epsilon(self, src: int, dst: int) -> None:
        self.edges[src].append((None, dst))

    def transition(self, src: int, predicate: Callable[[str], bool], dst: int) -> None:
        self.edges[src].append((predicate, dst))

    # -- simulation ------------------------------------------------------
    def closure(self, states: Iterable[int]) -> frozenset[int]:
        seen = set(states)
        stack = list(seen)
        while stack:
            state = stack.pop()
            for predicate, target in self.edges[state]:
                if predicate is None and target not in seen:
                    seen.add(target)
                    stack.append(target)
        return frozenset(seen)


# ---------------------------------------------------------------------------
# parse-tree -> NFA
# ---------------------------------------------------------------------------

_MAX_REPEAT_EXPANSION = 4096


class _Builder:
    def __init__(self, nfa: _Nfa) -> None:
        self.nfa = nfa

    # returns (entry, exit) states
    def build(self, subpattern: Any) -> tuple[int, int]:
        return self._sequence(list(subpattern))

    def _sequence(self, items: Sequence[tuple[Any, Any]]) -> tuple[int, int]:
        entry = self.nfa.new_state()
        current = entry
        for op, av in items:
            start, end = self._item(op, av)
            self.nfa.epsilon(current, start)
            current = end
        return entry, current

    def _item(self, op: Any, av: Any) -> tuple[int, int]:
        nfa = self.nfa
        if op is _sre_parse.LITERAL:
            start, end = nfa.new_state(), nfa.new_state()
            nfa.transition(start, lambda ch, value=chr(av): ch == value, end)
            return start, end
        if op is _sre_parse.NOT_LITERAL:
            start, end = nfa.new_state(), nfa.new_state()
            nfa.transition(start, lambda ch, value=chr(av): ch != value, end)
            return start, end
        if op is _sre_parse.ANY:
            start, end = nfa.new_state(), nfa.new_state()
            # ``.`` does not match a newline unless DOTALL is set; flags are rejected
            # elsewhere, so this is the non-DOTALL reading.
            nfa.transition(start, lambda ch: ch != "\n", end)
            return start, end
        if op is _sre_parse.IN:
            klass = self._character_class(av)
            start, end = nfa.new_state(), nfa.new_state()
            nfa.transition(start, klass, end)
            return start, end
        if op is _sre_parse.SUBPATTERN:
            group, add_flags, del_flags, child = av
            if add_flags or del_flags:
                raise UnsupportedPattern("inline regex flags are not supported for masking")
            return self.build(child)
        if op in (_sre_parse.MAX_REPEAT, _sre_parse.MIN_REPEAT) or (
            hasattr(_sre_parse, "POSSESSIVE_REPEAT") and op is _sre_parse.POSSESSIVE_REPEAT
        ):
            low, high, child = av
            return self._repeat(int(low), high, child)
        if op is _sre_parse.BRANCH:
            _group, branches = av
            start, end = nfa.new_state(), nfa.new_state()
            for branch in branches:
                branch_start, branch_end = self._sequence(list(branch))
                nfa.epsilon(start, branch_start)
                nfa.epsilon(branch_end, end)
            return start, end
        if op is _sre_parse.AT:
            if av in (_sre_parse.AT_BEGINNING, _sre_parse.AT_BEGINNING_STRING,
                      _sre_parse.AT_END, _sre_parse.AT_END_STRING):
                start, end = nfa.new_state(), nfa.new_state()
                nfa.epsilon(start, end)
                return start, end
            raise UnsupportedPattern(f"anchor {av!r} is not supported for masking")
        if op is _sre_parse.ATOMIC_GROUP if hasattr(_sre_parse, "ATOMIC_GROUP") else False:
            return self.build(av)
        raise UnsupportedPattern(f"regex opcode {op!r} is not supported for masking")

    def _repeat(self, low: int, high: Any, child: Any) -> tuple[int, int]:
        nfa = self.nfa
        unbounded = high == _sre_parse.MAXREPEAT
        upper = low if unbounded else int(high)
        if upper < low:
            raise UnsupportedPattern("repeat upper bound is below its lower bound")
        if not unbounded and upper > _MAX_REPEAT_EXPANSION:
            raise UnsupportedPattern("repeat expansion is too large to mask exactly")
        start = nfa.new_state()
        current = start
        # Mandatory copies.
        for _ in range(low):
            body_start, body_end = self.build(child)
            nfa.epsilon(current, body_start)
            current = body_end
        tail = nfa.new_state()
        nfa.epsilon(current, tail)
        if unbounded:
            # One shared copy that loops onto itself: any number of extra repeats.
            body_start, body_end = self.build(child)
            nfa.epsilon(current, body_start)
            nfa.epsilon(body_end, current)
            return start, tail
        # Optional copies: copy ``i`` may be entered from the end of copy ``i-1`` and may
        # either continue to the next optional copy or skip straight to the tail.
        for _ in range(upper - low):
            body_start, body_end = self.build(child)
            nfa.epsilon(current, body_start)
            nfa.epsilon(body_end, tail)
            current = body_end
        return start, tail

    def _character_class(self, items: Any) -> _Class:
        predicates: list[Callable[[str], bool]] = []
        negated = False
        for op, av in items:
            if op is _sre_parse.NEGATE:
                negated = True
                continue
            if op is _sre_parse.LITERAL:
                predicates.append(lambda ch, value=chr(av): ch == value)
            elif op is _sre_parse.RANGE:
                low, high = av
                predicates.append(lambda ch, lo=low, hi=high: chr(lo) <= ch <= chr(hi))
            elif op is _sre_parse.CATEGORY:
                predicate = _CATEGORIES.get(av)
                if predicate is None:
                    raise UnsupportedPattern(f"character category {av!r} is not supported")
                predicates.append(predicate)
            else:
                raise UnsupportedPattern(f"character-class opcode {op!r} is not supported")
        return _Class(tuple(predicates), negated)


def _strip_anchors(subpattern: Any) -> tuple[Any, bool, bool]:
    """Return ``(items, start_anchored, end_anchored)`` for a parsed pattern."""

    items = list(subpattern)
    start_anchored = False
    end_anchored = False
    while items and items[0][0] is _sre_parse.AT and items[0][1] in (
        _sre_parse.AT_BEGINNING, _sre_parse.AT_BEGINNING_STRING
    ):
        start_anchored = True
        items.pop(0)
    while items and items[-1][0] is _sre_parse.AT and items[-1][1] in (
        _sre_parse.AT_END, _sre_parse.AT_END_STRING
    ):
        end_anchored = True
        items.pop()
    return items, start_anchored, end_anchored


# ---------------------------------------------------------------------------
# public recognizer
# ---------------------------------------------------------------------------

@dataclass
class PatternConstraint:
    """An incremental recognizer for one JSON-Schema ``pattern``."""

    pattern: str
    start_anchored: bool = False
    end_anchored: bool = False
    _nfa: _Nfa | None = field(default=None, repr=False)
    _initial: frozenset[int] = field(default=frozenset(), repr=False)
    _entry: int = field(default=0, repr=False)

    # -- construction ----------------------------------------------------
    @classmethod
    @lru_cache(maxsize=512)
    def compile(cls, pattern: str) -> PatternConstraint:
        if not isinstance(pattern, str) or not pattern:
            raise UnsupportedPattern("pattern must be a non-empty string")
        try:
            parsed = _sre_parse.parse(pattern)
        except Exception as exc:  # re.error and friends
            raise UnsupportedPattern(f"invalid regex {pattern!r}: {exc}") from exc
        if parsed.state.flags & 0x02 or parsed.state.flags & 0x40:  # IGNORECASE | LOCALE
            raise UnsupportedPattern("case-insensitive patterns are not supported for masking")
        items, start_anchored, end_anchored = _strip_anchors(parsed)
        nfa = _Nfa()
        builder = _Builder(nfa)
        entry, exit_state = builder._sequence(items)
        accept = nfa.new_state()
        nfa.epsilon(exit_state, accept)
        nfa.accept = accept
        return cls(
            pattern=pattern,
            start_anchored=start_anchored,
            end_anchored=end_anchored,
            _nfa=nfa,
            _initial=nfa.closure([entry]),
            _entry=entry,
        )

    # -- state -----------------------------------------------------------
    def _step(self, state: PatternState, ch: str) -> PatternState:
        nfa = self._nfa
        assert nfa is not None
        seeds: set[int] = set()
        for current in state.states:
            for predicate, target in nfa.edges[current]:
                if predicate is not None and predicate(ch):
                    seeds.add(target)
        if not self.start_anchored and not state.matched_any:
            # An unanchored pattern may begin a fresh attempt at this position.
            seeds |= set(self._initial)
        if not seeds:
            return PatternState(frozenset(), state.matched_any, state.length + 1)
        states = nfa.closure(seeds)
        matched_any = state.matched_any or (not self.end_anchored and nfa.accept in states)
        return PatternState(states, matched_any, state.length + 1)

    def initial_state(self) -> PatternState:
        nfa = self._nfa
        assert nfa is not None
        states = nfa.closure([self._entry])
        matched = (not self.end_anchored) and nfa.accept in states
        return PatternState(states, matched, 0)

    def run(self, text: str) -> PatternState:
        state = self.initial_state()
        for ch in text:
            state = self._step(state, ch)
            if not state.states and not state.matched_any:
                return state
        return state

    # -- queries ---------------------------------------------------------
    def satisfied(self, text: str) -> bool:
        return self.run(text).satisfied(self)

    def prefix_feasible(self, prefix: str) -> bool:
        return self.run(prefix).feasible(self)

    def allowed_next(self, prefix: str, alphabet: Sequence[str] | None = None) -> list[str]:
        state = self.run(prefix)
        if not state.feasible(self):
            return []
        if alphabet is None:
            alphabet = [chr(code) for code in range(0x20, 0x7F)]
        out = []
        for ch in alphabet:
            candidate = self._step(state, ch)
            if candidate.feasible(self):
                out.append(ch)
        return out

    def can_stop(self, prefix: str) -> bool:
        return self.run(prefix).satisfied(self)

    def witness(
        self,
        alphabet: Sequence[str] | None = None,
        *,
        min_length: int = 0,
        max_length: int = 64,
    ) -> str | None:
        """Shortest string accepted by the pattern inside an optional length window.

        Breadth-first over the recognizer's own state space, so the result is a genuine
        witness: it is re-checked with :meth:`satisfied` before being returned.  Used by
        pre-flight to build a concrete instance for schemas whose non-numeric fields are
        pattern-constrained strings.
        """

        chars = list(dict.fromkeys(alphabet or _DEFAULT_ALPHABET))
        start = self.initial_state()
        frontier: list[tuple[PatternState, str]] = [(start, "")]
        # Length is part of the key: a length window (``minLength``) needs the search to
        # keep extending even after the automaton has reached a saturated state.
        seen: set[tuple[frozenset[int], bool, int]] = {(start.states, start.matched_any, 0)}
        nodes = 0
        floor = max(0, int(min_length))
        ceiling = max(floor, int(max_length))
        while frontier:
            state, prefix = frontier.pop(0)
            if len(prefix) >= floor and state.satisfied(self):
                return prefix if self.satisfied(prefix) else None
            if len(prefix) >= ceiling:
                continue
            for ch in chars:
                nodes += 1
                if nodes > 200_000:
                    return None
                candidate = self._step(state, ch)
                if not candidate.states:
                    continue
                key = (candidate.states, candidate.matched_any, len(prefix) + 1)
                if key in seen:
                    continue
                seen.add(key)
                frontier.append((candidate, prefix + ch))
        return None


@dataclass(frozen=True)
class PatternState:
    states: frozenset[int]
    matched_any: bool
    length: int = 0

    def satisfied(self, constraint: PatternConstraint) -> bool:
        nfa = constraint._nfa
        assert nfa is not None
        if constraint.end_anchored:
            return nfa.accept in self.states
        return self.matched_any

    def feasible(self, constraint: PatternConstraint) -> bool:
        if self.satisfied(constraint):
            return True
        return bool(self.states)


# ---------------------------------------------------------------------------
# helpers used by the compiler / decoder
# ---------------------------------------------------------------------------

_DEFAULT_ALPHABET = (
    "0123456789"
    + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    + "abcdefghijklmnopqrstuvwxyz"
    + " -_.@/:+,()"
)


def compiles(pattern: str) -> bool:
    """True when the pattern is inside the maskable subset."""

    try:
        PatternConstraint.compile(pattern)
        return True
    except UnsupportedPattern:
        return False


def matches(pattern: str, text: str) -> bool:
    """Reference ``re``-semantics check (used for tests / non-masked paths)."""

    import re

    return re.search(pattern, text) is not None


__all__ = [
    "PatternConstraint",
    "PatternState",
    "UnsupportedPattern",
    "compiles",
    "matches",
]
