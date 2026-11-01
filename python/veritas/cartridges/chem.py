"""``veritas.chem`` — SMILES valence, charge and ring-closure verification.

A small but real SMILES reader: organic-subset atoms, bracket atoms (isotopes, explicit H,
formal charge), branches, ring closures, bond orders and disconnected fragments.  Valence
limits are checked with charge-adjusted allowed valences; implicit hydrogens are inferred
for the organic subset so molecular weights are meaningful.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

VALENCE: Dict[str, Tuple[int, ...]] = {
    "H": (1,),
    "B": (3,),
    "C": (4,),
    "N": (3, 5),
    "O": (2,),
    "F": (1,),
    "Si": (4,),
    "P": (3, 5),
    "S": (2, 4, 6),
    "Cl": (1,),
    "Br": (1,),
    "I": (1,),
    "Se": (2, 4, 6),
}
AROMATIC_VALENCE: Dict[str, Tuple[int, ...]] = {
    "c": (3,),
    "n": (3,),
    "o": (2,),
    "p": (3,),
    "s": (2,),
    "b": (3,),
}
ATOMIC_WEIGHT = {
    "H": 1.008, "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998,
    "Na": 22.990, "Mg": 24.305, "Si": 28.085, "P": 30.974, "S": 32.06, "Cl": 35.45,
    "K": 39.098, "Ca": 40.078, "Fe": 55.845, "Br": 79.904, "I": 126.904, "Se": 78.971,
}

_BRACKET_RE = re.compile(
    r"^\[(?:\d+)?([A-Z][a-z]?|[a-z])(?:@+)?(H(\d*))?([+-]\d*)?(?::\d+)?\]$"
)


class SmilesError(ValueError):
    pass


@dataclass
class Atom:
    index: int
    symbol: str
    aromatic: bool = False
    charge: int = 0
    explicit_h: int = 0
    implicit_h: int = 0
    bond_sum: float = 0.0
    bracket: bool = False

    @property
    def element(self) -> str:
        return self.symbol.capitalize()


@dataclass
class SmilesAnalysis:
    atoms: List[Atom] = field(default_factory=list)
    bonds: List[Tuple[int, int, float]] = field(default_factory=list)
    formal_charge: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    formula: Dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def molecular_weight(self) -> Optional[float]:
        if self.errors:
            return None
        return sum(ATOMIC_WEIGHT.get(el, 0.0) * count for el, count in self.formula.items())

    def as_dict(self) -> Dict[str, object]:
        return {
            "ok": self.ok,
            "atoms": len(self.atoms),
            "bonds": len(self.bonds),
            "formal_charge": self.formal_charge,
            "formula": dict(self.formula),
            "molecular_weight": self.molecular_weight(),
            "errors": list(self.errors),
        }


def _tokenize(smiles: str) -> List[str]:
    tokens: List[str] = []
    i = 0
    while i < len(smiles):
        ch = smiles[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "[":
            end = smiles.find("]", i)
            if end < 0:
                raise SmilesError("unclosed bracket atom")
            tokens.append(smiles[i:end + 1])
            i = end + 1
            continue
        if ch == "%":
            if i + 2 >= len(smiles) or not smiles[i + 1:i + 3].isdigit():
                raise SmilesError("invalid %nn ring closure")
            tokens.append(smiles[i:i + 3])
            i += 3
            continue
        if ch.isdigit():
            tokens.append(ch)
            i += 1
            continue
        if smiles.startswith(("Cl", "Br"), i):
            tokens.append(smiles[i:i + 2])
            i += 2
            continue
        if ch in "BCNOPSFI":
            tokens.append(ch)
            i += 1
            continue
        if ch in "bcnops":
            tokens.append(ch)
            i += 1
            continue
        if ch in "-=#:/\\.()":
            tokens.append(ch)
            i += 1
            continue
        raise SmilesError(f"unexpected character {ch!r}")
    return tokens


def _bracket_atom(index: int, text: str) -> Atom:
    match = _BRACKET_RE.match(text)
    if not match:
        raise SmilesError(f"invalid bracket atom {text!r}")
    symbol = match.group(1)
    aromatic = symbol.islower()
    h_group = match.group(2)
    if h_group is None:
        explicit_h = 0
    else:
        digits = match.group(3)
        explicit_h = int(digits) if digits else 1
    charge_text = match.group(4) or ""
    charge = 0
    if charge_text:
        sign = 1 if charge_text[0] == "+" else -1
        amount = charge_text[1:]
        charge = sign * (int(amount) if amount else 1)
    return Atom(
        index=index,
        symbol=symbol,
        aromatic=aromatic,
        charge=charge,
        explicit_h=explicit_h,
        bracket=True,
    )


def analyze_smiles(smiles: str) -> SmilesAnalysis:
    analysis = SmilesAnalysis()
    if not isinstance(smiles, str) or not smiles.strip():
        analysis.errors.append("empty SMILES")
        return analysis
    try:
        tokens = _tokenize(smiles)
    except SmilesError as exc:
        analysis.errors.append(str(exc))
        return analysis

    atoms: List[Atom] = analysis.atoms
    current: Optional[int] = None
    pending_bond: Optional[float] = None
    branch_stack: List[Optional[int]] = []
    rings: Dict[str, Tuple[int, float]] = {}

    for token in tokens:
        if token == "(":
            branch_stack.append(current)
            continue
        if token == ")":
            if not branch_stack:
                analysis.errors.append("unmatched ')'")
                continue
            current = branch_stack.pop()
            continue
        if token == ".":
            current = None
            pending_bond = None
            continue
        if token in "-=#:/\\":
            pending_bond = {"-": 1.0, "=": 2.0, "#": 3.0, ":": 1.0, "/": 1.0, "\\": 1.0}[token]
            continue
        if token.isdigit() or (token.startswith("%") and token[1:].isdigit()):
            if current is None:
                analysis.errors.append("ring closure without an atom")
                continue
            number = token.lstrip("%")
            order = pending_bond if pending_bond is not None else 1.0
            pending_bond = None
            if number in rings:
                other, other_order = rings.pop(number)
                order = order if order != 1.0 else other_order
                _add_bond(analysis, other, current, order)
            else:
                rings[number] = (current, order)
            continue

        # atom
        if token.startswith("["):
            try:
                atom = _bracket_atom(len(atoms), token)
            except SmilesError as exc:
                analysis.errors.append(str(exc))
                continue
        else:
            atom = Atom(index=len(atoms), symbol=token, aromatic=token.islower())
        atoms.append(atom)
        if current is not None:
            order = pending_bond if pending_bond is not None else 1.0
            _add_bond(analysis, current, atom.index, order)
        pending_bond = None
        current = atom.index

    if rings:
        analysis.errors.append(f"unclosed ring closure(s): {sorted(rings)}")
    if branch_stack:
        analysis.errors.append("unmatched '('")
    if not atoms:
        analysis.errors.append("SMILES contains no atoms")

    for atom in atoms:
        _finish_atom(analysis, atom)
    analysis.formal_charge = sum(atom.charge for atom in atoms)
    _formula(analysis)
    return analysis


def _add_bond(analysis: SmilesAnalysis, i: int, j: int, order: float) -> None:
    if i == j:
        analysis.errors.append("self bond")
        return
    analysis.bonds.append((i, j, float(order)))
    if 0 <= i < len(analysis.atoms):
        analysis.atoms[i].bond_sum += float(order)
    if 0 <= j < len(analysis.atoms):
        analysis.atoms[j].bond_sum += float(order)


def _allowed(atom: Atom) -> Tuple[int, ...]:
    symbol = atom.symbol
    if atom.aromatic:
        base = AROMATIC_VALENCE.get(symbol)
    else:
        base = VALENCE.get(symbol)
    if base is None:
        return ()
    return tuple(max(0, value + atom.charge) for value in base)


def _finish_atom(analysis: SmilesAnalysis, atom: Atom) -> None:
    allowed = _allowed(atom)
    if not allowed:
        if atom.bracket:
            analysis.warnings.append(f"unknown element {atom.symbol!r}; valence unchecked")
        return
    if atom.bracket:
        total = atom.bond_sum + atom.explicit_h
        # Bracket atoms are allowed to be under-valent (radicals, carbenes, ions); they
        # may never exceed the charge-adjusted maximum.
        if total > max(allowed):
            analysis.errors.append(
                f"atom {atom.index} ({atom.symbol}{atom.charge:+d}) has valence {total:g}; allowed {allowed}"
            )
        atom.implicit_h = 0
        return
    if atom.aromatic and atom.symbol in {"n", "o", "p", "s"}:
        atom.implicit_h = 0
        total = atom.bond_sum
    else:
        target = next((value for value in sorted(allowed) if value >= atom.bond_sum), None)
        if target is None:
            analysis.errors.append(
                f"atom {atom.index} ({atom.symbol}) bond order {atom.bond_sum:g} exceeds allowed valence {allowed}"
            )
            atom.implicit_h = 0
            return
        atom.implicit_h = int(target - atom.bond_sum)
        total = atom.bond_sum + atom.implicit_h
    if total not in allowed and not (atom.aromatic and total <= max(allowed)):
        analysis.errors.append(
            f"atom {atom.index} ({atom.symbol}) valence {total:g} not in {allowed}"
        )


def _formula(analysis: SmilesAnalysis) -> None:
    formula: Dict[str, int] = {}
    for atom in analysis.atoms:
        element = atom.element
        formula[element] = formula.get(element, 0) + 1
        hydrogens = atom.explicit_h + atom.implicit_h
        if hydrogens:
            formula["H"] = formula.get("H", 0) + hydrogens
    analysis.formula = dict(sorted(formula.items()))


# ---------------------------------------------------------------------------
# public helpers
# ---------------------------------------------------------------------------

def valence_ok(symbol: str, bonds: float, charge: int = 0) -> bool:
    base = VALENCE.get(symbol)
    if base is None:
        return True
    return bonds in {max(0, value + charge) for value in base}


def smiles_valence_ok(smiles: str) -> bool:
    return analyze_smiles(smiles).ok


def formal_charge_neutral(charges: Sequence[int]) -> bool:
    return sum(int(charge) for charge in charges) == 0


def molecular_weight(smiles: str) -> Optional[float]:
    return analyze_smiles(smiles).molecular_weight()


__all__ = [
    "ATOMIC_WEIGHT",
    "AROMATIC_VALENCE",
    "Atom",
    "SmilesAnalysis",
    "SmilesError",
    "VALENCE",
    "analyze_smiles",
    "formal_charge_neutral",
    "molecular_weight",
    "smiles_valence_ok",
    "valence_ok",
]
