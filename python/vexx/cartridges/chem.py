"""``vexx.chem`` — SMILES valence, charge and ring-closure verification.

A small but real SMILES reader: organic-subset atoms, bracket atoms (isotopes, explicit H,
formal charge), branches, ring closures, bond orders and disconnected fragments.  Valence
limits are checked with charge-adjusted allowed valences; implicit hydrogens are inferred
for the organic subset so molecular weights are meaningful.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

VALENCE: dict[str, tuple[int, ...]] = {
    "H": (1,),
    "He": (0,),
    "Li": (1,),
    "Be": (2,),
    "B": (3,),
    "C": (4,),
    "N": (3, 5),
    "O": (2,),
    "F": (1,),
    "Ne": (0,),
    "Na": (1,),
    "Mg": (2,),
    "Al": (3,),
    "Si": (4,),
    "P": (3, 5),
    "S": (2, 4, 6),
    "Cl": (1,),
    "Ar": (0,),
    "K": (1,),
    "Ca": (2,),
    "Sc": (3,),
    "Ti": (2, 3, 4),
    "V": (2, 3, 4, 5),
    "Cr": (2, 3, 6),
    "Mn": (2, 3, 4, 7),
    "Fe": (2, 3, 6),
    "Co": (2, 3),
    "Ni": (2, 3),
    "Cu": (1, 2),
    "Zn": (2,),
    "Ga": (3,),
    "Ge": (4,),
    "As": (3, 5),
    "Se": (2, 4, 6),
    "Br": (1,),
    "Kr": (0,),
    "Rb": (1,),
    "Sr": (2,),
    "Y": (3,),
    "Zr": (4,),
    "Nb": (3, 5),
    "Mo": (2, 3, 4, 6),
    "Tc": (4, 7),
    "Ru": (2, 3, 4, 8),
    "Rh": (3,),
    "Pd": (2, 4),
    "Ag": (1,),
    "Cd": (2,),
    "In": (3,),
    "Sn": (2, 4),
    "Sb": (3, 5),
    "Te": (2, 4, 6),
    "I": (1,),
    "Xe": (0,),
    "Cs": (1,),
    "Ba": (2,),
    "La": (3,),
    "Ce": (3, 4),
    "Pr": (3,),
    "Nd": (3,),
    "Pm": (3,),
    "Sm": (2, 3),
    "Eu": (2, 3),
    "Gd": (3,),
    "Tb": (3, 4),
    "Dy": (3,),
    "Ho": (3,),
    "Er": (3,),
    "Tm": (2, 3),
    "Yb": (2, 3),
    "Lu": (3,),
    "Hf": (4,),
    "Ta": (5,),
    "W": (4, 6),
    "Re": (4, 7),
    "Os": (4, 8),
    "Ir": (3, 4),
    "Pt": (2, 4),
    "Au": (1, 3),
    "Hg": (1, 2),
    "Tl": (1, 3),
    "Pb": (2, 4),
    "Bi": (3, 5),
    "Po": (2, 4, 6),
    "At": (1,),
    "Rn": (0,),
    "Fr": (1,),
    "Ra": (2,),
    "Ac": (3,),
    "Th": (4,),
    "Pa": (5,),
    "U": (3, 4, 6),
    "Np": (3, 4, 5, 6),
    "Pu": (3, 4, 5, 6),
    "Am": (3,),
    "Cm": (3,),
    "Bk": (3, 4),
    "Cf": (3,),
    "Es": (3,),
    "Fm": (3,),
    "Md": (3,),
    "No": (2, 3),
    "Lr": (3,),
    "Rf": (4,),
    "Db": (5,),
    "Sg": (6,),
    "Bh": (7,),
    "Hs": (8,),
    "Mt": (1,),
    "Ds": (2,),
    "Rg": (1,),
    "Cn": (2,),
    "Nh": (1,),
    "Fl": (2,),
    "Mc": (1,),
    "Lv": (2,),
    "Ts": (1,),
    "Og": (0,),
}
AROMATIC_VALENCE: dict[str, tuple[int, ...]] = {
    "c": (3,),
    "n": (3,),
    "o": (2,),
    "p": (3,),
    "s": (2,),
    "b": (3,),
    "se": (2,),
    "as": (3,),
}
ATOMIC_WEIGHT = {
    "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974, "S": 32.06, "Cl": 35.45, "Ar": 39.948, "K": 39.098, "Ca": 40.078,
    "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938, "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.630, "As": 74.922, "Se": 78.971, "Br": 79.904, "Kr": 83.798, "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224,
    "Nb": 92.906, "Mo": 95.95, "Tc": 98, "Ru": 101.07, "Rh": 102.91, "Pd": 106.42, "Ag": 107.87, "Cd": 112.41, "In": 114.82, "Sn": 118.71,
    "Sb": 121.76, "Te": 127.60, "I": 126.904, "Xe": 131.29, "Cs": 132.91, "Ba": 137.33, "La": 138.91, "Ce": 140.12, "Pr": 140.91, "Nd": 144.24,
    "Pm": 145, "Sm": 150.36, "Eu": 151.96, "Gd": 157.25, "Tb": 158.93, "Dy": 162.50, "Ho": 164.93, "Er": 167.26, "Tm": 168.93, "Yb": 173.05,
    "Lu": 174.97, "Hf": 178.49, "Ta": 180.95, "W": 183.84, "Re": 186.21, "Os": 190.23, "Ir": 192.22, "Pt": 195.08, "Au": 196.97, "Hg": 200.59,
    "Tl": 204.38, "Pb": 207.2, "Bi": 208.98, "Po": 209, "At": 210, "Rn": 222, "Fr": 223, "Ra": 226, "Ac": 227, "Th": 232.04,
    "Pa": 231.04, "U": 238.03, "Np": 237, "Pu": 244, "Am": 243, "Cm": 247, "Bk": 247, "Cf": 251, "Es": 252, "Fm": 257, "Md": 258,
    "No": 259, "Lr": 262, "Rf": 267, "Db": 268, "Sg": 269, "Bh": 270, "Hs": 277, "Mt": 276, "Ds": 281, "Rg": 280, "Cn": 285,
    "Nh": 284, "Fl": 289, "Mc": 288, "Lv": 293, "Ts": 294, "Og": 294,
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
    atoms: list[Atom] = field(default_factory=list)
    bonds: list[tuple[int, int, float]] = field(default_factory=list)
    formal_charge: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    formula: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def molecular_weight(self) -> float | None:
        if self.errors:
            return None
        if any(el not in ATOMIC_WEIGHT for el in self.formula):
            return None
        return sum(ATOMIC_WEIGHT[el] * count for el, count in self.formula.items())

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "atoms": len(self.atoms),
            "bonds": len(self.bonds),
            "formal_charge": self.formal_charge,
            "formula": dict(self.formula),
            "molecular_weight": self.molecular_weight(),
            "errors": list(self.errors),
        }


# All 118 element symbols (two-letter and one-letter)
_ELEMENT_SYMBOLS_2 = frozenset({
    "He", "Li", "Be", "Ne", "Na", "Mg", "Al", "Si", "Cl", "Ar", "Ca", "Sc", "Ti", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
    "Tm", "Yb", "Lu", "Hf", "Ta", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra",
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh",
    "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
})
_ELEMENT_SYMBOLS_1 = frozenset("BCNOPSFIHKVYIWU")  # Single-letter symbols that can appear alone

def _tokenize(smiles: str) -> list[str]:
    tokens: list[str] = []
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
        # Check for two-letter element symbols first
        if i + 1 < len(smiles):
            two_char = smiles[i:i+2]
            if two_char in _ELEMENT_SYMBOLS_2:
                tokens.append(two_char)
                i += 2
                continue
        # Single-letter element symbols (uppercase)
        if ch in _ELEMENT_SYMBOLS_1:
            tokens.append(ch)
            i += 1
            continue
        # Aromatic (lowercase) symbols
        if ch in "bcnopsseas":
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

    atoms: list[Atom] = analysis.atoms
    current: int | None = None
    pending_bond: float | None = None
    branch_stack: list[int | None] = []
    rings: dict[str, tuple[int, float]] = {}

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


def _allowed(atom: Atom) -> tuple[int, ...]:
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
    formula: dict[str, int] = {}
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


def molecular_weight(smiles: str) -> float | None:
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
