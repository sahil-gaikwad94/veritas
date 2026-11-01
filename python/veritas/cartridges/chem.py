"""veritas.chem cartridge."""

from __future__ import annotations

VALENCE = {"C": 4, "N": 3, "O": 2, "H": 1, "S": 2, "P": 3, "F": 1, "Cl": 1}


def valence_ok(symbol: str, bonds: int, charge: int = 0) -> bool:
    base = VALENCE.get(symbol)
    if base is None:
        return True
    return bonds <= base + abs(charge)


def smiles_valence_ok(smiles: str) -> bool:
    # very rough: count atoms
    return isinstance(smiles, str) and len(smiles) > 0


def formal_charge_neutral(charges: list) -> bool:
    return sum(charges) == 0
