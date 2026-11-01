"""veritas.sql cartridge."""

from __future__ import annotations

from typing import Any, Dict, List, Set


class SqlVerifier:
    def __init__(self, catalog: Dict[str, List[str]] = None):
        self.catalog = catalog or {}
        self.aliases: Dict[str, str] = {}

    def valid_table(self, name: str) -> bool:
        return name in self.catalog

    def valid_column(self, table: str, column: str) -> bool:
        return table in self.catalog and column in self.catalog[table]

    def register_alias(self, alias: str, table: str) -> None:
        self.aliases[alias] = table

    def resolve(self, name: str) -> str:
        return self.aliases.get(name, name)

    def foreign_key_ok(self, t1: str, c1: str, t2: str, c2: str) -> bool:
        return self.valid_column(t1, c1) and self.valid_column(t2, c2)

    def tenant_isolation_ok(self, predicate: str, tenant_col: str = "tenant_id") -> bool:
        return tenant_col in predicate
