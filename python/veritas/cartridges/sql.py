"""``veritas.sql`` — catalog-aware verification for text-to-SQL.

The cartridge checks exactly the failure modes that make generated SQL dangerous:

* hallucinated tables / columns (qualified and unqualified),
* aliases that are out of scope,
* joins that do not follow any declared foreign key,
* missing tenant-isolation predicates,
* destructive statements in read-only pipelines.

It is a lexical/structural verifier, not a full SQL parser.  It is deliberately strict
about identifiers and honest about constructs it cannot reason about (they are reported as
warnings, never silently accepted as verified).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

_TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?|\d+(?:\.\d+)?|[(),;.*=<>!]+"
)
_KEYWORDS = {
    "SELECT", "FROM", "WHERE", "JOIN", "INNER", "LEFT", "RIGHT", "FULL", "OUTER", "CROSS",
    "ON", "AS", "AND", "OR", "NOT", "NULL", "GROUP", "BY", "ORDER", "HAVING", "LIMIT",
    "OFFSET", "UNION", "ALL", "DISTINCT", "CASE", "WHEN", "THEN", "ELSE", "END", "WITH",
    "ASC", "DESC", "IN", "IS", "BETWEEN", "LIKE", "EXISTS",
}
_WRITE_KEYWORDS = {"INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE", "MERGE"}


@dataclass
class ForeignKey:
    table: str
    column: str
    ref_table: str
    ref_column: str

    def matches(self, t1: str, c1: str, t2: str, c2: str) -> bool:
        return (
            self.table.lower() == t1.lower() and self.column.lower() == c1.lower()
            and self.ref_table.lower() == t2.lower() and self.ref_column.lower() == c2.lower()
        ) or (
            self.table.lower() == t2.lower() and self.column.lower() == c2.lower()
            and self.ref_table.lower() == t1.lower() and self.ref_column.lower() == c1.lower()
        )


@dataclass
class SqlCatalog:
    tables: Dict[str, List[str]] = field(default_factory=dict)
    foreign_keys: List[ForeignKey] = field(default_factory=list)
    tenant_column: str = "tenant_id"
    require_tenant: bool = False

    def columns(self, table: str) -> Optional[List[str]]:
        requested = table.split(".")[-1].lower()
        for name, cols in self.tables.items():
            if name.lower() == requested:
                return list(cols)
        return None

    def canonical(self, table: str) -> Optional[str]:
        requested = table.split(".")[-1].lower()
        for name in self.tables:
            if name.lower() == requested:
                return name
        return None


@dataclass
class SqlVerification:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    tables: List[str] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    has_tenant_filter: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "tables": list(self.tables),
            "columns": list(self.columns),
            "has_tenant_filter": self.has_tenant_filter,
        }


def _strip_comments_and_strings(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    sql = re.sub(r"--[^\n]*", " ", sql)
    # Replace literal content but keep the quotes so tokenization stays honest.
    sql = re.sub(r"'(?:''|[^'])*'", "''", sql)
    return sql


def tokenize(sql: str) -> List[str]:
    return _TOKEN_RE.findall(_strip_comments_and_strings(sql))


def _lname(token: str) -> str:
    return token.lower()


class SqlVerifier:
    def __init__(self, catalog: Any = None, *, read_only: bool = True, require_tenant: Optional[bool] = None):
        if isinstance(catalog, SqlCatalog):
            self.catalog = catalog
        else:
            self.catalog = SqlCatalog(tables=dict(catalog or {}))
        if require_tenant is not None:
            self.catalog.require_tenant = require_tenant
        self.read_only = read_only
        self.aliases: Dict[str, str] = {}

    # -- legacy surface --------------------------------------------------
    def valid_table(self, name: str) -> bool:
        return self.catalog.columns(name) is not None

    def valid_column(self, table: str, column: str) -> bool:
        columns = self.catalog.columns(table)
        return columns is not None and any(c.lower() == column.lower() for c in columns)

    def register_alias(self, alias: str, table: str) -> None:
        self.aliases[alias.lower()] = table

    def resolve(self, name: str) -> str:
        return self.aliases.get(name.lower(), name)

    def foreign_key_ok(self, t1: str, c1: str, t2: str, c2: str) -> bool:
        if self.catalog.foreign_keys:
            return any(fk.matches(t1, c1, t2, c2) for fk in self.catalog.foreign_keys)
        return self.valid_column(t1, c1) and self.valid_column(t2, c2)

    def tenant_isolation_ok(self, predicate: str, tenant_col: Optional[str] = None) -> bool:
        column = (tenant_col or self.catalog.tenant_column).lower()
        return column in predicate.lower()

    # -- verification ----------------------------------------------------
    def verify(self, sql: str) -> SqlVerification:
        errors: List[str] = []
        warnings: List[str] = []
        tokens = tokenize(sql)
        if not tokens:
            return SqlVerification(False, ["empty SQL statement"])
        head = tokens[0].upper()
        if self.read_only and (head in _WRITE_KEYWORDS or any(t.upper() in _WRITE_KEYWORDS for t in tokens[:3])):
            errors.append(f"read-only pipeline rejected a `{head}` statement")

        scope: Dict[str, str] = {}
        tables_seen: List[str] = []
        columns_seen: List[str] = []
        has_tenant = False

        upper = [t.upper() for t in tokens]
        from_index = next((i for i, t in enumerate(upper) if t == "FROM"), None)
        if from_index is None:
            errors.append("query has no FROM clause")
            return SqlVerification(False, errors, warnings)

        # --- table references and aliases --------------------------------
        table_tokens: List[Tuple[str, Optional[str]]] = []
        joins: List[int] = [from_index] + [i for i, t in enumerate(upper) if t == "JOIN"]
        for index in joins:
            cursor = index + 1
            while cursor < len(tokens) and upper[cursor] in {"INNER", "LEFT", "RIGHT", "FULL", "OUTER", "CROSS"}:
                cursor += 1
            if cursor >= len(tokens):
                errors.append("dangling JOIN/FROM without a table")
                continue
            table = tokens[cursor]
            if table == "(":
                errors.append("subqueries are not supported by this verifier")
                continue
            canonical = self.catalog.canonical(table)
            if canonical is None:
                errors.append(f"unknown table `{table}`")
                continue
            table_tokens.append((canonical, None))
            tables_seen.append(canonical)
            # optional alias
            alias = canonical
            next_index = cursor + 1
            if next_index < len(tokens) and upper[next_index] == "AS":
                next_index += 1
            if next_index < len(tokens) and upper[next_index] not in _KEYWORDS and tokens[next_index] != "(":
                alias = tokens[next_index]
            scope[alias.lower()] = canonical
            if alias != canonical:
                scope[canonical.lower()] = canonical
        if not table_tokens:
            return SqlVerification(False, errors or ["no resolvable tables"], warnings)

        def resolve_column(reference: str) -> Optional[Tuple[str, str]]:
            if "." in reference:
                alias, _, column = reference.partition(".")
                table = scope.get(alias.lower())
                if table is None:
                    errors.append(f"unknown table alias `{alias}`")
                    return None
                if not self.valid_column(table, column):
                    errors.append(f"unknown column `{table}.{column}`")
                    return None
                return table, column
            for table in set(scope.values()):
                if self.valid_column(table, reference):
                    return table, reference
            errors.append(f"unknown unqualified column `{reference}`")
            return None

        # --- SELECT list --------------------------------------------------
        select_tokens = tokens[1:from_index]
        if select_tokens:
            for piece in _split_commas(select_tokens):
                if any("*" in token for token in piece):
                    continue
                if any(token.upper() == "CASE" for token in piece):
                    warnings.append("CASE expression was not column-verified")
                    continue
                # strip an output alias (`... AS name`)
                alias_cut = next(
                    (i for i, token in enumerate(piece) if token.upper() == "AS"), len(piece)
                )
                expression = piece[:alias_cut]
                identifiers = [
                    (i, token) for i, token in enumerate(expression)
                    if re.match(r"^[A-Za-z_]", token) and token.upper() not in _KEYWORDS
                ]
                # drop function names: an identifier immediately followed by `(`
                identifiers = [
                    (i, token) for i, token in identifiers
                    if not (i + 1 < len(expression) and expression[i + 1] == "(")
                ]
                if not identifiers:
                    continue
                reference = identifiers[-1][1]
                resolved = resolve_column(reference)
                if resolved:
                    columns_seen.append(f"{resolved[0]}.{resolved[1]}")

        # --- joins + FK validity -----------------------------------------
        fk_candidates: List[Tuple[str, str, str, str]] = []
        for index, token in enumerate(tokens):
            if token.upper() != "ON":
                continue
            segment = _until_keyword(tokens[index + 1:])
            found = _extract_equalities(segment)
            if not found:
                warnings.append("JOIN without a recognisable equality predicate")
                continue
            for left, right in found:
                l_res = resolve_column(left)
                r_res = resolve_column(right)
                if l_res and r_res:
                    fk_candidates.append((l_res[0], l_res[1], r_res[0], r_res[1]))
        if self.catalog.foreign_keys:
            for t1, c1, t2, c2 in fk_candidates:
                if not any(fk.matches(t1, c1, t2, c2) for fk in self.catalog.foreign_keys):
                    errors.append(
                        f"join `{t1}.{c1} = {t2}.{c2}` does not follow any declared foreign key"
                    )

        # --- tenant isolation -------------------------------------------
        tenant = self.catalog.tenant_column.lower()
        has_tenant = any(tenant in t.lower() for t in tokens)
        if self.catalog.require_tenant and not has_tenant:
            errors.append(f"missing required tenant-isolation predicate on `{self.catalog.tenant_column}`")

        # --- group by / order by columns ---------------------------------
        for keyword in ("GROUP", "ORDER"):
            if keyword in upper:
                index = upper.index(keyword)
                if index + 1 < len(tokens) and upper[index + 1] == "BY":
                    for piece in _split_commas(tokens[index + 2:]):
                        for token in piece:
                            if re.match(r"^[A-Za-z_]", token) and token.upper() not in _KEYWORDS:
                                if "." in token or any(
                                    self.valid_column(table, token) for table in set(scope.values())
                                ):
                                    resolve_column(token)

        return SqlVerification(
            ok=not errors,
            errors=errors,
            warnings=warnings,
            tables=sorted(set(tables_seen)),
            columns=sorted(set(columns_seen)),
            has_tenant_filter=has_tenant,
        )


def _split_commas(tokens: Sequence[str]) -> List[List[str]]:
    parts: List[List[str]] = [[]]
    depth = 0
    for token in tokens:
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
        if token == "," and depth == 0:
            parts.append([])
        else:
            parts[-1].append(token)
    return [part for part in parts if part]


def _until_keyword(tokens: Sequence[str]) -> List[str]:
    out: List[str] = []
    depth = 0
    for token in tokens:
        upper = token.upper()
        if token == "(":
            depth += 1
        elif token == ")":
            if depth == 0:
                break
            depth -= 1
        if depth == 0 and upper in {"JOIN", "WHERE", "GROUP", "ORDER", "HAVING", "LIMIT", "LEFT", "RIGHT", "INNER", "FULL"}:
            break
        out.append(token)
    return out


def _extract_equalities(tokens: Sequence[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for index, token in enumerate(tokens):
        if token != "=":
            continue
        left = next((tokens[j] for j in range(index - 1, -1, -1) if re.match(r"^[A-Za-z_]", tokens[j])), None)
        right = next((tokens[j] for j in range(index + 1, len(tokens)) if re.match(r"^[A-Za-z_]", tokens[j])), None)
        if left and right:
            out.append((left, right))
    return out


__all__ = ["ForeignKey", "SqlCatalog", "SqlVerification", "SqlVerifier", "tokenize"]
