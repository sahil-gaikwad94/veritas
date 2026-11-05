"""``vexx.sql`` — catalog-aware verification for text-to-SQL.

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
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

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
    tables: dict[str, list[str]] = field(default_factory=dict)
    foreign_keys: list[ForeignKey] = field(default_factory=list)
    tenant_column: str = "tenant_id"
    require_tenant: bool = False

    def columns(self, table: str) -> list[str] | None:
        requested = table.split(".")[-1].lower()
        for name, cols in self.tables.items():
            if name.lower() == requested:
                return list(cols)
        return None

    def canonical(self, table: str) -> str | None:
        requested = table.split(".")[-1].lower()
        for name in self.tables:
            if name.lower() == requested:
                return name
        return None


@dataclass
class SqlVerification:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    has_tenant_filter: bool = False

    def as_dict(self) -> dict[str, Any]:
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


def tokenize(sql: str) -> list[str]:
    return _TOKEN_RE.findall(_strip_comments_and_strings(sql))


def _lname(token: str) -> str:
    return token.lower()


class SqlVerifier:
    def __init__(self, catalog: Any = None, *, read_only: bool = True, require_tenant: bool | None = None):
        if isinstance(catalog, SqlCatalog):
            self.catalog = catalog
        else:
            self.catalog = SqlCatalog(tables=dict(catalog or {}))
        if require_tenant is not None:
            self.catalog.require_tenant = require_tenant
        self.read_only = read_only
        self.aliases: dict[str, str] = {}

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

    def tenant_isolation_ok(self, predicate: str, tenant_col: str | None = None) -> bool:
        column = (tenant_col or self.catalog.tenant_column).lower()
        return column in predicate.lower()

    # -- verification ----------------------------------------------------
    def verify(self, sql: str) -> SqlVerification:
        """AST-backed verification of a single read-only SELECT subset.

        Uses sqlglot when available; otherwise fails closed. Tenant isolation must be a
        top-level conjunctive equality on the tenant column, not a substring or an OR
        branch. Multiple statements, writes, subqueries and set operations are rejected.
        """
        errors: list[str] = []
        warnings: list[str] = []
        try:
            import sqlglot
            from sqlglot import expressions as exp
        except Exception:
            return SqlVerification(False, ["sqlglot is required for SQL verification; install vexx-ai[sql]"])
        try:
            statements = sqlglot.parse(sql)
        except Exception as exc:
            return SqlVerification(False, [f"SQL parse error: {exc}"])
        statements = [st for st in statements if st is not None]
        if len(statements) != 1:
            return SqlVerification(False, ["expected exactly one SQL statement"])
        tree = statements[0]
        if not isinstance(tree, exp.Select):
            if self.read_only:
                errors.append(f"read-only pipeline rejected a `{tree.key.upper()}` statement")
            else:
                errors.append("only SELECT statements are supported by this verifier")
            return SqlVerification(False, errors, warnings)
        if tree.find(exp.Union) or tree.find(exp.Subquery) or tree.find(exp.With):
            errors.append("subqueries, CTEs and set operations are not supported by this verifier")
        if self.read_only and tree.find(exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Alter, exp.Create):
            errors.append("read-only pipeline rejected a write/DDL expression")

        scope: dict[str, str] = {}
        tables_seen: list[str] = []
        columns_seen: list[str] = []

        def register_table(table: exp.Table) -> str | None:
            name = table.name
            canonical = self.catalog.canonical(name)
            if canonical is None:
                errors.append(f"unknown table `{name}`")
                return None
            alias = table.alias or canonical
            scope[alias.lower()] = canonical
            scope[canonical.lower()] = canonical
            tables_seen.append(canonical)
            return canonical

        from_clause = tree.args.get("from") or tree.args.get("from_")
        if from_clause is None:
            errors.append("query has no FROM clause")
            return SqlVerification(False, errors, warnings)
        for table in from_clause.find_all(exp.Table):
            register_table(table)
        joins = list(tree.find_all(exp.Join))
        for join in joins:
            for table in join.find_all(exp.Table):
                register_table(table)
        if not tables_seen:
            return SqlVerification(False, errors or ["no resolvable tables"], warnings)

        def resolve_column(col: exp.Column) -> tuple[str, str] | None:
            column = col.name
            if col.table:
                table = scope.get(col.table.lower())
                if table is None:
                    errors.append(f"unknown table alias `{col.table}`")
                    return None
            else:
                matches = [t for t in set(scope.values()) if self.valid_column(t, column)]
                if len(matches) != 1:
                    errors.append(f"unknown or ambiguous unqualified column `{column}`")
                    return None
                table = matches[0]
            if not self.valid_column(table, column):
                errors.append(f"unknown column `{table}.{column}`")
                return None
            return table, column

        for col in tree.find_all(exp.Column):
            if col.name == "*" or isinstance(col.this, exp.Star):
                if col.table and col.table.lower() not in scope:
                    errors.append(f"unknown table alias `{col.table}`")
                continue
            # Columns inside tenant predicates are validated too.
            resolved = resolve_column(col)
            if resolved and col.parent and isinstance(col.parent, exp.Dot):
                continue
            if resolved:
                columns_seen.append(f"{resolved[0]}.{resolved[1]}")
        for star in tree.find_all(exp.Star):
            parent = star.parent
            if isinstance(parent, exp.Column) and parent.table:
                if parent.table.lower() not in scope:
                    errors.append(f"unknown table alias `{parent.table}`")

        fk_candidates: list[tuple[str, str, str, str]] = []
        for join in joins:
            on = join.args.get("on")
            if on is None:
                warnings.append("JOIN without an ON predicate")
                continue
            for eq in on.find_all(exp.EQ):
                left, right = eq.left, eq.right
                if isinstance(left, exp.Column) and isinstance(right, exp.Column):
                    l_res, r_res = resolve_column(left), resolve_column(right)
                    if l_res and r_res:
                        fk_candidates.append((l_res[0], l_res[1], r_res[0], r_res[1]))
        if self.catalog.foreign_keys:
            for t1, c1, t2, c2 in fk_candidates:
                if not any(fk.matches(t1, c1, t2, c2) for fk in self.catalog.foreign_keys):
                    errors.append(f"join `{t1}.{c1} = {t2}.{c2}` does not follow any declared foreign key")

        tenant = self.catalog.tenant_column.lower()
        where = tree.args.get("where")
        has_tenant = False
        def conjuncts(node):
            if isinstance(node, exp.And):
                yield from conjuncts(node.left)
                yield from conjuncts(node.right)
            else:
                yield node
        if where is not None:
            for part in conjuncts(where.this):
                if not isinstance(part, exp.EQ):
                    continue
                cols = [c for c in (part.left, part.right) if isinstance(c, exp.Column)]
                literals = [v for v in (part.left, part.right) if isinstance(v, (exp.Literal, exp.Placeholder, exp.Parameter))]
                if len(cols) == 1 and len(literals) == 1 and cols[0].name.lower() == tenant:
                    resolved = resolve_column(cols[0])
                    if resolved:
                        has_tenant = True
        if self.catalog.require_tenant and not has_tenant:
            errors.append(f"missing required conjunctive tenant-isolation predicate on `{self.catalog.tenant_column}`")

        return SqlVerification(
            ok=not errors,
            errors=errors,
            warnings=warnings,
            tables=sorted(set(tables_seen)),
            columns=sorted(set(columns_seen)),
            has_tenant_filter=has_tenant,
        )


def _split_commas(tokens: Sequence[str]) -> list[list[str]]:
    parts: list[list[str]] = [[]]
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


def _until_keyword(tokens: Sequence[str]) -> list[str]:
    out: list[str] = []
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


def _extract_equalities(tokens: Sequence[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for index, token in enumerate(tokens):
        if token != "=":
            continue
        left = next((tokens[j] for j in range(index - 1, -1, -1) if re.match(r"^[A-Za-z_]", tokens[j])), None)
        right = next((tokens[j] for j in range(index + 1, len(tokens)) if re.match(r"^[A-Za-z_]", tokens[j])), None)
        if left and right:
            out.append((left, right))
    return out


__all__ = ["ForeignKey", "SqlCatalog", "SqlVerification", "SqlVerifier", "tokenize"]
