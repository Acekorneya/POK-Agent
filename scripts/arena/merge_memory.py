"""Merge POK-Ai memory copies into one application memory.

In the arena each VM learns into its own copy of an application's memory
(`<app>-w<n>`), because several VMs cannot write one SQLite file over the
shared folder safely. When a VM finishes a piece of work, and at the end of a
pass, its copy is merged back here, so the next piece and the next pass start
from everything learned so far, as a single agent would.

    python3 merge_memory.py <app memory dir> <copy dir> [<copy dir> ...]

Each copy was seeded from the application memory, so rows present in both
are the same item: counts take the larger value and content comes from
whichever side changed it last. Items only in a copy are added, deletions a
copy recorded (tombstones) are applied, and skill folders are copied over
when newer. Copies are deleted after a successful merge.
"""

from __future__ import annotations

import shutil
import sqlite3
import sys
from pathlib import Path

COUNTS = {"success_count", "retrieval_count", "failure_count", "reinforcement_count"}


def columns(db: sqlite3.Connection, schema: str, table: str) -> list[str]:
    return [row[1] for row in db.execute(f"PRAGMA {schema}.table_info({table})")]


def add_missing_columns(db: sqlite3.Connection, table: str) -> None:
    """Columns a newer agent added to its copy (for example a skill's replay
    program) are added to the main table first, so merging keeps them."""
    main = columns(db, "main", table)
    for row in db.execute(f"PRAGMA copy.table_info({table})").fetchall():
        name, kind = row[1], row[2] or "TEXT"
        if name not in main:
            db.execute(f"ALTER TABLE main.{table} ADD COLUMN {name} {kind}")


def merge_table(db: sqlite3.Connection, table: str, key: str, fts: tuple[str, list[str]] | None) -> tuple[int, int]:
    """Merge `copy.table` into `main.table` by `key`; returns (added, updated)."""
    add_missing_columns(db, table)
    shared = [name for name in columns(db, "copy", table) if name in columns(db, "main", table)]
    if key not in shared:
        return 0, 0
    names = ", ".join(shared)
    added = updated = 0
    for row in db.execute(f"SELECT {names} FROM copy.{table}").fetchall():
        record = dict(zip(shared, row))
        existing = db.execute(f"SELECT {names} FROM main.{table} WHERE {key} = ?", (record[key],)).fetchone()
        match = key
        # A skill rewritten in the copy has a new fingerprint but keeps its id:
        # it is the same row, so update it rather than insert a second one.
        if existing is None and key != "id" and "id" in shared:
            existing = db.execute(f"SELECT {names} FROM main.{table} WHERE id = ?", (record["id"],)).fetchone()
            match = "id" if existing is not None else key
        if existing is None:
            db.execute(f"INSERT INTO main.{table} ({names}) VALUES ({', '.join('?' for _ in shared)})", row)
            added += 1
        else:
            current = dict(zip(shared, existing))
            newer = (record.get("updated_at") or "") > (current.get("updated_at") or "")
            merged = {name: (max(record[name] or 0, current[name] or 0) if name in COUNTS
                             else record[name] if newer else current[name])
                      for name in shared}
            if merged == current:
                continue
            db.execute(f"UPDATE main.{table} SET {', '.join(f'{name} = ?' for name in shared)} WHERE {match} = ?",
                       [merged[name] for name in shared] + [current[match]])
            updated += 1
        if fts:
            fts_table, fts_columns = fts
            item = db.execute(f"SELECT id, {', '.join(fts_columns)} FROM main.{table} WHERE id = ?",
                              (record["id"],)).fetchone()
            db.execute(f"DELETE FROM main.{fts_table} WHERE id = ?", (item[0],))
            db.execute(f"INSERT INTO main.{fts_table} (id, {', '.join(fts_columns)}) VALUES "
                       f"({', '.join('?' for _ in item)})", item)
    return added, updated


def merge_copy(main_dir: Path, copy_dir: Path) -> str:
    copy_db = copy_dir / "memory.db"
    if not copy_db.exists():
        shutil.rmtree(copy_dir, ignore_errors=True)
        return f"{copy_dir.name}: empty"
    if not (main_dir / "memory.db").exists():
        main_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(copy_dir, main_dir, dirs_exist_ok=True)
        shutil.rmtree(copy_dir)
        return f"{copy_dir.name}: became {main_dir.name}"
    db = sqlite3.connect(main_dir / "memory.db")
    try:
        db.execute("ATTACH DATABASE ? AS copy", (str(copy_db),))
        with db:
            procedures = merge_table(db, "procedures", "fingerprint", (
                "procedures_fts", ["task_signature", "title", "summary", "applications", "command_template"]))
            memories = merge_table(db, "memories", "id", ("memories_fts", ["text"]))
            for table in ("memory_tombstones", "memory_merges", "memory_semantic_suppressions"):
                shared = [name for name in columns(db, "copy", table) if name in columns(db, "main", table)]
                if shared:
                    names = ", ".join(shared)
                    db.execute(f"INSERT OR IGNORE INTO main.{table} ({names}) SELECT {names} FROM copy.{table}")
            # Deletions the copy made apply here too.
            for (fingerprint,) in db.execute("SELECT fingerprint FROM copy.memory_tombstones").fetchall():
                for table, fts_table in (("procedures", "procedures_fts"), ("memories", "memories_fts")):
                    for (item_id,) in db.execute(f"SELECT id FROM main.{table} WHERE fingerprint = ?",
                                                 (fingerprint,)).fetchall():
                        db.execute(f"DELETE FROM main.{fts_table} WHERE id = ?", (item_id,))
                        db.execute(f"DELETE FROM main.{table} WHERE id = ?", (item_id,))
        db.execute("DETACH DATABASE copy")
    finally:
        db.close()
    skills = 0
    for source in (copy_dir / "skills").glob("*/*"):
        target = main_dir / "skills" / source.parent.name / source.name
        if not target.exists() or source.stat().st_mtime > target.stat().st_mtime:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            skills += 1
    shutil.rmtree(copy_dir)
    return (f"{copy_dir.name}: procedures +{procedures[0]} ~{procedures[1]}, "
            f"facts +{memories[0]} ~{memories[1]}, skill files {skills}")


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    main_dir = Path(sys.argv[1])
    for copy in sys.argv[2:]:
        print(merge_copy(main_dir, Path(copy)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
