"""Persistent, indexed lexical/skeleton overlap gate (not a semantic plagiarism proof)."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

from .models import Unit


def normalize(text: str) -> list[str]:
    text = re.sub(r"\{\{[^}]+\}\}|\[\[[1-4]\]\]", " BLANK ", text)
    # Mask capitalized names except sentence-initial words to catch simple name swaps.
    text = re.sub(r"(?<!^)(?<![.!?] )\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", " NAME ", text)
    text = re.sub(r"\d+(?:[.,:]\d+)*", " NUM ", text)
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def shingles(tokens: list[str], n: int = 5) -> set[str]:
    return {hashlib.sha256(" ".join(tokens[i:i+n]).encode()).hexdigest()[:24] for i in range(max(0, len(tokens)-n+1))}


def unit_texts(unit: Unit) -> list[tuple[str, str]]:
    if unit.part == 5:
        return [(q.id, q.prompt) for q in unit.questions]
    return [(d.id, "\n".join(b.readable() for b in d.blocks)) for d in unit.documents]


class History:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS texts (id INTEGER PRIMARY KEY, source TEXT UNIQUE, tokens TEXT, digest TEXT);
            CREATE TABLE IF NOT EXISTS grams (gram TEXT, text_id INTEGER, PRIMARY KEY(gram,text_id));
            CREATE TABLE IF NOT EXISTS accepted (id TEXT PRIMARY KEY, summary TEXT, plan TEXT, content_sha256 TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS corpus (path TEXT PRIMARY KEY, digest TEXT);
            CREATE INDEX IF NOT EXISTS text_digest ON texts(digest);
        """)
        if "content_sha256" not in {r[1] for r in self.db.execute("PRAGMA table_info(accepted)")}:
            self.db.execute("ALTER TABLE accepted ADD COLUMN content_sha256 TEXT")
            self.db.commit()
        if "revises" not in {r[1] for r in self.db.execute("PRAGMA table_info(accepted)")}:
            self.db.execute("ALTER TABLE accepted ADD COLUMN revises TEXT")
            self.db.commit()

    def close(self):
        self.db.close()

    def revision_parent(self, requested: str, digest: str, fallback: str | None = None) -> str:
        """Recover the accepted identity of a saved draft or older audit alias."""
        if self.db.execute("SELECT 1 FROM accepted WHERE id=?", (requested,)).fetchone():
            return requested
        row = self.db.execute("SELECT id FROM accepted WHERE content_sha256=? ORDER BY created_at DESC, id LIMIT 1", (digest,)).fetchone()
        return row[0] if row else (fallback or requested)

    def _add(self, source: str, text: str):
        tokens = normalize(text)
        digest = hashlib.sha256(" ".join(tokens).encode()).hexdigest()
        cursor = self.db.execute("INSERT INTO texts(source,tokens,digest) VALUES (?,?,?)", (source, json.dumps(tokens), digest))
        self.db.executemany("INSERT OR IGNORE INTO grams VALUES (?,?)", [(g, cursor.lastrowid) for g in shingles(tokens)])

    def import_corpus(self, root: Path) -> int:
        paths = sorted(root.glob("test-*/test-*.json"))
        if len(paths) != 10:
            raise ValueError(f"Expected ten benchmark test files in {root}; found {len(paths)}")
        count = 0
        with self.db:
            for path in paths:
                raw = path.read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                key = str(path.resolve())
                existing = self.db.execute("SELECT digest FROM corpus WHERE path=?", (key,)).fetchone()
                if existing and existing[0] == digest:
                    continue
                prefix = f"corpus:{key}:"
                ids = [r[0] for r in self.db.execute("SELECT id FROM texts WHERE substr(source,1,?)=?", (len(prefix), prefix))]
                for row_id in ids:
                    self.db.execute("DELETE FROM grams WHERE text_id=?", (row_id,))
                    self.db.execute("DELETE FROM texts WHERE id=?", (row_id,))
                data = json.loads(raw)
                for q in data["questions"]:
                    if q["part"] == 5:
                        self._add(prefix + q["id"], q["prompt"])
                        count += 1
                for d in data["documents"]:
                    # Include table text even if the extractor stores it separately.
                    txt = "\n".join(str(b.get("text", "")) + " " + json.dumps(b.get("rows", []), ensure_ascii=False) for b in d["blocks"])
                    self._add(prefix + d["id"], txt)
                    count += 1
                self.db.execute("INSERT OR REPLACE INTO corpus VALUES (?,?)", (key, digest))
        return count

    @staticmethod
    def overlap(a: list[str], b: list[str]) -> tuple[bool, float]:
        if a == b:
            return True, 1.0
        ga, gb = shingles(a), shingles(b)
        common = len(ga & gb)
        coverage = common / max(1, min(len(ga), len(gb)))
        long_copy = bool(shingles(a, 14) & shingles(b, 14))
        return (common >= 3 and coverage >= .48) or long_copy, round(coverage, 3)

    def check(self, unit: Unit, exclude_source: str | None = None) -> list[dict]:
        findings = []
        current = []
        # A second revision may retain good items from every earlier version.
        # Exempt only its recorded ancestry, never unrelated accepted content.
        excluded = set()
        ancestor = exclude_source
        while ancestor and ancestor not in excluded:
            excluded.add(ancestor)
            row = self.db.execute("SELECT revises FROM accepted WHERE id=?", (ancestor,)).fetchone()
            ancestor = row[0] if row else None
        for local_id, text in unit_texts(unit):
            tokens = normalize(text)
            grams = shingles(tokens)
            candidate_ids: set[int] = set()
            # Bounded chunks avoid SQLite variable limits for long documents.
            values = list(grams)
            for offset in range(0, len(values), 400):
                batch = values[offset:offset+400]
                sql = "SELECT DISTINCT text_id FROM grams WHERE gram IN (" + ",".join("?" for _ in batch) + ")"
                candidate_ids.update(r[0] for r in self.db.execute(sql, batch))
            digest = hashlib.sha256(" ".join(tokens).encode()).hexdigest()
            candidate_ids.update(r[0] for r in self.db.execute("SELECT id FROM texts WHERE digest=?", (digest,)))
            candidates = [self.db.execute("SELECT source,tokens FROM texts WHERE id=?", (i,)).fetchone() for i in candidate_ids]
            for source, encoded in candidates:
                if any(source.startswith(f"generated:{item}:") for item in excluded):
                    continue
                hit, score = self.overlap(tokens, json.loads(encoded))
                if hit:
                    findings.append({"item": local_id, "source": source, "overlap": score})
            for prior_id, prior_tokens in current:
                hit, score = self.overlap(tokens, prior_tokens)
                if hit:
                    findings.append({"item": local_id, "source": f"same-unit:{prior_id}", "overlap": score})
            current.append((local_id, tokens))
        return findings

    def accept(self, unit_id: str, unit: Unit, plan: dict, revises: str | None = None):
        # Recheck under a write lock: simultaneous generators cannot both accept a duplicate.
        self.db.execute("BEGIN IMMEDIATE")
        try:
            if self.check(unit, exclude_source=revises):
                raise ValueError("Novelty collision discovered at commit")
            digest = hashlib.sha256(json.dumps(unit.model_dump(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            self.db.execute("INSERT INTO accepted(id,summary,plan,content_sha256,revises) VALUES (?,?,?,?,?)", (unit_id, unit.scenario_summary, json.dumps(plan), digest, revises))
            for key, txt in unit_texts(unit):
                self._add(f"generated:{unit_id}:{key}", txt)
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

    def has_accepted(self, unit_id: str, digest: str) -> bool:
        row = self.db.execute("SELECT content_sha256 FROM accepted WHERE id=?", (unit_id,)).fetchone()
        return bool(row and row[0] == digest)

    def recent(self, limit: int = 40) -> list[str]:
        return [r[0] for r in self.db.execute("SELECT summary FROM accepted ORDER BY rowid DESC LIMIT ?", (limit,))]

    def usage(self) -> dict[str, int]:
        counter: Counter = Counter()
        for row in self.db.execute("SELECT plan FROM accepted"):
            plan = json.loads(row[0])
            for key in ("scenario", "event"):
                counter[f"{key}:{plan[key]}"] += 1
            for slot in plan["slots"]:
                if "scenario" in slot:
                    counter[f"scenario:{slot['scenario']}"] += 1
            if plan["part"] == 6:
                counter[f"p6_genre:{plan['genres'][0]}"] += 1
                for slot in plan["slots"]:
                    counter[f"p6_grammar:{slot['skill']}"] += 1
                    counter[f"p6_third:{slot['skill']}"] += 1
            if plan["part"] == 7:
                key = plan["genres"][0] if plan["mode"] == "single" else tuple(plan["genres"])
                counter[f"p7_{plan['mode']}:{key}"] += 1
        return dict(counter)
