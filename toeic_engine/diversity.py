"""Offline editorial history, distinct from plagiarism or semantic-equivalence proof.

New material supplies author fingerprints confirmed by a reviewer. Archives use
small, inspectable surface rules; unobserved reasoning operations stay unknown.
This module never adds keys or diagnostics to student/blind material.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from .models import Evidence, Model, Unit, check_evidence
from .novelty import History, normalize

VERSION = 1
WORD = re.compile(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*")
READING_OPERATIONS = (
    "local_grammar", "lexical_selection", "direct_retrieval", "paraphrase",
    "reference_resolution", "pragmatic_intent", "sequence_reconstruction",
    "rule_application", "implicit_inference", "purpose_identification",
    "sentence_cohesion", "cross_document_join", "vocabulary_in_context",
    "main_purpose", "direct_location", "negative_detail", "cross_sentence_inference",
    "chronology_inference", "lexical_substitution", "discourse_cohesion", "cross_document_integration",
    "lexical_discrimination_in_discourse", "discourse_relation", "grammatical_completion",
)
FINGERPRINT_FIELDS = ("microconstruct", "option_family", "answer_lexeme", "reading_operation", "evidence_join", "solution_template")


def content_hash(unit: Unit) -> str:
    return hashlib.sha256(json.dumps(unit.model_dump(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def literal(text: str) -> str:
    """Literal word normalization, deliberately not a semantic lemmatizer."""
    return " ".join(WORD.findall(text.casefold()))


class EvidenceJoin(Model):
    relation: str = Field(min_length=4, description="Specific relation, e.g. rule_and_request_select_eligible_candidate; not just inference.")
    premise_roles: list[str] = Field(min_length=1, description="Ordered logical roles, independent of names and document IDs.")
    document_count: int = Field(ge=1, le=3)

    @model_validator(mode="after")
    def meaningful_roles(self):
        if any(len(role.strip()) < 3 for role in self.premise_roles):
            raise ValueError("Premise roles must be explicit nonempty descriptions")
        return self


class FingerprintAnnotation(Model):
    question_id: str
    microconstruct: str | None = None
    option_family: str = Field(min_length=4)
    answer_lexeme: str = Field(min_length=1, description="Literal normalized keyed option, not a guessed lemma.")
    reading_operation: Literal[
        "local_grammar", "lexical_selection", "direct_retrieval", "paraphrase",
        "reference_resolution", "pragmatic_intent", "sequence_reconstruction",
        "rule_application", "implicit_inference", "purpose_identification",
        "sentence_cohesion", "cross_document_join", "vocabulary_in_context",
        "main_purpose", "direct_location", "negative_detail", "cross_sentence_inference",
        "chronology_inference", "lexical_substitution", "discourse_cohesion", "cross_document_integration",
    "lexical_discrimination_in_discourse", "discourse_relation", "grammatical_completion",
    ] | None = None
    evidence_join: EvidenceJoin | None = None
    solution_template: str = Field(min_length=8)
    evidence: list[Evidence] = Field(default_factory=list)
    basis: str = Field(min_length=20)
    reviewer_id: str = ""
    reviewer_note: str = ""

    @model_validator(mode="after")
    def specific_codes(self):
        generic = {"unknown", "other", "grammar", "vocabulary", "inference", "detail", "purpose", "cross_document", *READING_OPERATIONS}
        for field in ("microconstruct", "option_family", "solution_template"):
            value = getattr(self, field)
            if value is not None and (len(value) < 4 or value.casefold() in generic):
                raise ValueError(f"{field} must identify a specific pattern, not a broad skill")
        if self.reviewer_id and len(self.reviewer_note) < 20:
            raise ValueError("Reviewer confirmation needs a concrete note of at least 20 characters")
        return self


def validate_annotations(unit: Unit, annotations: list[dict] | None, *, require_review: bool = True) -> list[str]:
    """Validate coverage and cited support, not the truth of a semantic judgment."""
    if not isinstance(annotations, list):
        return ["Explicit fingerprint annotations are required for new material"]
    parsed = []
    errors = []
    for index, item in enumerate(annotations):
        try:
            parsed.append(FingerprintAnnotation.model_validate(item))
        except (ValidationError, TypeError) as exc:
            errors.append(f"Annotation {index + 1}: {exc}")
    if [a.question_id for a in parsed] != [q.id for q in unit.questions]:
        errors.append("Annotate every question exactly once in question order")
    by_id = {q.id: q for q in unit.questions}
    for a in parsed:
        q = by_id.get(a.question_id)
        if q is None:
            continue
        if a.answer_lexeme != literal(next(o.text for o in q.options if o.label == q.answer)):
            errors.append(f"{q.id}: answer_lexeme must equal the literal normalized keyed option")
        if require_review and not a.reviewer_id:
            errors.append(f"{q.id}: reviewer confirmation is required")
        if unit.part == 5:
            if not a.microconstruct:
                errors.append(f"{q.id}: Part 5 needs a specific microconstruct")
            if a.evidence or a.evidence_join:
                errors.append(f"{q.id}: Part 5 cannot cite passage evidence")
        else:
            if not a.reading_operation or not a.evidence_join or not a.evidence:
                errors.append(f"{q.id}: passage items need reading_operation, evidence_join and exact evidence")
            errors.extend(f"{q.id}: {e}" for e in check_evidence(a.evidence, unit))
            if a.evidence_join and a.evidence_join.document_count != len({e.document_id for e in a.evidence}):
                errors.append(f"{q.id}: evidence_join document_count must match cited documents")
    return errors


def _heuristic(unit: Unit, q) -> dict:
    """Narrow observable signatures. No kind/skill label is a reasoning proof."""
    answer = literal(next(o.text for o in q.options if o.label == q.answer))
    options = sorted(literal(o.text) for o in q.options)
    option_set = set(options)
    text = re.sub(r"\{\{[^}]+\}\}", " BLANK ", q.prompt).casefold()
    fields = {key: None for key in FINGERPRINT_FIELDS}
    fields["answer_lexeme"] = answer
    confidence = {key: "unknown" for key in FINGERPRINT_FIELDS}
    confidence["answer_lexeme"] = "observed"
    observations = []
    # These cues are descriptions of visible forms, not parsed grammar claims.
    if unit.part == 5:
        if "approval" in option_set and all(x.startswith("approv") for x in options):
            fields.update(microconstruct="approval_derivation", option_family="approve_derivations", solution_template="approval_word_family_fill_blank")
            observations.append("All four choices share approv- and approval is present.")
        elif "whose" == answer and {"who", "whom", "whose"} <= option_set:
            fields.update(microconstruct="relative_clause.possessive_relative", option_family="relative_pronoun_forms", solution_template="whose_before_nominal_material")
            observations.append("Key is whose and choices include who/whom; syntax still needs review.")
        elif {"although", "despite"} <= option_set:
            fields.update(microconstruct="conjunction_vs_preposition.concessive_clause_vs_phrase", option_family="although_despite_contrast", solution_template=f"concession_{answer.replace(' ', '_')}_blank")
            observations.append("Although and despite are competing choices; clause/NP structure is not automatically parsed.")
        elif re.search(r"\beach of\b.*\bblank\b", text) and answer in {"is", "has", "was", "has been"}:
            fields.update(microconstruct="subject_verb_agreement.each_of_head", option_family="finite_agreement_forms", solution_template="each_of_plural_modifier_singular_predicate")
            observations.append("Each of precedes the blank and key is a singular finite verb.")
        elif "each" in option_set and answer == "each":
            fields.update(microconstruct="determiner_quantifier.distributive_reference", option_family="distributive_determiners", solution_template="each_before_nominal_material")
            observations.append("Key is each; determiner/number contrast requires review.")
        elif re.search(r"\bblank\s+by\b", text) and re.search(r"\b(?:be|been|being|is|are|was|were)\s+\w+(?:ed|en)\b", answer):
            fields.update(microconstruct="passive_explicit_by_phrase", option_family="active_passive_verb_forms", solution_template="passive_blank_followed_by_by_phrase")
            observations.append("Passive-looking keyed form followed by by; by-agent interpretation needs review.")
        for key in ("microconstruct", "option_family", "solution_template"):
            if fields[key]:
                confidence[key] = "heuristic"
    if unit.part != 5:
        observations.append("Question kind, cited-document count and tables do not establish a reading operation or evidence-join template.")
    return {**fields, "provenance": "archive_heuristic", "confidence": confidence,
            "unknown_fields": [k for k, v in fields.items() if v is None], "basis": observations,
            "option_set": options, "reviewer_id": None}


def _fingerprints(unit: Unit, annotations: list[dict] | None) -> dict[str, dict]:
    if annotations is None:
        return {q.id: _heuristic(unit, q) for q in unit.questions}
    result = {}
    for raw, q in zip(annotations, unit.questions):
        a = FingerprintAnnotation.model_validate(raw)
        fields = {key: getattr(a, key) for key in FINGERPRINT_FIELDS}
        fields["evidence_join"] = a.evidence_join.model_dump() if a.evidence_join else None
        result[q.id] = {**fields, "provenance": "reviewer_confirmed" if a.reviewer_id else "author_annotation",
                       "confidence": {key: ("reviewer_confirmed" if a.reviewer_id else "author_annotation") if value is not None else "unknown" for key, value in fields.items()},
                       "unknown_fields": [key for key, value in fields.items() if value is None],
                       "basis": a.basis, "reviewer_id": a.reviewer_id, "reviewer_note": a.reviewer_note,
                       "evidence": [e.model_dump() for e in a.evidence], "option_set": sorted(literal(o.text) for o in q.options)}
    return result


def _texts(unit: Unit) -> list[dict]:
    texts = []
    for q in unit.questions:
        options = sorted(o.text for o in q.options)
        texts.append({"id": q.id, "kind": "question", "part": unit.part, "prompt": q.prompt, "options": options,
                      "text": q.prompt + "\n" + "\n".join(options)})
    for d in unit.documents:
        text = "\n".join(b.readable() for b in d.blocks)
        texts.append({"id": d.id, "kind": "document", "text": text})
    return texts


@lru_cache(maxsize=4096)
def _tokens(text: str, skeleton: bool = True) -> tuple[str, ...]:
    return tuple(normalize(text) if skeleton else literal(text).split())


@lru_cache(maxsize=32768)
def _overlap(a: tuple[str, ...], b: tuple[str, ...]) -> tuple[bool, float]:
    return History.overlap(a, b)


def _lexical_match(a: dict, b: dict) -> dict | None:
    if a["kind"] != b["kind"]:
        return None
    if a["kind"] == "question":
        # Reused short stems and insertion position choices are ordinary format.
        ap, bp = _tokens(a["prompt"]), _tokens(b["prompt"])
        ao, bo = _tokens(" ".join(a["options"])), _tokens(" ".join(b["options"]))
        # Short finite-verb/word-family sets belong in pattern history. They
        # must not become unwaivable lexical allegations on their own.
        options_hit = len(ao) >= 16 and len(bo) >= 16 and _overlap(ao, bo)[0]
        # Reading-question stems are often conventional even when lengthy.
        # Require substantial option support for P6/P7 question overlap;
        # standalone P5 sentences can carry distinctive copied content.
        prompt_hit = a.get("part") == b.get("part") == 5 and min(len(ap), len(bp)) >= 14 and _overlap(ap, bp)[0]
        if not options_hit and not prompt_hit:
            return None
    else:
        if min(len(_tokens(a["text"])), len(_tokens(b["text"]))) < 14:
            return None
    component = "document" if a["kind"] == "document" else "question_and_options"
    raw_a, raw_b = _tokens(a["text"], False), _tokens(b["text"], False)
    raw_hit, raw_score = _overlap(raw_a, raw_b)
    skeleton_hit, skeleton_score = _overlap(_tokens(a["text"]), _tokens(b["text"]))
    if not raw_hit and not skeleton_hit:
        if a["kind"] != "question":
            return None
        component = "options" if options_hit else "prompt"
        lhs = " ".join(a["options"]) if options_hit else a["prompt"]
        rhs = " ".join(b["options"]) if options_hit else b["prompt"]
        raw_a, raw_b = _tokens(lhs, False), _tokens(rhs, False)
        raw_hit, raw_score = _overlap(raw_a, raw_b)
        skeleton_hit, skeleton_score = _overlap(_tokens(lhs), _tokens(rhs))
    # Different literal dates/times share an unavoidable answer format. Keep
    # the signal visible, but do not treat numeric masking alone as copied
    # language. Identical literal choices and semantic-template matches still
    # require review through their existing gates.
    date_time = re.compile(
        r"At\s+\d{1,2}(?::\d{2})?\s+[AP]\.?M\.?\s+on\s+"
        r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{1,2}", re.IGNORECASE)
    generic_time_format = (
        not raw_hit and a["kind"] == "question" and a.get("part") == b.get("part") == 7
        and len(a["options"]) == len(b["options"]) == 4
        and all(date_time.fullmatch(o.strip()) for o in a["options"] + b["options"]))
    return {"category": "lexical_overlap" if raw_hit else "skeleton_overlap",
            "component": component,
            "confidence": "exact" if raw_a == raw_b else "surface_overlap",
            "overlap": raw_score if raw_hit else skeleton_score,
            "requires_review": not generic_time_format,
            "note": ("Different literal date/time choices share a conventional numeric format; retained as a diagnostic. Semantic reasoning-template review remains required."
                     if generic_time_format else "Visible text/options overlap; this is an editorial review signal, not a plagiarism verdict.")}


def option_length_audit(units: list[Unit], unit_ids: list[str] | None = None) -> dict:
    """Choose uniformly among tied longest choices; all items form denominator."""
    ids = unit_ids if unit_ids is not None else [f"u{i + 1:03d}" for i in range(len(units))]
    if len(ids) != len(units) or len(set(ids)) != len(ids):
        raise ValueError("Supply one unique unit ID per unit")
    items = []
    for uid, unit in zip(ids, units):
        for q in unit.questions:
            lengths = {o.label: len(WORD.findall(o.text)) for o in q.options}
            longest = [label for label, n in lengths.items() if n == max(lengths.values())]
            credit = 1 / len(longest) if q.answer in longest else 0.0
            shortest = [label for label, n in lengths.items() if n == min(lengths.values())]
            short_credit = 1 / len(shortest) if q.answer in shortest else 0.0
            items.append({"unit_id": uid, "question_id": q.id, "part": unit.part, "lengths": lengths,
                          "longest_labels": longest, "tie_count": len(longest),
                          "shortest_labels": shortest, "shortest_tie_count": len(shortest), "shortest_expected_correct": short_credit,
                          "correct_unique_shortest": len(shortest) == 1 and q.answer in shortest,
                          "expected_correct": credit, "correct_unique_longest": len(longest) == 1 and q.answer in longest,
                          "flags": ["correct_unique_longest"] if len(longest) == 1 and q.answer in longest else []})

    def stats(rows):
        n = len(rows)
        unique = sum(r["tie_count"] == 1 for r in rows)
        correct_unique = sum(r["correct_unique_longest"] for r in rows)
        credit = sum(r["expected_correct"] for r in rows)
        short_credit = sum(r["shortest_expected_correct"] for r in rows)
        return {"item_count": n, "unique_longest_item_count": unique, "tied_longest_item_count": n - unique,
                "correct_unique_longest_count": correct_unique, "expected_correct_sum": round(credit, 8),
                "longest_strategy_expected_accuracy": round(credit / n, 8) if n else None,
                "shortest_strategy_expected_accuracy": round(short_credit / n, 8) if n else None,
                "shortest_expected_correct_sum": round(short_credit, 8),
                "unique_shortest_item_count": sum(r["shortest_tie_count"] == 1 for r in rows),
                "correct_unique_shortest_count": sum(r["correct_unique_shortest"] for r in rows),
                "correct_unique_longest_over_all": round(correct_unique / n, 8) if n else None,
                "correct_unique_longest_over_unique": round(correct_unique / unique, 8) if unique else None,
                "random_choice_baseline": 0.25 if n else None}

    return {"schema_version": VERSION, "diagnostic_only": True,
            "method": "English alphanumeric word tokens; internal apostrophe/hyphen retained. Uniform random choice among ALL tied longest or shortest options, evaluated separately. Accuracy denominator is ALL questions, including all-equal choices. No calibrated item rejection threshold.",
            "overall": stats(items), "parts": {str(p): stats([r for r in items if r["part"] == p]) for p in (5, 6, 7)}, "items": items}


class DiversityHistory:
    """Portable JSON history for offline, single-writer editorial workflows.

    Do not use concurrently as a production reservation database. Source IDs and
    hashes make bootstrap idempotent; replacing an edition/unit requires explicit
    deletion/rebuild, so older content is never silently hidden from comparison.
    """
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path is not None else None
        self.records: dict[str, dict] = {}
        if self.path and self.path.exists():
            data = json.loads(self.path.read_text())
            if data.get("schema_version") != VERSION:
                raise ValueError("Unsupported diversity history version")
            self.records = data["records"]
            # Earlier history snapshots stored the section only on the unit.
            # Hydrate it in memory so P5 stem overlap still works, without
            # rewriting the frozen snapshot merely by reading it.
            for record in self.records.values():
                for text in record["texts"]:
                    if text["kind"] == "question":
                        text.setdefault("part", record["part"])

    def export(self) -> dict:
        return {"schema_version": VERSION, "method": "Explicit reviewer annotations plus disclosed archive surface heuristics; no embeddings or semantic-equivalence proof.", "records": self.records}

    def save(self, path: Path | None = None):
        target = Path(path) if path is not None else self.path
        if target is None:
            raise ValueError("A destination path is required")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(self.export(), ensure_ascii=False, indent=2) + "\n")
        temporary.replace(target)

    @staticmethod
    def _key(edition_id: str, unit_id: str) -> str:
        if not edition_id or not unit_id or ":" in edition_id or ":" in unit_id:
            raise ValueError("Edition/unit IDs must be nonempty and cannot contain colons")
        return f"{edition_id}:{unit_id}"

    def _record(self, unit: Unit, edition_id: str, unit_id: str, annotations: list[dict] | None, *, archive: bool = False):
        key = self._key(edition_id, unit_id)
        digest = content_hash(unit)
        if key in self.records:
            if self.records[key]["content_sha256"] != digest:
                raise ValueError(f"History identity already has different content: {key}")
            if annotations is not None:
                # A content-bound reviewer sidecar can enrich an earlier
                # heuristic import of the exact same archived material.
                self.records[key]["fingerprints"] = _fingerprints(unit, annotations)
            return
        self.records[key] = {"edition_id": edition_id, "unit_id": unit_id, "part": unit.part,
                             "content_sha256": digest, "archive": archive, "scenario_summary": unit.scenario_summary,
                             "fingerprints": _fingerprints(unit, annotations), "texts": _texts(unit)}

    def bootstrap_archives(self, examples_root: Path) -> dict:
        paths = sorted(Path(examples_root).glob("full-reading-*/questions.json"))
        imported = []
        pending = []
        for path in paths:
            data = json.loads(path.read_text())
            sidecar_path = path.with_name("diversity-annotations.json")
            sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else None
            if sidecar is not None:
                if not isinstance(sidecar, dict) or not isinstance(sidecar.get("units"), dict):
                    raise ValueError(f"Invalid diversity annotation sidecar: {sidecar_path}")
                if set(sidecar["units"]) != {entry["id"] for entry in data["units"]}:
                    raise ValueError(f"Diversity sidecar must cover every archived unit exactly once: {sidecar_path}")
            for entry in data["units"]:
                unit = Unit.model_validate(entry["content"])
                annotations = None
                exceptions = []
                if sidecar is not None:
                    binding = sidecar["units"][entry["id"]]
                    if not isinstance(binding, dict) or binding.get("content_sha256") != content_hash(unit):
                        raise ValueError(f"Stale or invalid diversity sidecar content hash: {sidecar_path}:{entry['id']}")
                    annotations = binding.get("annotations")
                    errors = validate_annotations(unit, annotations)
                    if errors:
                        raise ValueError(f"Invalid archived fingerprint annotations: {sidecar_path}:{entry['id']}: {errors}")
                    exceptions = binding.get("exceptions", [])
                    if not isinstance(exceptions, list) or any(not isinstance(e, dict) or not e.get("finding_id") or e.get("question_id") not in {q.id for q in unit.questions} or not e.get("reviewer_id") or not isinstance(e.get("reason"), str) or len(e["reason"]) < 30 for e in exceptions):
                        raise ValueError(f"Invalid archived core-skill reuse exception: {sidecar_path}:{entry['id']}")
                pending.append((unit, data["run_id"], entry["id"], annotations, exceptions))
            imported.append({"edition_id": data["run_id"], "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "unit_count": len(data["units"]), "annotation_sidecar": str(sidecar_path) if sidecar is not None else None})
        # Validate all source sidecars before changing the in-memory or saved
        # history. A stale/malformed sidecar must not fall back to heuristics.
        for unit, edition_id, unit_id, annotations, exceptions in pending:
            self._record(unit, edition_id, unit_id, annotations, archive=True)
            if annotations is not None:
                self.records[self._key(edition_id, unit_id)]["editorial_exceptions"] = exceptions
        if self.path:
            self.save()
        return {"archives": imported, "record_count": len(self.records), "limitations": "Archive signatures are surface heuristics; reading operations and evidence joins remain unknown without explicit review."}

    def usage(self) -> dict[str, int]:
        counts = Counter()
        for record in self.records.values():
            for fp in record["fingerprints"].values():
                for field in ("microconstruct", "option_family", "answer_lexeme", "reading_operation", "solution_template"):
                    if fp[field] is not None:
                        counts[f"{field}:{fp[field]}"] += 1
        return dict(counts)

    def planning_context(self, part: int, limit: int = 12) -> list[dict]:
        examples = []
        for key, record in reversed(list(self.records.items())):
            if record["part"] != part:
                continue
            texts = {t["id"]: t for t in record["texts"] if t["kind"] == "question"}
            for qid, fp in record["fingerprints"].items():
                if not any(fp[k] is not None for k in ("microconstruct", "solution_template", "reading_operation")):
                    continue
                examples.append({"source": f"{key}:{qid}", "prompt": texts[qid]["prompt"], "fingerprint": fp})
                if len(examples) >= limit:
                    return examples
        return examples

    def audit_unit(self, unit: Unit, edition_id: str, unit_id: str, annotations: list[dict] | None = None,
                   exceptions: list[dict] | None = None, *, archive: bool = False) -> dict:
        key = self._key(edition_id, unit_id)
        errors = [] if archive else validate_annotations(unit, annotations)
        fingerprints = _fingerprints(unit, annotations if not errors else None)
        lexical, repetitions = [], []
        current_texts = _texts(unit)
        for old_key, record in self.records.items():
            if old_key == key:
                if record["content_sha256"] != content_hash(unit):
                    errors.append(f"History identity already has different content: {key}")
                continue
            for text in current_texts:
                for prior in record["texts"]:
                    match = _lexical_match(text, prior)
                    if match:
                        lexical.append({**match, "question_id": text["id"], "source": f"{old_key}:{prior['id']}", "prior_example": prior["text"]})
            if record["part"] != unit.part:
                continue
            for qid, fp in fingerprints.items():
                for prior_id, old in record["fingerprints"].items():
                    exact_options = fp["option_set"] == old["option_set"]
                    same_micro = fp["microconstruct"] is not None and fp["microconstruct"] == old["microconstruct"]
                    same_template = fp["solution_template"] is not None and fp["solution_template"] == old["solution_template"]
                    same_join = fp["evidence_join"] is not None and fp["evidence_join"] == old["evidence_join"]
                    same_operation = fp["reading_operation"] is not None and fp["reading_operation"] == old["reading_operation"]
                    same_family = fp["option_family"] is not None and fp["option_family"] == old["option_family"]
                    same_answer = fp["answer_lexeme"] == old["answer_lexeme"]
                    # Old archives often lack semantic annotations. A changed
                    # distractor set must not hide reuse of a lexical answer.
                    # This is a review candidate, not proof that its sense or
                    # decision rule repeats; grammatical function words recur.
                    lexical_answer_candidate = (unit.part == 5 and same_answer
                                                and next(q.kind for q in unit.questions if q.id == qid) == "vocabulary")
                    if not (same_micro or same_template or exact_options or (same_family and same_answer) or (same_join and same_operation) or lexical_answer_candidate):
                        continue
                    reviewed = fp["provenance"] == old["provenance"] == "reviewer_confirmed"
                    detailed = same_template and ((same_micro and same_family) if unit.part == 5 else (same_join and same_operation))
                    required = lexical_answer_candidate or (reviewed and detailed) or (same_micro and exact_options and same_answer)
                    prior_text = next(t for t in record["texts"] if t["kind"] == "question" and t["id"] == prior_id)
                    repetitions.append({"category": "lexical_answer_reuse_candidate" if lexical_answer_candidate and not detailed else "reasoning_template_repetition" if detailed else "microconstruct_or_option_reuse",
                                        "question_id": qid, "source": f"{old_key}:{prior_id}", "prior_example": prior_text["text"],
                                        "matching_dimensions": [name for name, match in [("microconstruct", same_micro), ("solution_template", same_template), ("evidence_join", same_join), ("reading_operation", same_operation), ("option_family", same_family), ("answer_lexeme", same_answer), ("option_set", exact_options)] if match],
                                        "confidence": "reviewer_confirmed" if reviewed else "observed_answer_only_semantics_unknown" if lexical_answer_candidate else "observed_options_with_heuristic_context" if exact_options else "heuristic",
                                        "requires_review": required, "prior_fingerprint": old,
                                        "note": "Core skills may recur. Review the specific cue/option/reasoning pattern; this is not a ban on the broad grammar skill."})
        # Same-unit textual duplicates also matter; ordinary generic question stems do not.
        for i, text in enumerate(current_texts):
            for prior in current_texts[:i]:
                match = _lexical_match(text, prior)
                if match:
                    lexical.append({**match, "question_id": text["id"], "source": f"same-unit:{prior['id']}", "prior_example": prior["text"]})
        all_findings = lexical + repetitions
        for finding in all_findings:
            stable = [key, finding["question_id"], finding["source"], finding["category"], content_hash(unit)]
            finding["finding_id"] = hashlib.sha256(json.dumps(stable).encode()).hexdigest()[:20]
            finding["disposition"] = "review_required" if finding["requires_review"] else "diagnostic"
        allowed = {f["finding_id"]: f for f in repetitions if f["requires_review"]}
        seen = set()
        for exception in exceptions or []:
            fid = exception.get("finding_id")
            f = allowed.get(fid)
            if fid in seen or f is None or exception.get("question_id") != f["question_id"] or not exception.get("reviewer_id") or len(exception.get("reason", "")) < 30:
                errors.append("Core-skill reuse exception must bind one current repetition finding/question, reviewer and concrete reason (30+ characters)")
                continue
            seen.add(fid)
            f.update(disposition="reviewed_core_skill_reuse", exception=exception)
        coverage = {"question_count": len(unit.questions), "reviewer_confirmed_count": sum(f["provenance"] == "reviewer_confirmed" for f in fingerprints.values()),
                    "unknown_by_field": {field: sum(fp[field] is None for fp in fingerprints.values()) for field in FINGERPRINT_FIELDS}}
        return {"schema_version": VERSION, "edition_id": edition_id, "unit_id": unit_id, "content_sha256": content_hash(unit),
                "annotation_errors": errors, "coverage": coverage, "fingerprints": fingerprints,
                "lexical_findings": lexical, "repetition_findings": repetitions,
                "review_required": bool(errors or any(f["disposition"] == "review_required" for f in all_findings)),
                "option_length": option_length_audit([unit], [unit_id])}

    def record_unit(self, unit: Unit, edition_id: str, unit_id: str, annotations: list[dict], exceptions: list[dict] | None = None) -> dict:
        audit = self.audit_unit(unit, edition_id, unit_id, annotations, exceptions)
        if audit["review_required"]:
            raise ValueError("Diversity review has unresolved annotation or repetition/overlap findings")
        self._record(unit, edition_id, unit_id, annotations)
        self.records[self._key(edition_id, unit_id)]["editorial_exceptions"] = exceptions or []
        if self.path:
            self.save()
        return audit

    def audit_test(self, entries: list[dict], edition_id: str, annotations_by_unit: dict[str, list[dict]] | None = None,
                   exceptions_by_unit: dict[str, list[dict]] | None = None, *, archive: bool = False) -> dict:
        """Audit existing final entries without changing history or calling models."""
        staged = DiversityHistory()
        staged.records = dict(self.records)
        annotations_by_unit = annotations_by_unit or {}
        exceptions_by_unit = exceptions_by_unit or {}
        audits, units, ids = [], [], []
        for entry in entries:
            unit = Unit.model_validate(entry["content"])
            uid = entry["id"]
            if uid in ids:
                raise ValueError("Duplicate unit ID in test")
            if entry.get("content_sha256") and entry["content_sha256"] != content_hash(unit):
                raise ValueError(f"Content hash mismatch in {uid}")
            annotations = annotations_by_unit.get(uid)
            audits.append(staged.audit_unit(unit, edition_id, uid, annotations, exceptions_by_unit.get(uid), archive=archive))
            # Invalid annotations cannot become trusted evidence for later units.
            staged._record(unit, edition_id, uid, annotations if not audits[-1]["annotation_errors"] else None, archive=archive)
            units.append(unit)
            ids.append(uid)
        return {"schema_version": VERSION, "edition_id": edition_id, "units": audits,
                "review_required": any(a["review_required"] for a in audits), "option_length": option_length_audit(units, ids)}
