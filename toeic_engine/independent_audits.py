"""Independent, bounded evidence-removal and teaching-quality checks.

The omitted documents never enter the model request. Expected keys and previous
review opinions are excluded from ablation calls. Explanation reviews receive
the authored teaching text, but no solver, challenger, critic or external review.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from typing import Literal

from pydantic import Field, model_validator

from .models import Evidence, Label, Model, Unit, blind_payload, check_evidence

PROTOCOL = "independent-audits-v1"
EXPLANATION_BATCH_SIZE = 5
TEACHING_FIELDS = ["summary_zh", "reasoning_zh", "A", "B", "C", "D", "evidence", "takeaway_zh"]


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


class EvidenceOption(Model):
    label: Label
    status: Literal["supported", "contradicted", "not_established", "uncertain"]
    reason: str = Field(min_length=10)
    evidence: list[Evidence]


class EvidenceDecision(Model):
    question_id: str
    decision: Literal["sufficient", "insufficient", "uncertain"]
    answer: Label | None
    options: list[EvidenceOption] = Field(min_length=4, max_length=4)
    justification: str = Field(min_length=20)

    @model_validator(mode="after")
    def unique_options(self):
        if sorted(o.label for o in self.options) != list("ABCD"):
            raise ValueError("Evidence review must cover A/B/C/D exactly once")
        self.options.sort(key=lambda o: o.label)
        return self


class TeachingField(Model):
    field: Literal["summary_zh", "reasoning_zh", "A", "B", "C", "D", "evidence", "takeaway_zh"]
    verdict: Literal["pass", "revise", "uncertain"]
    note: str = Field(min_length=10)


class TeachingIssue(Model):
    field: Literal["summary_zh", "reasoning_zh", "A", "B", "C", "D", "evidence", "takeaway_zh"]
    claim: str = Field(min_length=1, description="Exact problematic authored substring, not a paraphrase")
    category: Literal["grammar", "unsupported_fact", "scope", "logic", "translation", "evidence", "other"]
    problem: str = Field(min_length=15)
    correction: str = Field(min_length=10)


class TeachingItem(Model):
    question_id: str
    verdict: Literal["pass", "revise", "uncertain"]
    fields: list[TeachingField] = Field(min_length=8, max_length=8)
    issues: list[TeachingIssue]

    @model_validator(mode="after")
    def unique_fields(self):
        if sorted(f.field for f in self.fields) != sorted(TEACHING_FIELDS):
            raise ValueError("Teaching review must cover all eight fields exactly once")
        self.fields.sort(key=lambda f: TEACHING_FIELDS.index(f.field))
        return self


class TeachingReview(Model):
    items: list[TeachingItem]

    @model_validator(mode="after")
    def unique_questions(self):
        if len({i.question_id for i in self.items}) != len(self.items):
            raise ValueError("Teaching review must not duplicate a question")
        return self


def evidence_schema(question_id: str):
    """Bind response identity locally, without leaking a key into the schema."""
    class RequestedEvidenceDecision(EvidenceDecision):
        @model_validator(mode="after")
        def requested_question(self):
            if self.question_id != question_id:
                raise ValueError(f"Evidence review must cover question {question_id}")
            return self
    return RequestedEvidenceDecision


def teaching_schema(question_ids: list[str]):
    requested = tuple(question_ids)
    class RequestedTeachingReview(TeachingReview):
        @model_validator(mode="after")
        def requested_questions(self):
            if sorted(i.question_id for i in self.items) != sorted(requested):
                raise ValueError(f"Teaching review must cover exactly {list(requested)}")
            self.items.sort(key=lambda item: requested.index(item.question_id))
            return self
    return RequestedTeachingReview


def document_subsets(unit: Unit) -> list[list[str]]:
    """All subsets plus a full-evidence positive control; 4/8 for double/triple."""
    ids = [d.id for d in unit.documents]
    if len(ids) not in (2, 3):
        raise ValueError("Document ablation requires a double/triple passage")
    return [list(c) for size in range(len(ids) + 1) for c in itertools.combinations(ids, size)]


def ablation_payload(unit: Unit, qid: str, visible: list[str]) -> dict:
    q = next(q for q in unit.questions if q.id == qid)
    ids = [d.id for d in unit.documents]
    if len(set(visible)) != len(visible) or not set(visible) <= set(ids):
        raise ValueError("Invalid visible document IDs")
    # Strict allowlist: no unit title/summary, key, kind, skill, difficulty,
    # explanation, other questions, removed document titles or prior results.
    return {"part": unit.part,
            "documents": [d for d in blind_payload(unit)["documents"] if d["id"] in visible],
            "question": {"id": q.id, "prompt": q.prompt,
                         "options": [o.model_dump() for o in q.options]}}


def explanation_payload(unit: Unit, qids: list[str]) -> dict:
    if not qids or len(qids) > EXPLANATION_BATCH_SIZE or len(set(qids)) != len(qids):
        raise ValueError("Explanation batches must contain one to five unique questions")
    by_id = {q.id: q for q in unit.questions}
    return {"part": unit.part, "documents": [d.model_dump() for d in unit.documents],
            "questions": [{"id": q.id, "prompt": q.prompt,
                           "options": [o.model_dump() for o in q.options],
                           "answer": q.answer, "explanation": q.explanation.model_dump()}
                          for q in (by_id[qid] for qid in qids)]}


def evidence_errors(result: EvidenceDecision, unit: Unit, qid: str, visible: list[str]) -> list[str]:
    errors = []
    if result.question_id != qid or [o.label for o in result.options] != list("ABCD"):
        errors.append("Evidence audit must cover the exact question and A/B/C/D in order")
    supported = [o.label for o in result.options if o.status == "supported"]
    if result.decision == "sufficient":
        if result.answer is None or supported != [result.answer] or any(o.status == 'uncertain' for o in result.options):
            errors.append("Sufficient decision lacks a unique supported answer")
    elif result.answer is not None or len(supported) == 1:
        errors.append("Insufficient/uncertain decision conflicts with answer/support fields")
    reduced = unit.model_copy(update={"documents": [d for d in unit.documents if d.id in visible]})
    for option in result.options:
        errors.extend(check_evidence(option.evidence, reduced))
    return errors


def teaching_errors(result: TeachingReview, unit: Unit, qids: list[str]) -> list[str]:
    errors = []
    if [i.question_id for i in result.items] != qids:
        errors.append("Teaching audit must cover every requested question exactly once in order")
    questions = {q.id: q for q in unit.questions}
    for item in result.items:
        if item.question_id not in questions:
            continue
        if [f.field for f in item.fields] != TEACHING_FIELDS:
            errors.append(f"{item.question_id}: incomplete teaching field coverage")
        if item.verdict != 'pass' or item.issues or any(f.verdict != 'pass' for f in item.fields):
            errors.append(f"{item.question_id}: teaching review requires correction or resolution")
        # A finding must quote actual learner text. It is never allowed to
        # manufacture a defect in a reviewer/blueprint rather than that text.
        explanation = questions[item.question_id].explanation
        authored = {k: getattr(explanation, k) for k in ['summary_zh', 'reasoning_zh', 'takeaway_zh']}
        authored.update({o.label: o.reason_zh for o in explanation.option_analysis})
        authored['evidence'] = '\n'.join(e.quote for e in explanation.evidence)
        for issue in item.issues:
            if issue.claim not in authored[issue.field]:
                errors.append(f"{item.question_id}: finding does not quote authored teaching text")
    return errors


def run_ablation(unit: Unit, qids: list[str], call, system: str, label: str, save) -> list[dict]:
    records = []
    for qid in qids:
        for visible in document_subsets(unit):
            payload = ablation_payload(unit, qid, visible)
            request_label = f"{label}-ablation-{qid}-{'-'.join(visible) or 'none'}"
            result = call(system, payload, evidence_schema(qid), request_label, max_tokens=8192)
            record = {"question_id": qid, "visible_document_ids": visible,
                      "input_sha256": digest(payload),
                      "request_input_sha256": result._request_input_sha256 or digest(payload),
                      "base_request_label": request_label,
                      "request_label": result._request_label or request_label,
                      "result": result.model_dump()}
            records.append(record)
            save(request_label, record)
    return records


def run_teaching(unit: Unit, qids: list[str], call, system: str, label: str, save) -> list[dict]:
    records = []
    for start in range(0, len(qids), EXPLANATION_BATCH_SIZE):
        batch = qids[start:start + EXPLANATION_BATCH_SIZE]
        payload = explanation_payload(unit, batch)
        request_label = f"{label}-teaching-{start // EXPLANATION_BATCH_SIZE + 1}"
        result = call(system, payload, teaching_schema(batch), request_label, max_tokens=16384)
        record = {"question_ids": batch, "input_sha256": digest(payload),
                  "request_input_sha256": result._request_input_sha256 or digest(payload),
                  "base_request_label": request_label,
                  "request_label": result._request_label or request_label,
                  "result": result.model_dump()}
        records.append(record)
        save(request_label, record)
    return records


def validate_ablation(records: list[dict], unit: Unit, qids: list[str]) -> list[str]:
    expected = [(qid, ids) for qid in qids for ids in document_subsets(unit)] if qids else []
    actual = [(r['question_id'], r['visible_document_ids']) for r in records]
    if actual != expected:
        return ["Missing, duplicate or out-of-order physical document-removal checks"]
    errors = []
    questions = {q.id: q for q in unit.questions}
    for record in records:
        qid, visible = record['question_id'], record['visible_document_ids']
        if record['input_sha256'] != digest(ablation_payload(unit, qid, visible)):
            errors.append(f"{qid}: evidence audit input hash mismatch")
        result = EvidenceDecision.model_validate(record['result'])
        errors.extend(evidence_errors(result, unit, qid, visible))
        if result.decision == 'uncertain' or any(o.status == 'uncertain' for o in result.options):
            errors.append(f"{qid}: document-removal verdict uncertain for {visible}")
        if len(visible) == len(unit.documents) and (result.decision != 'sufficient' or result.answer != questions[qid].answer):
            errors.append(f"{qid}: full-evidence positive control does not establish the key")
        # Removing an update/correction can make an obsolete distractor appear
        # uniquely supported. That is a dependency on the omitted document,
        # not a way to solve the complete question. Preserve the raw decision
        # in the audit, but only the complete item's key counts as a shortcut.
        # The full-evidence positive control above must still establish the key.
        if (result.decision == 'sufficient'
                and result.answer == questions[qid].answer and len(visible) < 2):
            errors.append(f"{qid}: solvable with {visible or 'stem/options only'}; cross-document shortcut")
    return errors


def validate_teaching(records: list[dict], unit: Unit, qids: list[str]) -> list[str]:
    expected = [qids[i:i + EXPLANATION_BATCH_SIZE] for i in range(0, len(qids), EXPLANATION_BATCH_SIZE)]
    if [r['question_ids'] for r in records] != expected:
        return ["Missing, duplicate or out-of-order independent teaching checks"]
    errors = []
    for record in records:
        ids = record['question_ids']
        if record['input_sha256'] != digest(explanation_payload(unit, ids)):
            errors.append("Teaching audit input hash mismatch")
        errors.extend(teaching_errors(TeachingReview.model_validate(record['result']), unit, ids))
    return errors


def validate_independent(audit: dict, unit: Unit) -> list[str]:
    if audit.get('protocol') != PROTOCOL or audit.get('content_sha256') != digest(unit.model_dump()):
        return ["Independent audit protocol/content hash mismatch"]
    qids = [q.id for q in unit.questions]
    cross = [q.id for q in unit.questions if q.kind == 'cross_document']
    return (validate_ablation(audit.get('document_ablation', []), unit, cross)
            + validate_teaching(audit.get('teaching_review', []), unit, qids))


def minimum_calls(plans: list[dict]) -> int:
    return sum(4 + math.ceil(p['question_count'] / EXPLANATION_BATCH_SIZE)
               + sum(s['kind'] == 'cross_document' for s in p['slots']) * 2 ** len(p['genres'])
               for p in plans)
