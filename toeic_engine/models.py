from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

Label = Literal["A", "B", "C", "D"]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    # Runtime provenance only; never part of a schema, prompt, or content hash.
    _request_input_sha256: str | None = PrivateAttr(default=None)
    _request_label: str | None = PrivateAttr(default=None)


class Option(Model):
    label: Label
    text: str = Field(min_length=1)


class Evidence(Model):
    document_id: str
    block_id: str
    quote: str = Field(min_length=1, description="Exact contiguous quotation from the cited block, retaining blank markers.")


class OptionReason(Model):
    label: Label
    reason_zh: str = Field(min_length=8, description="Specific Traditional Chinese explanation of why this option works or fails here.")


class Explanation(Model):
    summary_zh: str = Field(min_length=10)
    reasoning_zh: str = Field(min_length=30, description="Concise teachable solution, not private chain of thought; grammar or evidence-to-answer reasoning.")
    option_analysis: list[OptionReason] = Field(min_length=4, max_length=4)
    evidence: list[Evidence]
    takeaway_zh: str = Field(min_length=8)


class Block(Model):
    id: str = Field(min_length=1)
    type: Literal["heading", "field", "paragraph", "list_item", "table", "message"]
    text: str
    speaker: str = ""
    timestamp: str = ""
    columns: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_block(self):
        if self.type == "table":
            if not self.columns or not self.rows or any(len(r) != len(self.columns) for r in self.rows):
                raise ValueError("Table must have columns and rectangular nonempty rows")
        elif not self.text:
            raise ValueError("Non-table blocks require text")
        if self.type == "message" and not self.speaker:
            raise ValueError("Messages require speakers")
        return self

    def readable(self) -> str:
        return "\n".join([self.text, " | ".join(self.columns), *(" | ".join(r) for r in self.rows)]).strip()


class Document(Model):
    id: str
    genre: str
    title: str
    blocks: list[Block] = Field(min_length=1)


class Question(Model):
    id: str = Field(description="q1, q2, ... in order, scoped to this unit")
    kind: Literal["grammar", "vocabulary", "text_completion", "sentence_completion", "purpose", "detail", "inference", "vocabulary_in_context", "intent", "sentence_insertion", "cross_document"]
    skill: str
    difficulty: Literal["easy", "medium", "hard"]
    prompt: str = Field(min_length=1)
    options: list[Option] = Field(min_length=4, max_length=4)
    answer: Label
    target_document_id: str = ""
    target_text: str = Field(default="", description="Quoted target word/utterance, or sentence to insert; empty if not applicable")
    explanation: Explanation

    @model_validator(mode="after")
    def choices(self):
        if [o.label for o in self.options] != list("ABCD"):
            raise ValueError("Options must be in A/B/C/D order")
        if len({o.text.casefold() for o in self.options}) != 4:
            raise ValueError("Options must be distinct")
        if sorted(r.label for r in self.explanation.option_analysis) != list("ABCD"):
            raise ValueError("Explain each of A, B, C, D exactly once")
        return self


class Unit(Model):
    part: Literal[5, 6, 7]
    title: str
    scenario_summary: str = Field(description="English description of the unique situation, conflict and resolution for future diversity planning")
    documents: list[Document]
    questions: list[Question] = Field(min_length=1, max_length=30)


class ExplanationReplacement(Model):
    question_id: str
    explanation: Explanation


class ExplanationRevision(Model):
    replacements: list[ExplanationReplacement]


class BlindAnswer(Model):
    question_id: str
    answer: Label | None
    defensible_answers: list[Label] = Field(max_length=4, description="Empty if none works; answer must then be null")
    rationale_zh: str = Field(min_length=15)
    evidence: list[Evidence]


class BlindSolution(Model):
    answers: list[BlindAnswer]


class ChallengeOption(Model):
    label: Label
    viable: bool
    uncertain: bool
    test: str = Field(min_length=15)


class ChallengeItem(Model):
    question_id: str
    options: list[ChallengeOption] = Field(min_length=4, max_length=4)


class Challenge(Model):
    items: list[ChallengeItem]


class Finding(Model):
    severity: Literal["minor", "major", "critical"]
    question_id: str = Field(description="qN or unit")
    category: Literal["ambiguity", "answer", "evidence", "naturalness", "explanation", "authenticity", "distractor", "coherence", "difficulty"]
    problem: str
    fix: str


class OptionCheck(Model):
    label: Label
    defensible: bool
    explanation_supported: bool
    note: str = Field(min_length=8, description="Check this option in the actual sentence/text and the factual accuracy of its Chinese explanation")


class ItemReview(Model):
    question_id: str
    acceptable_answers: list[Label] = Field(description="All options that work under a reasonable reading; empty if none")
    strongest_rival: Label
    rival_test: str = Field(min_length=20, description="Brief concrete counterexample/substitution test for the strongest wrong option; do not invent an absolute dictionary rule")
    distractor_quality: int = Field(ge=1, le=5)
    explanation_accuracy: int = Field(ge=1, le=5)
    requires_cross_sentence_context: bool
    context_justification: str
    necessary_document_ids: list[str] = Field(description="One inclusion-minimal sufficient document set: this set establishes the answer, but removing any member from this set does not. Alternative minimal sets are allowed; do not take their intersection. Independently check for any zero/one-document shortcut. Empty for Part 5.")
    option_checks: list[OptionCheck] = Field(min_length=4, max_length=4)


class Review(Model):
    verdict: Literal["pass", "revise"]
    naturalness: int = Field(ge=1, le=5)
    toeic_alignment: int = Field(ge=1, le=5)
    explanation_quality: int = Field(ge=1, le=5)
    findings: list[Finding]
    item_reviews: list[ItemReview]

    def passes(self) -> bool:
        return self.verdict == "pass" and min(self.naturalness, self.toeic_alignment, self.explanation_quality) >= 4 and not any(f.severity != "minor" for f in self.findings)


def blind_payload(unit: Unit) -> dict:
    """Remove every author-supplied key, skill and rationale from solver input."""
    return {"part": unit.part, "documents": [d.model_dump() for d in unit.documents], "questions": [{"id": q.id, "prompt": q.prompt, "options": [o.model_dump() for o in q.options]} for q in unit.questions]}


def check_evidence(evidence: list[Evidence], unit: Unit) -> list[str]:
    blocks = {(d.id, b.id): b.readable() for d in unit.documents for b in d.blocks}
    errors = []
    for e in evidence:
        source = blocks.get((e.document_id, e.block_id))
        if source is None or e.quote not in source:
            errors.append(f"Invalid evidence {e.document_id}/{e.block_id}: quote is not an exact substring")
    return errors


def validate_review(review: Review, unit: Unit) -> list[str]:
    errors = []
    if not review.passes():
        errors.append("Editorial scores/verdict/findings fail the quality threshold")
    if [r.question_id for r in review.item_reviews] != [q.id for q in unit.questions]:
        return errors + ["Editorial review must explicitly assess each question exactly once in order"]
    doc_ids = {d.id for d in unit.documents}
    covered = set()
    for r, q in zip(review.item_reviews, unit.questions):
        if r.acceptable_answers != [q.answer]:
            errors.append(f"{q.id}: editorial acceptable answers {r.acceptable_answers} do not establish the unique key")
        if min(r.distractor_quality, r.explanation_accuracy) < 4:
            errors.append(f"{q.id}: distractors or explanation need substantive revision")
        if [o.label for o in r.option_checks] != list("ABCD"):
            errors.append(f"{q.id}: review each option A/B/C/D exactly once")
        if [o.label for o in r.option_checks if o.defensible] != [q.answer] or not all(o.explanation_supported for o in r.option_checks):
            errors.append(f"{q.id}: option-level audit found ambiguity or an unsupported teaching claim")
        if r.strongest_rival == q.answer:
            errors.append(f"{q.id}: strongest rival must challenge the key, not repeat it")
        needed = set(r.necessary_document_ids)
        if not needed <= doc_ids or (unit.part != 5 and not needed):
            errors.append(f"{q.id}: invalid/minimal evidence document list")
        if q.kind == "cross_document" and len(needed) < 2:
            errors.append(f"{q.id}: hiding a document does not change answerability; not a genuine cross-document question")
        covered |= needed
        if r.requires_cross_sentence_context and len(r.context_justification) < 20:
            errors.append(f"{q.id}: cross-sentence dependence needs a concrete justification")
    if unit.part == 6 and sum(r.requires_cross_sentence_context for r in review.item_reviews) < 2:
        errors.append("Part 6 needs at least two items that genuinely require context beyond their own sentence")
    if unit.part == 7 and covered != doc_ids:
        errors.append("Every document must contribute necessary evidence to at least one question")
    return errors


def validate_unit(unit: Unit, plan: dict) -> list[str]:
    errors: list[str] = []
    expected_ids = [f"q{i+1}" for i in range(plan["question_count"])]
    if unit.part != plan["part"] or [q.id for q in unit.questions] != expected_ids:
        errors.append("Part or ordered question IDs/count do not match blueprint")
    if len(unit.documents) != len(plan["genres"]):
        errors.append("Wrong document count; only one passage set per call is allowed")
    if len({d.id for d in unit.documents}) != len(unit.documents):
        errors.append("Duplicate document ID")
    if len({b.id for d in unit.documents for b in d.blocks}) != sum(len(d.blocks) for d in unit.documents):
        errors.append("Block IDs must be unique in the unit")
    for d, genre in zip(unit.documents, plan["genres"]):
        if d.genre != genre:
            errors.append(f"Expected genre {genre}, got {d.genre}")
    text = "\n".join(b.readable() for d in unit.documents for b in d.blocks)
    blanks = re.findall(r"\{\{(q\d+)\}\}", text)
    if unit.part == 6 and blanks != expected_ids:
        errors.append("Part 6 must contain each of its four {{qN}} blanks exactly once in order")
    if unit.part == 7 and blanks:
        errors.append("Part 7 cannot contain completion blanks")
    if unit.part != 5:
        wc = sum(any(c.isalnum() for c in word) for word in text.split())
        low, high = plan["word_range"]
        if not low <= wc <= high:
            errors.append(f"Passage word count {wc} outside {low}..{high}")
    for q, slot in zip(unit.questions, plan["slots"]):
        if q.skill != slot["skill"] or q.difficulty != slot["difficulty"]:
            errors.append(f"{q.id}: skill/difficulty must match blueprint")
        if q.kind != slot["kind"]:
            errors.append(f"{q.id}: kind must be {slot['kind']}")
        if unit.part == 5 and re.findall(r"\{\{(q\d+)\}\}", q.prompt) != [q.id]:
            errors.append(f"{q.id}: Part 5 stem needs exactly one matching blank")
        if unit.part != 5 and not q.explanation.evidence:
            errors.append(f"{q.id}: passage-based explanation needs evidence")
        errors.extend(f"{q.id}: {e}" for e in check_evidence(q.explanation.evidence, unit))
        if q.kind == "cross_document" and len({e.document_id for e in q.explanation.evidence}) < 2:
            errors.append(f"{q.id}: cross-document answer must cite at least two documents")
        if q.kind in ("sentence_insertion", "vocabulary_in_context", "intent"):
            target = next((d for d in unit.documents if d.id == q.target_document_id), None)
            # Quoting an utterance inside a question often omits its final period.
            # Keep source matching exact; tolerate only terminal sentence punctuation
            # in the question's quotation, not paraphrases or changed wording.
            quoted_target = q.target_text.rstrip('.!?')
            if target is None or not quoted_target or (q.target_text not in q.prompt and quoted_target not in q.prompt):
                errors.append(f"{q.id}: target document and quoted target in prompt required")
            elif q.kind == "sentence_insertion":
                target_source = "\n".join(b.readable() for b in target.blocks)
                marks = re.findall(r"\[\[([1-4])\]\]", target_source)
                if marks != list("1234") or [o.text for o in q.options] != [f"[{n}]" for n in range(1, 5)]:
                    errors.append(f"{q.id}: insertion needs ordered [[1]]..[[4]] and options [1]..[4]")
                if q.target_text in target_source:
                    errors.append(f"{q.id}: insertion sentence is already in the passage")
            elif q.target_text not in "\n".join(b.readable() for b in target.blocks):
                errors.append(f"{q.id}: target phrase missing from passage")
    if unit.part == 6 and sum(q.kind == "sentence_completion" for q in unit.questions) != 1:
        errors.append("Part 6 needs exactly one whole-sentence completion")
    return errors


def validate_blind(solution: BlindSolution, unit: Unit) -> list[str]:
    errors = []
    if [s.question_id for s in solution.answers] != [q.id for q in unit.questions]:
        return ["Blind solver did not answer every question exactly once in order"]
    for solved, q in zip(solution.answers, unit.questions):
        if solved.answer != q.answer or solved.defensible_answers != [q.answer]:
            errors.append(f"{q.id}: author={q.answer}, blind={solved.answer}, defensible={solved.defensible_answers}")
        if unit.part != 5 and not solved.evidence:
            errors.append(f"{q.id}: blind solver lacks evidence")
        errors.extend(check_evidence(solved.evidence, unit))
    return errors


def validate_challenge(challenge: Challenge, unit: Unit) -> list[str]:
    if [i.question_id for i in challenge.items] != [q.id for q in unit.questions]:
        return ["Ambiguity challenger must test every question in order"]
    errors = []
    for item, q in zip(challenge.items, unit.questions):
        if [o.label for o in item.options] != list("ABCD"):
            errors.append(f"{q.id}: challenger must test each option once")
        viable = [o.label for o in item.options if o.viable]
        if viable != [q.answer] or any(o.uncertain for o in item.options):
            errors.append(f"{q.id}: adversarial viable answers={viable}; tests={item.model_dump()}")
    return errors
