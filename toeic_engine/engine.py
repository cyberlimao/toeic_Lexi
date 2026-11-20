from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Callable
from pydantic import ValidationError

from .models import BlindSolution, Challenge, ExplanationRevision, Review, Unit, blind_payload, validate_blind, validate_unit, validate_review, validate_challenge
from .novelty import History
from .provider import MODEL, OutputError, ProviderError
from .independent_audits import PROTOCOL, digest as input_digest, run_ablation, run_teaching, validate_independent
from .review_format import part7_missing_document_ids, document_id_repair_errors


def prompt(name: str) -> str:
    return files("toeic_engine").joinpath("prompts", f"{name}.txt").read_text()


PROMPT_NAMES = ("author", "solver", "challenger", "critic", "explanation_editor", "evidence_probe", "teaching_reviewer")


def prompt_version() -> str:
    return hashlib.sha256("\n".join(prompt(n) for n in PROMPT_NAMES).encode()).hexdigest()[:16]


def content_hash(unit: Unit) -> str:
    return hashlib.sha256(json.dumps(unit.model_dump(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def model_provenance(provider, role="author") -> dict:
    return dict(getattr(provider, f'{role}_provenance', {
        'model': getattr(provider, 'model', MODEL),
        **getattr(provider, 'reasoning', {'thinking_level': 'HIGH'}),
    }))


def critic_payload(plan: dict, unit: Unit, solved: BlindSolution, challenge: Challenge, detected_issues: list) -> dict:
    """Expose planner-field semantics without changing the plan or verdict.

    The planner allocates two mandatory contextual items but also uses false
    for reference slots whose realization explicitly calls for another
    sentence. Actual context dependence remains the critic's factual judgment.
    Older blueprints without this field keep their original payload shape.
    """
    payload = {"blueprint": plan, "unit": unit.model_dump(),
               "independent_solution": solved.model_dump(),
               "option_challenge": challenge.model_dump(), "detected_issues": detected_issues}
    if plan['part'] == 6 and any('requires_cross_sentence_context' in slot for slot in plan['slots']):
        payload['blueprint_field_semantics'] = {
            'version': 'part6-context-allocation-v1',
            'field': 'Part 6 blueprint.slots[].requires_cross_sentence_context',
            'true': 'This slot is explicitly allocated to the minimum set that must require context beyond its own sentence.',
            'false': 'This slot is not counted in that minimum allocation. It is not a prohibition on using neighboring sentences and does not require that the item be solvable locally.',
            'actual_review_field': 'ItemReview.requires_cross_sentence_context is a factual judgment of the realized item. Assess the actual loss of a necessary referent/chronology cue or the survival of multiple viable alternatives when neighboring sentences are hidden; do not copy the blueprint allocation flag.',
            'constraints': 'Retain the planned microconstruct, realization_strategy and reading_operation. Reference-cohesion items may require earlier context. At least two realized items must genuinely require surrounding discourse. Answer uniqueness, distractor quality and explanation accuracy still require independent assessment.',
            'neutrality': 'This clarifies metadata semantics only. It requests no particular verdict, score or answer; report any substantive content or construct defect you find.',
        }
    return payload


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


class QualityError(RuntimeError):
    pass


class Engine:
    def __init__(self, provider, history: History, output: Path, max_attempts: int = 3, progress: Callable[[str], None] = print):
        self.provider = provider
        self.history = history
        self.output = output
        self.max_attempts = max_attempts
        self.progress = progress
        self.prompts = {n: prompt(n) for n in PROMPT_NAMES}
        self.version = hashlib.sha256("\n".join(self.prompts.values()).encode()).hexdigest()[:16]

    def _review_call(self, system, payload, schema, label, max_tokens):
        """A malformed reviewer response must not rewrite a sound question set."""
        locked_part7 = None
        allowed_part7_ids = []
        for retry in range(3):
            suffix = '' if retry == 0 else f'-format{retry+1}'
            try:
                result = self.provider.generate(system, payload, schema, label + suffix, max_tokens=max_tokens)
                result = schema.model_validate(result.model_dump())
                if locked_part7 is not None:
                    errors = document_id_repair_errors(locked_part7, result, allowed_part7_ids, payload['unit'])
                    if errors:
                        save_json(self.output / 'attempts' / f'{label}{suffix}.rejected-format-review.json',
                                  {'errors': errors, 'review': result.model_dump()})
                        # Do not keep sampling after a locked judgment changes.
                        raise ProviderError(f'{label}: Part 7 metadata-only retry violated its locked review contract: {errors}')
                elif schema is Review:
                    allowed_part7_ids = part7_missing_document_ids(result, payload.get('unit', {}))
                    if allowed_part7_ids:
                        locked_part7 = result.model_copy(deep=True)
                        save_json(self.output / 'attempts' / f'{label}{suffix}.pre-format-review.json',
                                  result.model_dump())
                        payload = {**payload, 'locked_review_response': result.model_dump(),
                                   'allowed_document_id_repairs': allowed_part7_ids}
                        raise OutputError(
                            f'Part 7 review metadata omitted necessary_document_ids for {allowed_part7_ids}, '
                            'although these non-cross_document items explicitly require supplied passage context. '
                            'Return the complete locked_review_response EXACTLY unchanged except that each listed '
                            'item must receive its inclusion-minimal necessary_document_ids using only supplied document IDs. '
                            'Do not change verdict, scores, findings, acceptable answers, option checks, context '
                            'justifications, other items, or any other field. Do not repair a substantive evidence '
                            'defect or claim a document is needed when the stem/options suffice. If you cannot fill '
                            'these lists while retaining every original judgment, leave the lists empty; the draft '
                            'will remain rejected. This is not a request to make the review pass.'
                        )
                if schema is Review and payload.get('unit', {}).get('part') == 6:
                    missing = [item.question_id for item in result.item_reviews
                               if not item.necessary_document_ids]
                    if missing:
                        raise OutputError(
                            f"Part 6 evidence document IDs are missing for {missing}. "
                            "The sentence containing a blank is supplied inside its document. "
                            "Report that document when its local sentence is needed, even for "
                            "a grammar item that needs no neighboring sentence. This is separate "
                            "from requires_cross_sentence_context. Reassess the actual evidence; "
                            "do not change any question, score, finding or verdict merely to fill "
                            "this field. If genuinely answerable without any document, retain "
                            "that finding; the unit must not pass the existing quality gate."
                        )
                result._request_input_sha256 = input_digest(payload)
                result._request_label = label + suffix
                return result
            except (OutputError, ValidationError) as exc:
                save_json(self.output / 'attempts' / f'{label}{suffix}.format-error.json', {'error':str(exc)})
                payload = {**payload, 'response_format_correction':str(exc)}
        raise ProviderError(f'{label}: reviewer returned invalid structured output three times; draft retained')

    def generate_unit(self, plan: dict, unit_id: str, initial: Unit | None = None, initial_feedback: list | None = None, revises: str | None = None, explanation_only: bool = False, repair_questions: list[str] | None = None, initial_author_provenance: dict | None = None) -> dict:
        feedback: list = initial_feedback or []
        previous = initial
        author_provenance = dict(initial_author_provenance or model_provenance(self.provider))
        previous_attempts = [int(m[1]) for p in (self.output / "attempts").glob(f"{unit_id}-a*.json") if (m := re.match(rf"{unit_id}-a(\d+)", p.name))]
        offset = max(previous_attempts, default=0)
        if offset:
            draft = self.output / "attempts" / f"{unit_id}-a{offset}.json"
            rejection = self.output / "attempts" / f"{unit_id}-a{offset}.rejection.json"
            if draft.exists():
                previous = Unit.model_validate_json(draft.read_text())
            if rejection.exists():
                feedback = json.loads(rejection.read_text())
        for attempt in range(1, self.max_attempts + 1):
            label = f"{unit_id}-a{offset + attempt}"
            self.progress(f"{unit_id} Part {plan['part']} · attempt {attempt}/{self.max_attempts} · author")
            payload = {"blueprint": plan, "avoid_recent_situations": self.history.recent()}
            if feedback:
                payload["required_fixes"] = feedback
            if previous:
                payload["rejected_draft"] = previous.model_dump()
            try:
                if explanation_only:
                    if previous is None or not feedback:
                        raise ValueError("Explanation revision requires a draft and feedback")
                    targets = repair_questions or [q.id for q in previous.questions]
                    if len(set(targets)) != len(targets) or not set(targets) <= {q.id for q in previous.questions}:
                        raise ValueError("Invalid explanation repair question IDs")
                    payload['repair_question_ids'] = targets
                    replacement = self.provider.generate(self.prompts['explanation_editor'], payload, ExplanationRevision, f"{label}-explanations", max_tokens=49152 if len(targets) > 10 else 24576)
                    if [r.question_id for r in replacement.replacements] != targets:
                        raise OutputError("Explanation editor must return exactly the requested IDs in order")
                    changes = {r.question_id: r.explanation for r in replacement.replacements}
                    unit = previous.model_copy(deep=True)
                    for question in unit.questions:
                        if question.id in changes:
                            question.explanation = changes[question.id]
                elif initial is not None and attempt == 1 and not feedback:
                    unit = initial
                    self.progress(f"{unit_id} · rechecking supplied draft; no author call")
                else:
                    unit = self.provider.generate(self.prompts["author"], payload, Unit, f"{label}-author", max_tokens=65536 if plan["part"] == 5 else 32768)
                    author_provenance = model_provenance(self.provider)
                previous = unit
                save_json(self.output / "attempts" / f"{label}.json", unit.model_dump())
                feedback = validate_unit(unit, plan)
                novelty = self.history.check(unit, exclude_source=revises)
                if novelty:
                    feedback.append({"novelty_rejection": novelty})
                if feedback:
                    self._reject(label, feedback)
                    continue
                self.progress(f"{unit_id} · blind solve")
                solved = self._review_call(self.prompts["solver"], blind_payload(unit), BlindSolution, f"{label}-blind", max_tokens=32768)
                save_json(self.output / "attempts" / f"{label}.blind.json", solved.model_dump())
                feedback = validate_blind(solved, unit)
                self.progress(f"{unit_id} · answer-blind option challenge")
                challenge = self._review_call(self.prompts["challenger"], blind_payload(unit), Challenge, f"{label}-challenge", max_tokens=49152 if plan["part"] == 5 else 24576)
                save_json(self.output / "attempts" / f"{label}.challenge.json", challenge.model_dump())
                feedback.extend(validate_challenge(challenge, unit))
                if feedback:
                    self._reject(label, feedback)
                    continue
                self.progress(f"{unit_id} · editorial review")
                review = self._review_call(self.prompts["critic"], critic_payload(plan, unit, solved, challenge, feedback), Review, f"{label}-critic", max_tokens=49152 if plan["part"] == 5 else 24576)
                save_json(self.output / "attempts" / f"{label}.review.json", review.model_dump())
                editorial_errors = validate_review(review, unit)
                if editorial_errors:
                    feedback.append({"editorial_gate": editorial_errors, "editorial_review": review.model_dump()})
                if feedback:
                    self._reject(label, feedback)
                    continue
                digest = content_hash(unit)
                self.progress(f"{unit_id} · physically isolated document checks and independent teaching review")
                save_audit = lambda name, record: save_json(self.output / 'attempts' / f'{name}.json', record)
                independent = {
                    'protocol': PROTOCOL, 'content_sha256': digest,
                    'document_ablation': run_ablation(
                        unit, [q.id for q in unit.questions if q.kind == 'cross_document'],
                        self._review_call, self.prompts['evidence_probe'], label, save_audit),
                    'teaching_review': run_teaching(
                        unit, [q.id for q in unit.questions], self._review_call,
                        self.prompts['teaching_reviewer'], label, save_audit),
                }
                save_json(self.output / 'attempts' / f'{label}.independent.json', independent)
                independent_errors = validate_independent(independent, unit)
                if independent_errors:
                    feedback = [{'independent_gate': independent_errors, 'independent_audit': independent}]
                    self._reject(label, feedback)
                    continue
                entry = {"checkpoint_version": 2, "id": unit_id, "content_sha256": digest, "blueprint": plan, "content": unit.model_dump(),
                         "quality": {"status": "accepted_automated", "attempts": offset + attempt, "shape_and_evidence": "pass", "novelty": "pass", "blind_solution": solved.model_dump(), "option_challenge": challenge.model_dump(), "editorial_review": review.model_dump(), "independent_audit": independent, "external_review": None},
                         "provenance": {**author_provenance, "quality_protocol": PROTOCOL, "prompt_version": self.version, "accepted_at": datetime.now(timezone.utc).isoformat(), "revises": revises, "revision_mode": "explanations_only" if explanation_only else "full_unit"}}
                entry['provenance']['reviewer'] = model_provenance(self.provider, 'review')
                if explanation_only:
                    entry['provenance']['explanation_editor'] = model_provenance(self.provider, 'review')
                # Prepare durable output before reserving novelty; checkpoint recovers accepted work.
                save_json(self.output / "units" / f"{unit_id}.pending.json", entry)
                try:
                    self.history.accept(f"{self.output.resolve()}/{unit_id}", unit, plan, revises=revises)
                except ValueError as exc:
                    feedback = [str(exc)]
                    self._reject(label, feedback)
                    continue
                (self.output / "units" / f"{unit_id}.pending.json").replace(self.output / "units" / f"{unit_id}.json")
                self.progress(f"{unit_id} · accepted ({len(unit.questions)} questions)")
                return entry
            except OutputError as exc:
                feedback = [str(exc)]
                self._reject(label, feedback)
        raise QualityError(f"{unit_id} rejected after {self.max_attempts} attempts; see attempts/*.rejection.json. No failed content was published.")

    def _reject(self, label: str, feedback: list):
        save_json(self.output / "attempts" / f"{label}.rejection.json", feedback)
        self.progress(f"{label} · rejected: {str(feedback)[:450]}")


def validate_entry_content(entry: dict, unit: Unit | None = None) -> list[str]:
    """Check a unit's structure, evidence, and content hash without its review records."""
    if unit is None:
        unit = Unit.model_validate(entry["content"])
    errors = validate_unit(unit, entry["blueprint"])
    if content_hash(unit) != entry["content_sha256"]:
        errors.append("Content hash mismatch")
    return errors


def verify_entry(entry: dict) -> list[str]:
    unit = Unit.model_validate(entry["content"])
    errors = validate_entry_content(entry, unit)
    if entry.get('provenance', {}).get('scoped_editorial_evidence') is not None:
        from .scoped_editorial import summary_errors
        errors.extend(summary_errors(entry))
    errors += validate_blind(BlindSolution.model_validate(entry["quality"]["blind_solution"]), unit)
    errors += validate_challenge(Challenge.model_validate(entry["quality"]["option_challenge"]), unit)
    errors += validate_review(Review.model_validate(entry["quality"]["editorial_review"]), unit)
    if entry["quality"]["status"] != "accepted_automated":
        errors.append("Unit not accepted")
    protocol = entry.get('provenance', {}).get('quality_protocol')
    independent = entry['quality'].get('independent_audit')
    checkpoint_version = entry.get('checkpoint_version', 1)
    if checkpoint_version not in (1, 2):
        errors.append('Unsupported checkpoint version')
    # Legacy checkpoints retain their original certification scope. New ones
    # cannot silently lose either new gate or accept a mismatched input hash.
    if checkpoint_version == 2 or protocol is not None or independent is not None:
        if protocol != PROTOCOL or not isinstance(independent, dict):
            errors.append('Missing or unsupported independent audit protocol')
        else:
            errors.extend(validate_independent(independent, unit))
    elif entry.get('provenance', {}).get('prompt_version') == prompt_version():
        errors.append('Current prompt version requires independent audit records')
    external = entry["quality"].get("external_review")
    if external:
        if external.get("content_sha256") != entry["content_sha256"]:
            errors.append("External review applies to a different content hash")
        if external.get("verdict") != "pass":
            errors.append("External review requires revision")
    return errors


def bundle(run: dict, entries: list[dict]) -> dict:
    from .anthropic_provider import AnthropicProvider
    from .openai_provider import OpenAIProvider
    defaults = {'claude': AnthropicProvider.author_provenance, 'openai': OpenAIProvider.author_provenance}
    default_author = defaults.get(run.get('author_provider'), {'model': MODEL, 'thinking_level': 'HIGH'})
    authors = [{k: v for k, v in e['provenance'].items() if k in ('model', 'thinking_level', 'thinking', 'effort')} for e in entries]
    authors = list({json.dumps(a, sort_keys=True): a for a in authors}.values()) or [default_author]
    author = authors[0] if len(authors) == 1 else {'model': 'mixed: see unit provenance', 'author_models': authors}
    return {"schema_version": "1.0", "run_id": run["run_id"], "created_at": run["created_at"], "seed": run["seed"],
            **author, "prompt_version": run["prompt_version"],
            "status": "complete" if len(entries) == len(run["plans"]) else "partial",
            "expected_question_count": sum(p["question_count"] for p in run["plans"]), "planned_unit_count": len(run["plans"]),
            "question_count": sum(len(e["content"]["questions"]) for e in entries),
            "quality_protocols": sorted({e.get('provenance', {}).get('quality_protocol', 'legacy-self-report') for e in entries}),
            "quality_note": "Review scope and reviewer model are recorded per unit in provenance. independent-audits-v1 adds physical document removal and separate teaching review; legacy checkpoints were not retroactively checked. Automated reviewers can share blind spots. External review applies only to matching content hashes.",
            "units": entries}
