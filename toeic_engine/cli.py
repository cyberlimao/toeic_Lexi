from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from .engine import Engine, QualityError, bundle, prompt, prompt_version, save_json, verify_entry, validate_entry_content
from .anthropic_provider import AnthropicProvider, ReviewerRouter
from .openai_provider import OpenAIProvider
from .models import Unit, blind_payload
from .novelty import History
from .planning import Planner
from .provider import MODEL, ProviderError, VertexProvider
from .render import render
from .independent_audits import minimum_calls


def positive(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return n


def arguments():
    parser = argparse.ArgumentParser(description="Original TOEIC practice via Gemini HIGH, Claude Opus 5.5 xhigh, or GPT-6.1 Sol xhigh")
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "plan"):
        p = subs.add_parser(name)
        p.add_argument("--part", choices=["5", "6", "7", "full", "sample"], required=True)
        p.add_argument("--count", type=positive, help="Part 5: 1..30 (default 30); Part 7 single: 2..4 (default 3)")
        p.add_argument("--groups", type=positive, default=1)
        p.add_argument("--mode", choices=["single", "double", "triple"], default="single")
        p.add_argument("--special", choices=["sentence_insertion", "vocabulary_in_context", "intent"])
        p.add_argument("--seed", type=int)
        p.add_argument("--history", type=Path, default=Path(".toeic/history.sqlite3"))
        p.add_argument("--out", type=Path)
        if name == "generate":
            p.add_argument("--corpus", type=Path, help="Optional external comparison corpus with ten test JSON files")
            limits(p)
    p = subs.add_parser("resume")
    p.add_argument("directory", type=Path)
    limits(p)
    p = subs.add_parser("audit", help="Recheck/revise one saved unit with the current prompts; feedback forces regeneration")
    p.add_argument("directory", type=Path)
    p.add_argument("--unit", default="u001")
    p.add_argument("--attempt", type=positive, help="Use this draft from attempts instead of an accepted unit")
    p.add_argument("--feedback", type=Path, help="UTF-8 reviewer feedback; generate a full replacement before rechecking")
    p.add_argument("--explanations-only", action="store_true", help="With --feedback, preserve all student material and keys; revise explanations only")
    p.add_argument("--questions", help="Comma-separated qIDs to revise, only with --explanations-only; default all")
    p.add_argument("--out", type=Path, required=True)
    limits(p)
    p = subs.add_parser("render")
    p.add_argument("file", type=Path)
    p.add_argument("--out", type=Path)
    p = subs.add_parser("validate")
    p.add_argument("file", type=Path)
    p = subs.add_parser("schema")
    p.add_argument("--out", type=Path, default=Path("generated/unit.schema.json"))
    return parser.parse_args()


def limits(p):
    p.add_argument('--author-provider', choices=['gemini','claude','openai'], default=None, help='Author provider (new runs default to Gemini HIGH); resume/audit inherit the saved choice')
    p.add_argument('--review-provider', choices=['gemini','claude','openai'], default=None, help='Reviewer provider (new runs default to Gemini HIGH); Claude Opus 5.5 and OpenAI GPT-6.1 Sol use xhigh; resume/audit inherit the saved choice')
    p.add_argument("--max-attempts", type=positive, default=3, help="Whole-unit generation attempts; rejected drafts are retained")
    p.add_argument("--max-api-calls", type=positive, default=240, help="Hard cap including network retries, for this invocation; full test minimum is 169")
    p.add_argument("--timeout", type=positive, default=900, help="Per-request timeout, seconds; HIGH reasoning plus 30 explanations can take several minutes")


def select_providers(args, inherited):
    """Legacy runs default to Gemini; only new audit runs may override lineage."""
    selected = {}
    for role in ('author', 'review'):
        key = f'{role}_provider'
        previous = inherited.get(key, 'gemini')
        choice = getattr(args, key, None) or previous
        if choice not in ('gemini', 'claude', 'openai'):
            raise ValueError(f'Unknown {role} provider: {choice}')
        if args.command == 'resume' and choice != previous:
            raise ValueError(f'Cannot change {role} provider on resume; create a new audit run')
        selected[key] = choice
    return selected


def create_provider(author_choice, review_choice, audit_dir, max_calls, timeout):
    """Keep role logs separate while charging retries against one shared cap."""
    providers = {'gemini': VertexProvider, 'claude': AnthropicProvider, 'openai': OpenAIProvider}
    if author_choice not in providers or review_choice not in providers:
        raise ValueError('Providers must be gemini, claude or openai')
    author = providers[author_choice](audit_dir / 'author', max_calls, timeout)
    try:
        reviewer = providers[review_choice](audit_dir / 'reviewer', max_calls, timeout)
    except Exception:
        author.close()
        raise
    return ReviewerRouter(author, reviewer, prompt('author'), max_calls)


def initial_author_provenance(raw, original, is_attempt):
    """Keep a reused draft's author distinct from the new audit transports."""
    providers = {'claude': AnthropicProvider.author_provenance, 'openai': OpenAIProvider.author_provenance}
    fallback = dict(providers.get(original.get('author_provider'), {'model': MODEL, 'thinking_level': 'HIGH'}))
    if is_attempt:
        return fallback
    source = raw.get('provenance', {})
    saved = {k: source[k] for k in ('model', 'effort', 'thinking', 'thinking_level') if k in source}
    return saved if saved.get('model') else fallback


def validate_bundle(data: dict) -> list[str]:
    errors = []
    ids = [e["id"] for e in data["units"]]
    if len(ids) != len(set(ids)):
        errors.append("Duplicate unit IDs")
    if sum(len(e["content"]["questions"]) for e in data["units"]) != data["question_count"]:
        errors.append("Question total mismatch")
    if data["status"] == "complete" and (data["question_count"] != data["expected_question_count"] or len(data["units"]) != data["planned_unit_count"]):
        errors.append("Complete status does not match the planned question/unit counts")
    validate_entry = validate_entry_content if data.get("document_type") == "practice" else verify_entry
    for entry in data["units"]:
        errors.extend(f"{entry['id']}: {e}" for e in validate_entry(entry))
    return errors


def export(output: Path, run: dict, entries: list[dict]):
    data = bundle(run, entries)
    save_json(output / "questions.json", data)
    (output / "practice.html").write_text(render(data))
    save_json(output / "review-blind.json", {"run_id": run["run_id"], "units": [{"id": e["id"], "content_sha256": e["content_sha256"], "student_material": blind_payload(Unit.model_validate(e["content"]))} for e in entries]})


def generate(args):
    output = args.directory.resolve() if args.command == "resume" else (args.out or Path("generated") / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])).resolve()
    if args.command == "resume":
        run = json.loads((output / "run.json").read_text())
        if run["prompt_version"] != prompt_version():
            raise ValueError("Prompts changed since this run. Start a new run so review provenance stays consistent.")
    elif args.command == "audit":
        if args.explanations_only and not args.feedback:
            raise ValueError("--explanations-only requires --feedback")
        if args.questions and not args.explanations_only:
            raise ValueError("--questions requires --explanations-only")
        if output.exists() and any(output.iterdir()):
            raise ValueError("Use a new --out directory for the audit")
        source = args.directory.resolve()
        original = json.loads((source / "run.json").read_text())
        if not re.fullmatch(r"u\d{3}", args.unit):
            raise ValueError("Unit must be u001, u002, ...")
        index = int(args.unit[1:]) - 1
        if not 0 <= index < len(original["plans"]):
            raise ValueError("Unit is outside this run")
        input_path = source / "attempts" / f"{args.unit}-a{args.attempt}.json" if args.attempt else source / "units" / f"{args.unit}.json"
        raw = json.loads(input_path.read_text())
        initial = Unit.model_validate(raw if args.attempt else raw["content"])
        # A rejected revision has no accepted-history row of its own yet.
        # Preserve its accepted ancestor instead of losing the revision lineage.
        lineage = History(Path(original['history']))
        try:
            from .engine import content_hash
            revises = lineage.revision_parent(f"{source}/{args.unit}", content_hash(initial), original.get('revises'))
        finally:
            lineage.close()
        run = {"run_id": output.name, "created_at": datetime.now(timezone.utc).isoformat(), "seed": original["seed"], "prompt_version": prompt_version(), "part": "audit",
               "history": original["history"], "corpus": original.get("corpus"), "plans": [original["plans"][index]], "initial_draft": initial.model_dump(),
               "initial_feedback": [args.feedback.read_text()] if args.feedback else [], "revises": revises, "audited_draft_source":str(input_path),
               "initial_author_provenance": initial_author_provenance(raw, original, bool(args.attempt)),
               "explanation_only": args.explanations_only, "repair_questions": [q.strip() for q in args.questions.split(',')] if args.questions else None,
               "author_provider": original.get('author_provider', 'gemini'), "review_provider": original.get('review_provider', 'gemini')}
        output.mkdir(parents=True, exist_ok=True)
        save_json(output / "run.json", run)
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output directory is not empty. Use a new --out or resume DIRECTORY.")
        output.mkdir(parents=True, exist_ok=True)
        corpus = args.corpus.resolve() if args.corpus else None
        history = History(args.history.resolve())
        try:
            if corpus is not None:
                history.import_corpus(corpus)
            planner = Planner(args.seed, history.usage())
            plans = planner.build(args.part, args.count, args.groups, args.mode, args.special)
        finally:
            history.close()
        run = {"run_id": output.name, "created_at": datetime.now(timezone.utc).isoformat(), "seed": planner.seed, "prompt_version": prompt_version(), "part": args.part,
               "history": str(args.history.resolve()), "corpus": str(corpus) if corpus is not None else None, "plans": plans}
        save_json(output / "run.json", run)
    selections = select_providers(args, run)
    run.update(selections)
    save_json(output / 'run.json', run)
    print(f"Output: {output}", flush=True)
    history = History(Path(run["history"]))
    provider = None
    entries = []
    try:
        if run.get("corpus"):
            history.import_corpus(Path(run["corpus"]))
        invocation = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
        provider = create_provider(selections['author_provider'], selections['review_provider'],
                                   output / 'api' / invocation, args.max_api_calls, args.timeout)
        engine = Engine(provider, history, output, args.max_attempts, lambda s: print(s, flush=True))
        for i, plan in enumerate(run["plans"], 1):
            unit_id = f"u{i:03d}"
            checkpoint = output / "units" / f"{unit_id}.json"
            pending = output / "units" / f"{unit_id}.pending.json"
            if not checkpoint.exists() and pending.exists():
                prepared = json.loads(pending.read_text())
                if prepared["blueprint"] == plan and not verify_entry(prepared) and history.has_accepted(f"{output}/{unit_id}", prepared["content_sha256"]):
                    pending.replace(checkpoint)
            if checkpoint.exists():
                entry = json.loads(checkpoint.read_text())
                if entry["blueprint"] != plan or verify_entry(entry):
                    raise ValueError(f"Invalid accepted checkpoint {checkpoint}")
                print(f"{unit_id} · using accepted checkpoint", flush=True)
            else:
                entry = engine.generate_unit(plan, unit_id, Unit.model_validate(run["initial_draft"]) if "initial_draft" in run else None, run.get("initial_feedback"), run.get("revises"), run.get("explanation_only", False), run.get("repair_questions"), initial_author_provenance=run.get('initial_author_provenance'))
            if run.get('editorial_preparation'):
                entry['provenance']['editorial_preparation'] = run['editorial_preparation']
                save_json(checkpoint, entry)
            entries.append(entry)
            export(output, run, entries)
        print(f"Complete: {sum(len(e['content']['questions']) for e in entries)} questions · {provider.calls} API calls", flush=True)
        return 0
    finally:
        if entries:
            export(output, run, entries)
        history.close()
        if provider:
            provider.close()


def main():
    args = arguments()
    try:
        if args.command in ("generate", "plan"):
            if args.special and (args.part != "7" or args.mode != "single"):
                raise ValueError("--special applies to Part 7 single-passage groups")
            args.count = args.count if args.count is not None else (30 if args.part == "5" else 3)
            if args.part == "7" and args.mode == "single" and args.count not in (2, 3, 4):
                raise ValueError("Part 7 single-passage --count must be 2, 3 or 4")
        if args.command in ("generate", "resume", "audit"):
            return generate(args)
        if args.command == "plan":
            history = History(args.history)
            try:
                planner = Planner(args.seed, history.usage())
                plans = planner.build(args.part, args.count, args.groups, args.mode, args.special)
            finally:
                history.close()
            data = {"seed": planner.seed, "plans": plans, "question_count": sum(p["question_count"] for p in plans), "minimum_api_calls": minimum_calls(plans)}
            if args.out:
                save_json(args.out, data)
                print(args.out.resolve())
            else:
                print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        if args.command == "schema":
            save_json(args.out, Unit.model_json_schema())
            print(args.out.resolve())
            return 0
        data = json.loads(args.file.read_text())
        errors = validate_bundle(data)
        if errors:
            raise ValueError("\n".join(errors))
        if args.command == "render":
            target = args.out or args.file.with_suffix(".html")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(render(data))
            print(target.resolve())
        else:
            print(f"Valid: {len(data['units'])} units / {data['question_count']} questions / {data['status']}")
        return 0
    except (ValueError, ValidationError, QualityError, ProviderError, OSError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
