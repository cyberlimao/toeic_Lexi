"""Prepare answer-blind packets and assemble a hash-bound, externally reviewed full test.

No model calls are made here. All released checkpoints must already pass the
engine's checks and have matching, complete GPT-6-astra review records at the
external reasoning effort declared by the source run (historically high).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from string import Template

from toeic_engine.cli import validate_bundle
from toeic_engine.engine import bundle, save_json, verify_entry
from toeic_engine.models import Unit, blind_payload
from toeic_engine.render import render


DEFAULT_TITLE = '完整閱讀回 01'
DEFAULT_LANDING_PATH = '../../full-test.html'


def read(path):
    return json.loads(Path(path).read_text())


def packets(run_path: Path, out: Path, release_unit: str | None = None, start_number: int = 101):
    run = read(run_path / 'run.json')
    if release_unit and len(run['plans']) != 1:
        raise ValueError('A release unit override requires a single-unit audit run')
    number = start_number
    count = 0
    for i, plan in enumerate(run['plans'], 1):
        uid = f'u{i:03d}'
        checkpoint = run_path / 'units' / f'{uid}.json'
        if checkpoint.exists():
            entry = read(checkpoint)
            if errors := verify_entry(entry):
                raise ValueError(f'{uid}: {errors}')
            uid = release_unit or uid
            payload = {'unit_id':uid, 'content_sha256':entry['content_sha256'],
                       'display_numbers':{q['id']:number+j for j,q in enumerate(entry['content']['questions'])},
                       'student_material':blind_payload(Unit.model_validate(entry['content']))}
            target = out / f"{uid}-{entry['content_sha256'][:12]}.blind.json"
            if target.exists() and read(target) != payload:
                raise ValueError(f'Existing packet changed: {target}')
            if not target.exists():
                save_json(target, payload)
            count += 1
        number += plan['question_count']
    print(f'{count}/{len(run["plans"])} accepted units have answer-blind packets: {out}')


def external_identity(run: dict) -> tuple[str, str]:
    """Require the source run's external actor; retain historical defaults."""
    model = run.get('external_review_model', 'gpt-6-astra')
    effort = run.get('external_review_effort', 'high')
    if model != 'gpt-6-astra' or effort not in ('high', 'xhigh'):
        raise ValueError('Unsupported external review model/effort identity')
    return model, effort


def validate_external(entry: dict, uid: str, review: dict, *, model='gpt-6-astra', effort='high'):
    external_identity({'external_review_model': model, 'external_review_effort': effort})
    if review.get('unit_id') != uid or review.get('content_sha256') != entry['content_sha256']:
        raise ValueError(f'{uid}: stale or mismatched external review')
    if (review.get('model'), review.get('reasoning_effort'), review.get('verdict')) != (model, effort, 'pass'):
        raise ValueError(f'{uid}: a passing {model}/{effort} review is required')
    if review.get('issues'):
        raise ValueError(f'{uid}: unresolved review issues')
    questions = entry['content']['questions']
    rows = review.get('items', [])
    if [r.get('question_id') for r in rows] != [q['id'] for q in questions]:
        raise ValueError(f'{uid}: external review must cover every question in order')
    for q, row in zip(questions, rows):
        if row.get('verdict') != 'pass' or row.get('independent_answer') != q['answer'] or row.get('correct_answer') != q['answer']:
            raise ValueError(f"{uid}/{q['id']}: answer or item verdict mismatch")
        checks = row.get('option_checks', [])
        if [c.get('label') for c in checks] != list('ABCD') or any(c.get('verdict') != 'pass' or not c.get('comment') for c in checks):
            raise ValueError(f"{uid}/{q['id']}: every option explanation needs a documented check")


def review_html(entries, reviews, ledger, *, title=DEFAULT_TITLE, landing_path=DEFAULT_LANDING_PATH,
                external_model='gpt-6-astra', external_effort='high'):
    external_identity({'external_review_model': external_model, 'external_review_effort': external_effort})
    sections = []
    number = 101
    for entry, review, record in zip(entries, reviews, ledger):
        rows = []
        for question, item in zip(entry['content']['questions'], review['items']):
            options = ''.join(f"<li><b>{c['label']}</b> {escape(c['comment'])}</li>" for c in item['option_checks'])
            notes = item.get('notes', '')
            if isinstance(notes, list): notes = '；'.join(notes)
            rows.append(f'<details><summary><span>{number}</span> {escape(question["id"])} · 正解 {question["answer"]} · 複核通過</summary><p>{escape(notes)}</p><ul>{options}</ul></details>')
            number += 1
        caveats = review.get('automated_review_caveats', [])
        caveat_html = ''
        if caveats:
            caveat_rows = ''.join('<li>' + escape(json.dumps(c, ensure_ascii=False) if isinstance(c, dict) else str(c)) + '</li>' for c in caveats)
            caveat_html = '<details><summary>自動審查原始紀錄的註記</summary><p>以下是審查過程的歷史註記，與最終給學習者的詳解分開保存。</p><ul>' + caveat_rows + '</ul></details>'
        links=' · '.join(f'<a href="{escape(r["path"],quote=True)}">{"結構化報告" if i==0 else "盲解紀錄" if i==1 else "詳解紀錄"}</a>' for i,r in enumerate(record['reports']))
        sections.append(f'<section id="{entry["id"]}"><h2>{entry["id"]} / Part {entry["content"]["part"]}</h2><p class="hash">SHA-256 {entry["content_sha256"]}</p><p class="method">{escape(review.get("method_note", ""))}</p><p>{links}</p>{"".join(rows)}{caveat_html}</section>')
    page = '''<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>$release_title · 逐題審查紀錄</title><style>
    :root{font-family:system-ui,-apple-system,sans-serif;background:#f8f9f4;color:#263d31;line-height:1.75}body{margin:0}main{max-width:1000px;margin:auto;padding:40px 24px}a{color:#286a4e}h1{font-size:32px;line-height:1.4}header{padding-bottom:30px;border-bottom:1px solid #d6dfcf}.eyebrow{font-size:11px;letter-spacing:2px;color:#82906f}.intro,.method{color:#687565;font-size:14px}.summary{display:flex;gap:24px;flex-wrap:wrap;margin:24px 0}.summary b{font-size:26px}.summary span{font-size:12px;margin-left:8px}section{padding:32px 0;border-bottom:1px solid #d6dfcf}h2{font-size:23px}.hash{overflow-wrap:anywhere;font:11px monospace;color:#7a8770}details{border:1px solid #d9e1d2;border-radius:5px;padding:14px 18px;margin:9px 0;background:#fff}summary{cursor:pointer;font-size:14px;font-weight:600}summary span{display:inline-block;min-width:45px;color:#788d4b}details p,li{font-size:13px}li{margin:7px 0}footer{font-size:12px;color:#7a8770;padding-top:25px}.nav{display:flex;gap:22px;flex-wrap:wrap;font-size:13px}@media(max-width:600px){h1{font-size:26px}main{padding:26px 18px}details{padding:13px}}</style></head><body><main><header><div class="eyebrow">TOEIC FORGE / EXTERNAL REVIEW LEDGER</div><h1>$release_title · 逐題審查紀錄</h1><p class="intro">$external_model / reasoning effort $external_effort subagents 的審查結果。先存答案隱藏的作答紀錄，再逐題檢查作者答案、四個選項解析、證據與解題提醒。修訂後的知情範圍列在各組說明。本頁僅收錄自動關卡與外部審查皆通過、完整內容 hash 相符的最終版本；原始報告保留審查當時的狀態措辭。</p><div class="summary"><div><b>100</b><span>題</span></div><div><b>400</b><span>選項詳解</span></div><div><b>20</b><span>內容 hash</span></div></div><div class="nav"><a href="practice.html">開啟完整試題 →</a><a href="questions.json" download>下載 JSON ↓</a><a href="$landing_path">回到閱讀回介紹 ↗</a></div></header>'''
    page = Template(page).substitute(release_title=escape(title), landing_path=escape(landing_path, quote=True),
                                     external_model=escape(external_model), external_effort=escape(external_effort))
    return page + ''.join(sections) + '<footer>通過代表本次審查未發現待修正缺陷，不是正確性的數學保證或官方認證。原始報告保留確切版本、獨立答案及複核範圍。</footer></main></body></html>'


def release(manifest_path: Path, out: Path):
    root = Path(__file__).resolve().parents[1]
    manifest = read(manifest_path)
    title = manifest.get('title', DEFAULT_TITLE)
    landing_path = manifest.get('landing_path', DEFAULT_LANDING_PATH)
    original = read(root / manifest['source_run'] / 'run.json')
    external_model, external_effort = external_identity(original)
    specs = manifest['units']
    if [s['id'] for s in specs] != [f'u{i:03d}' for i in range(1,21)]:
        raise ValueError('A full test requires exactly u001 through u020 in order')
    entries, ledger, copies, reviews = [], [], [], []
    for index, spec in enumerate(specs):
        uid = spec['id']
        entry = copy.deepcopy(read(root / spec['checkpoint']))
        if entry['blueprint'] != original['plans'][index]:
            expected = copy.deepcopy(original['plans'][index])
            adjusted = copy.deepcopy(entry['blueprint'])
            for plan in (expected, adjusted):
                for slot in plan['slots']: slot.pop('difficulty', None)
            if expected != adjusted or not spec.get('difficulty_recalibration'):
                raise ValueError(f'{uid}: undeclared blueprint change from the planned full test')
        if errors := verify_entry(entry):
            raise ValueError(f'{uid}: {errors}')
        review_path = root / spec['review']
        review = read(review_path)
        validate_external(entry, uid, review, model=external_model, effort=external_effort)
        reviews.append(review)
        attachments = [review_path, root / review['blind_report'], root / review['explanation_report']]
        reports = []
        for report in attachments:
            raw = report.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            filename = f'{uid}-{digest[:12]}-{report.name}'
            reports.append({'path':f'reports/{filename}', 'sha256':digest})
            copies.append((report, out / 'reports' / filename))
        entry['id'] = uid
        entry['provenance']['source_artifact'] = spec['checkpoint']
        if spec.get('difficulty_recalibration'):
            entry['provenance']['difficulty_recalibration'] = spec['difficulty_recalibration']
        entry['quality']['external_review'] = {
            'model':external_model, 'reasoning_effort':external_effort, 'verdict':'pass',
            'content_sha256':entry['content_sha256'], 'scope':'Every question, all four option explanations, evidence and takeaways',
            'reports':reports, 'method_note':review.get('method_note','Student-only review saved before opening answers; see reports for revision exposure.'),
        }
        if errors := verify_entry(entry):
            raise ValueError(f'{uid}: {errors}')
        entries.append(entry)
        ledger.append({'unit_id':uid, 'content_sha256':entry['content_sha256'],
                       'part':entry['content']['part'], 'question_count':len(entry['content']['questions']),
                       'source':spec['checkpoint'], 'reports':reports})
    totals = {p:sum(len(e['content']['questions']) for e in entries if e['content']['part']==p) for p in (5,6,7)}
    if totals != {5:30,6:16,7:54}:
        raise ValueError(f'Wrong section totals: {totals}')
    diversity = None
    if original.get('diversity_protocol'):
        if original['diversity_protocol'] == 'api-reviewed-fingerprints-and-whole-test-v1':
            from scripts.diverse_api_run import audit as audit_diversity
        elif original['diversity_protocol'] == 'reviewed-fingerprints-and-whole-test-v1':
            from scripts.diverse_native_run import audit as audit_diversity
        else:
            raise ValueError('Unsupported diversity certification protocol')
        diversity = audit_diversity(root / manifest['source_run'], require_pass=True)
        if {a['unit_id']: a['content_sha256'] for a in diversity['units']} != {e['id']: e['content_sha256'] for e in entries}:
            raise ValueError('Released entries differ from the independently reviewed diversity audit')
    run = {**original, 'run_id':manifest['release_id'], 'created_at':datetime.now(timezone.utc).isoformat(), 'plans':[e['blueprint'] for e in entries]}
    # Keep author/reviewer provenance from source run and accepted entries.
    data = bundle(run, entries)
    data['title'] = title
    data['landing_path'] = landing_path
    versions = sorted({e['provenance']['prompt_version'] for e in entries})
    data['prompt_versions'] = versions
    data['prompt_version'] = versions[0] if len(versions)==1 else 'mixed: see unit provenance'
    data['external_review_summary'] = {'model':external_model,'reasoning_effort':external_effort,'question_count':100,'option_explanation_count':400,'units':20}
    data['quality_note'] += f' All 100 questions and 400 option explanations passed external {external_model}/{external_effort} review for these exact content hashes. Difficulty labels are not psychometrically calibrated.'
    if errors := validate_bundle(data):
        raise ValueError(errors)
    out.mkdir(parents=True, exist_ok=True)
    for source, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    if diversity is not None:
        if original['diversity_protocol'] == 'api-reviewed-fingerprints-and-whole-test-v1':
            from scripts.diverse_api_run import entries_and_annotations
        else:
            from scripts.diverse_native_run import entries_and_annotations
        _, confirmed = entries_and_annotations(root / manifest['source_run'])
        exception_path = root / manifest['source_run'] / 'audit/diversity-exceptions.json'
        exceptions = read(exception_path) if exception_path.exists() else {}
        save_json(out/'diversity-annotations.json', {'units': {
            e['id']: {'content_sha256': e['content_sha256'], 'annotations': confirmed[e['id']],
                      'exceptions': exceptions.get(e['id'], [])} for e in entries}})
        save_json(out/'diversity-report.json', diversity)
        data['diversity_review'] = {'protocol': original['diversity_protocol'], 'report': 'diversity-report.json',
                                    'annotations': 'diversity-annotations.json', 'scope': 'Whole-test editorial review and hash-bound reviewed fingerprints; not psychometric validation.'}
    save_json(out/'questions.json', data)
    (out/'practice.html').write_text(render(data))
    (out/'review.html').write_text(review_html(entries, reviews, ledger, title=title, landing_path=landing_path,
                                             external_model=external_model, external_effort=external_effort))
    save_json(out/'review-blind.json', {'run_id':run['run_id'], 'units':[
        {'id':e['id'], 'content_sha256':e['content_sha256'], 'student_material':blind_payload(Unit.model_validate(e['content']))} for e in entries]})
    save_json(out/'manifest.json', {'release_id':run['run_id'],'source_run':manifest['source_run'], 'title':title, 'landing_path':landing_path, 'question_count':100, 'units':ledger})
    print(f'Released 20 units / 100 questions / 400 option checks: {out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    p=subs.add_parser('packets'); p.add_argument('run',type=Path); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--release-unit'); p.add_argument('--start-number',type=int,default=101)
    p=subs.add_parser('release'); p.add_argument('manifest',type=Path); p.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='packets': packets(args.run,args.out,args.release_unit,args.start_number)
    else: release(args.manifest,args.out)


if __name__=='__main__':
    main()
