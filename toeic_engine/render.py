"""Standalone, escaped HTML: no CDN, server, framework, or untrusted HTML."""
from __future__ import annotations

import re
from html import escape

from .models import Block

SKILL_LABELS = {
    'word_form_noun': '名詞詞形', 'word_form_adjective': '形容詞詞形',
    'word_form_adverb': '副詞詞形', 'word_form_participle': '分詞用法',
    'subject_verb_agreement': '主詞與動詞一致', 'tense_aspect': '時態與完成／進行觀念',
    'passive_voice': '被動語態', 'infinitive_gerund': '不定詞與動名詞',
    'relative_clause': '關係子句', 'nominal_clause': '名詞子句', 'fused_relative': '融合關係子句', 'pronoun_case_reference': '代名詞格位與指涉',
    'determiner_quantifier': '限定詞與數量詞', 'comparison': '比較句型',
    'preposition': '介系詞', 'conjunction_vs_preposition': '連接詞與介系詞',
    'subordinate_conjunction': '從屬連接詞', 'conjunctive_adverb': '連接副詞',
    'verb_collocation': '動詞搭配', 'noun_collocation': '名詞搭配',
    'adjective_collocation': '形容詞搭配', 'adverb_collocation': '副詞搭配',
    'business_verb_meaning': '商務動詞語意', 'business_noun_meaning': '商務名詞語意',
    'business_adjective_meaning': '商務形容詞語意', 'phrasal_verb': '片語動詞',
    'word_form': '詞性與詞形', 'pronoun_reference': '代名詞指涉',
    'logical_connector': '邏輯連接', 'contextual_collocation': '篇章搭配',
    'reference_cohesion': '指涉與銜接', 'sentence_cohesion': '句子銜接',
    'contextual_vocabulary': '語境詞彙', 'purpose': '寫作目的',
    'detail': '細節理解', 'negative_detail': '未提及／不符合原文的資訊',
    'inference': '文意推論', 'vocabulary_in_context': '文中詞義',
    'intent': '話語意圖', 'sentence_insertion': '插句與篇章銜接',
    'cross_document': '跨篇資訊整合',
}

CSS = """
:root{font-family:Inter,system-ui,-apple-system,'Noto Sans TC',sans-serif;color:#1b2c37;background:#edf2f4;line-height:1.65}
*{box-sizing:border-box}body{margin:0}header{background:#102e3b;color:#fff;padding:40px max(24px,calc((100vw - 1120px)/2));border-bottom:5px solid #e6ae5b}
.unit-nav{display:flex;flex-wrap:wrap;gap:8px;margin:20px 0 28px}.unit-nav a{color:#176b60;background:#fff;border:1px solid #cdd9dd;border-radius:5px;padding:7px 12px;text-decoration:none;font-size:14px}.unit-nav a:hover{background:#e3f4ec}.unit{scroll-margin-top:95px}
.eyebrow{font-size:12px;letter-spacing:.16em;color:#8fe1d1;font-weight:700}h1{margin:10px 0;font-size:32px;line-height:1.25}h2{font-size:23px;line-height:1.3}h3{font-size:18px}p{margin:0 0 16px}
header p{color:#c4d7df;margin:8px 0 0}.toolbar{position:sticky;top:0;z-index:5;display:flex;align-items:center;gap:12px;background:#fff;padding:12px max(20px,calc((100vw - 1120px)/2));box-shadow:0 2px 12px #1232;flex-wrap:wrap}
button{border:0;border-radius:6px;background:#176b60;color:white;padding:10px 17px;font:inherit;cursor:pointer}button.secondary{background:#e6eef0;color:#193e4b}#score{margin-left:auto;font-weight:700}main{max-width:1160px;margin:32px auto;padding:0 20px}.unit{margin-bottom:38px}.unit-header{display:flex;gap:15px;align-items:baseline;border-bottom:1px solid #bccdd3;margin-bottom:18px}.badge{font-size:12px;color:#536a74}
.layout{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:24px}.layout.part5{display:block;max-width:880px;margin:auto}.documents,.questions{min-width:0}.document{background:white;border:1px solid #cdd9dd;border-radius:8px;padding:28px;margin-bottom:18px;box-shadow:0 3px 15px #173b4c07}.genre{font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:#46776b;border-bottom:1px solid #dde8e8;padding-bottom:9px;margin-bottom:18px}.field{margin:3px 0;font-size:14px}.block{white-space:pre-wrap}.message{margin:8px 0;padding:12px 15px;border-radius:10px;background:#eef5f7}.message small{display:block;color:#56707a;margin-bottom:4px}.blank{display:inline-block;font-weight:bold;letter-spacing:.03em;border-bottom:1px solid #204b5d;min-width:50px;text-align:center}.marker{color:#08755b;font-weight:700}.question{background:white;padding:22px;border:1px solid #d1dde0;border-radius:8px;margin-bottom:16px}.stem{font-weight:600;font-size:16px}.number{color:#117565;font-weight:800;margin-right:8px}.choices{display:grid;gap:6px}.choice{display:flex;align-items:flex-start;gap:9px;border:1px solid #e4ebee;border-radius:5px;padding:9px 12px;cursor:pointer}.choice:hover{background:#f3f8f9}.choice input{margin-top:6px;accent-color:#147361}.choice.correct{background:#e3f4ec;border-color:#5cad89}.choice.wrong{background:#fcebed;border-color:#d18588}details{margin-top:16px;border-top:1px solid #e3ebed;padding-top:12px;font-size:14px}summary{cursor:pointer;font-weight:600;color:#176b60}.answer{background:#edf6f2;padding:12px;margin:12px 0;border-radius:6px}.evidence{border-left:3px solid #8aafa2;padding-left:12px;color:#435b64;font-size:13px}table{border-collapse:collapse;width:100%;font-size:14px;margin:12px 0}th,td{border:1px solid #c6d7db;padding:8px;text-align:left;white-space:pre-wrap}th{background:#edf4f5}.table-scroll{overflow-x:auto}li{margin:6px 0}footer{font-size:12px;text-align:center;color:#5d727c;padding:30px}.no-js{padding:12px;background:#fff4d6}
@media(max-width:850px){.layout{grid-template-columns:1fr}header{padding:28px 20px}h1{font-size:26px}.document,.question{padding:20px}#score{margin-left:0}.toolbar{position:static}}
@media print{body{background:white;font-size:10pt}header{background:white;color:black;padding:0 0 12px;border-bottom:2px solid #555}header p{color:#444}.toolbar,footer,.no-print{display:none}main{margin:0;max-width:none;padding:0}.layout{display:block}.document,.question{box-shadow:none;break-inside:avoid;border:1px solid #bbb}.unit{break-before:page}.unit:first-child{break-before:auto}details:not([open]){display:none}.choice{padding:3px;border:0}input{display:none}}
"""
JS = """
const questions=[...document.querySelectorAll('.question')];
document.getElementById('grade').addEventListener('click',()=>{let answered=0,correct=0;for(const q of questions){const chosen=q.querySelector('input:checked');if(chosen)answered++;if(chosen?.value===q.dataset.answer)correct++;for(const c of q.querySelectorAll('.choice')){const input=c.querySelector('input');c.classList.toggle('correct',input.value===q.dataset.answer);c.classList.toggle('wrong',input.checked&&input.value!==q.dataset.answer)}q.querySelector('details').open=true}document.getElementById('score').textContent=`答對 ${correct} / ${questions.length}　已作答 ${answered}`});
document.getElementById('reset').addEventListener('click',()=>{for(const i of document.querySelectorAll('input'))i.checked=false;for(const c of document.querySelectorAll('.choice'))c.classList.remove('correct','wrong');for(const d of document.querySelectorAll('details'))d.open=false;document.getElementById('score').textContent='尚未批改'});
document.getElementById('explain').addEventListener('click',()=>{const all=[...document.querySelectorAll('details')];const open=all.some(d=>!d.open);all.forEach(d=>d.open=open)});
document.getElementById('print').addEventListener('click',()=>window.print());
"""


def text(value: str, numbers: dict[str, int]) -> str:
    value = escape(value)
    value = re.sub(r"\{\{(q\d+)\}\}", lambda m: f'<span class="blank">({numbers.get(m[1], m[1])}) _____</span>', value)
    return re.sub(r"\[\[([1-4])\]\]", r'<span class="marker">[\1]</span>', value)


def block_html(block: dict, numbers: dict) -> str:
    b = Block.model_validate(block)
    content = text(b.text, numbers)
    if b.type == "table":
        head = "".join(f"<th>{text(x,numbers)}</th>" for x in b.columns)
        rows = "".join("<tr>" + "".join(f"<td>{text(x,numbers)}</td>" for x in row) + "</tr>" for row in b.rows)
        return f'<div class="table-scroll"><table><caption>{content}</caption><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'
    if b.type == "message":
        return f'<div class="message"><small>{escape(b.speaker)} · {escape(b.timestamp)}</small><div class="block">{content}</div></div>'
    if b.type == "heading":
        return f"<h3>{content}</h3>"
    if b.type == "list_item":
        return f'<p class="block">• {content}</p>'
    return f'<p class="block {"field" if b.type == "field" else ""}">{content}</p>'


def render(data: dict) -> str:
    sections = []
    navigation = []
    number = 101
    for entry in data["units"]:
        u = entry["content"]
        numbers = {q["id"]: number+i for i, q in enumerate(u["questions"])}
        number += len(numbers)
        docs = []
        for d in u["documents"]:
            blocks = "".join(block_html(b, numbers) for b in d["blocks"])
            docs.append(f'<article class="document" aria-label="{escape(d["title"])}"><div class="genre">{escape(d["genre"].replace("_", " "))}</div>{blocks}</article>')
        questions = []
        for q in u["questions"]:
            name = f'{entry["id"]}-{q["id"]}'
            opts = "".join(f'<label class="choice"><input type="radio" name="{escape(name)}" value="{o["label"]}"><span>({o["label"]}) {text(o["text"],numbers)}</span></label>' for o in q["options"])
            ex = q["explanation"]
            skill = SKILL_LABELS.get(q['skill'], q['skill'].replace('_', ' '))
            analysis = "".join(f'<li><b>({r["label"]})</b> {escape(r["reason_zh"])}</li>' for r in ex["option_analysis"])
            evidence = "".join(f'<p class="evidence">{escape(e["document_id"])} / {escape(e["block_id"])}<br>“{text(e["quote"], numbers)}”</p>' for e in ex["evidence"])
            explanation = f'<details><summary>答案與詳解</summary><div class="answer">答案：{q["answer"]} · {escape(skill)}<br>{escape(ex["summary_zh"])}</div><p>{escape(ex["reasoning_zh"])}</p>{evidence}<ul>{analysis}</ul><p><b>解題提醒：</b>{escape(ex["takeaway_zh"])}</p></details>'
            stem = "" if u["part"] == 6 else text(q["prompt"], numbers)
            questions.append(f'<article class="question" data-answer="{q["answer"]}"><p class="stem"><span class="number">{numbers[q["id"]]}.</span>{stem}</p><div class="choices" role="group" aria-label="Question {numbers[q["id"]]}">{opts}</div>{explanation}</article>')
        instruction = {5: "Choose the word or phrase that best completes each sentence.", 6: "Choose the best answer for each blank in the text.", 7: "Read the document(s) and choose the best answer to each question."}[u["part"]]
        # Author titles often reveal a tested point; headings use only part/genre.
        title = {5: "Incomplete Sentences", 6: "Text Completion", 7: "Reading Comprehension"}[u["part"]]
        anchor = escape(entry['id'])
        detail = {5: '句子填空', 6: '篇章填空'}.get(u['part'])
        if detail is None:
            detail = {1: '單篇', 2: '雙篇', 3: '三篇'}[len(docs)]
            kinds = {q['kind'] for q in u['questions']}
            for kind, label in [('sentence_insertion', '插句'), ('intent', '語意'), ('vocabulary_in_context', '詞義')]:
                if kind in kinds:
                    detail += '・' + label
        navigation.append(f'<a href="#{anchor}">Part {u["part"]} {detail} · {min(numbers.values())}–{max(numbers.values())}</a>')
        sections.append(f'<section class="unit" id="{anchor}"><div class="unit-header"><h2>Part {u["part"]} · {title}</h2><span class="badge">{len(numbers)} questions</span></div><p>{instruction}</p><div class="layout {"part5" if u["part"] == 5 else ""}">{"<div class=documents>" + "".join(docs) + "</div>" if docs else ""}<div class="questions">{"".join(questions)}</div></div></section>')
    practice = data.get('document_type') == 'practice'
    page_title = escape(data.get('title') or ('英文閱讀練習' if practice else 'TOEIC 閱讀練習'))
    status = "完整練習" if data["status"] == "complete" else "部分完成：僅列出已通過檢查的題目"
    if data.get('sample_scope') and data['status'] == 'complete':
        status = '練習範例' if practice else '審查範例（非完整 100 題測驗）'
    footer = 'AI 生成的英文閱讀練習' if practice else '原創練習，非 ETS 官方試題。自動品質審查不代表官方認證。'
    nav = '<nav class="unit-nav no-print" aria-label="題組導航">' + ''.join(navigation) + '</nav>'
    return f'<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{page_title}</title><style>{CSS}</style></head><body><header><div class="eyebrow">READING PRACTICE / ORIGINAL EDITION</div><h1>{page_title}</h1><p>{data["question_count"]} 題 · {status} · 英文試題／繁體中文詳解</p></header><div class="toolbar"><button id="grade">批改答案</button><button id="explain" class="secondary">展開／收合詳解</button><button id="reset" class="secondary">重新作答</button><button id="print" class="secondary">列印</button><span id="score" aria-live="polite">尚未批改</span></div><noscript><p class="no-js">瀏覽器停用 JavaScript。仍可閱讀題目，逐題展開詳解。</p></noscript><main>{nav}{"".join(sections)}</main><footer>{footer}<br>{escape(data["run_id"])}</footer><script>{JS}</script></body></html>'
