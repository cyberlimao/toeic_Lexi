import copy
import json
from types import SimpleNamespace

import pytest

from toeic_engine import cli
from toeic_engine.engine import Engine, QualityError, content_hash, verify_entry
from toeic_engine.models import BlindSolution, Challenge, Review, Unit, blind_payload, validate_blind, validate_unit, validate_review, validate_challenge
from toeic_engine.novelty import History, normalize
from toeic_engine.planning import Planner
from toeic_engine.render import render


@pytest.fixture
def example():
    plan = Planner(1).unit(5, 1)
    plan['slots'] = [{'id': 'q1', 'kind': 'grammar', 'skill': 'word_form_adverb', 'difficulty': 'medium', 'scenario': 'museum exhibition'}]
    data = {'part': 5, 'title': 'Example', 'scenario_summary': 'Museum attendance rose after a local outreach campaign.', 'documents': [], 'questions': [{
        'id': 'q1', 'kind': 'grammar', 'skill': 'word_form_adverb', 'difficulty': 'medium',
        'prompt': 'Attendance at the museum increased {{q1}} after its outreach campaign began last spring.',
        'options': [{'label': l, 'text': t} for l, t in zip('ABCD', ['significantly', 'significance', 'significant', 'signify'])], 'answer': 'A',
        'explanation': {'summary_zh': '空格修飾動詞 increased，因此需要副詞 significantly，表示參觀人數顯著增加。',
                        'reasoning_zh': '句子的主詞是 Attendance，動詞是 increased。空格說明增加的程度，因此以副詞 significantly 修飾動詞，不能使用名詞或形容詞。',
                        'option_analysis': [{'label': l, 'reason_zh': r} for l, r in zip('ABCD', ['副詞 significantly 可以修飾動詞 increased。', 'significance 是名詞，無法在此修飾 increased。', 'significant 是形容詞，此處需要修飾動詞的副詞。', 'signify 是另一個動詞，不能直接放在 increased 後。'])],
                        'evidence': [], 'takeaway_zh': '先找出空格所修飾的對象，再判斷所需詞性。'}}]}
    return Unit.model_validate(data), plan


def solution(unit):
    return BlindSolution.model_validate({'answers': [{'question_id': q.id, 'answer': q.answer, 'defensible_answers': [q.answer], 'rationale_zh': '空格需要副詞以修飾動詞，因此選項 A 最符合句法要求。', 'evidence': []} for q in unit.questions]})


def good_review():
    return Review.model_validate({'verdict': 'pass', 'naturalness': 5, 'toeic_alignment': 4, 'explanation_quality': 5, 'findings': [], 'item_reviews': [{
        'question_id': 'q1', 'acceptable_answers': ['A'], 'strongest_rival': 'C',
        'rival_test': 'Significant is an adjective and cannot modify the intransitive verb increased here.',
        'distractor_quality': 4, 'explanation_accuracy': 5, 'requires_cross_sentence_context': False,
        'context_justification': 'Standalone grammar item.', 'necessary_document_ids': [],
        'option_checks': [{'label': label, 'defensible': label == 'A', 'explanation_supported': True, 'note': 'Correct part-of-speech analysis in this sentence.'} for label in 'ABCD']}]})


def good_teaching(unit):
    from toeic_engine.independent_audits import TeachingReview, TEACHING_FIELDS
    return TeachingReview.model_validate({'items': [
        {'question_id':q.id, 'verdict':'pass', 'issues':[],
         'fields':[{'field':field, 'verdict':'pass', 'note':'Checked against the actual wording and source evidence.'} for field in TEACHING_FIELDS]}
        for q in unit.questions]})


def challenge_for(unit):
    return Challenge.model_validate({'items': [{'question_id': q.id, 'options': [{'label': label, 'viable': label == q.answer, 'uncertain': False, 'test': 'Substitution confirms the grammatical role in this exact sentence.'} for label in 'ABCD']} for q in unit.questions]})


def test_full_blueprint_has_exact_exam_structure():
    for seed in range(50):
        plans = Planner(seed).build('full')
        assert len(plans) == 20
        assert sum(p['question_count'] for p in plans) == 100
        assert [sum(p['question_count'] for p in plans if p['part'] == k) for k in (5, 6, 7)] == [30, 16, 54]
        assert [len(p['genres']) for p in plans if p['part'] == 7] == [1]*10 + [2]*2 + [3]*3
        assert sum(s['kind'] == 'sentence_insertion' for p in plans for s in p['slots']) == 2
        assert sum(s['kind'] == 'intent' for p in plans for s in p['slots']) >= 2
        assert all(sum(s['kind'] == 'sentence_completion' for s in p['slots']) == 1 for p in plans if p['part'] == 6)


def test_seed_is_reproducible_but_new_seeds_change_plan():
    assert Planner(9).build('sample') == Planner(9).build('sample')
    assert Planner(9).build('sample') != Planner(10).build('sample')
    assert Planner(9).unit(7, 2, special='intent')['slots'][1]['kind'] == 'intent'


def test_blind_payload_cannot_leak_key_or_explanation(example):
    unit, _ = example
    hidden = blind_payload(unit)
    serialized = json.dumps(hidden)
    for token in ('answer', 'explanation', 'skill', 'difficulty', 'scenario_summary', 'takeaway'):
        assert token not in serialized
    assert hidden['questions'][0]['options'][0]['text'] == 'significantly'


def test_ambiguous_or_unanswerable_blind_results_reject(example):
    unit, _ = example
    solved = solution(unit)
    solved.answers[0].defensible_answers = ['A', 'C']
    assert validate_blind(solved, unit)
    solved.answers[0].answer = None
    solved.answers[0].defensible_answers = []
    assert validate_blind(solved, unit)


def test_critic_cannot_pass_major_findings():
    reviewed = good_review()
    reviewed.naturalness = 3
    assert not reviewed.passes()
    reviewed = good_review()
    reviewed = Review.model_validate({**reviewed.model_dump(), 'findings': [{'severity': 'major', 'question_id': 'q1', 'category': 'ambiguity', 'problem': 'Both A and B work', 'fix': 'Rewrite'}]})
    assert not reviewed.passes()


def test_aggregate_praise_cannot_override_option_errors(example):
    unit, _ = example
    reviewed = good_review()
    assert not validate_review(reviewed, unit)
    reviewed.item_reviews[0].option_checks[2].explanation_supported = False
    assert validate_review(reviewed, unit)
    reviewed.item_reviews = []
    assert validate_review(reviewed, unit)


def test_challenger_uncertainty_or_alternate_reading_blocks(example):
    unit, _ = example
    challenge = challenge_for(unit)
    assert not validate_challenge(challenge, unit)
    challenge.items[0].options[1].uncertain = True
    assert validate_challenge(challenge, unit)
    challenge.items[0].options[1].uncertain = False
    challenge.items[0].options[1].viable = True
    assert validate_challenge(challenge, unit)


def test_wrong_counts_and_blank_markers_reject(example):
    unit, plan = example
    assert not validate_unit(unit, plan)
    unit.questions[0].prompt = unit.questions[0].prompt.replace('{{q1}}', '{{q2}}')
    assert validate_unit(unit, plan)
    plan['question_count'] = 30
    assert any('count' in e for e in validate_unit(unit, plan))


def test_history_rejects_repeat_and_name_number_swaps(tmp_path, example):
    unit, plan = example
    history = History(tmp_path / 'history.sqlite')
    try:
        history.accept('first', unit, plan)
        assert history.check(unit)
        with pytest.raises(ValueError):
            history.accept('second', unit, plan)
        assert len(history.recent()) == 1
        assert normalize('The shipment from Maple Harbor contains 300 custom-made items.') == normalize('The shipment from Willow Grove contains 900 custom-made items.')
        history.close()
        history = History(tmp_path / 'history.sqlite')
        assert history.check(unit)
    finally:
        history.close()


def test_evidence_must_resolve_to_exact_document_block(example):
    unit, plan = example
    payload = unit.model_dump()
    payload['questions'][0]['explanation']['evidence'] = [{'document_id': 'd9', 'block_id': 'b9', 'quote': 'fabricated'}]
    unit = Unit.model_validate(payload)
    assert any('Invalid evidence' in e for e in validate_unit(unit, plan))


class FakeProvider:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.payloads = []

    def generate(self, system, payload, schema, label, max_tokens):
        self.payloads.append((label, payload))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response

    @property
    def calls(self):
        return len(self.payloads)

    def close(self):
        pass


def test_part6_missing_document_ids_retry_review_without_rewriting(tmp_path):
    empty = good_review()
    complete = good_review()
    complete.item_reviews[0].necessary_document_ids = ['d1']
    # A substantive problem remains a problem after the format is corrected.
    complete.item_reviews[0].explanation_accuracy = 3
    provider = FakeProvider([empty, complete])
    engine = Engine(provider, None, tmp_path, progress=lambda _: None)
    payload = {'unit': {'part': 6, 'documents': [{'id': 'd1'}]}}
    result = engine._review_call('critic', payload, Review, 'u001-a1-critic', 1000)
    assert result.item_reviews[0].necessary_document_ids == ['d1']
    assert result.item_reviews[0].explanation_accuracy == 3
    assert [label for label, _ in provider.payloads] == ['u001-a1-critic', 'u001-a1-critic-format2']
    assert 'response_format_correction' in provider.payloads[1][1]
    assert provider.payloads[1][1]['unit'] == payload['unit']
    assert 'response_format_correction' not in payload
    assert (tmp_path/'attempts/u001-a1-critic.format-error.json').exists()
    assert not list(tmp_path.glob('units/*'))


def test_part6_missing_document_ids_stop_after_bounded_retries(tmp_path):
    from toeic_engine.provider import ProviderError
    provider = FakeProvider([good_review(), good_review(), good_review()])
    engine = Engine(provider, None, tmp_path, progress=lambda _: None)
    with pytest.raises(ProviderError, match='three times'):
        engine._review_call('critic', {'unit': {'part': 6}}, Review, 'critic', 1000)
    assert len(provider.payloads) == 3
    assert not list(tmp_path.glob('units/*'))


@pytest.mark.parametrize('part', [5, 7])
def test_empty_evidence_for_other_parts_is_not_rewritten_as_format_error(tmp_path, part):
    provider = FakeProvider([good_review()])
    engine = Engine(provider, None, tmp_path, progress=lambda _: None)
    result = engine._review_call('critic', {'unit': {'part': part}}, Review, 'critic', 1000)
    assert result.item_reviews[0].necessary_document_ids == []
    assert len(provider.payloads) == 1


def test_pipeline_rejects_ambiguity_and_reaudits_replacement(tmp_path, example):
    unit, plan = example
    ambiguous = solution(unit)
    ambiguous.answers[0].defensible_answers = ['A', 'C']
    provider = FakeProvider([unit, ambiguous, challenge_for(unit), unit, solution(unit), challenge_for(unit), good_review(), good_teaching(unit)])
    history = History(tmp_path / 'history.sqlite')
    try:
        engine = Engine(provider, history, tmp_path / 'run', progress=lambda _: None)
        accepted = engine.generate_unit(plan, 'u001')
        assert accepted['quality']['attempts'] == 2
        assert not verify_entry(accepted)
        assert 'required_fixes' in provider.payloads[3][1]
        assert (tmp_path / 'run/attempts/u001-a1.rejection.json').exists()
        accepted['content']['questions'][0]['answer'] = 'D'
        assert verify_entry(accepted)
    finally:
        history.close()


def test_rejected_content_never_becomes_accepted_checkpoint(tmp_path, example):
    unit, plan = example
    bad = solution(unit)
    bad.answers[0].answer = 'B'
    provider = FakeProvider([unit, bad, challenge_for(unit)])
    history = History(tmp_path / 'history.sqlite')
    try:
        engine = Engine(provider, history, tmp_path / 'run', max_attempts=1, progress=lambda _: None)
        with pytest.raises(QualityError):
            engine.generate_unit(plan, 'u001')
        assert not history.recent()
        assert not (tmp_path / 'run/units/u001.json').exists()
    finally:
        history.close()


def test_html_escapes_untrusted_content_and_preserves_answers(example):
    unit, _ = example
    unit.questions[0].prompt += '<script>alert("test")</script>'
    data = {'units': [{'id': 'u1', 'content': unit.model_dump()}], 'status': 'complete', 'question_count': 1, 'run_id': 'test'}
    html = render(data)
    assert '<script>alert(' not in html
    assert '&lt;script&gt;' in html
    assert 'data-answer="A"' in html
    assert '(101) _____' in html
    assert '答案與詳解' in html


def test_revision_exemption_is_limited_to_recorded_ancestry(tmp_path, example):
    unit, plan = example
    history = History(tmp_path / 'history.sqlite')
    try:
        history.accept('first', unit, plan)
        revision = unit.model_copy(deep=True)
        revision.questions[0].explanation.takeaway_zh += '先分析完整句子的結構。'
        history.accept('second', revision, plan, revises='first')
        assert not history.check(revision, exclude_source='second')
        assert history.check(revision, exclude_source='unrelated')
        third = revision.model_copy(deep=True)
        third.questions[0].prompt = 'The catering staff will deliver the freshly prepared meals to the conference center {{q1}}.'
        history.accept('unrelated', third, plan)
        assert history.check(third, exclude_source='second')
    finally:
        history.close()


def test_reaudit_skips_author_and_checks_external_hash(tmp_path, example):
    unit, plan = example
    provider = FakeProvider([solution(unit), challenge_for(unit), good_review(), good_teaching(unit)])
    history = History(tmp_path / 'history.sqlite')
    try:
        history.accept('old', unit, plan)
        engine = Engine(provider, history, tmp_path / 'run', progress=lambda _: None)
        entry = engine.generate_unit(plan, 'u001', initial=unit, revises='old')
        assert history.has_accepted(f'{(tmp_path / "run").resolve()}/u001', entry['content_sha256'])
        assert len(provider.payloads) == 4
        assert not any(label.endswith('-author') for label, _ in provider.payloads)
        assert not verify_entry(entry)
        entry['quality']['external_review'] = {'verdict': 'pass', 'content_sha256': 'different'}
        assert any('External review' in e for e in verify_entry(entry))
        entry['quality']['external_review']['content_sha256'] = entry['content_sha256']
        assert not verify_entry(entry)
        entry['quality']['external_review']['verdict'] = 'revise'
        assert any('requires revision' in e for e in verify_entry(entry))
    finally:
        history.close()


def test_revision_parent_recovers_exact_accepted_draft(tmp_path, example):
    from toeic_engine.engine import content_hash
    unit, plan = example
    history = History(tmp_path/'history.sqlite')
    try:
        history.accept('accepted-copy', unit, plan, revises='original-draft')
        assert history.revision_parent('original-draft', content_hash(unit)) == 'accepted-copy'
        assert history.revision_parent('unknown', 'different', 'known-parent') == 'known-parent'
        assert history.revision_parent('accepted-copy', 'different', 'other') == 'accepted-copy'
    finally:
        history.close()


def test_quoted_utterance_may_omit_terminal_period(example):
    unit, _ = example
    unit.part = 7
    question = unit.questions[0]
    question.kind = 'intent'
    question.skill = 'intent'
    question.target_document_id = 'd1'
    question.target_text = 'That takes a load off.'
    question.prompt = 'What does the speaker mean by "That takes a load off"?'
    from toeic_engine.models import Document, Block, Evidence
    unit.documents = [Document(id='d1', genre='chat', title='Team chat', blocks=[Block(id='d1-b1', type='message', speaker='Alex', text='That takes a load off.')])]
    question.explanation.evidence = [Evidence(document_id='d1', block_id='d1-b1', quote='That takes a load off.')]
    plan = {'part':7, 'question_count':1, 'genres':['chat'], 'word_range':[1,100], 'slots':[{'kind':'intent','skill':'intent','difficulty':question.difficulty}]}
    assert not validate_unit(unit, plan)
    question.target_text = 'That resolves the issue.'
    assert validate_unit(unit, plan)


def test_explanation_only_revision_preserves_student_content(tmp_path, example):
    from toeic_engine.models import ExplanationRevision
    unit, plan = example
    revised = unit.questions[0].explanation.model_copy(deep=True)
    revised.takeaway_zh = '先分析句子結構，確認此處需要修飾動詞的副詞。'
    patch = ExplanationRevision.model_validate({'replacements':[{'question_id':'q1','explanation':revised.model_dump()}]})
    provider = FakeProvider([patch, solution(unit), challenge_for(unit), good_review(), good_teaching(unit)])
    provider.author_provenance = {'model': 'claude-opus-5-5', 'effort': 'xhigh', 'thinking': 'adaptive'}
    provider.review_provenance = provider.author_provenance
    history = History(tmp_path / 'history.sqlite')
    try:
        result = Engine(provider, history, tmp_path/'run', progress=lambda _: None).generate_unit(plan,'u001',initial=unit,initial_feedback=['Clarify the takeaway.'],explanation_only=True,initial_author_provenance={'model':'gemini-3.8-flash','thinking_level':'HIGH'})
        final = Unit.model_validate(result['content'])
        assert blind_payload(final) == blind_payload(unit)
        assert final.questions[0].answer == unit.questions[0].answer
        assert final.questions[0].explanation.takeaway_zh == revised.takeaway_zh
        assert result['provenance']['revision_mode'] == 'explanations_only'
        assert result['provenance']['model'] == 'gemini-3.8-flash'
        assert result['provenance']['explanation_editor']['model'] == 'claude-opus-5-5'
        assert not verify_entry(result)
    finally:
        history.close()


def test_claude_author_provenance_survives_bundle(tmp_path, example):
    from toeic_engine.engine import bundle
    unit, plan = example
    provider = FakeProvider([unit, solution(unit), challenge_for(unit), good_review(), good_teaching(unit)])
    provider.author_provenance = {'model': 'claude-opus-5-5', 'effort': 'xhigh', 'thinking': 'adaptive'}
    provider.review_provenance = provider.author_provenance
    history = History(tmp_path/'history.sqlite')
    try:
        entry = Engine(provider, history, tmp_path/'run', progress=lambda _: None).generate_unit(plan,'u001')
        run = {'run_id':'claude','created_at':'2026-10-20','seed':1,'prompt_version':'test','plans':[plan]}
        result = bundle(run,[entry])
        assert result['model'] == entry['provenance']['model'] == 'claude-opus-5-5'
        assert result['effort'] == 'xhigh' and 'thinking_level' not in result
        assert entry['provenance']['reviewer']['effort'] == 'xhigh'
    finally:
        history.close()


def test_malformed_review_retries_reviewer_not_author(tmp_path, example):
    from toeic_engine.provider import OutputError
    unit, plan = example
    provider = FakeProvider([unit, OutputError('Malformed blind response'), solution(unit), challenge_for(unit), good_review(), good_teaching(unit)])
    history = History(tmp_path/'history.sqlite')
    try:
        result = Engine(provider, history, tmp_path/'run', progress=lambda _: None).generate_unit(plan,'u001')
        assert result['quality']['attempts'] == 1
        assert sum(label.endswith('-author') for label,_ in provider.payloads) == 1
        assert any(label.endswith('-blind-format2') for label,_ in provider.payloads)
        assert not verify_entry(result)
    finally:
        history.close()


def test_cli_generates_and_resumes_without_a_reference_corpus(tmp_path, example, monkeypatch):
    unit, plan = example
    provider = FakeProvider([unit, solution(unit), challenge_for(unit), good_review(), good_teaching(unit)])
    monkeypatch.setattr(cli, "create_provider", lambda *args: provider)
    monkeypatch.setattr(cli, "Planner", lambda *args: SimpleNamespace(seed=1, build=lambda *_: [plan]))
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "generated"
    monkeypatch.setattr(
        "sys.argv", ["toeic", "generate", "--part", "5", "--count", "1", "--out", str(output)]
    )

    assert cli.main() == 0
    run = json.loads((output / "run.json").read_text())
    assert run.get("corpus") is None
    assert provider.calls == 5

    monkeypatch.setattr("sys.argv", ["toeic", "resume", str(output)])
    assert cli.main() == 0
    assert provider.calls == 5
    assert (output / "practice.html").is_file()


def test_practice_export_validates_without_review_records_and_detects_edits(example):
    unit, plan = example
    entry = {"id": "u001", "content": unit.model_dump(), "blueprint": plan, "content_sha256": content_hash(unit)}
    data = {
        "document_type": "practice",
        "run_id": "clean-sample",
        "status": "complete",
        "question_count": 1,
        "expected_question_count": 1,
        "planned_unit_count": 1,
        "units": [entry],
    }

    assert cli.validate_bundle(data) == []
    entry["content"]["questions"][0]["prompt"] += " (edited)"
    assert any("Content hash mismatch" in error for error in cli.validate_bundle(data))
