"""Whole-test constraints and measured variation, without generating model output."""
from collections import Counter
from typing import get_args

import pytest

from toeic_engine.models import Question
from toeic_engine.planning import INSERTION_GENRES, P5_FAMILIES, P5_MICROCONSTRUCTS, TABLE_GENRES, Planner


@pytest.fixture(scope="module")
def full_plans():
    return [Planner(seed).build("full") for seed in range(128)]


def test_full_format_and_author_metadata_across_seeds(full_plans):
    allowed_kinds = set(get_args(Question.model_fields["kind"].annotation))
    for plans in full_plans:
        assert len(plans) == 20
        assert [sum(p["question_count"] for p in plans if p["part"] == part) for part in (5, 6, 7)] == [30, 16, 54]
        assert [sum(p["question_count"] for p in plans if p["part"] == 7 and p["mode"] == mode)
                for mode in ("single", "double", "triple")] == [29, 10, 15]
        assert [len(p["genres"]) for p in plans[5:]] == [1] * 10 + [2] * 2 + [3] * 3
        for plan in plans:
            assert [s["id"] for s in plan["slots"]] == [f"q{i + 1}" for i in range(plan["question_count"])]
            for slot in plan["slots"]:
                assert slot["kind"] in allowed_kinds
                assert slot["microconstruct"]
                assert len(slot["realization_strategy"]) > 30
                if plan["part"] != 5:
                    assert slot["reading_operation"]
        for plan in plans[1:5]:
            assert Counter(s["kind"] for s in plan["slots"]) == {"sentence_completion": 1, "text_completion": 3}
            assert sum(s["requires_cross_sentence_context"] for s in plan["slots"]) >= 2


def test_part5_bounded_mix_and_concrete_microconstructs(full_plans):
    family_signatures = set()
    micro_signatures = set()
    observed = set()
    for plans in full_plans:
        slots = plans[0]["slots"]
        family_counts = Counter(s["family"] for s in slots)
        family_signatures.add(tuple(sorted(family_counts.items())))
        micro_signatures.add(tuple(sorted(s["microconstruct"] for s in slots)))
        assert len({s["microconstruct"] for s in slots}) == 30
        for family, (minimum, maximum, kind, skills) in P5_FAMILIES.items():
            assert minimum <= family_counts[family] <= maximum
            for slot in (s for s in slots if s["family"] == family):
                assert slot["kind"] == kind and slot["skill"] in skills
                assert slot["microconstruct"] in P5_MICROCONSTRUCTS[slot["skill"]]
        assert any(s["skill"] in ("nominal_clause", "fused_relative") for s in slots)
        observed.update(s["microconstruct"] for s in slots)
    # These signatures ignore names, scenarios, ordering and random tokens.
    assert len(family_signatures) >= 50
    assert len(micro_signatures) >= 120
    assert {
        "nominal_clause.embedded_question", "nominal_clause.whether_subject",
        "fused_relative.whatever_noun", "fused_relative.whichever_choice",
        "pronoun_case_reference.reflexive_coreference", "pronoun_case_reference.reciprocal_reference",
        "pronoun_case_reference.others_contrast", "pronoun_case_reference.independent_possessive",
        "passive_voice.omitted_agent", "subject_verb_agreement.intervening_modifier",
    } <= observed
    for family in P5_FAMILIES:
        assert len({sum(s["family"] == family for s in plans[0]["slots"]) for plans in full_plans}) >= 2


def test_part7_whole_test_quotas_and_genre_compatibility(full_plans):
    for plans in full_plans:
        reading = plans[5:]
        kinds = Counter(s["kind"] for p in reading for s in p["slots"])
        assert 5 <= kinds["purpose"] <= 8  # Editorial range, not an ETS rule.
        assert kinds["intent"] == kinds["sentence_insertion"] == 2
        assert kinds["vocabulary_in_context"] >= 2
        assert kinds["cross_document"] == 10
        assert sum(p["genres"] == ["chat"] for p in reading) == 2
        for plan in reading:
            local = Counter(s["kind"] for s in plan["slots"])
            assert len({s["microconstruct"] for s in plan["slots"]}) == plan["question_count"]
            assert local["purpose"] <= 1
            if local["intent"]:
                assert plan["genres"] == ["chat"]
            if local["sentence_insertion"]:
                assert plan["genres"][0] in INSERTION_GENRES
                assert plan["question_count"] in (3, 4)
            if plan["mode"] != "single":
                assert local["cross_document"] == 2


def test_multi_document_semantic_and_table_budgets(full_plans):
    for plans in full_plans:
        multi = plans[-5:]
        relations = {p["semantic_relation"] for p in multi}
        assert {"role_relationship", "chronology_change", "experience_contrast"} <= relations
        assert len(relations) == 5
        assert sum(p["semantic_relation"] == "candidate_selection" for p in multi) <= 1
        assert 2 <= sum(p["table_policy"]["max_table_blocks"] for p in multi) <= 3
        for plan in multi:
            policy = plan["table_policy"]
            assert policy["max_table_blocks"] in (0, 1)
            assert policy["candidate_selection_allowed"] == (plan["semantic_relation"] == "candidate_selection")
            if policy["max_table_blocks"] == 0:
                assert not TABLE_GENRES.intersection(plan["genres"])
            cross = [s for s in plan["slots"] if s["kind"] == "cross_document"]
            assert len({s["microconstruct"] for s in cross}) == 2
            assert all(s["minimum_documents"] == 2 and s["semantic_relation"] == plan["semantic_relation"] for s in cross)
            assert plan["evidence_requirements"]["all_documents_contribute"]
            assert plan["evidence_requirements"]["check_all_documents_for_single_source_shortcuts"]


def test_positions_vary_beyond_topic_or_token_changes(full_plans):
    chat_positions, insertion_positions, cross_positions = set(), set(), set()
    first_kinds = Counter()
    intent_local_positions, insertion_local_positions, sentence_blank_positions = set(), set(), set()
    for plans in full_plans:
        reading = plans[5:]
        chat_positions.add(tuple(i for i, p in enumerate(reading) if p["genres"] == ["chat"]))
        insertion_positions.add(tuple(i for i, p in enumerate(reading) if any(s["kind"] == "sentence_insertion" for s in p["slots"])))
        first_kinds.update(p["slots"][0]["kind"] for p in reading)
        for plan in reading:
            for i, slot in enumerate(plan["slots"]):
                if slot["kind"] == "intent":
                    intent_local_positions.add(i)
                elif slot["kind"] == "sentence_insertion":
                    insertion_local_positions.add(i)
            if plan["mode"] != "single":
                cross_positions.add(tuple(i for i, s in enumerate(plan["slots"]) if s["kind"] == "cross_document"))
        for plan in plans[1:5]:
            sentence_blank_positions.add(next(i for i, s in enumerate(plan["slots"]) if s["kind"] == "sentence_completion"))
    assert len(chat_positions) >= 15
    assert len(insertion_positions) >= 10
    assert len(cross_positions) == 10  # Every pair of five positions is reachable.
    assert intent_local_positions == {0, 1, 2}
    assert insertion_local_positions == sentence_blank_positions == {0, 1, 2, 3}
    assert {"purpose", "detail", "inference", "intent", "vocabulary_in_context", "sentence_insertion", "cross_document"} <= first_kinds.keys()
    assert first_kinds["purpose"] / sum(first_kinds.values()) < .35


def test_history_avoids_used_microtemplates_without_concentrating_new_ones():
    usage = {f"microconstruct:{code}": 1000 for variants in P5_MICROCONSTRUCTS.values() for code in variants}
    for variants in P5_MICROCONSTRUCTS.values():
        usage[f"microconstruct:{next(iter(variants))}"] = 0
    usage["microconstruct:subject_verb_agreement.each_of_head"] = 1000000
    usage["microconstruct:relative_clause.possessive_relative"] = 1000000
    original = dict(usage)
    for seed in range(32):
        plan = Planner(seed, usage).unit(5)
        constructs = [s["microconstruct"] for s in plan["slots"]]
        assert len(set(constructs)) == 30
        assert "subject_verb_agreement.each_of_head" not in constructs
        assert "relative_clause.possessive_relative" not in constructs
        assert plan == Planner(seed, usage).unit(5)
    assert usage == original


@pytest.mark.parametrize("seed", [0, 1, 9, 2026102905])
def test_existing_entry_points_remain_seeded_and_schema_compatible(seed):
    assert Planner(seed).build("full") == Planner(seed).build("full")
    assert Planner(seed).build("sample") == Planner(seed).build("sample")
    assert sum(p["question_count"] for p in Planner(seed).build("sample")) == 54
    for special in ("intent", "sentence_insertion", "vocabulary_in_context"):
        plan = Planner(seed).unit(7, 2, special=special)
        assert len(plan["slots"]) == 2
        assert sum(s["kind"] == special for s in plan["slots"]) == 1
    assert len(Planner(seed).build("5", count=7, groups=2)) == 2
    assert len(Planner(seed).unit(5, 7)["slots"]) == 7


@pytest.mark.parametrize("count", [0, 31])
def test_part5_invalid_counts_still_reject(count):
    with pytest.raises(ValueError, match="1..30"):
        Planner(1).unit(5, count)
