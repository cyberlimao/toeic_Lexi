"""Seeded content blueprints, independent of model sampling parameters."""
from __future__ import annotations

import random
import secrets
from collections import Counter

SCENARIOS = [
    "museum exhibition", "independent bookshop", "hotel guest services", "rail travel",
    "office relocation", "staff training", "community garden", "equipment rental",
    "conference planning", "restaurant catering", "product packaging", "warehouse dispatch",
    "software subscription", "theater production", "commercial photography", "fitness center",
    "public library", "building maintenance", "retail returns", "recruitment",
    "landscape design", "bicycle repair", "trade fair", "print shop", "employee benefits",
    "airport shuttle", "local newspaper", "furniture delivery", "volunteer program",
    "manufacturing inspection", "business awards", "customer research", "architectural studio",
    "farmers market", "translation service", "recycling service", "professional association",
    "bakery wholesale", "art supplies", "harbor tour", "audio equipment", "coworking space",
]
EVENTS = [
    "a revised schedule", "a replacement item", "a newly introduced service", "an eligibility condition",
    "a venue change", "a limited capacity", "a trial period", "a correction to earlier information",
    "an unexpected delay", "a customer preference", "a missing attachment", "a successful pilot",
    "a resource allocation", "a changed specification", "a deadline extension", "a follow-up request",
    "a customer's experience", "a founder's change of direction", "an employee's new responsibility",
    "a local success story", "a recommendation after a trial", "a difference between a promise and an experience",
    "an unexpected benefit", "a colleague's contribution", "a professional milestone", "a reader's reaction",
]
# Editorial sampling ranges, not official exam quotas. The total remains 30,
# while both family counts and the constructions within each family can change.
P5_FAMILIES = {
    "word_formation": (4, 6, "grammar", ["word_form_noun", "word_form_adjective", "word_form_adverb", "word_form_participle"]),
    "verb_system": (3, 5, "grammar", ["subject_verb_agreement", "tense_aspect", "passive_voice", "infinitive_gerund"]),
    "clause_structure": (2, 4, "grammar", ["relative_clause", "nominal_clause", "fused_relative"]),
    "reference_quantity": (3, 5, "grammar", ["pronoun_case_reference", "determiner_quantifier", "comparison"]),
    "relations": (3, 5, "grammar", ["preposition", "conjunction_vs_preposition", "subordinate_conjunction", "conjunctive_adverb"]),
    "lexical_choice": (5, 8, "vocabulary", ["business_verb_meaning", "business_noun_meaning", "business_adjective_meaning", "phrasal_verb"]),
    "collocation": (4, 6, "vocabulary", ["verb_collocation", "noun_collocation", "adjective_collocation", "adverb_collocation"]),
}
# Kept as a public skill catalog for callers that imported the previous constant.
# It no longer represents an invariant 30-question allocation.
P5_SKILLS = [(skill, kind) for _, _, kind, skills in P5_FAMILIES.values() for skill in skills]

# Stable microconstruct IDs are also usable as history keys: microconstruct:ID.
# The text specifies an observable construction, not a topic/name substitution.
_P5_VARIANTS = {
    "word_form_noun": {
        "subject_head": "Use a derived noun as the head of a subject, with natural intervening modification.",
        "object_head": "Require a noun as the object of a transitive verb, not just after an article.",
        "prepositional_complement": "Require a derived noun inside a meaningful prepositional phrase.",
    },
    "word_form_adjective": {
        "predicative_complement": "Require an adjective as a subject complement after a linking verb.",
        "attributive_modifier": "Require an adjective before a business noun; check that no noun modifier also works.",
        "object_complement": "Use an adjective as an object complement, as in keep an item accessible.",
    },
    "word_form_adverb": {
        "verb_modification": "Use an adverb to modify an action; avoid reusing increased significantly as a frame.",
        "adjective_modification": "Require an adverb modifying an adjective, not an adjacent verb.",
        "clause_modification": "Use a natural clause-level evaluative adverb with appropriate punctuation.",
    },
    "word_form_participle": {
        "experiencer_vs_stimulus": "Distinguish the experiencer's state from the thing causing that state.",
        "reduced_active_modifier": "Require an active participial modifier of a noun with a clear action relationship.",
        "reduced_passive_modifier": "Require a passive participial modifier without an explicit by-agent cue.",
    },
    "subject_verb_agreement": {
        "each_of_head": "Use an each-of subject only once; agreement follows the singular head, not the nearest plural.",
        "intervening_modifier": "Separate a subject head and its verb with a natural modifier containing another number.",
        "relative_clause_subject": "Make agreement depend on the actual relative-clause subject and its antecedent.",
        "coordinated_subject": "Use a clearly coordinated subject whose verb cannot agree with just the nearest noun.",
    },
    "tense_aspect": {
        "prior_past_event": "Use two past events to establish which event was already complete.",
        "ongoing_past_interval": "Establish an unfinished activity during a specific past interval.",
        "present_result": "Use a completed event with a current consequence and a defensible present-perfect frame.",
        "future_arrangement": "Establish a future arrangement without treating a future word as a mechanical tense rule.",
    },
    "passive_voice": {
        "omitted_agent": "Require passive voice from the subject's relation to the verb; omit any by-agent phrase.",
        "modal_passive": "Use a modal plus passive construction, with a natural object undergoing an action.",
        "perfect_passive": "Combine a clear completed-result time relation with passive voice.",
    },
    "infinitive_gerund": {
        "verb_complement": "Use a verb that selects the intended complement; do not contrast two accepted patterns.",
        "preposition_gerund": "Use a gerund after a meaningful preposition, not an infinitive marker.",
        "purpose_infinitive": "Use an infinitive expressing the purpose of an action with an unambiguous attachment.",
    },
    "relative_clause": {
        "subject_relative": "Use a relative pronoun as subject; the relative clause must lack its own subject.",
        "object_relative": "Use an object relative with an explicit subject, and avoid making omission another answer.",
        "possessive_relative": "Use whose plus a possessed noun; the following noun must have its own clause role.",
        "fronted_preposition": "Use a fronted preposition whose complement is the relative pronoun.",
    },
    "nominal_clause": {
        "embedded_question": "Use an embedded how/why/which question as a verb complement with statement word order.",
        "whether_subject": "Use a whether clause as a singular subject describing an unresolved issue.",
        "that_complement": "Use a finite that clause as a complement; distinguish a clause from a noun phrase.",
    },
    "fused_relative": {
        "whatever_noun": "Use whatever plus a noun for unrestricted choice within a nominal constituent.",
        "whichever_choice": "Use whichever for selection from a stated set; make the clause role decisive.",
        "whoever_subject": "Use whoever as both the head of a nominal constituent and its internal clause subject.",
    },
    "pronoun_case_reference": {
        "object_case": "Make the pronoun's object role and antecedent clear without a nearby case shortcut alone.",
        "independent_possessive": "Require an independent possessive replacing an established noun phrase.",
        "possessive_determiner": "Require a possessive determiner before an overt possessed noun.",
        "reflexive_coreference": "Use a reflexive because subject and object refer to the same participant.",
        "reciprocal_reference": "Use each other/one another for a reciprocal relationship among participants.",
        "others_contrast": "Contrast members of a group with others, distinguishing a pronoun from a determiner.",
    },
    "determiner_quantifier": {
        "count_mass": "Use countability plus the intended quantity, not countability alone when two choices fit.",
        "distributive_reference": "Distinguish distribution over members from collective quantity in the given context.",
        "restricted_set": "Use a stated restricted set to distinguish another, other, or the other naturally.",
    },
    "comparison": {
        "comparative_than": "Use a comparative with a clear two-way comparison and appropriate than phrase.",
        "equal_comparison": "Use an as...as comparison with the correct adjective/adverb role.",
        "superlative_set": "Establish a comparison set for a superlative; do not teach a universal article rule.",
    },
    "preposition": {
        "spatial_relationship": "Use an ordinary spatial relation that the sentence uniquely establishes.",
        "time_boundary": "Distinguish deadline, duration, and starting point using an actual event relation.",
        "abstract_relationship": "Use a natural abstract relationship such as support, responsibility, or contact.",
    },
    "conjunction_vs_preposition": {
        "concessive_clause_vs_phrase": "Distinguish a concession clause from a concession noun phrase without extra negative guardrails.",
        "causal_clause_vs_phrase": "Distinguish because/because of-type structures through the complement's syntax.",
        "temporal_clause_vs_phrase": "Distinguish a time clause from a prepositional noun/gerund phrase.",
    },
    "subordinate_conjunction": {
        "condition": "Establish a real condition for an outcome; avoid restating every rejected condition.",
        "concession": "Connect an unexpected outcome to a contrasting fact in a natural sentence.",
        "time_sequence": "Require the correct sequence between two explicitly established events.",
    },
    "conjunctive_adverb": {
        "result_relation": "Require a result relationship, checking that an ordinary concurrent reading is not also valid.",
        "contrast_relation": "Connect two contrasted facts using syntax and discourse meaning.",
        "addition_relation": "Add a further supporting point with a contextually decisive discourse relation.",
    },
    "business_verb_meaning": {
        "change_of_state": "Distinguish verbs describing a concrete change, using same-part-of-speech plausible alternatives.",
        "information_action": "Distinguish actions on information, such as verify, summarize, or distribute, by the actual task.",
        "interpersonal_action": "Distinguish a request, recommendation, or acknowledgement without relying on assumed etiquette.",
    },
    "business_noun_meaning": {
        "document_function": "Distinguish common document nouns by what the document actually does.",
        "process_or_result": "Distinguish a business process from its output or consequence through context.",
        "resource_or_role": "Distinguish relevant resource or role nouns without specialist knowledge.",
    },
    "business_adjective_meaning": {
        "functional_property": "Use a concrete need to distinguish close functional properties.",
        "status_or_availability": "Establish a status or availability that rules out plausible neighboring meanings.",
        "evaluation": "Use a natural evaluation with evidence rather than a dictionary-definition tail clause.",
    },
    "phrasal_verb": {
        "operational_action": "Use a common phrasal verb for a concrete action with credible same-form distractors.",
        "follow_up_action": "Use a phrasal verb for continuation, investigation, or response, with a decisive contextual cue.",
        "change_or_replacement": "Use a common phrasal verb for a change; do not eliminate a valid wider dictionary sense.",
    },
    "verb_collocation": {
        "verb_object": "Test a natural verb-object combination with semantically related alternatives.",
        "verb_preposition": "Test a verb's complement pattern in a complete, idiomatic phrase.",
        "two_object_relation": "Use a natural relation between two complements, such as weigh one factor against another.",
    },
    "noun_collocation": {
        "noun_preposition": "Require a noun-preposition combination in its actual business meaning.",
        "compound_noun": "Use a common business compound whose modifier has a distinct meaning.",
        "support_verb_noun": "Test the noun in an ordinary support-verb phrase with contextually plausible rivals.",
    },
    "adjective_collocation": {
        "adjective_noun": "Test a natural adjective-noun combination without mixing unrelated semantic dimensions.",
        "adjective_complement": "Require an adjective's complement pattern, not a loose frequency preference.",
        "degree_property": "Use a gradable property in context; avoid two equally natural adjectives.",
    },
    "adverb_collocation": {
        "adverb_verb": "Choose between semantically close adverbs modifying one action; establish the decisive distinction.",
        "degree_adjective": "Test an idiomatic degree modifier with an adjective in a natural business sentence.",
        "manner_discrimination": "Distinguish manners of an action; do not mix manner, frequency, and degree distractors.",
    },
}
P5_MICROCONSTRUCTS = {
    skill: {f"{skill}.{variant}": strategy for variant, strategy in variants.items()}
    for skill, variants in _P5_VARIANTS.items()
}
SINGLES = ["email", "notice", "advertisement", "article", "web_page", "letter", "chat", "review", "instructions", "schedule"]
DOUBLES = [("advertisement", "email"), ("notice", "email"), ("email", "email"), ("schedule", "memo"), ("web_page", "review"), ("order_form", "email"), ("receipt", "email"), ("booking_confirmation", "notice")]
TRIPLES = [("web_page", "email", "memo"), ("advertisement", "order_form", "email"), ("notice", "schedule", "email"), ("email", "invoice", "email"), ("article", "email", "notice"), ("booking_confirmation", "email", "schedule"), ("web_page", "feedback_form", "email")]

INSERTION_GENRES = ["email", "notice", "article", "web_page", "letter", "review", "instructions"]
TABLE_GENRES = {"schedule", "order_form", "receipt", "invoice"}
MULTI_RELATIONS = {
    "role_relationship": (
        [DOUBLES[2], DOUBLES[1]], [TRIPLES[0], TRIPLES[4]],
        {
            "infer_role": "Combine one document's actions or responsibilities with another's correspondence to identify a role; no single document states the answer.",
            "responsibility_handoff": "Trace responsibility from one participant to another across documents, using a different evidence join from the role question.",
        },
    ),
    "chronology_change": (
        [DOUBLES[2], DOUBLES[3], DOUBLES[7]], [TRIPLES[2], TRIPLES[4], TRIPLES[5]],
        {
            "earlier_later_state": "Compare earlier and later documents to infer what changed; neither alone states the change.",
            "event_sequence": "Reconstruct the relationship between two events using distinct dated or narrative clues across documents, without a calendar puzzle.",
        },
    ),
    "experience_contrast": (
        [DOUBLES[4], DOUBLES[0]], [TRIPLES[6], TRIPLES[0]],
        {
            "claim_vs_experience": "Compare a promised feature with a user's actual experience; neither document alone should identify the discrepancy.",
            "preference_vs_outcome": "Relate a person's stated preference to a later outcome, using evidence different from the advertised-claim comparison.",
        },
    ),
    "request_fulfillment": (
        [DOUBLES[5], DOUBLES[6], DOUBLES[0]], [TRIPLES[1], TRIPLES[3], TRIPLES[6]],
        {
            "requested_vs_supplied": "Compare what was requested with what was supplied; use an ordinary feature or service, not a four-candidate matrix.",
            "remaining_follow_up": "Infer the remaining follow-up from one document's request and another's completed action without either stating the final answer.",
        },
    ),
    "cause_response": (
        [DOUBLES[1], DOUBLES[2], DOUBLES[7]], [TRIPLES[0], TRIPLES[2], TRIPLES[4]],
        {
            "trigger_for_response": "Use an earlier circumstance and a later response to infer why that response occurred.",
            "action_and_consequence": "Connect an action in one document with its consequence in another using a second, independent evidence join.",
        },
    ),
    "candidate_selection": (
        [DOUBLES[0], DOUBLES[3], DOUBLES[5]], [TRIPLES[1], TRIPLES[2], TRIPLES[5]],
        {
            "requirements_match": "Match a genuine request to relevant offerings; avoid engineering four rows that each fail exactly one separate condition.",
            "selected_follow_up": "Combine the selected offering with a separate request or response to infer a follow-up; do not repeat the first elimination exercise.",
        },
    ),
}

P7_VARIANTS = {
    "purpose": {
        "overall_function": ("main_purpose", "Identify the document's main communicative function; include a true subordinate purpose among plausible distractors."),
        "reason_for_contact": ("main_purpose", "Identify why this communication was sent, distinguishing its trigger from incidental details."),
    },
    "detail": {
        "paraphrased_fact": ("paraphrase", "Ask about an explicitly supported fact using a clear paraphrase and nearby plausible competing facts."),
        "participant_role": ("direct_location", "Ask about an explicitly established participant's role or identity, without labeling a paraphrase as inference."),
        "intended_reader": ("direct_location", "Identify the readership from explicit audience cues; keep alternatives at a comparable level of specificity."),
        "time_or_place": ("direct_location", "Ask about a meaningful time or location; distractors may attach another real time/place to the wrong event."),
        "stated_reason": ("paraphrase", "Ask about an explicitly stated reason for an action, distinguishing that reason from an outcome."),
        "negative_detail": ("negative_detail", "Ask a natural NOT mentioned/stated question: support three choices and make the fourth uniquely unsupported or contradicted."),
    },
    "inference": {
        "role_from_actions": ("cross_sentence_inference", "Infer a role or relationship from at least two complementary clues; do not merely rephrase an explicit title."),
        "chronology_implication": ("chronology_inference", "Infer a consequence of two established time relations without requiring external calendar facts or arithmetic."),
        "motivation_from_context": ("cross_sentence_inference", "Infer a specific motivation from two contextual clues; do not assume standard business practices."),
        "attitude_from_evaluation": ("cross_sentence_inference", "Infer an attitude from a concrete evaluation and response, without making the key an explicit synonym of one sentence."),
    },
    "vocabulary_in_context": {
        "contextual_sense": ("lexical_substitution", "Target a common polysemous word whose intended sense is fixed by the passage; alternatives preserve its grammatical role."),
        "phrase_meaning": ("lexical_substitution", "Target a natural phrase whose local meaning differs from a plausible literal or alternative reading."),
    },
    "intent": {
        "indirect_request": ("pragmatic_intent", "Use an indirect request whose purpose is established by the preceding problem and the other speaker's response."),
        "reassurance": ("pragmatic_intent", "Use an utterance that reassures a worried participant; recover what concern it addresses from the exchange."),
        "polite_refusal": ("pragmatic_intent", "Use a tactful refusal or qualification; identify the rejected action from context without a literal no."),
        "correct_misunderstanding": ("pragmatic_intent", "Use a conversational correction whose intended meaning depends on an earlier misunderstanding."),
        "offer_responsibility": ("pragmatic_intent", "Use an offer to take responsibility; the neighboring turns establish which unresolved task is accepted."),
    },
    "sentence_insertion": {
        "cause_to_consequence": ("discourse_cohesion", "Insert an event or action motivated by the left context and needed to interpret the right-hand consequence."),
        "new_referent_bridge": ("reference_resolution", "Insert a new entity grounded in prior text and required by a later reference; check every competing antecedent."),
        "contrast_to_evaluation": ("discourse_cohesion", "Insert a contrast that develops the preceding claim and motivates the following evaluation; avoid optional/required guardrails."),
        "sequence_bridge": ("discourse_cohesion", "Insert a necessary intervening step or experience between two narrative events, using both sides to constrain placement."),
    },
}

P6_VARIANTS = {
    "tense_aspect": {
        "earlier_event": "Establish the decisive time relation in a different sentence from the blank.",
        "ongoing_or_complete": "Use the surrounding account to establish whether the activity is ongoing or complete.",
    },
    "passive_voice": {
        "agent_omitted": "Determine voice from the participant's role; do not require an explicit by-agent phrase.",
        "modal_passive": "Use a natural modal-passive construction embedded in the document's actual task.",
    },
    "word_form": {
        "complement_role": "Require a noun or adjective complement in a natural sentence, not an isolated grammar example.",
        "modifier_scope": "Require the correct modifier form and scope within a phrase relevant to the document.",
    },
    "pronoun_reference": {
        "antecedent_tracking": "Resolve the pronoun's reference from a different sentence with two plausible nearby participants.",
        "set_reference": "Track which previously introduced group or objects a pronoun refers to without gratuitous former/latter wording.",
    },
    "relative_clause": {
        "subject_or_object": "Distinguish the relative pronoun's actual clause role in a natural supporting sentence.",
        "possessed_entity": "Use a possessive relative only when the possessed noun and clause structure make it necessary.",
    },
    "contextual_vocabulary": {
        "cause_result": "Make a preceding or following sentence's cause/result necessary to select the word; local collocation alone must leave rivals plausible.",
        "participant_goal": "Use a participant's goal established elsewhere in the passage to distinguish plausible same-part-of-speech words.",
        "evaluation_context": "Require another sentence's evaluation or experience to select the intended lexical meaning.",
    },
    "logical_connector": {
        "contrast": "Connect genuinely contrasting propositions in different sentences; test concurrent and additive rivals fairly.",
        "consequence": "Connect a stated cause and its actual consequence without treating mere preference for therefore as uniqueness.",
        "supporting_point": "Add a supporting point whose relation to the preceding sentence is decisive.",
    },
    "contextual_collocation": {
        "discourse_selected_action": "Use another sentence to establish which otherwise plausible collocation expresses the intended action.",
        "outcome_selected_phrase": "Make the document's established outcome select between plausible same-form phrases.",
    },
    "reference_cohesion": {
        "entity_chain": "Track a person or object across sentences without repeating its name beside the blank.",
        "event_reference": "Refer back to an established action or event; do not use a generic this process as the only context claim.",
    },
    "sentence_cohesion": {
        "causal_bridge": "Supply an action or result that connects concrete neighboring facts; all choices perform the same communicative function.",
        "reference_bridge": "Introduce or resolve an entity needed by the adjacent discourse; all choices concern the same local participants.",
        "evaluation_bridge": "Develop a concrete evaluation or concession while preserving the document's purpose and voice.",
        "next_step": "Supply a natural next step justified by the surrounding exchange, avoiding a repetitive two-by-two condition matrix.",
    },
}


class Planner:
    def __init__(self, seed: int | None = None, usage: dict[str, int] | None = None):
        self.seed = seed if seed is not None else secrets.randbits(64)
        self.rng = random.Random(self.seed)
        self.usage = Counter(usage or {})
        self._micro_usage: Counter = Counter()

    def choose(self, category: str, values: list):
        floor = min(self.usage[f"{category}:{v}"] for v in values)
        value = self.rng.choice([v for v in values if self.usage[f"{category}:{v}"] == floor])
        self.usage[f"{category}:{value}"] += 1
        return value

    def _choose_microconstruct(self, values: list[str]) -> str:
        # A heavily used archive pattern should lose ties, but one previously
        # unseen pattern must not then fill a whole new test before catching up.
        floor = min(self._micro_usage[value] for value in values)
        code = self.choose("microconstruct", [v for v in values if self._micro_usage[v] == floor])
        self._micro_usage[code] += 1
        return code

    def _microconstruct(self, variants: dict[str, str]) -> dict:
        code = self._choose_microconstruct(list(variants))
        return {"microconstruct": code, "realization_strategy": variants[code]}

    def _part5_slots(self, count: int) -> list[dict]:
        allocations = {family: bounds[0] for family, bounds in P5_FAMILIES.items()}
        while sum(allocations.values()) < 30:
            family = self.rng.choice([name for name, bounds in P5_FAMILIES.items() if allocations[name] < bounds[1]])
            allocations[family] += 1
        slots = []
        for family, n in allocations.items():
            _, _, kind, skills = P5_FAMILIES[family]
            local_counts: Counter = Counter()
            for _ in range(n):
                floor = min(local_counts[skill] for skill in skills)
                skill = self.choose("p5_skill", [s for s in skills if local_counts[s] == floor])
                local_counts[skill] += 1
                slots.append({"skill": skill, "kind": kind, "family": family,
                              **self._microconstruct(P5_MICROCONSTRUCTS[skill]),
                              "scenario": self.choose("scenario", SCENARIOS)})
        self.rng.shuffle(slots)
        return slots[:count]

    def _reading_slot(self, kind: str, genre: str, used: set[str]) -> dict:
        variants = P7_VARIANTS[kind]
        # Do not prescribe an attitude question for a bare schedule or an
        # audience question about a private chat. Keep the source genre usable.
        if kind == "detail" and genre == "chat":
            variants = {k: v for k, v in variants.items() if k not in ("intended_reader", "negative_detail")}
        if kind == "inference" and genre == "schedule":
            variants = {"chronology_implication": variants["chronology_implication"]}
        choices = {f"{kind}.{name}": value for name, value in variants.items() if f"{kind}.{name}" not in used}
        if not choices:
            # A schedule has only one suitable inference variant. Ask a new
            # direct-detail question instead of prescribing the same operation.
            return self._reading_slot("detail", genre, used)
        code = self._choose_microconstruct(list(choices))
        operation, strategy = choices[code]
        return {"kind": kind, "skill": "negative_detail" if code == "detail.negative_detail" else kind,
                "microconstruct": code, "reading_operation": operation, "realization_strategy": strategy}

    def _part6_slot(self, skill: str, kind: str = "text_completion") -> dict:
        variants = {f"part6.{skill}.{name}": strategy for name, strategy in P6_VARIANTS[skill].items()}
        operation = {
            "contextual_vocabulary": "lexical_discrimination_in_discourse",
            "contextual_collocation": "lexical_discrimination_in_discourse",
            "pronoun_reference": "reference_resolution", "reference_cohesion": "reference_resolution",
            "logical_connector": "discourse_relation", "sentence_cohesion": "discourse_cohesion",
        }.get(skill, "grammatical_completion")
        return {"skill": skill, "kind": kind, **self._microconstruct(variants), "reading_operation": operation,
                "requires_cross_sentence_context": skill in ("contextual_vocabulary", "sentence_cohesion")}

    def _part7(self, count: int, mode: str, special: str | None, controls: dict) -> tuple:
        supported_specials = {None, "intent", "sentence_insertion", "vocabulary_in_context"}
        if special not in supported_specials:
            raise ValueError("Part 7 special must be intent, sentence_insertion or vocabulary_in_context")
        include_purpose = controls.get("purpose", self.rng.random() < .45)
        kinds = []
        metadata = {}
        if mode == "single":
            count = count if count in (2, 3, 4) else 3
            available = list(SINGLES)
            if special == "intent":
                available = ["chat"]
            elif special == "sentence_insertion":
                available = INSERTION_GENRES
            else:
                if not controls.get("allow_chat", True):
                    available.remove("chat")
                if special == "vocabulary_in_context" or controls.get("vocabulary"):
                    available = [g for g in available if g not in ("chat", "schedule")]
            genre = self.choose("p7_single", available)
            genres = [genre]
            if genre == "chat":
                special = "intent"
            if special:
                kinds.append(special)
            word_range = {2: [65, 180], 3: [120, 240], 4: [170, 300]}[count]
        elif mode in ("double", "triple"):
            count = 5
            relation = controls.get("relation") or self.choose("p7_relation", list(MULTI_RELATIONS))
            allow_table = controls.get("allow_table", self.rng.random() < .6)
            # Candidate selection may use a table; it still does not require one.
            if relation == "candidate_selection":
                allow_table = True
            doubles, triples, joins = MULTI_RELATIONS[relation]
            available = doubles if mode == "double" else triples
            if not allow_table:
                available = [bundle for bundle in available if not TABLE_GENRES.intersection(bundle)]
            genres = list(self.choose(f"p7_{mode}", available))
            genre = genres[0]
            word_range = [230, 420] if mode == "double" else [300, 510]
            kinds = ["cross_document", "cross_document"]
            metadata = {
                "semantic_relation": relation,
                "table_policy": {
                    "max_table_blocks": int(allow_table),
                    "candidate_selection_allowed": relation == "candidate_selection",
                    "guidance": "Use a table only for information a real document would tabulate. Do not turn the set into four candidates that each fail one custom constraint.",
                },
                "evidence_requirements": {
                    "cross_document_questions": 2,
                    "minimum_documents_per_cross_question": 2,
                    "all_documents_contribute": True,
                    "distinct_evidence_joins": True,
                    "check_all_documents_for_single_source_shortcuts": True,
                },
            }
        else:
            raise ValueError("Part 7 mode must be single, double or triple")
        if include_purpose:
            kinds.append("purpose")
        if controls.get("vocabulary") and "vocabulary_in_context" not in kinds:
            kinds.append("vocabulary_in_context")
        # Preserve room for the requested special even in a two-question unit.
        while len(kinds) < count:
            if "detail" not in kinds:
                kinds.append("detail")
            elif "inference" not in kinds:
                kinds.append("inference")
            else:
                kinds.append(self.rng.choice(["detail", "inference", "vocabulary_in_context"] if genre != "schedule" else ["detail", "inference"]))
        self.rng.shuffle(kinds)
        slots = []
        for kind in kinds:
            if kind == "cross_document":
                variants = {f"cross_document.{relation}.{name}": strategy for name, strategy in joins.items()}
                used = {s["microconstruct"] for s in slots}
                variants = {k: v for k, v in variants.items() if k not in used}
                slots.append({"skill": kind, "kind": kind, **self._microconstruct(variants),
                              "reading_operation": "cross_document_integration", "semantic_relation": relation,
                              "minimum_documents": 2})
            else:
                slots.append(self._reading_slot(kind, genre, {s["microconstruct"] for s in slots}))
        return count, genres, slots, word_range, metadata

    def unit(self, part: int, count: int = 30, mode: str = "single", special: str | None = None) -> dict:
        return self._unit(part, count, mode, special)

    def _unit(self, part: int, count: int = 30, mode: str = "single", special: str | None = None,
              *, reading: dict | None = None) -> dict:
        scenario = self.choose("scenario", SCENARIOS)
        event = self.choose("event", EVENTS)
        genres = []
        metadata = {}
        if part == 5:
            if not 1 <= count <= 30:
                raise ValueError("Part 5 count must be 1..30")
            slots = self._part5_slots(count)
            word_range = [0, 0]
        elif part == 6:
            count = 4
            genres = [self.choose("p6_genre", ["email", "letter", "memo", "notice", "article", "advertisement", "review"])]
            grammar = self.choose("p6_grammar", ["tense_aspect", "passive_voice", "word_form", "pronoun_reference", "relative_clause"])
            third = self.choose("p6_third", ["logical_connector", "contextual_collocation", "reference_cohesion"])
            slots = [self._part6_slot(grammar), self._part6_slot("contextual_vocabulary"),
                     self._part6_slot(third), self._part6_slot("sentence_cohesion", "sentence_completion")]
            self.rng.shuffle(slots)
            word_range = [95, 180]
        elif part == 7:
            count, genres, slots, word_range, metadata = self._part7(count, mode, special, reading or {})
        else:
            raise ValueError("Only Parts 5, 6, 7 are supported")
        levels = ["easy"] * max(1, round(count * .25)) + ["medium"] * max(1, round(count * .5))
        levels += ["hard"] * max(0, count - len(levels))
        self.rng.shuffle(levels)
        for i, slot in enumerate(slots):
            slot.update(id=f"q{i+1}", difficulty=levels[i])
        return {"part": part, "question_count": count, "mode": mode, "scenario": scenario, "event": event,
                "genres": genres, "word_range": word_range, "slots": slots,
                "variation_token": f"{self.rng.getrandbits(64):016x}", **metadata}

    def build(self, part: str, count: int = 30, groups: int = 1, mode: str = "single", special: str | None = None) -> list[dict]:
        if part == "full":
            result = [self.unit(5)] + [self.unit(6) for _ in range(4)]
            counts = [2, 2, 2, 2, 3, 3, 3, 4, 4, 4]
            chat_indices = set(self.rng.sample([i for i, n in enumerate(counts) if n <= 3], 2))
            insertion_indices = set(self.rng.sample([i for i, n in enumerate(counts) if n >= 3 and i not in chat_indices], 2))
            purpose_indices = set(self.rng.sample(range(15), self.rng.randint(5, 8)))
            vocabulary_candidates = [i for i in range(15) if i >= 10 or counts[i] >= 3 and i not in chat_indices | insertion_indices]
            vocabulary_indices = set(self.rng.sample(vocabulary_candidates, self.rng.randint(2, 4)))
            relations = ["role_relationship", "chronology_change", "experience_contrast"]
            relations += self.rng.sample(["request_fulfillment", "cause_response", "candidate_selection"], 2)
            self.rng.shuffle(relations)
            table_indices = {i for i, relation in enumerate(relations) if relation == "candidate_selection"}
            table_indices.update(self.rng.sample([i for i in range(5) if i not in table_indices], self.rng.randint(2, 3) - len(table_indices)))
            for i, n in enumerate(counts):
                feature = "intent" if i in chat_indices else "sentence_insertion" if i in insertion_indices else None
                result.append(self._unit(7, n, special=feature, reading={
                    "purpose": i in purpose_indices, "vocabulary": i in vocabulary_indices, "allow_chat": False,
                }))
            for i, relation in enumerate(relations):
                result.append(self._unit(7, mode="double" if i < 2 else "triple", reading={
                    "purpose": i + 10 in purpose_indices, "vocabulary": i + 10 in vocabulary_indices,
                    "relation": relation, "allow_table": i in table_indices,
                }))
            return result
        if part == "sample":
            return [self.unit(5), self.unit(6), self.unit(7, 4, special="sentence_insertion"), self.unit(7, 3, special="intent"), self.unit(7, 3, special="vocabulary_in_context"), self.unit(7, mode="double"), self.unit(7, mode="triple")]
        return [self.unit(int(part), count, mode, special) for _ in range(groups)]
