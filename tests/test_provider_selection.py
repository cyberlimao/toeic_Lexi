from types import SimpleNamespace

import pytest

from toeic_engine import cli
from toeic_engine.anthropic_provider import ANTHROPIC_MODEL, AnthropicProvider, ReviewerRouter
from toeic_engine.openai_provider import OPENAI_MODEL, OpenAIProvider
from toeic_engine.engine import PROMPT_NAMES, prompt, prompt_version
from toeic_engine.provider import MODEL, ProviderError


class FakeTransport:
    def __init__(self, directory, max_calls, timeout):
        self.directory, self.max_calls, self.timeout = directory, max_calls, timeout
        self.calls, self.labels, self.closed = 0, [], 0

    def generate(self, system, payload, schema, label, **kwargs):
        # A retrying transport must also obey the router's remaining allowance.
        for _ in range(payload.get('attempts', 1)):
            if self.calls >= self.max_calls:
                raise ProviderError('transport call limit')
            self.calls += 1
        self.labels.append(label)
        return label

    def close(self):
        self.closed += 1


class GeminiTransport(FakeTransport):
    model = MODEL
    reasoning = {'thinking_level': 'HIGH'}


class ClaudeTransport(FakeTransport):
    model = ANTHROPIC_MODEL
    reasoning = {'effort': 'xhigh', 'thinking': 'adaptive'}


class OpenAITransport(FakeTransport):
    model = OPENAI_MODEL
    reasoning = {'effort': 'xhigh'}


@pytest.fixture
def transports(monkeypatch):
    monkeypatch.setattr(cli, 'VertexProvider', GeminiTransport)
    monkeypatch.setattr(cli, 'AnthropicProvider', ClaudeTransport)
    monkeypatch.setattr(cli, 'OpenAIProvider', OpenAITransport)


@pytest.mark.parametrize('author', ['gemini', 'claude', 'openai'])
@pytest.mark.parametrize('review', ['gemini', 'claude', 'openai'])
def test_provider_combinations_route_roles_and_report_provenance(tmp_path, transports, author, review):
    routed = cli.create_provider(author, review, tmp_path, 7, 123)
    classes = {'gemini': GeminiTransport, 'claude': ClaudeTransport, 'openai': OpenAITransport}
    assert type(routed.author) is classes[author]
    assert type(routed.reviewer) is classes[review]
    assert routed.author.directory == tmp_path / 'author'
    assert routed.reviewer.directory == tmp_path / 'reviewer'
    assert routed.author.timeout == routed.reviewer.timeout == 123
    for role in PROMPT_NAMES:
        routed.generate(prompt(role), {}, None, role)
    assert routed.author.labels == ['author']
    assert set(routed.reviewer.labels) == set(PROMPT_NAMES) - {'author'}
    assert routed.author_provenance == {'model': classes[author].model, **classes[author].reasoning}
    assert routed.review_provenance == {'model': classes[review].model, **classes[review].reasoning}
    assert routed.model == classes[author].model
    assert routed.reasoning == classes[author].reasoning
    assert routed.calls == 7
    with pytest.raises(ProviderError, match='Combined'):
        routed.generate(prompt('author'), {}, None, 'extra')
    routed.close()
    assert routed.author.closed == routed.reviewer.closed == 1


def test_network_retries_share_cap_between_roles(tmp_path, transports):
    routed = cli.create_provider('claude', 'gemini', tmp_path, 4, 123)
    routed.generate(prompt('author'), {'attempts': 3}, None, 'author')
    with pytest.raises(ProviderError, match='transport call limit'):
        routed.generate(prompt('solver'), {'attempts': 3}, None, 'solver')
    assert routed.calls == 4
    with pytest.raises(ProviderError, match='Combined'):
        routed.generate(prompt('author'), {}, None, 'extra')


def test_shared_transport_not_double_counted_or_closed(tmp_path):
    transport = ClaudeTransport(tmp_path, 2, 123)
    routed = ReviewerRouter(transport, transport, prompt('author'), 2)
    routed.generate(prompt('author'), {}, None, 'author')
    routed.generate(prompt('solver'), {}, None, 'solver')
    assert routed.calls == 2
    routed.close()
    assert transport.closed == 1


def test_factory_closes_author_when_reviewer_setup_fails(tmp_path, monkeypatch):
    author = ClaudeTransport(tmp_path, 7, 123)
    monkeypatch.setattr(cli, 'AnthropicProvider', lambda *args: author)
    def fail(*args):
        raise ProviderError('missing reviewer credential')
    monkeypatch.setattr(cli, 'VertexProvider', fail)
    with pytest.raises(ProviderError, match='missing reviewer'):
        cli.create_provider('claude', 'gemini', tmp_path, 7, 123)
    assert author.closed == 1


@pytest.mark.parametrize('command', ['generate', 'resume', 'audit'])
def test_legacy_defaults_and_persisted_choices(command):
    args = SimpleNamespace(command=command, author_provider=None, review_provider=None)
    assert cli.select_providers(args, {}) == {'author_provider': 'gemini', 'review_provider': 'gemini'}
    saved = {'author_provider': 'claude', 'review_provider': 'claude'}
    assert cli.select_providers(args, saved) == saved
    assert cli.select_providers(args, {'review_provider': 'claude'}) == {'author_provider': 'gemini', 'review_provider': 'claude'}


@pytest.mark.parametrize('role', ['author', 'review'])
def test_resume_rejects_changed_role(role):
    args = SimpleNamespace(command='resume', author_provider=None, review_provider=None)
    setattr(args, role + '_provider', 'claude')
    with pytest.raises(ValueError, match=f'Cannot change {role}'):
        cli.select_providers(args, {})


@pytest.mark.parametrize('command, positional', [('generate', ['--part', '5']), ('resume', ['saved']), ('audit', ['saved', '--out', 'new'])])
def test_cli_accepts_author_selection(monkeypatch, command, positional):
    monkeypatch.setattr('sys.argv', ['toeic', command, *positional, '--author-provider', 'claude', '--review-provider', 'gemini'])
    args = cli.arguments()
    assert (args.author_provider, args.review_provider) == ('claude', 'gemini')


def test_initial_author_uses_checkpoint_not_audit_provider():
    raw = {'provenance': {'model': MODEL, 'thinking_level': 'HIGH', 'reviewer': {'model': ANTHROPIC_MODEL}, 'accepted_at': 'old'}}
    assert cli.initial_author_provenance(raw, {'author_provider': 'claude'}, False) == {'model': MODEL, 'thinking_level': 'HIGH'}
    assert cli.initial_author_provenance({}, {'author_provider': 'claude'}, True) == AnthropicProvider.author_provenance
    assert cli.initial_author_provenance({}, {}, True) == {'model': MODEL, 'thinking_level': 'HIGH'}
    assert cli.initial_author_provenance({}, {'author_provider': 'openai'}, True) == OpenAIProvider.author_provenance


@pytest.mark.parametrize('command,positional', [('generate', ['--part', 'full']), ('resume', ['saved']), ('audit', ['saved', '--out', 'new'])])
def test_openai_choice_is_explicit_and_inherited(monkeypatch, command, positional):
    monkeypatch.setattr('sys.argv', ['toeic', command, *positional, '--author-provider', 'openai', '--review-provider', 'openai'])
    args = cli.arguments()
    assert cli.select_providers(args, {'author_provider': 'openai', 'review_provider': 'openai'}) == {'author_provider': 'openai', 'review_provider': 'openai'}
    assert cli.select_providers(SimpleNamespace(command=command, author_provider=None, review_provider=None), {'author_provider':'openai','review_provider':'openai'}) == {'author_provider':'openai','review_provider':'openai'}
