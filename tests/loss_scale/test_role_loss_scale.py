# Copyright (c) ModelScope Contributors. All rights reserved.
import json

import pytest

from swift.loss_scale import get_loss_scale
from swift.loss_scale.role import inject_role_loss_scale_jsonl, load_role_loss_config
from swift.template import ContextType
from swift.template.base import Template


CONFIG = {
    'role_weights': {
        'system': 0.0,
        'user': 0.2,
        'assistant': 1.0,
        'tool_call': 1.5,
        'tool_response': 0.3,
    },
    'assistant_weights': {
        'think_tag': 0.8,
        'think_content': 0.5,
        'answer': 1.0,
    },
}


def _write_config(tmp_path):
    """写入测试所需的统一角色权重配置。"""
    path = tmp_path / 'role_loss.json'
    path.write_text(json.dumps(CONFIG), encoding='utf-8')
    return str(path)


def test_role_loss_config_validation(tmp_path):
    """验证配置加载及非法连续权重拒绝逻辑。"""
    config = load_role_loss_config(_write_config(tmp_path))
    assert config.role_weights['tool_response'] == 0.3
    assert config.assistant_weights['think_content'] == 0.5

    invalid = dict(CONFIG)
    invalid['role_weights'] = dict(CONFIG['role_weights'], assistant=-1)
    path = tmp_path / 'invalid.json'
    path.write_text(json.dumps(invalid), encoding='utf-8')
    with pytest.raises(ValueError):
        load_role_loss_config(str(path))


def test_inject_role_loss_scale_jsonl(tmp_path):
    """验证JSONL流式注入及原始消息字段保留。"""
    config_path = _write_config(tmp_path)
    source = tmp_path / 'input.jsonl'
    output = tmp_path / 'output.jsonl'
    roles = [*CONFIG['role_weights'], 'tool']
    row = {
        'tools': '[]',
        'messages': [{
            'role': role,
            'content': '',
            'reasoning_content': '',
            'tool_call_id': '',
        } for role in roles]
    }
    source.write_text(json.dumps(row) + '\n', encoding='utf-8')

    summary = inject_role_loss_scale_jsonl(str(source), config_path, str(output))
    result = json.loads(output.read_text(encoding='utf-8'))

    expected_weights = [*CONFIG['role_weights'].values(), CONFIG['role_weights']['tool_response']]
    assert summary['lines'] == 1
    assert summary['roles']['tool_response'] == 2
    assert [message['role'] for message in result['messages']] == roles
    assert [message['loss_scale'] for message in result['messages']] == expected_weights
    assert all(message['reasoning_content'] == '' and message['tool_call_id'] == ''
               for message in result['messages'])


def test_assistant_semantic_weights_and_empty_think(tmp_path):
    """验证assistant语义分段及空think固定mask。"""
    loss_scale = get_loss_scale('role+ignore_empty_think', _write_config(tmp_path))
    messages = [{
        'role': 'user',
        'content': 'question',
        '_source_role': 'user',
        'loss_scale': 0.2,
    }, {
        'role': 'assistant',
        'content': '<think>\nreasoning\n</think>\n\nanswer',
        '_source_role': 'assistant',
        'loss_scale': 1.0,
    }]
    contexts, weights = loss_scale(
        ['system', 'question', messages[1]['content'], '<|im_end|>'],
        [ContextType.SYSTEM, ContextType.QUERY, ContextType.RESPONSE, ContextType.SUFFIX],
        messages,
        system_loss_scale=0.0)

    assert ''.join(contexts) == 'systemquestion<think>\nreasoning\n</think>\n\nanswer<|im_end|>'
    assert weights == [0.0, 0.2, 0.8, 0.5, 0.8, 1.0, 1.0]

    messages[1]['content'] = '<think>\n\n</think>\n\nanswer'
    contexts, weights = loss_scale([messages[1]['content']], [ContextType.RESPONSE], messages)
    assert ''.join(contexts) == messages[1]['content']
    assert weights == [0.0, 1.0]


def test_template_placeholder_boundaries():
    """验证模板固定字符与system/query内容边界不会混淆。"""
    contexts, context_types = [], []
    Template._concat_context_list(
        ['<s>{{SYSTEM}}</s><u>{{QUERY}}</u><a>'],
        contexts,
        context_types,
        system='system',
        query='query')

    assert contexts == ['<s>', 'system', '</s><u>', 'query', '</u><a>']
    assert context_types == [
        ContextType.OTHER,
        ContextType.SYSTEM,
        ContextType.OTHER,
        ContextType.QUERY,
        ContextType.OTHER,
    ]


def test_tool_response_masks_template_wrapper(tmp_path):
    """验证tool_response按role加权且外围模板字符保持mask。"""
    loss_scale = get_loss_scale('role', _write_config(tmp_path))
    messages = [{
        'role': 'tool',
        'content': [],
        '_source_role': 'tool_response',
        'loss_scale': 0.3,
    }, {
        'role': 'assistant',
        'content': 'done',
        '_source_role': 'assistant',
        'loss_scale': 1.0,
    }]
    context = '<|im_start|>user\n<tool_response>\nresult\n</tool_response><|im_end|>\n<|im_start|>assistant\n'
    contexts, weights = loss_scale([context], [ContextType.QUERY], messages)

    assert ''.join(contexts) == context
    assert weights == [0.0, 0.3, 0.0]


def test_direct_tool_call_keeps_injected_empty_think_masked(tmp_path):
    """验证直接tool_call前自动插入的空think不会参与训练。"""
    loss_scale = get_loss_scale('role', _write_config(tmp_path))
    content = '<think>\n\n</think>\n\n<tool_call>call</tool_call>'
    messages = [{
        'role': 'user',
        'content': 'question',
        '_source_role': 'user',
        'loss_scale': 0.2,
    }, {
        'role': 'assistant',
        'content': content,
        '_source_role': 'tool_call',
        'loss_scale': 1.5,
    }]
    contexts, weights = loss_scale([content], [ContextType.RESPONSE], messages)

    assert ''.join(contexts) == content
    assert weights == [0.0, 1.5]


def test_role_composes_with_hermes(tmp_path):
    """验证role基础权重与Hermes工具调用权重按乘法组合。"""
    loss_scale = get_loss_scale('role+hermes', _write_config(tmp_path))
    tool_call = '<tool_call>\n{"name":"search","arguments":{}}\n</tool_call>'
    messages = [{
        'role': 'user',
        'content': 'question',
        '_source_role': 'user',
        'loss_scale': 0.2,
    }, {
        'role': 'assistant',
        'content': tool_call,
        '_source_role': 'tool_call',
        'loss_scale': 1.5,
    }]
    contexts, weights = loss_scale([tool_call], [ContextType.RESPONSE], messages)

    assert contexts == [tool_call]
    assert weights == [3.0]


def test_historical_and_final_response_suffix(tmp_path):
    """验证历史和最终assistant结束符保持框架固定监督语义。"""
    loss_scale = get_loss_scale('role', _write_config(tmp_path))
    messages = [{
        'role': 'user',
        'content': 'q1',
        '_source_role': 'user',
        'loss_scale': 0.2,
    }, {
        'role': 'assistant',
        'content': 'a1',
        '_source_role': 'assistant',
        'loss_scale': 0.4,
    }, {
        'role': 'user',
        'content': 'q2',
        '_source_role': 'user',
        'loss_scale': 0.2,
    }, {
        'role': 'assistant',
        'content': 'a2',
        '_source_role': 'assistant',
        'loss_scale': 0.6,
    }]
    _, weights = loss_scale(
        ['a1', '<|im_end|>', 'q2', 'a2', '<|im_end|>'],
        [ContextType.RESPONSE, ContextType.RESPONSE_SUFFIX, ContextType.QUERY, ContextType.RESPONSE,
         ContextType.SUFFIX], messages)

    assert weights == [0.4, 1.0, 0.2, 0.6, 1.0]


def test_injection_normalizes_openai_before_assigning_roles(tmp_path):
    config_path = _write_config(tmp_path)
    source = tmp_path / 'openai.jsonl'
    output = tmp_path / 'injected.jsonl'
    row = {'messages': [
        {'role': 'user', 'content': 'q'},
        {'role': 'assistant', 'content': '<think>reason</think>', 'tool_calls': [
            {'id': 'call_a', 'type': 'function', 'function': {'name': 'search', 'arguments': '{}'}},
            {'id': 'call_b', 'type': 'function', 'function': {'name': 'read', 'arguments': '{}'}},
        ]},
        {'role': 'tool', 'content': 'result_b', 'tool_call_id': 'call_b'},
        {'role': 'tool', 'content': 'result_a', 'tool_call_id': 'call_a'},
        {'role': 'assistant', 'content': 'answer'},
    ]}
    source.write_text(json.dumps(row) + '\n')
    summary = inject_role_loss_scale_jsonl(str(source), config_path, str(output))
    actual = json.loads(output.read_text())['messages']
    assert [message['role'] for message in actual] == [
        'user', 'assistant', 'tool_call', 'tool_call', 'tool', 'tool', 'assistant']
    assert [message['loss_scale'] for message in actual] == [0.2, 1., 1.5, 1.5, 0.3, 0.3, 1.]
    assert [message['content'] for message in actual if message['role'] == 'tool'] == ['result_a', 'result_b']
    assert summary['roles']['tool_call'] == 2
    assert json.loads(source.read_text()) == row


def test_role_supports_official_leading_close_think_prefix(tmp_path):
    strategy = get_loss_scale('role+ignore_empty_think', _write_config(tmp_path))
    content = '</think>\nanswer'
    messages = [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': content}]
    contexts, weights = strategy([content], [ContextType.RESPONSE], messages)
    assert contexts == ['</think>\n', 'answer']
    assert weights == [0., 1.]


def test_injection_does_not_hide_preexisting_openai_weights(tmp_path):
    config_path = _write_config(tmp_path)
    source = tmp_path / 'openai.jsonl'
    output = tmp_path / 'injected.jsonl'
    source.write_text(json.dumps({'messages': [{'role': 'assistant', 'content': '', 'loss_scale': 0.1,
        'tool_calls': [{'function': {'name': 'search', 'arguments': '{}'}}]}]}) + '\n')
    with pytest.raises(ValueError, match='已包含loss_scale'):
        inject_role_loss_scale_jsonl(str(source), config_path, str(output))
    assert not output.exists()
    inject_role_loss_scale_jsonl(str(source), config_path, str(output), overwrite=True)
    message = json.loads(output.read_text())['messages'][0]
    assert message['role'] == 'tool_call'
    assert message['loss_scale'] == 1.5


@pytest.mark.parametrize('block', [
    '<tool_response>result</tool_response>',
    '<tool_result>result</tool_result>',
    '<｜tool▁output▁begin｜>result<｜tool▁output▁end｜>',
])
def test_native_tool_result_regions_keep_nonzero_weights(tmp_path, block):
    strategy = get_loss_scale('role+ignore_empty_think', _write_config(tmp_path))
    contexts, weights = strategy._split_tool_response('prefix' + block + 'suffix', .3)
    assert contexts == ['prefix', block, 'suffix']
    assert weights == [0., .3, 0.]


def test_empty_historical_response_does_not_supervise_eos(tmp_path):
    strategy = get_loss_scale('role+ignore_empty_think', _write_config(tmp_path))
    assert not strategy._response_is_trainable({'role': 'assistant', 'content': ''})
    assert not strategy._response_is_trainable({'role': 'assistant', 'content': '<think></think>'})
    assert strategy._response_is_trainable({'role': 'assistant', 'content': 'answer'})


def test_tool_call_leading_close_think_is_masked(tmp_path):
    strategy = get_loss_scale('role+ignore_empty_think', _write_config(tmp_path))
    content = '</think>\n\ncall'
    messages = [{'role': 'user', 'content': 'q'},
                {'role': 'assistant', 'content': content, '_source_role': 'tool_call'}]
    contexts, weights = strategy([content], [ContextType.RESPONSE], messages)
    assert contexts == ['</think>\n\n', 'call']
    assert weights == [0., 1.5]


@pytest.mark.parametrize('agent_name', ['glm4_5', 'glm4_7', 'glm5_1'])
@pytest.mark.parametrize('contents', [('', ''), ('same', 'same'), ('first', 'second')])
def test_native_followup_preserves_user_provenance(agent_name, contents):
    from types import SimpleNamespace
    from swift.agent_template import agent_template_map
    from swift.loss_scale.role import RoleContent

    agent = agent_template_map[agent_name]()
    agent.template_meta = SimpleNamespace(prompt=['<user>{{QUERY}}<assistant>'])
    messages = [{'role': 'user', 'content': RoleContent(content), '_source_role': 'user'} for content in contents]
    tools = [{'role': 'tool', 'content': 'result'}]
    prompt = agent._format_tool_user_followup(tools, messages)
    annotated = Template._role_followup_prompt(prompt, {'role': 'tool', 'content': ''}, messages)
    assert [part['message'] for part in annotated if part['message']['role'] == 'user'] == messages
    _, original_tools = agent._format_tool_responses('', tools)
    prefix, suffix = ''.join(agent.template_meta.prompt).split('{{QUERY}}', 1)
    assert ''.join(prompt) == ''.join(original_tools[:-1]) + ''.join(prefix + content for content in contents) + suffix


@pytest.mark.parametrize('as_list', [False, True])
def test_removing_history_thinking_keeps_tool_separator(as_list):
    from types import SimpleNamespace

    template = object.__new__(Template)
    template.template_meta = SimpleNamespace(history_thinking_prefix='')
    content = '<think>reason</think>answer\n\n'
    message = {'role': 'assistant', 'content': [content, 'tool_call'] if as_list else content,
               '_role_tool_call_prefix': '\n\n'}
    inputs = SimpleNamespace(messages=[{'role': 'user', 'content': 'q1'}, message,
                                       {'role': 'user', 'content': 'q2'}, {'role': 'assistant', 'content': 'a2'}])
    template._remove_history_thinking(inputs)
    actual = message['content'][0] if as_list else message['content']
    assert actual == 'answer\n\n'
