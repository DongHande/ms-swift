# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from swift.loss_scale import get_loss_scale
from swift.template import ContextType, StdTemplateInputs, Template, TemplateInputs, get_template
from swift.trainers.utils import per_token_loss_func

CONFIG = {
    'role_weights': {'system': 0.1, 'user': 0.2, 'assistant': 1.0, 'tool_call': 1.5, 'tool_response': 0.3},
    'assistant_weights': {'think_tag': 0.8, 'think_content': 0.5, 'answer': 1.0},
}


def _tokenizer():
    vocab = {char: i for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token='<|im_end|>', pad_token='<|endoftext|>',
        additional_special_tokens=['<|im_start|>', '<think>', '</think>'])
    tokenizer.model_info = SimpleNamespace(config=SimpleNamespace(), task_type='causal_lm', max_model_len=8192)
    tokenizer.model_meta = SimpleNamespace(is_multimodal=False)
    return tokenizer


def _config(tmp_path, config=CONFIG):
    path = tmp_path / 'role.json'
    path.write_text(json.dumps(config))
    return str(path)


def _template(tokenizer, config_path, **kwargs):
    template = get_template(
        tokenizer, template_type='qwen3_5', loss_scale='role', role_loss_config=config_path,
        preserve_thinking=True, add_non_thinking_prefix=False, **kwargs)
    template.set_mode('train')
    return template


def _assert_weight(tokenizer, encoded, text, weight):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    positions = [i for i in range(len(encoded['input_ids']) - len(tokens) + 1)
                 if encoded['input_ids'][i:i + len(tokens)] == tokens]
    assert len(positions) == 1
    start = positions[0]
    assert encoded['loss_scale'][start:start + len(tokens)] == pytest.approx([weight] * len(tokens))


@pytest.mark.parametrize('tool_role', ['tool', 'tool_response'])
@pytest.mark.parametrize('n_users', [0, 1, 2])
def test_tool_followup_preserves_roles_and_literal_placeholders(tmp_path, tool_role, n_users):
    tokenizer = _tokenizer()
    template = _template(tokenizer, _config(tmp_path))
    data = {'messages': [
        {'role': 'user', 'content': 'initial_question'},
        {'role': 'tool_call', 'content': '{"name":"search","arguments":{}}'},
        {'role': tool_role, 'content': 'tool_result'},
        *[{'role': 'user', 'content': f'followup_{i}_{{{{QUERY}}}}', 'loss_scale': 0.4 + i}
          for i in range(n_users)],
        {'role': 'assistant', 'content': '<think>reasoning_text</think>final_answer'},
    ]}
    original = copy.deepcopy(data)
    encoded = template.encode(data)
    assert data == original
    assert len(encoded['input_ids']) == len(encoded['labels']) == len(encoded['loss_scale'])
    _assert_weight(tokenizer, encoded, 'initial_question', 0.2)
    _assert_weight(tokenizer, encoded, 'tool_result', 0.3)
    _assert_weight(tokenizer, encoded, 'reasoning_text', 0.5)
    _assert_weight(tokenizer, encoded, 'final_answer', 1.0)
    for i in range(n_users):
        _assert_weight(tokenizer, encoded, f'followup_{i}_{{{{QUERY}}}}', 0.4 + i)


def test_consecutive_users_preserve_individual_weights(tmp_path):
    tokenizer = _tokenizer()
    template = _template(tokenizer, _config(tmp_path))
    encoded = template.encode({'messages': [
        {'role': 'user', 'content': 'first_question', 'loss_scale': 0.25},
        {'role': 'user', 'content': 'second_question', 'loss_scale': 0.75},
        {'role': 'assistant', 'content': 'answer'},
    ]})
    _assert_weight(tokenizer, encoded, 'first_question', 0.25)
    _assert_weight(tokenizer, encoded, 'second_question', 0.75)


@pytest.mark.parametrize('followups', [[], ['', ''], ['same', 'same']])
def test_followup_source_tracking_handles_empty_and_duplicate_content(tmp_path, followups):
    tokenizer = _tokenizer()
    template = _template(tokenizer, _config(tmp_path))
    encoded = template.encode({'messages': [
        {'role': 'user', 'content': 'question'},
        {'role': 'tool_call', 'content': '{"name":"search","arguments":{}}'},
        {'role': 'tool', 'content': 'result'},
        *[{'role': 'user', 'content': text} for text in followups],
        {'role': 'assistant', 'content': 'answer'},
    ]})
    assert len(encoded['input_ids']) == len(encoded['labels']) == len(encoded['loss_scale'])


def test_role_equivalent_output_only_preserves_upstream_labels(tmp_path):
    tokenizer = _tokenizer()
    config = copy.deepcopy(CONFIG)
    config['role_weights'] = dict(system=0, user=0, assistant=1, tool_call=1, tool_response=0)
    config['assistant_weights'] = dict(think_tag=1, think_content=1, answer=1)
    role_template = _template(tokenizer, _config(tmp_path, config))
    upstream_template = get_template(
        tokenizer, template_type='qwen3_5', loss_scale='default+ignore_empty_think',
        preserve_thinking=True, add_non_thinking_prefix=False)
    upstream_template.set_mode('train')
    data = {'messages': [
        {'role': 'user', 'content': 'q1'}, {'role': 'assistant', 'content': '<think>reason</think>answer1'},
        {'role': 'user', 'content': 'q2'}, {'role': 'assistant', 'content': '<think>\n\n</think>answer2'},
    ]}
    encoded = role_template.encode(data)
    expected = upstream_template.encode(data)
    assert encoded['input_ids'] == expected['input_ids']
    assert encoded['labels'] == expected['labels']
    assert all(weight in {0.0, 1.0} for weight in encoded['loss_scale'])


@pytest.mark.parametrize('value', [-1, float('nan'), float('inf'), True, '0.5'])
def test_message_weights_reject_invalid_values(tmp_path, value):
    strategy = get_loss_scale('role', _config(tmp_path))
    messages = [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'a', 'loss_scale': value}]
    with pytest.raises((ValueError, TypeError)):
        strategy(['a'], [ContextType.RESPONSE], messages)


def test_token_backed_mask_multiplies_role_weight_and_masks_history_eos(tmp_path):
    strategy = get_loss_scale('role', _config(tmp_path))
    message = {'role': 'assistant', 'content': {'token_ids': [10, 11, 12], 'loss_scale': [0, 0.5, 1]},
               'loss_scale': 2.0}
    messages = [{'role': 'user', 'content': 'q'}, message]
    contexts, weights = strategy([message['content']], [ContextType.RESPONSE], messages)
    assert contexts == [[10], [11], [12]]
    assert weights == [0, 1, 2]
    message['content']['loss_scale'] = [1, 0.5, 0]
    assert not strategy._response_is_trainable(message)
    message['content']['loss_scale'] = [0, 0, 0]
    assert not strategy._response_is_trainable(message)
    message['content']['loss_scale'] = [1]
    with pytest.raises(ValueError, match='identical lengths'):
        strategy([message['content']], [ContextType.RESPONSE], messages)


def test_weighted_loss_matches_target_shift_and_gradients():
    torch.manual_seed(7)
    logits = torch.randn(2, 5, 13, requires_grad=True)
    labels = torch.tensor([[-100, 1, 2, -100, 4], [-100, 3, 4, 5, 6]])
    weights = torch.tensor([[0., 0.2, 0.5, 0., 2.], [0., 1., 0.3, 1.5, 1.]])
    token_losses = per_token_loss_func(SimpleNamespace(logits=logits), labels).reshape_as(weights)
    actual = (token_losses * weights.roll(-1, dims=-1)).sum()
    actual = actual / (labels != -100).sum()
    reference = F.cross_entropy(logits[:, :-1].reshape(-1, 13), labels[:, 1:].reshape(-1), reduction='none')
    reference = (reference.reshape(2, 4) * weights[:, 1:]).sum() / (labels != -100).sum()
    torch.testing.assert_close(actual, reference)
    actual_grad, = torch.autograd.grad(actual, logits, retain_graph=True)
    reference_grad, = torch.autograd.grad(reference, logits)
    torch.testing.assert_close(actual_grad, reference_grad)


def test_role_padding_and_packing_keep_array_alignment(tmp_path):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    tokenizer = _tokenizer()
    template = _template(tokenizer, _config(tmp_path))
    config = Qwen3_5TextConfig(
        vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, layer_types=['full_attention'])
    template.model = Qwen3_5ForCausalLM(config)
    rows = [template.encode({'messages': [{'role': 'user', 'content': text},
                                         {'role': 'assistant', 'content': 'answer'}]}, return_length=True)
            for text in ['q', 'a longer question']]
    for row in rows:
        row['length'] = len(row['input_ids'])
    batch = template.data_collator(rows)
    assert batch['input_ids'].shape == batch['labels'].shape == batch['loss_scale'].shape
    assert torch.all(batch['loss_scale'][batch['attention_mask'] == 0] == 0)
    packed = template.packing_row(rows)
    assert len(packed['input_ids']) == len(packed['labels']) == len(packed['loss_scale'])
    boundary = len(rows[0]['input_ids'])
    assert packed['labels'][boundary] == -100
    assert packed['loss_scale'][boundary] == 0


def test_standalone_tool_preserves_user_and_tool_weights(tmp_path):
    tokenizer = _tokenizer()
    template = _template(tokenizer, _config(tmp_path))
    encoded = template.encode({'messages': [
        {'role': 'user', 'content': 'standalone_question', 'loss_scale': 0.25},
        {'role': 'tool', 'content': 'standalone_result', 'loss_scale': 0.75},
        {'role': 'assistant', 'content': 'answer'},
    ]})
    _assert_weight(tokenizer, encoded, 'standalone_question', 0.25)
    _assert_weight(tokenizer, encoded, 'standalone_result', 0.75)
    assert len(encoded['input_ids']) == len(encoded['labels']) == len(encoded['loss_scale'])


def test_direct_template_inputs_retain_assistant_semantics(tmp_path):
    tokenizer = _tokenizer()
    template = _template(tokenizer, _config(tmp_path))
    inputs = TemplateInputs(chosen=StdTemplateInputs(messages=[
        {'role': 'user', 'content': 'question'},
        {'role': 'assistant', 'content': '<think>direct_reason</think>'},
        {'role': 'assistant', 'content': 'direct_answer'},
    ]))
    encoded = template.encode(inputs)
    _assert_weight(tokenizer, encoded, 'direct_reason', 0.5)
    _assert_weight(tokenizer, encoded, 'direct_answer', 1.0)


def test_non_role_placeholder_replacement_keeps_upstream_segmentation():
    contexts, kinds = [], []
    Template._concat_context_list(
        ['<s>{{SYSTEM}}</s><u>{{QUERY}}</u>', '{{RESPONSE}}'], contexts, kinds,
        system='system', query='question{{ROUND0}}', response='answer', round0=0,
        preserve_role_context=False)
    assert contexts == ['<s>system</s><u>question0</u>', 'answer']
    assert kinds == [ContextType.OTHER, ContextType.RESPONSE]


@pytest.mark.parametrize('strategy', ['default', 'last_round'])
def test_non_role_history_loss_false_does_not_supervise_separator(strategy):
    tokenizer = _tokenizer()
    template = get_template(tokenizer, template_type='qwen3_5', loss_scale=strategy,
                            preserve_thinking=True, add_non_thinking_prefix=False)
    template.set_mode('train')
    encoded = template.encode({'messages': [
        {'role': 'user', 'content': 'q1'},
        {'role': 'assistant', 'content': 'masked_history', 'loss': False},
        {'role': 'user', 'content': 'q2'},
        {'role': 'assistant', 'content': 'final_answer'},
    ]})
    supervised = [token for token in encoded['labels'] if token != -100]
    expected = tokenizer.encode('final_answer<|im_end|>\n', add_special_tokens=False)
    assert supervised == expected
