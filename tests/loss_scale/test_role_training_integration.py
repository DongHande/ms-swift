# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from datasets import Dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3_5ForCausalLM, Qwen3_5TextConfig

from swift import get_processor, get_template
from swift.dataset import LazyLLMDataset
from swift.loss_scale.role import inject_role_loss_scale_jsonl
from swift.model.model_meta import ModelInfo, ModelMeta
from swift.pipelines.utils import get_cached_dataset
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments

ROOT = Path(__file__).resolve().parents[2]
CONFIG = {
    'role_weights': {'system': 0.1, 'user': 0.2, 'assistant': 1., 'tool_call': 1.5, 'tool_response': 0.3},
    'assistant_weights': {'think_tag': 0.8, 'think_content': 0.5, 'answer': 1.},
}


def _local_model():
    value = os.environ.get('ROLE_TEST_MODEL')
    if not value:
        pytest.skip('ROLE_TEST_MODEL is required for native local processor/export validation')
    path = Path(value).resolve(strict=True)
    if not path.is_dir() or list(path.glob('*.py')):
        raise ValueError('Use a native local model directory without custom Python files')
    for filename in ('config.json', 'tokenizer_config.json', 'processor_config.json'):
        config_path = path / filename
        if config_path.is_file() and json.loads(config_path.read_text()).get('auto_map'):
            raise ValueError(f'Custom auto_map is not allowed: {filename}')
    return path


@pytest.fixture(params=['byte', 'local'])
def processor(request):
    if request.param == 'local':
        return get_processor(str(_local_model()))
    vocab = {char: i for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token='<|im_end|>', pad_token='<|endoftext|>',
        additional_special_tokens=['<|im_start|>', '<think>', '</think>'])
    tokenizer.model_info = ModelInfo(
        model_type='tiny_qwen3_5_text', model_dir='tiny-qwen3.5-text', torch_dtype=torch.float32,
        max_model_len=8192, quant_method=None, quant_bits=None, task_type='causal_lm')
    tokenizer.model_meta = ModelMeta(model_type='tiny_qwen3_5_text', model_groups=[], template='qwen3_5')
    return tokenizer


@pytest.fixture
def role_config(tmp_path):
    path = tmp_path / 'role.json'
    path.write_text(json.dumps(CONFIG), encoding='utf-8')
    return path


def _template(processor, role_config, **kwargs):
    template = get_template(
        processor, template_type='qwen3_5', agent_template='qwen3_5', loss_scale='role',
        role_loss_config=str(role_config), preserve_thinking=True, add_non_thinking_prefix=False, **kwargs)
    template.set_mode('train')
    return template


def _row():
    return {'messages': [
        {'role': 'system', 'content': 'help'},
        {'role': 'user', 'content': 'question'},
        {'role': 'assistant', 'content': '<think>reason</think>answer'},
    ]}


@pytest.mark.parametrize('direction', ['left', 'right'])
def test_role_truncation_keeps_target_weights_aligned(processor, role_config, direction):
    data = _row()
    data['messages'][1]['content'] *= 10
    data['messages'][2]['content'] = '<think>' + 'reason ' * 12 + '</think>' + 'answer ' * 12
    original = copy.deepcopy(data)
    full = _template(processor, role_config, max_length=8192).encode(data)
    limit = len(full['input_ids']) - 17
    encoded = _template(processor, role_config, max_length=limit, truncation_strategy=direction).encode(data)
    selected = slice(-limit, None) if direction == 'left' else slice(None, limit)
    expected_labels = full['labels'][selected]
    expected_weights = full['loss_scale'][selected]
    expected_labels[0], expected_weights[0] = -100, 0.
    assert encoded['input_ids'] == full['input_ids'][selected]
    assert encoded['labels'] == expected_labels
    assert encoded['loss_scale'] == pytest.approx(expected_weights)
    assert len(encoded['input_ids']) == len(encoded['labels']) == len(encoded['loss_scale']) == limit
    assert any(label != -100 for label in encoded['labels'])
    assert data == original


def test_encoded_arrow_cache_trainer_one_step(tmp_path, role_config):
    torch.manual_seed(7)
    vocab = {char: i for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token='<|im_end|>', pad_token='<|endoftext|>',
        additional_special_tokens=['<|im_start|>', '<think>', '</think>'])
    model_config = Qwen3_5TextConfig(
        vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, layer_types=['full_attention'],
        attn_implementation='eager', attention_dropout=0.)
    model = Qwen3_5ForCausalLM(model_config)
    model.model_info = ModelInfo(
        model_type='tiny_qwen3_5_text', model_dir=str(tmp_path / 'tiny-qwen3.5-text'),
        torch_dtype=torch.float32, max_model_len=8192, quant_method=None, quant_bits=None,
        config=model_config, task_type='causal_lm')
    model.model_meta = ModelMeta(model_type='tiny_qwen3_5_text', model_groups=[], template='qwen3_5')
    tokenizer.model_info, tokenizer.model_meta = model.model_info, model.model_meta
    template = _template(tokenizer, role_config)
    template.model = model
    encoded = template.encode(_row(), return_length=True)
    cache = tmp_path / 'encoded-arrow'
    Dataset.from_list([encoded]).save_to_disk(str(cache))
    reload_args = SimpleNamespace(
        cached_dataset=[str(cache)], cached_val_dataset=[], max_length=8192,
        truncation_strategy='left', data_seed=7, dataset_shuffle=False)
    datasets, val_datasets = get_cached_dataset(reload_args)
    assert not val_datasets
    restored = datasets[0][0]
    for name in ('input_ids', 'labels', 'loss_scale'):
        assert restored[name] == encoded[name]
    args = Seq2SeqTrainingArguments(
        output_dir=str(tmp_path / 'trainer'), use_cpu=True, max_steps=1, seed=7,
        per_device_train_batch_size=1, gradient_accumulation_steps=1,
        dataloader_num_workers=0, dataloader_persistent_workers=False,
        gradient_checkpointing=False, vit_gradient_checkpointing=False,
        report_to=[], callbacks=[], check_model=False, predict_with_generate=False,
        use_logits_to_keep=False, remove_unused_columns=False,
        save_strategy='no', eval_strategy='no', optim='sgd', learning_rate=0.01,
        weight_decay=0., max_grad_norm=0., lr_scheduler_type='constant', disable_tqdm=True)
    trainer = Seq2SeqTrainer(model=model, args=args, template=template, train_dataset=datasets[0])
    batch = template.data_collator([restored])
    labels, weights = batch['labels'], batch['loss_scale']
    assert torch.any((weights > 0) & (weights != 1))
    assert torch.all(weights[labels == -100] == 0)
    actual, outputs = trainer.compute_loss(model, copy.deepcopy(batch), return_outputs=True)
    logits = outputs.logits
    reference = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), reduction='none')
    reference = (reference.reshape_as(labels[:, 1:]) * weights[:, 1:]).sum() / (labels != -100).sum()
    torch.testing.assert_close(actual, reference)
    parameter = model.lm_head.weight
    actual_grad, = torch.autograd.grad(actual, parameter, retain_graph=True)
    reference_grad, = torch.autograd.grad(reference, parameter)
    torch.testing.assert_close(actual_grad, reference_grad)
    before = parameter.detach().clone()
    assert torch.count_nonzero(reference_grad) > 0
    result = trainer.train()
    assert trainer.state.global_step == 1
    assert math.isfinite(result.training_loss)
    assert result.training_loss == pytest.approx(reference.item(), rel=1e-5)
    assert not torch.equal(parameter.detach(), before)
    torch.testing.assert_close(parameter.detach(), before - args.learning_rate * reference_grad)


def test_native_cli_raw_cache_export_reload(tmp_path, role_config):
    model = _local_model()
    source = tmp_path / 'source.jsonl'
    source.write_text(json.dumps(_row()) + '\n', encoding='utf-8')
    source_bytes = source.read_bytes()
    materialized = tmp_path / 'materialized.jsonl'
    inject_role_loss_scale_jsonl(str(source), str(role_config), str(materialized))
    output = tmp_path / 'export'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
               MODELSCOPE_OFFLINE='1', HF_DATASETS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
    command = [
        sys.executable, str(ROOT / 'swift/cli/export.py'), '--model', str(model),
        '--to_cached_dataset', 'true', '--dataset', str(materialized), '--output_dir', str(output),
        '--loss_scale', 'role', '--role_loss_config', str(role_config), '--template', 'qwen3_5',
        '--agent_template', 'qwen3_5', '--preserve_thinking', 'true', '--add_non_thinking_prefix', 'false',
        '--max_length', '8192', '--split_dataset_ratio', '0', '--dataset_num_proc', '1',
        '--dataset_shuffle', 'false', '--load_from_cache_file', 'false', '--strict', 'true',
        '--torch_dtype', 'float32', '--attn_impl', 'eager']
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout[-12000:] + result.stderr[-12000:]
    reload_args = SimpleNamespace(
        cached_dataset=[str(output / 'train')], cached_val_dataset=[], max_length=8192,
        truncation_strategy='left', data_seed=7, dataset_shuffle=False)
    datasets, val_datasets = get_cached_dataset(reload_args)
    assert not val_datasets and len(datasets) == 1 and len(datasets[0]) == 1
    raw = datasets[0][0]
    expected_messages = json.loads(materialized.read_text())['messages']
    assert [(message['role'], message['content'], message['loss_scale']) for message in raw['messages']] == [
        (message['role'], message['content'], message['loss_scale']) for message in expected_messages]
    assert 'lengths' in raw and 'input_ids' not in raw
    processor = get_processor(str(model))
    template = _template(processor, role_config)
    direct = template.encode(_row(), return_length=True)
    lazy = LazyLLMDataset(datasets[0], template.encode, strict=True)
    restored = lazy[0]
    for name in ('input_ids', 'labels', 'loss_scale'):
        assert restored[name] == direct[name]
    assert max(raw['lengths']) == len(direct['input_ids'])
    assert source.read_bytes() == source_bytes
