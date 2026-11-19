#!/usr/bin/env python3
# Copyright (c) ModelScope Contributors. All rights reserved.
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

BASELINE = {
    'role_weights': dict(system=0., user=0., assistant=1., tool_call=1., tool_response=0.),
    'assistant_weights': dict(think_tag=1., think_content=1., answer=1.),
}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configurations():
    settings = {
        'equivalent': (0., 0., 1., 1., 0., 1., 1., 1.),
        'historical': (0., .2, 1., 1., .2, 1., .5, 1.),
        'fractional': (.1, .25, .5, .75, .125, .8, .4, .6),
        'amplified': (.2, .3, 2., 3., .4, 1.5, .25, 2.),
        'answer_only': (0., 0., 1., 2., 0., 0., 0., 1.),
        'all_roles': (1., 1., 1., 1., 1., 1., 1., 1.),
    }
    return {name: {'role_weights': dict(zip(BASELINE['role_weights'], values[:5])),
                   'assistant_weights': dict(zip(BASELINE['assistant_weights'], values[5:]))}
            for name, values in settings.items()}


def examples(source):
    cases = [{'name': 'standard', 'data': copy.deepcopy(source), 'options': {}}]
    cases.append({'name': 'standard_preserve_thinking', 'data': copy.deepcopy(source),
                  'options': {'preserve_thinking': True}})
    alias = copy.deepcopy(source)
    for message in alias['messages']:
        if message['role'] == 'tool_response':
            message['role'] = 'tool'
    cases.append({'name': 'tool_alias', 'data': alias, 'options': {}})
    for texts in [('followup_1_{{QUERY}}',), ('followup_1', 'followup_2'), ('', ''), ('same', 'same')]:
        data = copy.deepcopy(source)
        index = next(i for i, message in enumerate(data['messages']) if message['role'] == 'tool_response')
        data['messages'][index + 1:index + 1] = [dict(role='user', content=text) for text in texts]
        cases.append({'name': f'followup_{len(cases)}', 'data': data, 'options': {}})
    for name, response in [('empty_think', '<think>\n\n</think>\nanswer'),
                           ('normal_think', '<think>reason</think>answer'),
                           ('leading_close_think', '</think>\nanswer')]:
        cases.append({'name': name, 'data': {'messages': [dict(role='user', content='question'),
                                                         dict(role='assistant', content=response)]},
                      'options': {'preserve_thinking': True, 'add_non_thinking_prefix': False}})
    cases.append({'name': 'direct_tool', 'data': {'tools': source['tools'], 'messages': [
        dict(role='user', content='question'),
        dict(role='tool_call', content='{"name":"Read","arguments":{"file_path":"example"}}'),
        dict(role='tool', content='result'), dict(role='assistant', content='answer')]}, 'options': {}})
    cases.append({'name': 'standalone_tool', 'data': {'messages': [
        dict(role='user', content='question'), dict(role='tool', content='result'),
        dict(role='assistant', content='answer')]}, 'options': {}})
    cases.append({'name': 'masked_history', 'data': {'messages': [
        dict(role='user', content='q1'), dict(role='assistant', content='a1', loss=False),
        dict(role='user', content='q2'), dict(role='assistant', content='a2')]}, 'options': {}})
    calls = [{'id': f'call_{i}', 'type': 'function', 'function': {
        'name': 'Read', 'arguments': json.dumps({'file_path': f'example_{i}'})}} for i in range(2)]
    cases.append({'name': 'parallel_openai', 'data': {'tools': source['tools'], 'messages': [
        dict(role='user', content='question'), dict(role='assistant', content='<think>reason</think>', tool_calls=calls),
        dict(role='tool', content='result_1', tool_call_id='call_1'),
        dict(role='tool', content='result_0', tool_call_id='call_0'),
        dict(role='assistant', content='answer')]}, 'options': {}})
    parallel_source = cases[-1]['data']
    for count in (1, 2):
        data = copy.deepcopy(parallel_source)
        data['messages'][-1:-1] = [dict(role='user', content=f'parallel_followup_{i}') for i in range(count)]
        cases.append({'name': f'parallel_followups_{count}', 'data': data, 'options': {}})
    for content in (None, '', 'answer', '<think>reason</think>', '<think>reason</think>answer'):
        data = copy.deepcopy(source)
        index = next(i for i, message in enumerate(data['messages']) if message['role'] == 'assistant')
        data['messages'][index]['content'] = content
        if content is None:
            del data['messages'][index]['content']
            call = json.loads(data['messages'].pop(index + 1)['content'])
            data['messages'][index]['tool_calls'] = [{'id': 'call_content', 'type': 'function', 'function': call}]
            data['messages'][index + 1]['tool_call_id'] = 'call_content'
        cases.append({'name': f'call_content_{len(cases)}', 'data': data,
                      'options': {'preserve_thinking': True}})
    for enabled in (False, True):
        for preserve in (False, True):
            cases.append({'name': f'thinking_{enabled}_{preserve}', 'data': copy.deepcopy(source),
                          'options': {'enable_thinking': enabled, 'preserve_thinking': preserve}})
    data = copy.deepcopy(source)
    index = next(i for i, message in enumerate(data['messages']) if message['role'] == 'user')
    data['messages'][index:index] = [dict(role='user', content='empty_history_question'),
                                   dict(role='assistant', content='')]
    cases.append({'name': 'empty_history_no_prefix', 'data': data,
                  'options': {'add_non_thinking_prefix': False, 'preserve_thinking': True}})
    data = copy.deepcopy(source)
    for message in data['messages']:
        if message['role'] in ('assistant', 'tool_call'):
            message['loss'] = False
            break
    cases.append({'name': 'masked_tool_history', 'data': data, 'options': {'preserve_thinking': True}})
    data = copy.deepcopy(source)
    index = next(i for i, message in enumerate(data['messages']) if message['role'] == 'assistant')
    data['messages'][index]['content'] = '<think></think>'
    cases.append({'name': 'empty_think_before_tool', 'data': data, 'options': {'preserve_thinking': True}})
    return cases


def region_check(encoded, tokenizer, config):
    ids, labels, weights = encoded['input_ids'], encoded['labels'], encoded['effective_weights']
    special = {name: tokenizer.convert_tokens_to_ids(name) for name in [
        '<|im_start|>', '<|im_end|>', '<think>', '</think>', '<tool_call>', '</tool_call>',
        '<tool_response>', '</tool_response>']}
    regions = []

    def add(category, start, end, expected):
        if end <= start:
            return
        regions.append(dict(category=category, start=start, end=end, expected=expected,
                            actual=sorted(set(weights[start:end])),
                            matches=all(abs(weight - expected) < 1e-9 for weight in weights[start:end])))

    starts = [i for i, token in enumerate(ids) if token == special['<|im_start|>']]
    newline = tokenizer.encode('\n', add_special_tokens=False)
    for turn, start in enumerate(starts):
        next_start = starts[turn + 1] if turn + 1 < len(starts) else len(ids)
        end = ids.index(special['<|im_end|>'], start + 1, next_start)
        role = tokenizer.decode(ids[start + 1:start + 2])
        header = tokenizer.encode(f'<|im_start|>{role}\n', add_special_tokens=False)
        assert ids[start:start + len(header)] == header, 'Unexpected ChatML header tokenization'
        cursor = start + len(header)
        turn_end = end + 1
        if ids[turn_end:turn_end + len(newline)] == newline:
            turn_end += len(newline)
        add(f'{role}_header', start, cursor, 0.)
        if role == 'system':
            add('system_content', cursor, end, config['role_weights']['system'])
            add('system_end', end, turn_end, 0.)
        elif role == 'user':
            tool_starts = [i for i in range(cursor, end) if ids[i] == special['<tool_response>']]
            if not tool_starts:
                add('user_content', cursor, end, config['role_weights']['user'])
            else:
                for tool_start in tool_starts:
                    add('tool_wrapper', cursor, tool_start, 0.)
                    tool_end = ids.index(special['</tool_response>'], tool_start + 1, end) + 1
                    add('tool_response', tool_start, tool_end, config['role_weights']['tool_response'])
                    cursor = tool_end
                add('tool_wrapper', cursor, end, 0.)
            add('user_end', end, turn_end, 0.)
        elif role == 'assistant':
            if cursor < end and ids[cursor] == special['<think>']:
                close = ids.index(special['</think>'], cursor + 1, end)
                if not tokenizer.decode(ids[cursor + 1:close]).strip():
                    text = tokenizer.decode(ids[cursor:end])
                    match = re.match(r'^<think>\s*</think>\s*', text, re.DOTALL)
                    prefix = tokenizer.encode(match.group(0), add_special_tokens=False)
                    assert ids[cursor:cursor + len(prefix)] == prefix
                    add('empty_think', cursor, cursor + len(prefix), 0.)
                    cursor += len(prefix)
                else:
                    base = config['role_weights']['assistant']
                    add('think_open', cursor, cursor + 1, base * config['assistant_weights']['think_tag'])
                    add('think_content', cursor + 1, close, base * config['assistant_weights']['think_content'])
                    add('think_close', close, close + 1, base * config['assistant_weights']['think_tag'])
                    cursor = close + 1
            while cursor < end:
                try:
                    tool_start = ids.index(special['<tool_call>'], cursor, end)
                except ValueError:
                    tool_start = end
                add('answer_or_separator', cursor, tool_start,
                    config['role_weights']['assistant'] * config['assistant_weights']['answer'])
                if tool_start == end:
                    break
                tool_end = ids.index(special['</tool_call>'], tool_start + 1, end) + 1
                add('tool_call', tool_start, tool_end, config['role_weights']['tool_call'])
                cursor = tool_end
            add('assistant_end', end, turn_end, 1.)
        else:
            raise AssertionError(f'Unexpected rendered role {role}')
    coverage = [0] * len(ids)
    for region in regions:
        for index in range(region['start'], region['end']):
            coverage[index] += 1
    mismatched_labels = [i for i, (token, label, weight) in enumerate(zip(ids, labels, weights))
                         if label != (token if weight > 0 else -100)]
    return dict(passed=all(region['matches'] for region in regions) and all(n == 1 for n in coverage)
                and not mismatched_labels, regions=regions, coverage_counts=dict(Counter(coverage)),
                label_mismatches=mismatched_labels, weight_counts=dict(Counter(weights)))


def worker(args):
    sys.path.insert(0, str(args.repo.resolve()))
    import swift
    from swift import get_processor, get_template
    assert Path(swift.__file__).resolve().is_relative_to(args.repo.resolve())
    if list(args.model.glob('*.py')):
        raise ValueError('Validation requires a native model without repository Python files.')
    for filename in ('config.json', 'tokenizer_config.json', 'processor_config.json'):
        path = args.model / filename
        if path.is_file() and json.loads(path.read_text()).get('auto_map'):
            raise ValueError(f'Validation does not execute custom auto_map code: {filename}')
    print(f'Loading native processor: {args.model}', flush=True)
    processor = get_processor(str(args.model))
    tokenizer = getattr(processor, 'tokenizer', processor)
    print(f'Processor ready: {args.template_type}/{args.agent_template}', flush=True)
    manifest = json.loads((args.output / 'cases.json').read_text())
    config_set = json.loads((args.output / 'configs.json').read_text())
    result = {'repo': str(args.repo), 'import_path': swift.__file__,
              'head': subprocess.check_output(['git', '-C', str(args.repo), 'rev-parse', 'HEAD'], text=True).strip(),
              'cases': {}, 'weighted': {}}

    def encode(case, strategy, config_path=None):
        options = dict(case['options'])
        if config_path is not None:
            options['role_loss_config'] = config_path
        template = get_template(processor, template_type=args.template_type, agent_template=args.agent_template,
                                max_length=262144, loss_scale=strategy, **options)
        template.set_mode('train')
        original = copy.deepcopy(case['data'])
        encoded = template.encode(case['data'])
        assert original == case['data'], 'Template mutated original data'
        ids, labels = encoded['input_ids'], encoded['labels']
        weights = encoded.get('loss_scale')
        assert len(ids) == len(labels) and (weights is None or len(weights) == len(ids))
        effective = [float(weight) if label != -100 else 0. for weight, label in
                     zip(weights if weights is not None else [1.] * len(ids), labels)]
        return dict(input_ids=ids, labels=labels, loss_scale=weights, effective_weights=effective,
                    decoded_text=tokenizer.decode(ids), length=len(ids), supervised=sum(n != -100 for n in labels))

    for case in manifest:
        try:
            config_path = str(args.output / 'config-equivalent.json') if args.mode == 'role' else None
            strategy = 'role+ignore_empty_think' if args.mode == 'role' else 'default+ignore_empty_think'
            result['cases'][case['name']] = encode(case, strategy, config_path)
        except Exception as error:
            result['cases'][case['name']] = {'error': f'{type(error).__name__}: {error}'}
    result['injection_roundtrips'] = {}
    if args.mode == 'role' and not args.equivalence_only:
        from swift.loss_scale.role import inject_role_loss_scale_jsonl
        from role_region_oracle import check_regions
        for name, config in config_set.items():
            config_path = args.output / f'config-{name}.json'
            original = manifest[0]
            source = args.output / 'standard.jsonl'
            save(source.with_suffix('.json'), original['data'])
            source.write_text(json.dumps(original['data'], ensure_ascii=False) + '\n')
            injected_path = args.output / f'injected-{name}.jsonl'
            try:
                summary = inject_role_loss_scale_jsonl(str(source), str(config_path), str(injected_path))
                injected = dict(original, data=json.loads(injected_path.read_text()))
                raw = encode(original, 'role+ignore_empty_think', str(config_path))
                materialized = encode(injected, 'role+ignore_empty_think', str(config_path))
                result['weighted'][name] = dict(raw=raw, injected=materialized, injection=summary,
                                               regions=check_regions(raw, tokenizer, config, args.template_type, original))
            except Exception as error:
                result['weighted'][name] = {'error': f'{type(error).__name__}: {error}'}
        result['injection_roundtrips'] = {}
        for name in config_set:
            config_path = args.output / f'config-{name}.json'
            source = args.output / f'cases-{name}.jsonl'
            source.write_text(''.join(json.dumps(case['data'], ensure_ascii=False) + '\n' for case in manifest))
            injected_path = args.output / f'cases-injected-{name}.jsonl'
            inject_role_loss_scale_jsonl(str(source), str(config_path), str(injected_path))
            injected_rows = [json.loads(line) for line in injected_path.read_text().splitlines()]
            assert len(injected_rows) == len(manifest)
            for case, row in zip(manifest, injected_rows):
                try:
                    raw = encode(case, 'role+ignore_empty_think', str(config_path))
                    injected = encode(dict(case, data=row), 'role+ignore_empty_think', str(config_path))
                    comparison = compare(raw, injected)
                    regions = check_regions(raw, tokenizer, config_set[name], args.template_type, case)
                    result['injection_roundtrips'][f'{name}/{case["name"]}'] = dict(
                        passed=comparison['passed'] and regions['passed'], injection_equal=comparison, regions=regions)
                except Exception as error:
                    result['injection_roundtrips'][f'{name}/{case["name"]}'] = {
                        'passed': False, 'error': f'{type(error).__name__}: {error}'}
    save(args.output / f'{args.mode}.json', result)


def compare(left, right):
    if 'error' in left or 'error' in right:
        return dict(passed=False, left_error=left.get('error'), right_error=right.get('error'))
    differences = {}
    for field in ('input_ids', 'labels', 'effective_weights', 'decoded_text'):
        a, b = left[field], right[field]
        if a != b:
            if isinstance(a, list):
                positions = [i for i in range(min(len(a), len(b))) if a[i] != b[i]]
                differences[field] = dict(left_length=len(a), right_length=len(b),
                                          mismatch_count=len(positions), first_positions=positions[:12])
            else:
                differences[field] = 'different'
    return dict(passed=not differences, differences=differences)


def main(args):
    source = json.loads(args.input.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    cases = examples(source)
    config_set = configurations()
    if args.collect_only:
        assert json.loads((args.output / 'cases.json').read_text()) == cases, 'Case manifest changed'
        assert json.loads((args.output / 'configs.json').read_text()) == config_set, 'Config manifest changed'
        for name, config in config_set.items():
            assert json.loads((args.output / f'config-{name}.json').read_text()) == config
    else:
        save(args.output / 'cases.json', cases)
        save(args.output / 'configs.json', config_set)
        for name, config in config_set.items():
            save(args.output / f'config-{name}.json', config)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
               HF_DATASETS_OFFLINE='1', MODELSCOPE_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    if args.prepare_only:
        return 0
    worker_repos = [] if args.collect_only else [('official', args.official), ('fork_default', args.fork), ('role', args.fork)]
    for mode, repo in worker_repos:
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--mode', mode, '--repo', str(repo),
                   '--model', str(args.model), '--output', str(args.output),
                   '--template-type', args.template_type, '--agent-template', args.agent_template]
        if args.equivalence_only:
            command.append('--equivalence-only')
        with (args.output / f'{mode}.log').open('w') as log:
            completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=args.worker_timeout)
        if completed.returncode:
            raise RuntimeError(f'{mode} worker failed; see {mode}.log')
    official, fork, role = [json.loads((args.output / f'{name}.json').read_text())
                            for name in ('official', 'fork_default', 'role')]
    report = dict(model=str(args.model), template_type=args.template_type, agent_template=args.agent_template,
                  scope='equivalence_only' if args.equivalence_only else 'equivalence_and_all_case_regions',
                  source_sha256=digest(args.input),
                  repositories={name: {key: value[key] for key in ('repo', 'head', 'import_path')}
                                for name, value in [('official', official), ('fork_default', fork), ('role', role)]},
                  environment={name: importlib.metadata.version(name) for name in
                               ['torch', 'transformers', 'tokenizers', 'qwen_vl_utils', 'decord', 'pytest', 'kernels']},
                  no_regression={}, equivalent={}, weighted={}, expected_rejections={})
    for name, expected in official['cases'].items():
        if (args.template_type == 'deepseek_v41' and name == 'standalone_tool'
                and expected.get('error', '').startswith('NotImplementedError:')):
            report['expected_rejections'][name] = dict(
                passed=expected == fork['cases'][name] == role['cases'][name], official_error=expected['error'],
                fork_error=fork['cases'][name].get('error'), role_error=role['cases'][name].get('error'),
                outcome='unsupported_by_official_template')
            continue
        report['no_regression'][name] = compare(expected, fork['cases'][name])
        report['equivalent'][name] = compare(expected, role['cases'][name])
    for name, entry in role['weighted'].items():
        if 'error' in entry:
            report['weighted'][name] = dict(passed=False, error=entry['error'])
        else:
            injection_equal = compare(entry['raw'], entry['injected'])
            report['weighted'][name] = dict(passed=injection_equal['passed'] and entry['regions']['passed'],
                                           injection_equal=injection_equal, regions=entry['regions'])
    report['injection_roundtrips'] = role['injection_roundtrips']
    if 'standalone_tool' in report['expected_rejections']:
        for name in list(report['injection_roundtrips']):
            if name.endswith('/standalone_tool'):
                entry = report['injection_roundtrips'].pop(name)
                expected_error = official['cases']['standalone_tool']['error']
                report['expected_rejections'][name] = dict(
                    passed=entry.get('error') == expected_error, error=entry.get('error'),
                    outcome='unsupported_by_official_template')
    report['passed'] = all(item['passed'] for group in
                           ('no_regression', 'equivalent', 'weighted', 'injection_roundtrips', 'expected_rejections')
                           for item in report[group].values())
    save(args.output / 'report.json', report)
    print(json.dumps({group: {name: item['passed'] for name, item in report[group].items()}
                      for group in ('no_regression', 'equivalent', 'weighted')}, indent=2), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Independent official/fork/role real-tokenizer validation')
    parser.add_argument('--mode', choices=['official', 'fork_default', 'role'])
    parser.add_argument('--repo', type=Path)
    parser.add_argument('--official', type=Path)
    parser.add_argument('--fork', type=Path)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--template-type', default='qwen3_5')
    parser.add_argument('--agent-template', default='qwen3_5')
    parser.add_argument('--worker-timeout', type=int, default=180)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--collect-only', action='store_true')
    parser.add_argument('--equivalence-only', action='store_true')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode:
        worker(args)
    else:
        if not all([args.official, args.fork, args.input]):
            parser.error('--official, --fork and --input are required for orchestration')
        sys.exit(main(args))
