# Copyright (c) ModelScope Contributors. All rights reserved.
import re
from collections import Counter


FORMATS = {
    'qwen3_5': (r'<\|im_start\|>(system|user|assistant)\n', r'<\|im_end\|>\n?',
                r'<tool_call>.*?</tool_call>'),
    'deepseek_v41': (r'<｜(User|Assistant)｜>', r'<｜end▁of▁sentence｜>',
                     r'<｜DSML｜ calls>.*?</｜DSML｜ calls>'),
    'glm5_next': (r'<\|(system|user|assistant|observation)\|>', None,
                  r'<tool_call>.*?</tool_call>'),
}
TOOL_RESULTS = re.compile(r'<tool_response>.*?</tool_response>|<tool_result>.*?</tool_result>', re.DOTALL)
EMPTY_THINK = re.compile(r'^(?:<think>\s*</think>|</think>)\s*', re.DOTALL)


def check_regions(encoded, tokenizer, config, template_type, case):
    ids = encoded['input_ids']
    text = tokenizer.decode(ids, clean_up_tokenization_spaces=False)
    header_pattern, end_pattern, call_pattern = FORMATS[template_type]
    headers = list(re.finditer(header_pattern, text))
    if not headers:
        raise AssertionError('No native role headers found')
    spans = []
    role_weights, assistant_weights = config['role_weights'], config['assistant_weights']

    def add(category, start, end, weight):
        if end > start:
            spans.append((category, start, end, float(weight)))

    if template_type == 'deepseek_v41':
        bos = '<｜begin▁of▁sentence｜>'
        assert text.startswith(bos)
        add('system_content', len(bos), headers[0].start(), role_weights['system'])
    masked = [message['content'] for message in case['data']['messages']
              if message.get('loss') is False and isinstance(message.get('content'), str) and message['content']]
    empty_assistants, current = [], None
    for message in case['data']['messages']:
        if message['role'] in ('assistant', 'tool_call'):
            if current is None:
                current = False
            content = message.get('content')
            if (message['role'] == 'assistant' and not (message.get('tool_calls') and not content)
                    and isinstance(content, str)
                    and (not content.strip() or re.fullmatch(r'<think>\s*</think>|</think>', content.strip()))):
                current = True
        elif current is not None:
            empty_assistants.append(current)
            current = None
    if current is not None:
        empty_assistants.append(current)
    assistant_index = 0
    for index, header in enumerate(headers):
        role = header.group(1).lower()
        start = header.end()
        next_start = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        end_match = re.search(end_pattern, text[start:next_start]) if end_pattern else None
        end = start + end_match.start() if end_match else next_start
        body = text[start:end]
        if role == 'system':
            if template_type == 'glm5_next' and index > 0 and headers[index - 1].group(1) == 'system':
                add('system_embedded_header', header.start(), start, role_weights['system'])
            add('system_content', start, end, role_weights['system'])
        elif role in ('user', 'observation'):
            if role == 'observation' and index > 0 and headers[index - 1].group(1) == 'assistant':
                previous_body = text[headers[index - 1].end():header.start()]
                if previous_body.rstrip().endswith('</tool_call>'):
                    add('tool_call_terminator', header.start(), start, role_weights['tool_call'])
            matches = list(TOOL_RESULTS.finditer(body))
            if matches:
                for match in matches:
                    add('tool_response', start + match.start(), start + match.end(), role_weights['tool_response'])
            else:
                add('user_content', start, end, role_weights['user'])
        elif role == 'assistant':
            preceding_empty = empty_assistants[assistant_index] if assistant_index < len(empty_assistants) else False
            assistant_index += 1
            base = role_weights['assistant']
            is_masked = any(content in body for content in masked)
            if is_masked:
                base = 0.
            cursor = start
            empty = EMPTY_THINK.match(body)
            if empty:
                cursor += empty.end()
                if template_type == 'deepseek_v41' and preceding_empty and re.search(call_pattern, body, re.DOTALL):
                    tag = re.match(r'^(?:<think>\s*</think>|</think>)', body, re.DOTALL)
                    add('tool_call_leading_separator', start + tag.end(), cursor, role_weights['tool_call'])
            elif body.startswith('<think>'):
                close = text.index('</think>', cursor, end)
                add('think_tag', cursor, cursor + len('<think>'), base * assistant_weights['think_tag'])
                add('think_content', cursor + len('<think>'), close, base * assistant_weights['think_content'])
                add('think_tag', close, close + len('</think>'), base * assistant_weights['think_tag'])
                cursor = close + len('</think>')
            answer_start = cursor
            calls = list(re.finditer(call_pattern, text[cursor:end], re.DOTALL))
            if calls:
                call_start, call_end = cursor + calls[0].start(), cursor + calls[-1].end()
                separator_start = call_start
                if template_type == 'deepseek_v41' and is_masked and text[call_start - 2:call_start] == '\n\n':
                    separator_start = max(cursor, call_start - 2)
                separator_weight = role_weights['tool_call'] if preceding_empty else base * assistant_weights['answer']
                add('answer_or_separator', cursor, separator_start, separator_weight)
                add('tool_call_batch', separator_start, call_end, role_weights['tool_call'])
                answer_start = call_end
            add('answer', answer_start, end, base * assistant_weights['answer'])
            if end_match:
                is_final = index == len(headers) - 1
                trainable = any(weight > 0 and a < end and b > header.end()
                                for _, a, b, weight in spans)
                weight = 1. if is_final or trainable else 0.
                add('assistant_end', end, header.end() + end_match.end(), weight)
        else:
            raise AssertionError(f'Unknown native role: {role}')
    if template_type == 'glm5_next' and text.endswith('<|user|>'):
        add('final_suffix', len(text) - len('<|user|>'), len(text), 1.)

    segments, cursor = [], 0
    for category, start, end, weight in sorted(spans, key=lambda span: span[1]):
        if start < cursor:
            raise AssertionError(f'Overlapping independent character regions: {category}')
        if start > cursor:
            segments.append(('fixed_wrapper_or_empty_think', cursor, start, 0.))
        segments.append((category, start, end, weight))
        cursor = end
    if cursor < len(text):
        segments.append(('fixed_wrapper_or_empty_think', cursor, len(text), 0.))
    spans = []
    for category, start, end, weight in segments:
        if spans and spans[-1][2] == start and spans[-1][3] == weight:
            previous = spans.pop()
            spans.append(('equal_weight_regions', previous[1], end, weight))
        else:
            spans.append((category, start, end, weight))
    boundaries = {}

    def token_boundary(position):
        if position not in boundaries:
            low, high = 0, len(ids)
            while low < high:
                middle = (low + high) // 2
                prefix = tokenizer.decode(ids[:middle], clean_up_tokenization_spaces=False)
                if len(prefix) < position:
                    low = middle + 1
                else:
                    high = middle
            prefix = tokenizer.decode(ids[:low], clean_up_tokenization_spaces=False)
            if prefix != text[:position]:
                raise AssertionError(f'Region boundary splits a token at character {position}')
            boundaries[position] = low
        return boundaries[position]

    expected, assigned = [0.] * len(ids), [False] * len(ids)
    regions = []
    for category, start, end, weight in spans:
        a, b = token_boundary(start), token_boundary(end)
        if any(assigned[a:b]):
            raise AssertionError(f'Overlapping independent regions: {category}')
        expected[a:b] = [weight] * (b - a)
        assigned[a:b] = [True] * (b - a)
        regions.append(dict(category=category, start=a, end=b, expected=weight))
    cursor = 0
    while cursor < len(ids):
        if assigned[cursor]:
            cursor += 1
            continue
        start = cursor
        while cursor < len(ids) and not assigned[cursor]:
            assigned[cursor] = True
            cursor += 1
        regions.append(dict(category='fixed_wrapper_or_empty_think', start=start, end=cursor, expected=0.))
    regions.sort(key=lambda region: region['start'])
    mismatch = [i for i, (wanted, actual) in enumerate(zip(expected, encoded['effective_weights']))
                if abs(wanted - actual) > 1e-9]
    label_mismatch = [i for i, (token, weight, label) in enumerate(zip(ids, expected, encoded['labels']))
                      if label != (token if weight > 0 else -100)]
    return dict(passed=not mismatch and not label_mismatch, regions=regions,
                coverage_counts={'1': len(ids)}, label_mismatches=label_mismatch[:20],
                weight_mismatches=[dict(index=i, expected=expected[i], actual=encoded['effective_weights'][i])
                                   for i in mismatch[:20]], weight_counts=dict(Counter(expected)))
