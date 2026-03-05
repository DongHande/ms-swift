# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import List, Literal, Optional, Tuple

import json

from swift.llm import Messages
from swift.llm.template import get_last_user_round
from swift.llm.template.utils import ContextType
from .utils import calculate_loss_scale

# loss_scale 策略说明：
# default: 只训练 response 和 suffix 部分的 token 的 loss
# last_round: 只训练最后一轮 response 和 suffix 部分的 token 的 loss
# all: 训练所有 token 的 loss
# ignore_system: 只训练除了 system 以外的部分的 token 的 loss
# response_only: 只训练 response 部分的 token 的 loss
# tool_only: 只训练 tool 部分的 token 的 loss
# tool_call_only: 只训练 tool_call 部分的 token 的 loss
# response_and_tool_call: 训练 response 和 tool_call 部分的 token 的 loss
ALL_BASE_STRATEGY = ['default', 'last_round', 'all', 'ignore_system', 'response_only', 'tool_only', 'tool_call_only', 'response_and_tool_call']


class LossScale:
    # Indicates whether loss_scale contains only 0 and 1.
    # If set to True, loss_scale will be replaced by labels to stay compatible with
    # acceleration techniques such as liger_kernel.
    # If set to False, an additional 'loss_scale' key will be stored and the
    # corresponding loss function will be used.
    loss_scale_config = None  # path
    is_binary = None

    def __init__(self, base_strategy: Literal['default',
                                              'last_round',
                                              'all',
                                              'ignore_system',
                                              'response_only',
                                              'tool_only',
                                              'tool_call_only',
                                              'response_and_tool_call'] = 'default'):
        assert base_strategy in ALL_BASE_STRATEGY, (
            f'ALL_BASE_STRATEGY: {ALL_BASE_STRATEGY}, base_strategy: {base_strategy}')
        self.base_strategy = base_strategy
        self.loss_scale_map = None
        if self.loss_scale_config is not None:
            path = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(path, 'config', self.loss_scale_config)
            with open(config_path, 'r', encoding='utf-8') as json_file:
                self.loss_scale_map = json.load(json_file)

    def get_loss_scale(self, context: str, **kwargs) -> Tuple[List[str], List[float]]:
        """Calculate loss scale

        Args:
            context: The input context
            query: The query of this round.

        Returns:
            A tuple, list of context and list of loss_scales
        """
        return [context], [1.]

    def __call__(self, context_list: List[str], context_types: List[ContextType], messages: Messages,
                 **kwargs) -> Tuple[List[str], List[float]]:
        res_context_list = []
        res_loss_scale = []
        i = 0
        last_user_round = get_last_user_round(messages)
        for context, context_type in zip(context_list, context_types):
            is_last_round = 2 * i >= last_user_round
            query, loss = None, None

            # 只有 ContextType.RESPONSE 类型的 context 对应的 loss 字段会被使用
            if context_type == ContextType.RESPONSE:
                query = messages[2 * i]['content']
                # Currently, we only support applying loss/mask to the response part.
                loss = messages[2 * i + 1].get('loss')
                assistant_content = messages[2 * i + 1]['content']
                if isinstance(assistant_content, str):
                    assert context == assistant_content
                else:
                    assert context in assistant_content, f'context: {context}, assistant_content: {assistant_content}'
                    # assert context.strip() in {c.strip() for c in assistant_content}, f'context: {context}, assistant_content: {assistant_content}'
                i += 1
            # ContextType.TOOL_CALL 原先是 ContextType.RESPONSE，目前区分出来，因此对应上一轮的 query 和 loss 字段
            if context_type == ContextType.TOOL_CALL:
                i -= 1
                query = messages[2 * i]['content']
                # Currently, we only support applying loss/mask to the response part.
                loss = messages[2 * i + 1].get('loss')
                assistant_content = messages[2 * i + 1]['content']
                if isinstance(assistant_content, str):
                    assert context == assistant_content
                else:
                    assert context in assistant_content, f'context: {context}, assistant_content: {assistant_content}'
                    # assert context.strip() in {c.strip() for c in assistant_content}, f'context: {context}, assistant_content: {assistant_content}'
                i += 1

            if isinstance(context, dict) and 'loss_scale' in context:
                new_context = [[token] for token in context['token_ids']]
                loss_scale = context['loss_scale']
            else:
                if isinstance(context, dict) and 'token_ids' in context:
                    context = context['token_ids']
                # RESPONSE 和 TOOL_CALL 都是 assistant 的生成内容，SUFFIX 为 sep 或 eos，训练时也属于 assistant 范畴
                is_assistant = context_type in {ContextType.RESPONSE, ContextType.TOOL_CALL, ContextType.SUFFIX}
                if self.base_strategy == 'all' \
                or (self.base_strategy == 'default' and is_assistant) \
                or (self.base_strategy == 'last_round' and is_assistant and is_last_round) \
                or (self.base_strategy == 'ignore_system' and context_type != ContextType.SYSTEM) \
                or (self.base_strategy == 'response_only' and context_type == ContextType.RESPONSE) \
                or (self.base_strategy == 'tool_only' and context_type == ContextType.TOOL) \
                or (self.base_strategy == 'tool_call_only' and context_type == ContextType.TOOL_CALL) \
                or (self.base_strategy == 'response_and_tool_call' and context_type in {ContextType.RESPONSE, ContextType.TOOL_CALL}):
                    new_context, loss_scale = self.get_loss_scale(context, query=query)
                    # 仅有 loss_scale 为 default 或 last_round 时，才乘以 loss 字段，其他类型不受 loss 字段影响
                    if loss is not None and self.base_strategy in {'default', 'last_round'}:
                        # 乘以自定义的 loss 系数
                        loss_scale = [s * float(loss) for s in loss_scale]
                else:
                        new_context, loss_scale = [context], [0.]
            res_context_list += new_context
            res_loss_scale += loss_scale
        return res_context_list, res_loss_scale

    @property
    def is_loss_scale_binary(self):
        if self.is_binary is not None:
            return self.is_binary
        if self.loss_scale_map is None:
            return True
        return all(scale in {0.0, 1.0} for lst in self.loss_scale_map.values() for scale in lst)


class AgentFlanLossScale(LossScale):
    loss_scale_config = 'agentflan.json'

    def get_loss_scale(self, context: str, *, query: Optional[str] = None):
        if isinstance(context, str):
            return calculate_loss_scale(query, context, self.loss_scale_map['response'], self.loss_scale_map['query'])
        return super().get_loss_scale(context)


class REACTLossScale(LossScale):
    loss_scale_config = 'react.json'

    def get_loss_scale(self, context: str, *, query: Optional[str] = None):
        if isinstance(context, str):
            return calculate_loss_scale(query, context, self.loss_scale_map)
        return super().get_loss_scale(context)


class QwenLossScale(REACTLossScale):
    loss_scale_config = 'qwen.json'


class HermesLossScale(REACTLossScale):
    loss_scale_config = 'hermes.json'


class AlphaUmiLossScale(REACTLossScale):
    loss_scale_config = 'alpha_umi.json'


class IgnoreEmptyThinkLossScale(REACTLossScale):
    loss_scale_config = 'ignore_empty_think.json'


# Add your loss scale here, use --loss_scale xxx to train
loss_scale_map = {
    '-': LossScale,
    'ignore_empty_think': IgnoreEmptyThinkLossScale,
    # agent
    'react': REACTLossScale,
    'hermes': HermesLossScale,
    'qwen': QwenLossScale,
    'agentflan': AgentFlanLossScale,
    'alpha_umi': AlphaUmiLossScale,
}


def get_loss_scale(loss_scale: str) -> LossScale:
    splited = loss_scale.split('+', 1)
    if len(splited) == 1:
        if splited[0] in ALL_BASE_STRATEGY:
            base_strategy, loss_scale = splited[0], '-'
        else:
            base_strategy, loss_scale = 'default', splited[0]
    else:
        base_strategy, loss_scale = splited
    return loss_scale_map[loss_scale](base_strategy)
