"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import os
import numpy as np
from string import Template
import re

from fastdeploy.utils import data_processor_logger
from paddlenlp.generation import GenerationConfig

if os.getenv("FLAG_TOKENIZER_V2"):
    from fastdeploy.input.ernie_tokenizer_v2 import ErnieBotTokenizer
else:
    from fastdeploy.input.ernie_tokenizer_v1 import ErnieBotTokenizer

from fastdeploy.input.text_processor import BaseDataProcessor
from fastdeploy.engine.config import ModelConfig
from fastdeploy.utils import data_processor_logger

class ErnieProcessor(BaseDataProcessor):
    """
    初始化模型实例。

    Args:
        model_name_or_path (str): 模型名称或路径。

    Attributes:
        model_name_or_path (str): 存储模型名称或路径。
        decode_status (dict): 存储解码状态信息。
        tokenizer (object): 存储分词器实例。
        eos_token_ids (list): 存储结束符号的token ID列表。
        eos_token_id_len (int): 存储结束符号的token ID列表的长度。
        pad_token_id (int): 存储填充符号的token ID。
    """
    def __init__(self, model_name_or_path):

        self.model_name_or_path = model_name_or_path
        data_processor_logger.info(f"model_name_or_path: {model_name_or_path}")
        self._init_config()
        self.model_name = ModelConfig(model_name_or_path).model_name

        self.is_thinking = (self.model_name == "x1")
        if self.is_thinking:
            self.thinking_template = "<|prefixoftext|>思考<|middleoftext|>"
            self.response_template = "<|prefixoftext|>开始回复<|middleoftext|>"

        data_processor_logger.info(f"Thinking mode is {self.is_thinking}")
        self.decode_status = dict()
        self._load_tokenizer()
        data_processor_logger.info(f"tokenizer information: bos_token is {self.tokenizer.bos_token} \
                                   {self.tokenizer.bos_token_id}, \
                                   eos_token is {self.tokenizer.eos_token}, {self.tokenizer.eos_token_id} ")
        self.eos_token_ids = [self.tokenizer.eos_token_id]
        self.eos_token_id_len = len(self.eos_token_ids)
        self.pad_token_id = self.get_pad_id()

    def _init_config(self):
        self.use_hf_tokenizer = int(os.getenv("USE_HF_TOKENIZER", 0)) == 1

        # Generation config
        try:
            self.generation_config = GenerationConfig.from_pretrained(self.model_name_or_path)
        except:
            data_processor_logger.warning(
                "Can't find generation config, so it will not use generation_config field in the model config"
            )
            self.generation_config = None


    def process_request(self, request, max_model_len=None):
        """
        Preprocess the request

        Args:
            request (Dict): may contain text and messages fields

        Returns:
            bool: Whether preprocessing is successful
            str: error message
        """
        if request.get("eos_token_ids") is None or len(request.eos_token_ids) == 0:
            request.eos_token_ids = self.eos_token_ids

        stop_sequences = request.get("stop", [])
        if stop_sequences is not None and len(stop_sequences) != 0:
            stop_seqs, stop_seqs_len = self.update_stop_seq(stop_sequences)
            request.set("stop_token_ids", stop_seqs)
            request.set("stop_seqs_len", stop_seqs_len)

        if request.prompt_token_ids is None or len(request.prompt_token_ids) == 0:
            system = request.get("system")
            if request.prompt is not None:
                request.prompt_token_ids = self.text2ids(request.prompt, max_model_len, system)
            elif request.messages is not None:
                request.prompt_token_ids = self.messages2ids(request.messages, max_model_len)
            else:
                raise ValueError(f"The request should have `input_ids`, `text` or `messages`: {request}.")
            if self.model_name == "base":
                assert (
                    system is None or system == ""
                ), "The loadding model is a base model, `system` is not supported."
                assert request.messages is None, "The loadding model is a base model, `messages` is not supported."

        if max_model_len is not None and len(request.prompt_token_ids) > max_model_len:
            request.prompt_token_ids = request.prompt_token_ids[:max_model_len - 1]
        data_processor_logger.info(f"processed request: {request}")
        return request

    def process_request_dict(self, request, max_model_len=None):
        """
        Preprocess the request

        Args:
            request (Dict): may contain text and messages fields

        Returns:
            bool: Whether preprocessing is successful
            str: error message
        """
        if not request.get('eos_token_ids'):
            request['eos_token_ids'] = self.eos_token_ids

        # 处理stop_sequences
        stop_sequences = request.get('stop', [])
        if stop_sequences:
            stop_seqs, stop_seqs_len = self.update_stop_seq(stop_sequences)
            request['stop_token_ids'] = stop_seqs
            request['stop_seqs_len'] = stop_seqs_len

        system = request.get("system")
        # 处理prompt_token_ids
        if not request.get('prompt_token_ids'):
            if 'prompt' in request:
                raw_request = request.get('raw_request', True)
                request['prompt_token_ids'] = self.text2ids(
                    request['prompt'],
                    raw_request,
                    max_model_len,
                    system
                )
            elif 'messages' in request:
                request['prompt_token_ids'] = self.messages2ids(request['messages'], max_model_len)
            else:
                raise ValueError(f"Request must contain 'prompt_token_ids', 'prompt', or 'messages': {request}")
        if self.model_name == "base":
            assert isinstance(
                request['prompt'], str
            ), "the loadding model is a base model, `prompt` must be a string type."
            assert (
                system is None or system == ""
            ), "The loadding model is a base model, `system` is not supported."
            assert request.get(
                'messages'
            ) is None, "The loadding model is a base model, `messages` is not supported."



        # 截断超过长度限制的prompt
        if max_model_len is not None and len(request['prompt_token_ids']) > max_model_len:
            request['prompt_token_ids'] = request['prompt_token_ids'][:max_model_len - 1]

        return request

    def process_response(self, response_dict, **kwargs):
        """
        Preprocess the response

        Args:
            response_dict (Dict): response for engine, contain ids fields

        Returns:
            Dict: response contain text fields
        """
        is_end = response_dict.finished
        req_id = response_dict.request_id


        token_ids = response_dict.outputs.token_ids
        if self.is_thinking:
            text, reasoning_content = self.ids2tokens_thinking(token_ids, req_id)
            response_dict.outputs.text = text
            response_dict.outputs.reasoning_content = reasoning_content
        else:
            response_dict.outputs.text = self.ids2tokens(token_ids, req_id)
        response_dict.usage = {"completion_tokens" : response_dict.outputs.index + 1}
        if is_end:
            if self.is_thinking:
                text, reasoning_content = self.ids2tokens_thinking(token_ids, req_id)
                response_dict.outputs.text = text
                pattern = re.compile(
                    r'^([\s\S]*?)<\|prefixoftext\|>开始回复<\|middleoftext\|>([\s\S]*)$',
                    flags=re.DOTALL | re.MULTILINE
                )
                if reasoning_content != "":
                    match = pattern.search(reasoning_content)
                    if match:
                        response_dict.outputs.text = match.group(2)
                        response_dict.outputs.reasoning_content = match.group(1)
            else:
                response_dict.outputs.text = self.ids2tokens(token_ids, req_id)

            data_processor_logger.debug("Request id: {} has been completed.".format(token_ids))
            self.clear_request_status(req_id)
        if response_dict.outputs.text == "" and response_dict.outputs.reasoning_content == "":
            return None
        return response_dict

    def process_response_dict(self, response_dict, stream=True):
        """
        Preprocess the response

        Args:
            response_dict (Dict): response for engine, contain ids fields

        Returns:
            Dict: response contain text fields
        """
        is_end = response_dict["finished"]
        req_id = response_dict["request_id"]

        token_ids = response_dict["outputs"]["token_ids"]
        if self.is_thinking:
            text, reasoning_content = self.ids2tokens_thinking(token_ids, req_id)
            response_dict["outputs"]["text"] = text
            response_dict["outputs"]["reasoning_content"] = reasoning_content
        else:
            response_dict["outputs"]["text"] = self.ids2tokens(token_ids, req_id)

        if is_end:
            if self.is_thinking:
                text, reasoning_content = self.ids2tokens_thinking(token_ids, req_id)
                response_dict["outputs"]["text"] = text
                pattern = re.compile(
                    r'^([\s\S]*?)<\|prefixoftext\|>开始回复<\|middleoftext\|>([\s\S]*)$',
                    flags=re.DOTALL | re.MULTILINE
                )
                if reasoning_content != "":
                    match = pattern.search(reasoning_content)
                    if match:
                        response_dict["outputs"]["text"] = match.group(2)
                        response_dict["outputs"]["reasoning_content"] = match.group(1)
            else:
                response_dict["outputs"]["text"] = self.ids2tokens(token_ids, req_id)
            data_processor_logger.debug("Request id: {} has been completed.".format(token_ids))
            full_text, reasoning_content = self.clear_request_status(req_id)
            if not stream:
                if self.is_thinking:
                    response_dict["outputs"]["text"] = full_text
                    response_dict["outputs"]["reasoning_content"] = reasoning_content
                else:
                    response_dict["outputs"]["text"] = reasoning_content

        return response_dict


    def text2ids(self, text, raw_request, max_model_len=None, system=None):
        """
        将文本转换为对应的 ID。

        Args:
            text (str): 待转换的文本。
            system (str): 系统设定，如“你是一位高超的程序员”

        Returns:
            List[int]: 转换后的 ID 列表。
        """
        messages = []
        if isinstance(text, list):
            messages.extend(text)
        else:
            messages.append(text)
        if self.is_thinking:
            system = "<sys_internal>\n【高优系统设定】必须最优先遵循<br/>\n启动思考模式：在采取任何行动前，\
都需要先写下自己的思考过程，为后续的决策或对用户的回复内容做铺垫。\n</sys_internal>\n\n"
            tokens = self._convert_to_ids_thinking(messages, max_model_len, system)
        else:
            tokens = self._convert_to_ids(messages, raw_request, max_model_len, system)
        data_processor_logger.debug(f"processed data : {''.join(tokens)}")
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return input_ids

    def _convert_to_ids(self, messages, raw_request, max_model_len=None, system=None):
        """
        将多轮对话转换为对话ID序列。

        Args:
            messages (List[str]): 包含所有对话轮的文本列表。
                messages示例[Q1, A1, Q2, A2, Q3, A3, Q4], Q3,A3表示最近时间的对话，Q4表示需要回答的问题

        Returns:
            List[int]: 对话ID序列，每个ID都是整数。
        """
        if len(messages) % 2 == 0:
            raise ValueError(f"The number of the messages context ({len(messages)}) must be odd.")

        if self.model_name == "base" or not raw_request:
            # for base model, the length of messages should be 1
            # and it only need to convert the input prompt to token ids
            tokens = self.tokenizer.tokenize(messages[0])
            return tokens

        prefix_tokens = [self.tokenizer.cls_token]
        suffix_tokens = self.tokenizer.tokenize("Assistant: ")
        system_tokens = []
        if system is not None:
            system_tokens = self.tokenizer.tokenize(system) + self.tokenizer.tokenize("\n")
        context_tokens = self.tokenizer.tokenize("User: ") + \
            self.tokenizer.tokenize(messages[-1]) + self.tokenizer.tokenize("\n")

        # process messages
        for idx in range(len(messages) - 2, -1, -2):
            cur_turn_tokens = self.tokenizer.tokenize("User: ") + \
                self.tokenizer.tokenize(messages[idx - 1]) + self.tokenizer.tokenize("\n")
            cur_turn_tokens += self.tokenizer.tokenize("Assistant: ") + \
                self.tokenizer.tokenize(messages[idx]) + [self.tokenizer.sep_token]
            if max_model_len is not None and len(prefix_tokens) + len(context_tokens) + len(suffix_tokens) + \
                                               len(cur_turn_tokens) >= max_model_len:
                data_processor_logger.warning(f"Truncate messages into: {messages[idx + 1:]}")
                break
            context_tokens = cur_turn_tokens + context_tokens
        new_length =  len(system_tokens) + len(prefix_tokens) + len(context_tokens) + len(suffix_tokens) + 1
        context_tokens = system_tokens + context_tokens
        if max_model_len is not None and len(prefix_tokens) + len(context_tokens) + \
                                             len(suffix_tokens) + 1 >= max_model_len:
            data_processor_logger.warning(
                "The length of the knowledge and the last user content "
                f"({len(prefix_tokens) + len(context_tokens) + len(suffix_tokens)}) is greater than "
                f"max input length ({max_model_len}). We will truncate it."
            )

            context_tokens = context_tokens[-(max_model_len - len(prefix_tokens) - len(suffix_tokens) - 1):]
            return prefix_tokens + context_tokens + suffix_tokens

        return prefix_tokens + context_tokens + suffix_tokens



    def _convert_to_ids_thinking(self, messages, max_model_len=None, system=None):
        """
        将多轮对话转换为对话ID序列。

        Args:
            messages (List[str]): 包含所有对话轮的文本列表。
                messages示例[Q1, A1, Q2, A2, Q3, A3, Q4], Q3,A3表示最近时间的对话，Q4表示需要回答的问题

        Returns:
            List[int]: 对话ID序列，每个ID都是整数。
        """
        if len(messages) % 2 == 0:
            raise ValueError(f"The number of the messages context ({len(messages)}) must be odd.")

        suffix_tokens = self.tokenizer.tokenize("<role>\nassistant<br/>\n<|prefixoftext|>思考<|middleoftext|>")

        system_tokens = self.tokenizer.tokenize(system)

        user_template = Template("""<role>\nuser<br/>\n${question}\n</role>\n\n""")

        assistant_template = Template("""<role>
assistant<br/>\n<|prefixoftext|>开始回复<|middleoftext|>${answer}<mask:1>\n</role>\n\n""")

        context_tokens = self.tokenizer.tokenize(user_template.safe_substitute({"question": messages[-1]}))

        # process messages
        for idx in range(len(messages) - 2, -1, -2):
            cur_turn_tokens = self.tokenizer.tokenize(user_template.safe_substitute({"question": messages[idx - 1]}))
            cur_turn_tokens += self.tokenizer.tokenize(assistant_template.safe_substitute({"answer": messages[idx]}))
            if max_model_len is not None and len(system_tokens) + len(context_tokens) + len(suffix_tokens) + \
                                               len(cur_turn_tokens) >= max_model_len:
                data_processor_logger.warning(f"Truncate messages into: {messages[idx + 1:]}")
                break
            context_tokens = cur_turn_tokens + context_tokens

        return system_tokens + context_tokens + suffix_tokens



    def messages2ids(self, raw_messages, max_model_len):
        """
        Convert multi-turn messages into ID sequences.

        Args:
            messages (List[Dict[str, Any]]): multi-turn messages.
            max_model_len : support max length

        Returns:
            List[int]: ID sequences
        """
        system = None
        if self.is_thinking:
            system = "<sys_internal>\n【高优系统设定】必须最优先遵循<br/>\n启动思考模式：在采取任何行动前，\
都需要先写下自己的思考过程，为后续的决策或对用户的回复内容做铺垫。\n</sys_internal>\n\n"
        else:
            if raw_messages[0]["role"] == "system" or raw_messages[0]["role"] == "developer":
                system = raw_messages[0]["content"]
                raw_messages = raw_messages[1:]
        messages = []
        messages_len = len(raw_messages)
        if messages_len % 2 == 0:
            raise ValueError(f"The number of the messages context (messages_len) must be odd.")
        for message in raw_messages:
            messages.append(message["content"])

        if self.is_thinking:
            tokens = self._convert_to_ids_thinking(messages, max_model_len, system)
        else:
            tokens = self._convert_to_ids(messages, raw_request=True, max_model_len=max_model_len, system=system)
        data_processor_logger.debug(f"processed data : {''.join(tokens)}")
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return input_ids


    def ids2tokens(self, token_id, task_id):
        """
        token ids to strings

        Args:
            token_ids (List[int]): token ids
			task_id (str): task id

        Returns:
            List[str]: strings
        """

        if task_id not in self.decode_status:
            # prefix offset & read offset & history token ids & history token strings
            self.decode_status[task_id] = [0, 0, [], "", ""]

        prefix_offset = self.decode_status[task_id][0]
        read_offset = self.decode_status[task_id][1]
        previous_token_ids = self.decode_status[task_id][2]
        decode_str, prefix_offset, read_offset = self.tokenizer.decode_token(
            previous_token_ids + token_id, prefix_offset, read_offset)
        self.decode_status[task_id][0] = prefix_offset
        self.decode_status[task_id][1] = read_offset
        self.decode_status[task_id][2] += token_id
        self.decode_status[task_id][3] += decode_str
        return decode_str

    def ids2tokens_thinking(self, token_id, task_id):
        """
        token ids to strings

        Args:
            token_ids (List[int]): token ids
			task_id (str): task id

        Returns:
            List[str]: strings
        """

        if task_id not in self.decode_status:
            # prefix offset & read offset & history token ids & history token strings
            self.decode_status[task_id] = [0, 0, [], "", ""]

        prefix_offset = self.decode_status[task_id][0]
        read_offset = self.decode_status[task_id][1]
        previous_token_ids = self.decode_status[task_id][2]
        decode_str, prefix_offset, read_offset = self.tokenizer.decode_token(
            previous_token_ids + token_id, prefix_offset, read_offset)
        self.decode_status[task_id][0] = prefix_offset
        self.decode_status[task_id][1] = read_offset
        self.decode_status[task_id][2] += token_id

        data_processor_logger.debug(f"{token_id}, {decode_str}")
        reasoning_content = ""
        content = ""
        if decode_str == "<|prefixoftext|>":
            self.decode_status[task_id][4] = decode_str
        elif self.decode_status[task_id][4] == "":
            self.decode_status[task_id][3] += decode_str
        else:
            self.decode_status[task_id][4] += decode_str

        if self.decode_status[task_id][4] == "":
            reasoning_content = decode_str
        elif '<|middleoftext|>' in self.decode_status[task_id][4] and decode_str != "<|middleoftext|>":
            content = decode_str
        return content, reasoning_content


    def _load_tokenizer(self):
        """
        load tokenizer

        Returns:
            tokenizer (AutoTokenizer)
        """
        vocab_file_names = ["tokenizer.model", "spm.model", "ernie_token_100k.model"]
        for i in range(len(vocab_file_names)):
            if os.path.exists(os.path.join(self.model_name_or_path, vocab_file_names[i])):
                ErnieBotTokenizer.resource_files_names["vocab_file"] = vocab_file_names[i]
                break
        self.tokenizer = ErnieBotTokenizer.from_pretrained(self.model_name_or_path)
    def clear_request_status(self, task_id):
        """
        clear request status

        Args:
            task_id (str): task id

        Returns:
            results_all (str): all token strings
        """
        results_all = ""
        reasoning_content = ""

        if task_id in self.decode_status:
            if self.use_hf_tokenizer:
                results_all = self.decode_status[task_id][2]
            else:
                reasoning_content = "".join(self.decode_status[task_id][3])
                results_all = "".join(self.decode_status[task_id][4][36:])
            del self.decode_status[task_id]
        return results_all, reasoning_content



    def get_pad_id(self):
        """
        get pad_token_id, if not pad_token_id, use eos_token

        Returns:
            int: pad_token_id
        """
        # if isinstance(self.tokenizer, (LlamaTokenizer, Llama3Tokenizer)) and not self.tokenizer.pad_token_id:
        #     return self.tokenizer.eos_token
        return self.tokenizer.pad_token_id

    def pad_batch_data(self, insts, pad_id=0, return_seq_len=False, return_array=True, pad_style="right"):
        """Pad the instances to the max sequence length in batch."""
        if len(insts) == 0:
            padded_insts = np.array([[]], dtype=np.int64) if return_array else [[]]
            if return_seq_len:
                seq_len = np.array([], dtype=np.int64) if return_array else []
                return padded_insts, seq_len
            return padded_insts

        max_len = max(map(len, insts))
        if pad_style == "left":
            padded_insts = [[pad_id] * (max_len - len(inst)) + list(inst) for inst in insts]
        else:
            padded_insts = [list(inst) + [pad_id] * (max_len - len(inst)) for inst in insts]
        if return_array:
            padded_insts = np.array(padded_insts, dtype=np.int64).reshape([-1, max_len])

        if return_seq_len:
            seq_len = [len(inst) for inst in insts]
            if return_array:
                seq_len = np.array(seq_len, dtype=np.int64).reshape(-1, 1)
            return padded_insts, seq_len
        return padded_insts

    def update_stop_seq(self, stop_sequences):
        """
        Update stop sequences from request.
        """
        stop_seqs =  []
        for seq in stop_sequences:
            if seq != self.tokenizer.eos_token_id:
                stop_seqs.append(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(seq)))
        stop_seqs, stop_seqs_len = self.pad_batch_data(
            stop_seqs,
            pad_id=-1,
            return_seq_len=True,
            return_array=False
        )
        data_processor_logger.debug(f"processed stop_seqs: {stop_seqs}, {stop_seqs_len}")
        return stop_seqs, stop_seqs_len
