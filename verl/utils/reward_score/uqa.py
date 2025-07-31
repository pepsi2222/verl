# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
import re

def format_reward(predict_str: str) -> float:
    """判断completion是否包含<think></think>格式"""
    m = re.match(r"^<think>[\s\S]*<\/think>[\s\S]+$", predict_str, flags=re.DOTALL)
    return 1.0 if m else 0.0


def _extract_tuple(s: str):
    pattern = r"<\|sid_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)><\|sid_end\|>"
    matches = re.findall(pattern, s)
    return matches[-1] if matches else ()  # 返回 (s_a, s_b, s_c) 或 ()


def acc_reward(predict_str: str, ground_truth: str, use_boxed: bool = True) -> float:
    prediction = _extract_tuple(predict_str)
    label = _extract_tuple(ground_truth)
    if len(prediction) == 0 or len(label) == 0:
        return 0.0
    if prediction == label:
        return 3.0
    elif prediction[:2] == label[:2]:
        return 2.0
    elif prediction[0] == label[0]:
        return 1.0
    else:
        return 0.0


def compute_score(predict_str: str, ground_truth: str, use_boxed: bool = True, format_score: float = 0.1) -> float:
    return (1.0 - format_score) * acc_reward(predict_str, ground_truth, use_boxed) + format_score * format_reward(predict_str) 