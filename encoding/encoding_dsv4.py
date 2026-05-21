"""
DeepSeek-V4 对话消息编码/解码模块

本模块为 DeepSeek-V4 提供自包含的 prompt 编解码实现，支持以下核心能力：
- 多轮对话消息的编码（encode）与解码（decode）
- Tool Calling：将 OpenAI 格式的 tool/tools 转换为 V4 的 DSML 格式
- Thinking Mode：支持 <think>...</think> 推理内容块的生成与解析
- Quick Instruction Task：内部分类任务（action/query/authority/domain/title/read_url）
- 上下文（context）拼接与 reasoning_content 丢弃策略

编码流程（encode_messages）：
    原始消息列表 → merge_tool_messages（合并 tool 到 user）
                → sort_tool_results_by_call_order（按调用顺序排序 tool_result）
                → _drop_thinking_messages（可选丢弃历史 reasoning）
                → render_message（逐条渲染为字符串）
                → 拼接 BOS + 各消息文本 → 最终 prompt

解码流程（parse_message_from_completion_text）：
    模型原始输出 → 提取 reasoning_content（<think> 块）
                → 提取 content（正文）
                → 提取 tool_calls（DSML 格式解析）
                → 返回结构化 assistant 消息

DSML（DeepSeek Markup Language）是 V4 用于描述 tool_call 的标记格式，
通过 <｜DSML｜invoke>、<｜DSML｜parameter> 等标签封装工具调用与参数。
"""

from typing import Any, Dict, List, Union, Optional, Tuple
import copy
import json
import re

# ============================================================
# 特殊 Token 定义
# ============================================================
# V4 使用一组特殊标记（special token）来界定句子边界、角色、推理块及工具调用。
# 这些 token 在编码时插入到 prompt 中，解码时作为定位锚点进行文本分割。

bos_token: str = "<｜begin▁of▁sentence｜>"
"""BOS（Begin of Sentence）：对话起始标记，encode_messages 在无前文 context 时自动附加。"""

eos_token: str = "<｜end▁of▁sentence｜>"
"""EOS（End of Sentence）：对话结束标记，assistant 消息模板默认以该 token 结尾。"""

thinking_start_token: str = "<think>"
"""Thinking 起始标记：在 thinking_mode='thinking' 时，模型需先输出推理过程。"""

thinking_end_token: str = "</think>"
"""Thinking 结束标记：标志 reasoning_content 块的结束，之后为正式回答或 tool_calls。"""

dsml_token: str = "｜DSML｜"
"""DSML 标记：用于构建 tool_call 的 XML-like 标签前缀，如 <｜DSML｜invoke>。"""

USER_SP_TOKEN = "<｜User｜>"
"""User 角色标记：在 user/developer 消息前附加，标识用户输入起始。"""

ASSISTANT_SP_TOKEN = "<｜Assistant｜>"
"""Assistant 角色标记：在 assistant 回复前附加，标识模型输出起始。"""

LATEST_REMINDER_SP_TOKEN = "<｜latest_reminder｜>"
"""Latest Reminder 标记：用于最新提醒类消息的角色标识。"""

# 内部分类任务（Quick Instruction Task）的特殊 token 映射表。
# 这些任务用于服务端内部对请求进行分类，不暴露给终端用户。
DS_TASK_SP_TOKENS = {
    "action": "<｜action｜>",      # 动作类任务
    "query": "<｜query｜>",        # 查询类任务
    "authority": "<｜authority｜>",# 权限类任务
    "domain": "<｜domain｜>",      # 领域类任务
    "title": "<｜title｜>",        # 标题类任务
    "read_url": "<｜read_url｜>",  # URL 读取类任务
}
VALID_TASKS = set(DS_TASK_SP_TOKENS.keys())
"""合法任务类型集合，用于校验 message['task'] 字段。"""

# ============================================================
# 模板常量
# ============================================================
# 以下模板定义了各角色消息在编码后的字符串结构。
# 通过 str.format() 填充 content / reasoning / tool_calls 等字段。

system_msg_template: str = "{content}"
"""System 消息模板：直接输出 content，无额外包装。"""

user_msg_template: str = "{content}"
"""User 消息模板：content 前已由 render_message 附加 USER_SP_TOKEN。"""

latest_reminder_msg_template: str = "{content}"
"""Latest Reminder 消息模板：content 前附加 LATEST_REMINDER_SP_TOKEN。"""

assistant_msg_template: str = "{reasoning}{content}{tool_calls}" + eos_token
"""Assistant 消息模板（含 EOS）：按 reasoning → content → tool_calls → EOS 顺序拼接。"""

assistant_msg_wo_eos_template: str = "{reasoning}{content}{tool_calls}"
"""Assistant 消息模板（不含 EOS）：用于流式输出或中间状态，避免提前终止。"""

thinking_template: str = "{reasoning_content}"
"""Thinking 内容模板：将 reasoning_content 原样嵌入，外层由 thinking_start/end_token 包裹。"""

response_format_template: str = (
    "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{schema}"
)
"""Response Format 模板：当消息包含 response_format 时，提示模型按指定 JSON Schema 回复。"""

tool_call_template: str = (
    "<{dsml_token}invoke name=\"{name}\">\n{arguments}\n</{dsml_token}invoke>"
)
"""单个 tool_call 的 DSML 模板：包含工具名与参数列表。"""

tool_calls_template = (
    "<{dsml_token}{tc_block_name}>\n{tool_calls}\n</{dsml_token}{tc_block_name}>"
)
"""tool_calls 块模板：将多个 tool_call 包裹在 <｜DSML｜tool_calls> 根标签内。"""

tool_calls_block_name: str = "tool_calls"
"""tool_calls 根标签名称，与 dsml_token 组合成 <｜DSML｜tool_calls>。"""

tool_output_template: str = (
    "<tool_result>{content}</tool_result>"
)
"""Tool 结果输出模板：将 tool 执行结果包裹为 <tool_result>，供模型读取。"""

REASONING_EFFORT_MAX = (
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n"
)
"""最大推理努力提示词：在 thinking_mode='thinking' 且 reasoning_effort='max' 时，
插入到 prompt 开头，强制模型进行最详尽的逐步推理。"""

TOOLS_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml_token}tool_calls>" block like the following:

<{dsml_token}tool_calls>
<{dsml_token}invoke name="$TOOL_NAME">
<{dsml_token}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml_token}parameter>
...
</{dsml_token}invoke>
<{dsml_token}invoke name="$TOOL_NAME2">
...
</{dsml_token}invoke>
</{dsml_token}tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {thinking_start_token}), you MUST output your complete reasoning inside {thinking_start_token}...{thinking_end_token} BEFORE any tool calls or final response.

Otherwise, output directly after {thinking_end_token} with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""
"""工具描述模板：向模型说明可用的工具列表、DSML 调用格式及参数类型规则。
在 system/developer 消息包含 tools 时，通过 render_tools() 格式化插入 prompt。"""

# ============================================================
# 工具函数：JSON / OpenAI 格式转换
# ============================================================

def to_json(value: Any) -> str:
    """将任意 Python 对象序列化为 JSON 字符串。

    优先使用 ensure_ascii=False 保持 Unicode 原样；若失败则回退到 ensure_ascii=True。
    该函数广泛用于将 tool schema、response_format、参数值等转为字符串嵌入 prompt。
    """
    try:
        return json.dumps(value, ensure_ascii=False)
    except:
        return json.dumps(value, ensure_ascii=True)


def tools_from_openai_format(tools):
    """从 OpenAI 格式的 tools 列表中提取 function 定义。

    OpenAI 格式：{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
    本函数返回 [function_dict, ...]，供后续 render_tools 使用。
    """
    return [tool["function"] for tool in tools]


def tool_calls_from_openai_format(tool_calls):
    """将 OpenAI 格式的 tool_calls 转换为内部简化格式。

    OpenAI 格式：{"id": ..., "type": "function", "function": {"name": ..., "arguments": ...}}
    内部格式：{"name": ..., "arguments": ...}（arguments 为 JSON 字符串）
    """
    return [
        {
            "name": tool_call["function"]["name"],
            "arguments": tool_call["function"]["arguments"],
        }
        for tool_call in tool_calls
    ]


def tool_calls_to_openai_format(tool_calls):
    """将内部 tool_calls 转换回 OpenAI 标准格式。

    用于 decode 阶段，将模型输出的 DSML tool_call 解析结果包装为 OpenAI API 兼容结构。
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool_call["name"],
                "arguments": tool_call["arguments"],
            }
        }
        for tool_call in tool_calls
    ]


# ============================================================
# DSML 参数编解码
# ============================================================

def encode_arguments_to_dsml(tool_call: Dict[str, str]) -> str:
    """将 tool_call 的 arguments 编码为 DSML parameter 格式。

    解析逻辑：
        1. 尝试将 tool_call["arguments"] 按 JSON 解析为字典；若失败则整体作为单字段 "arguments"。
        2. 遍历每个键值对：
           - 值为 str 时，string="true"，value 原样写入；
           - 值为其他类型（int/float/list/dict/bool）时，string="false"，value 经 to_json() 序列化。
        3. 按行拼接为 <｜DSML｜parameter name="..." string="...">...</｜DSML｜parameter>

    Args:
        tool_call: {"name": str, "arguments": str(JSON)} 格式的工具调用描述。

    Returns:
        DSML 格式的参数字符串，可直接嵌入 <｜DSML｜invoke> 块内。
    """
    p_dsml_template = '<{dsml_token}parameter name="{key}" string="{is_str}">{value}</{dsml_token}parameter>'
    P_dsml_strs = []

    try:
        arguments = json.loads(tool_call["arguments"])
    except Exception as err:
        arguments = {"arguments": tool_call["arguments"]}

    for k, v in arguments.items():
        p_dsml_str = p_dsml_template.format(
            dsml_token=dsml_token,
            key=k,
            is_str="true" if isinstance(v, str) else "false",
            value=v if isinstance(v, str) else to_json(v),
        )
        P_dsml_strs.append(p_dsml_str)

    return "\n".join(P_dsml_strs)


def decode_dsml_to_arguments(tool_name: str, tool_args: Dict[str, Tuple[str, str]]) -> Dict[str, str]:
    """将 DSML 参数解码回内部 tool_call 字典。

    解码逻辑：
        1. 遍历 tool_args（param_name → (value, is_string_flag)）。
        2. 若 is_string_flag == "true"，将 value 用 to_json() 包裹为 JSON 字符串；
           否则保持 value 原样（已是 JSON 字面量）。
        3. 拼接为 {"name": tool_name, "arguments": json_str} 返回。

    Args:
        tool_name: 工具名称。
        tool_args: 参数映射，值为 (参数值, 是否为字符串标记) 的二元组。

    Returns:
        {"name": str, "arguments": str(JSON)} 格式的内部 tool_call 字典。
    """
    def _decode_value(key: str, value: str, string: str):
        if string == "true":
            value = to_json(value)
        return f"{to_json(key)}: {value}"

    tool_args_json = "{" + ", ".join([_decode_value(k, v, string=is_str) for k, (v, is_str) in tool_args.items()]) + "}"
    return dict(name=tool_name, arguments=tool_args_json)


# ============================================================
# 工具渲染与辅助函数
# ============================================================

def render_tools(tools: List[Dict[str, Union[str, Dict[str, Any]]]]) -> str:
    """将工具 schema 列表渲染为 system prompt 中的 Tools 段落。

    每个工具 schema 先经 to_json() 序列化为一行 JSON，再填入 TOOLS_TEMPLATE。
    渲染结果包含：
        - DSML 调用示例（<｜DSML｜tool_calls> 完整结构）
        - 参数类型规则（string="true"/"false"）
        - thinking_mode 下推理块的位置要求
        - 可用工具 schema 列表（JSON 行）

    Args:
        tools: 工具定义列表，每项为 {"name": ..., "description": ..., "parameters": ...}。

    Returns:
        格式化后的 Tools 段落字符串，可直接追加到 system/developer 消息内容后。
    """
    tools_json = [to_json(t) for t in tools]

    return TOOLS_TEMPLATE.format(
        tool_schemas="\n".join(tools_json),
        dsml_token=dsml_token,
        thinking_start_token=thinking_start_token,
        thinking_end_token=thinking_end_token,
    )


def find_last_user_index(messages: List[Dict[str, Any]]) -> int:
    """查找对话中最后一条 user 或 developer 消息的索引。

    该索引用于判断 "历史消息" 与 "当前待生成回复" 的分界点：
    - 在 drop_thinking 策略中，最后一条 user 之前的 assistant reasoning_content 会被丢弃。
    - render_message 中，index > last_user_idx 的 assistant 消息保留 reasoning。

    Returns:
        最后一条 user/developer 消息的索引；若无则返回 -1。
    """
    last_user_index = -1
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") in ["user", "developer"]:
            last_user_index = idx
            break
    return last_user_index


# ============================================================
# 单条消息渲染（核心编码逻辑）
# ============================================================

def render_message(index: int, messages: List[Dict[str, Any]], thinking_mode: str, drop_thinking: bool = True, reasoning_effort: Optional[str] = None) -> str:
    """将指定索引的消息渲染为 DeepSeek-V4 编码字符串。

    这是编码流程的核心函数，负责将单条消息（system/user/developer/assistant/latest_reminder）
    转换为带特殊 token 和模板的文本片段。同时处理：
    - 角色标记（USER_SP_TOKEN / ASSISTANT_SP_TOKEN）的插入位置
    - thinking 块的保留/丢弃策略
    - tool_calls 的 DSML 格式化
    - content_blocks（text + tool_result）的混合渲染
    - 任务特殊 token（task sp token）的追加
    - 消息间过渡 token（thinking_start/end_token）的自动衔接

    Args:
        index: 当前消息在 messages 列表中的索引。
        messages: 完整对话消息列表（含 context）。
        thinking_mode: "chat" 或 "thinking"，决定模型是否输出推理过程。
        drop_thinking: 是否丢弃历史 assistant 消息中的 reasoning_content。
        reasoning_effort: "max" / "high" / None，仅在 thinking_mode='thinking' 时生效。

    Returns:
        该消息编码后的字符串片段。
    """
    assert 0 <= index < len(messages)
    assert thinking_mode in ["chat", "thinking"], f"Invalid thinking_mode `{thinking_mode}`"

    prompt = ""
    msg = messages[index]
    last_user_idx = find_last_user_index(messages)

    role = msg.get("role")
    content = msg.get("content")
    tools = msg.get("tools")
    response_format = msg.get("response_format")
    tool_calls = msg.get("tool_calls")
    reasoning_content = msg.get("reasoning_content")
    wo_eos = msg.get("wo_eos", False)

    # 若消息携带 tools/tool_calls，先转换为内部格式
    if tools:
        tools = tools_from_openai_format(tools)
    if tool_calls:
        tool_calls = tool_calls_from_openai_format(tool_calls)

    # 仅在 thinking_mode='thinking'、reasoning_effort='max'、且为第一条消息时，
    # 插入 REASONING_EFFORT_MAX 提示词，强制模型进行最详尽的推理。
    assert reasoning_effort in ['max', None, 'high'], f"Invalid reasoning effort: {reasoning_effort}"
    if index == 0 and thinking_mode == "thinking" and reasoning_effort == 'max':
        prompt += REASONING_EFFORT_MAX

    # -------------------------------
    # 按角色分支渲染
    # -------------------------------
    if role == "system":
        # System 消息：直接输出 content，可选追加 tools 和 response_format
        prompt += system_msg_template.format(content=content or "")
        if tools:
            prompt += "\n\n" + render_tools(tools)
        if response_format:
            prompt += "\n\n" + response_format_template.format(schema=to_json(response_format))

    elif role == "developer":
        # Developer 消息：语义上等价于 user，但需附加 USER_SP_TOKEN
        assert content, f"Invalid message for role `{role}`: {msg}"

        content_developer = USER_SP_TOKEN
        content_developer += content

        if tools:
            content_developer += "\n\n" + render_tools(tools)
        if response_format:
            content_developer += "\n\n" + response_format_template.format(schema=to_json(response_format))

        prompt += user_msg_template.format(content=content_developer)

    elif role == "user":
        # User 消息：附加 USER_SP_TOKEN，支持 content_blocks（text + tool_result 混合）
        prompt += USER_SP_TOKEN

        content_blocks = msg.get("content_blocks")
        if content_blocks:
            parts = []
            for block in content_blocks:
                block_type = block.get("type")
                if block_type == "text":
                    parts.append(block.get("text", ""))
                elif block_type == "tool_result":
                    tool_content = block.get("content", "")
                    # tool_content 可能为列表（多模态块），提取其中 text 类型内容
                    if isinstance(tool_content, list):
                        text_parts = []
                        for b in tool_content:
                            if b.get("type") == "text":
                                text_parts.append(b.get("text", ""))
                            else:
                                text_parts.append(f"[Unsupported {b.get('type')}]")
                        tool_content = "\n\n".join(text_parts)
                    parts.append(tool_output_template.format(content=tool_content))
                else:
                    parts.append(f"[Unsupported {block_type}]")
            prompt += "\n\n".join(parts)
        else:
            prompt += content or ""

    elif role == "latest_reminder":
        # Latest Reminder：在 content 前附加 LATEST_REMINDER_SP_TOKEN
        prompt += LATEST_REMINDER_SP_TOKEN + latest_reminder_msg_template.format(content=content)

    elif role == "tool":
        # V4 无独立 tool 角色，必须通过 merge_tool_messages 预处理合并到 user 消息中
        raise NotImplementedError("deepseek_v4 merges tool messages into user; please preprocess with merge_tool_messages()")

    elif role == "assistant":
        # Assistant 消息：处理 reasoning、content、tool_calls 三部分
        thinking_part = ""
        tc_content = ""

        # 若存在 tool_calls，按 DSML 格式渲染为 <｜DSML｜tool_calls> 块
        if tool_calls:
            tc_list = [
                tool_call_template.format(
                    dsml_token=dsml_token,
                    name=tc.get("name"),
                    arguments=encode_arguments_to_dsml(tc)
                )
                for tc in tool_calls
            ]
            tc_content += '\n\n' + tool_calls_template.format(
                dsml_token=dsml_token,
                tool_calls="\n".join(tc_list),
                tc_block_name=tool_calls_block_name,
            )

        summary_content = content or ""
        rc = reasoning_content or ""

        # 若前一条消息带有 task，则当前 assistant 输出为任务结果，不附加 thinking 块
        prev_has_task = index - 1 >= 0 and messages[index - 1].get("task") is not None

        if thinking_mode == "thinking" and not prev_has_task:
            # drop_thinking 策略：
            # - 不丢弃 或 当前消息在最后一条 user 之后：保留完整 reasoning + </think>
            # - 否则（历史消息）：清空 thinking_part，不输出推理内容
            if not drop_thinking or index > last_user_idx:
                thinking_part = thinking_template.format(reasoning_content=rc) + thinking_end_token
            else:
                thinking_part = ""

        # 根据 wo_eos 标志选择是否附加 EOS token
        if wo_eos:
            prompt += assistant_msg_wo_eos_template.format(
                reasoning=thinking_part,
                content=summary_content,
                tool_calls=tc_content,
            )
        else:
            prompt += assistant_msg_template.format(
                reasoning=thinking_part,
                content=summary_content,
                tool_calls=tc_content,
            )
    else:
        raise NotImplementedError(f"Unknown role: {role}")

    # -------------------------------
    # 消息间过渡 token 处理
    # -------------------------------
    # 若下一条消息为 assistant 或 latest_reminder，不追加过渡 token（由对方自行处理开头）
    if index + 1 < len(messages) and messages[index + 1].get("role") not in ["assistant", "latest_reminder"]:
        return prompt

    task = messages[index].get("task")
    if task is not None:
        # 内部分类任务：在消息末尾追加对应的 task special token
        assert task in VALID_TASKS, f"Invalid task: '{task}'. Valid tasks are: {list(VALID_TASKS)}"
        task_sp_token = DS_TASK_SP_TOKENS[task]

        if task != "action":
            # 非 action 任务：直接追加 task token
            prompt += task_sp_token
        else:
            # action 任务：追加 Assistant 标记 + thinking token + action token
            prompt += ASSISTANT_SP_TOKEN
            prompt += thinking_end_token if thinking_mode != "thinking" else thinking_start_token
            prompt += task_sp_token

    elif messages[index].get("role") in ["user", "developer"]:
        # 正常对话流转：user/developer 消息后追加 Assistant 标记 + thinking 控制 token
        prompt += ASSISTANT_SP_TOKEN
        if not drop_thinking and thinking_mode == "thinking":
            # 不丢弃 thinking 时，提示模型开始输出 <think>
            prompt += thinking_start_token
        elif drop_thinking and thinking_mode == "thinking" and index >= last_user_idx:
            # 丢弃历史 thinking，但当前在最后一条 user 位置，仍需提示开始 thinking
            prompt += thinking_start_token
        else:
            # chat 模式或历史位置：直接追加 </think>（表示无 thinking 输出）
            prompt += thinking_end_token

    return prompt


# ============================================================
# 预处理：Tool 消息合并与排序
# ============================================================

def merge_tool_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """将 OpenAI 格式的独立 tool 消息合并到相邻的 user 消息中。

    DeepSeek-V4 没有独立的 "tool" 角色；tool 执行结果需以 <tool_result> 块形式嵌入 user 消息。
    本函数将标准对话中的 "tool" 角色消息转换为 user 消息的 content_blocks，实现格式兼容。

    合并规则：
        1. "tool" 消息 → 生成 {"type": "tool_result", "tool_use_id": ..., "content": ...} 块。
        2. 若前一条消息已是带 content_blocks 的 user，则追加到其 content_blocks；
           否则新建一条 user 消息，仅含该 tool_result 块。
        3. "user" 消息 → 生成 {"type": "text", "text": content} 块。
           若前一条 user 无 task 且已有 content_blocks，则追加；否则新建 user 消息。
        4. 保留 task、wo_eos、mask 等扩展字段。

    Args:
        messages: OpenAI 格式的消息列表（可能包含 role="tool"）。

    Returns:
        合并后的消息列表，不含 role="tool"，tool 结果已嵌入 user 消息的 content_blocks。
    """
    merged: List[Dict[str, Any]] = []

    for msg in messages:
        msg = copy.deepcopy(msg)
        role = msg.get("role")

        if role == "tool":
            # 将 tool 消息转为 tool_result 块
            tool_block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": msg.get("content", ""),
            }
            # 尝试合并到前一条 user 消息
            if merged and merged[-1].get("role") == "user" and "content_blocks" in merged[-1]:
                merged[-1]["content_blocks"].append(tool_block)
            else:
                merged.append({
                    "role": "user",
                    "content_blocks": [tool_block],
                })
        elif role == "user":
            text_block = {"type": "text", "text": msg.get("content", "")}
            if merged and merged[-1].get("role") == "user" and "content_blocks" in merged[-1] and merged[-1].get("task") is None:
                merged[-1]["content_blocks"].append(text_block)
            else:
                new_msg = {
                    "role": "user",
                    "content": msg.get("content", ""),
                    "content_blocks": [text_block],
                }
                # 保留扩展字段
                for key in ("task", "wo_eos", "mask"):
                    if key in msg:
                        new_msg[key] = msg[key]
                merged.append(new_msg)
        else:
            merged.append(msg)

    return merged


def sort_tool_results_by_call_order(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按 assistant 发起 tool_call 的顺序，对 user 消息中的 tool_result 块排序。

    在多工具调用场景下，模型可能按特定顺序调用多个工具；为保证上下文一致性，
    tool_result 应严格对应 tool_call 的调用顺序。本函数通过 tool_use_id 关联：
        1. 遍历消息列表，记录每条 assistant 消息的 tool_calls 顺序（id → 序号）。
        2. 遇到 user 消息的 content_blocks 时，若包含多个 tool_result，
           按记录的 tool_call_order 重新排序。

    Args:
        messages: 已执行 merge_tool_messages 后的消息列表。

    Returns:
        tool_result 已按调用顺序排好序的消息列表（原地修改）。
    """
    last_tool_call_order: Dict[str, int] = {}

    for msg in messages:
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            # 重建 tool_call 顺序映射
            last_tool_call_order = {}
            for idx, tc in enumerate(msg["tool_calls"]):
                tc_id = tc.get("id") or tc.get("function", {}).get("id", "")
                if tc_id:
                    last_tool_call_order[tc_id] = idx

        elif role == "user" and msg.get("content_blocks"):
            tool_blocks = [b for b in msg["content_blocks"] if b.get("type") == "tool_result"]
            if len(tool_blocks) > 1 and last_tool_call_order:
                # 按 tool_use_id 在 last_tool_call_order 中的序号排序
                sorted_blocks = sorted(
                    tool_blocks,
                    key=lambda b: last_tool_call_order.get(b.get("tool_use_id", ""), 0)
                )
                sorted_idx = 0
                new_blocks = []
                for block in msg["content_blocks"]:
                    if block.get("type") == "tool_result":
                        new_blocks.append(sorted_blocks[sorted_idx])
                        sorted_idx += 1
                    else:
                        new_blocks.append(block)
                msg["content_blocks"] = new_blocks

    return messages


# ============================================================
# 主编码入口
# ============================================================

def encode_messages(
    messages: List[Dict[str, Any]],
    thinking_mode: str,
    context: Optional[List[Dict[str, Any]]] = None,
    drop_thinking: bool = True,
    add_default_bos_token: bool = True,
    reasoning_effort: Optional[str] = None,
) -> str:
    """将消息列表编码为 DeepSeek-V4 格式的完整 prompt 字符串。

    这是对外暴露的主编码入口，编排整个编码流水线：
        1. 预处理：merge_tool_messages + sort_tool_results_by_call_order
        2. 若存在 context，同样执行预处理并拼接到消息列表前
        3. 根据 add_default_bos_token 决定是否插入 BOS
        4. 若任意消息包含 tools，强制禁用 drop_thinking（避免丢失工具相关推理）
        5. 若启用 drop_thinking，调用 _drop_thinking_messages 丢弃历史 reasoning，
           并重新计算需渲染的消息范围（context_len / num_to_render）
        6. 逐条调用 render_message 生成文本片段并拼接

    Args:
        messages: 待编码的对话消息列表（OpenAI 格式）。
        thinking_mode: "chat" 或 "thinking"，控制模型是否输出推理过程。
        context: 可选的前置上下文消息（已编码前缀，不参与 drop_thinking 计算）。
        drop_thinking: 是否丢弃历史 assistant 的 reasoning_content，仅保留当前轮次。
        add_default_bos_token: 是否在对话开头添加 BOS token（无前文 context 时生效）。
        reasoning_effort: "max" / "high" / None，控制推理深度提示词。

    Returns:
        编码后的完整 prompt 字符串，可直接送入模型进行生成。
    """
    context = context if context else []

    # 预处理：合并 tool 消息并按调用顺序排序 tool_result
    messages = merge_tool_messages(messages)
    messages = sort_tool_results_by_call_order(context + messages)[len(context):]
    if context:
        context = merge_tool_messages(context)
        context = sort_tool_results_by_call_order(context)

    full_messages = context + messages

    # 无前文时附加 BOS token
    prompt = bos_token if add_default_bos_token and len(context) == 0 else ""

    # 若对话中定义了 tools，强制保留所有 reasoning（避免工具调用逻辑被截断）
    effective_drop_thinking = drop_thinking
    if any(m.get("tools") for m in full_messages):
        effective_drop_thinking = False

    if thinking_mode == "thinking" and effective_drop_thinking:
        # 丢弃历史 thinking 内容，并重新计算 context 与待渲染消息的边界
        full_messages = _drop_thinking_messages(full_messages)
        num_to_render = len(full_messages) - len(_drop_thinking_messages(context))
        context_len = len(full_messages) - num_to_render
    else:
        num_to_render = len(messages)
        context_len = len(context)

    for idx in range(num_to_render):
        prompt += render_message(
            idx + context_len,
            full_messages,
            thinking_mode=thinking_mode,
            drop_thinking=effective_drop_thinking,
            reasoning_effort=reasoning_effort,
        )

    return prompt


def _drop_thinking_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """丢弃最后一条 user 消息之前的非必要内容，用于压缩上下文长度。

    丢弃策略（以 last_user_idx 为分界）：
        - 保留角色为 user / system / tool / latest_reminder / direct_search_results 的消息。
        - 保留索引 >= last_user_idx 的所有消息（当前轮次及之后）。
        - 位于 last_user_idx 之前的 assistant 消息：删除 reasoning_content 字段，保留 content。
        - 位于 last_user_idx 之前的 developer 消息：整消息丢弃（developer 仅作为系统提示）。

    该策略确保模型在生成回复时，不会看到历史轮次中已完成的推理过程，
    同时保留用户输入、系统设定和工具结果等关键信息。

    Args:
        messages: 完整对话消息列表。

    Returns:
        压缩后的消息列表，reasoning_content 已按需清理。
    """
    last_user_idx = find_last_user_index(messages)
    result = []
    keep_roles = {"user", "system", "tool", "latest_reminder", "direct_search_results"}

    for idx, msg in enumerate(messages):
        role = msg.get("role")
        if role in keep_roles or idx >= last_user_idx:
            result.append(msg)
        elif role == "assistant":
            # 浅拷贝并移除 reasoning_content，保留其余字段
            msg = copy.copy(msg)
            msg.pop("reasoning_content", None)
            result.append(msg)
        # developer 及其他角色在 last_user_idx 之前被静默丢弃

    return result


# ============================================================
# 解码：模型输出解析
# ============================================================

def _read_until_stop(index: int, text: str, stop: List[str]) -> Tuple[int, str, Optional[str]]:
    """从指定索引开始读取文本，直到遇到任意一个 stop 字符串。

    扫描逻辑：
        1. 遍历所有 stop 字符串，查找其在 text[index:] 中的首次出现位置。
        2. 取最早出现的 stop 作为匹配项。
        3. 返回匹配 stop 之后的新索引、stop 之前的内容、匹配到的 stop 字符串。
        4. 若无任何 stop 匹配，则返回文本剩余全部内容，matched_stop 为 None。

    该函数是解码器的基础构件，用于按 token 边界切分模型输出。

    Args:
        index: 起始读取位置。
        text: 待解析的完整文本（模型原始输出）。
        stop: 终止标记字符串列表。

    Returns:
        (new_index, content_before_stop, matched_stop_or_None)
    """
    min_pos = len(text)
    matched_stop = None

    for s in stop:
        pos = text.find(s, index)
        if pos != -1 and pos < min_pos:
            min_pos = pos
            matched_stop = s

    if matched_stop:
        content = text[index:min_pos]
        return min_pos + len(matched_stop), content, matched_stop
    else:
        content = text[index:]
        return len(text), content, None


def parse_tool_calls(index: int, text: str) -> Tuple[int, Optional[str], List[Dict[str, str]]]:
    """从文本指定位置解析 DSML 格式的 tool_calls 块。

    解析状态机：
        1. 循环读取 <｜DSML｜invoke 或 </｜DSML｜tool_calls>：
           - 若匹配结束标签，退出循环。
           - 若匹配 invoke 标签，读取 name="...">\n 提取工具名。
        2. 循环读取 <｜DSML｜parameter 或 </｜DSML｜invoke>：
           - 匹配 parameter：提取 name/string/value 三元组，存入 tool_args。
           - 匹配 invoke 结束标签：将当前 tool_args 解码为 JSON，生成 tool_call 字典。
        3. 重复直到 tool_calls 块结束。

    格式校验：
        - invoke 标签后必须紧跟 ">\n"。
        - parameter 标签后必须紧跟 ">\n"（在读取完 parameter 内容后）。
        - 不允许重复参数名。

    Args:
        index: tool_calls 块在 text 中的起始位置（已越过 <｜DSML｜tool_calls 标签）。
        text: 模型完整输出文本。

    Returns:
        (new_index, last_stop_token, list_of_tool_call_dicts)
        每个 tool_call 为 {"name": str, "arguments": str(JSON)}。
    """
    tool_calls: List[Dict[str, Any]] = []
    stop_token = None
    tool_calls_end_token = f"</{dsml_token}{tool_calls_block_name}>"

    while index < len(text):
        index, _, stop_token = _read_until_stop(index, text, [f"<{dsml_token}invoke", tool_calls_end_token])
        if _ != ">\n":
            raise ValueError(f"Tool call format error: expected '>\\n' but got '{_}'")

        if stop_token == tool_calls_end_token:
            break

        if stop_token is None:
            raise ValueError("Missing special token in tool calls")

        index, tool_name_content, stop_token = _read_until_stop(index, text, [f"<{dsml_token}parameter", f"</{dsml_token}invoke"])

        p_tool_name = re.findall(r'^\s*name="(.*?)">\n$', tool_name_content, flags=re.DOTALL)
        if len(p_tool_name) != 1:
            raise ValueError(f"Tool name format error: '{tool_name_content}'")
        tool_name = p_tool_name[0]

        tool_args: Dict[str, Tuple[str, str]] = {}
        while stop_token == f"<{dsml_token}parameter":
            index, param_content, stop_token = _read_until_stop(index, text, [f"/{dsml_token}parameter"])

            param_kv = re.findall(r'^ name="(.*?)" string="(true|false)">(.*?)<$', param_content, flags=re.DOTALL)
            if len(param_kv) != 1:
                raise ValueError(f"Parameter format error: '{param_content}'")
            param_name, string, param_value = param_kv[0]

            if param_name in tool_args:
                raise ValueError(f"Duplicate parameter name: '{param_name}'")
            tool_args[param_name] = (param_value, string)

            index, content, stop_token = _read_until_stop(index, text, [f"<{dsml_token}parameter", f"</{dsml_token}invoke"])
            if content != ">\n":
                raise ValueError(f"Parameter format error: expected '>\\n' but got '{content}'")

        tool_call = decode_dsml_to_arguments(tool_name=tool_name, tool_args=tool_args)
        tool_calls.append(tool_call)

    return index, stop_token, tool_calls


def parse_message_from_completion_text(text: str, thinking_mode: str) -> Dict[str, Any]:
    """将模型单次生成的原始文本解析为结构化的 assistant 消息。

    解析流程：
        1. 若 thinking_mode='thinking'：
           - 先读取到 </think> 或 <｜DSML｜tool_calls 为止的内容作为 reasoning_content。
           - 必须命中 </think>，否则格式非法。
        2. 读取到 EOS 或 <｜DSML｜tool_calls 为止的内容作为 content（正文）。
           - 若命中 tool_calls 起始标记，进入工具调用解析分支。
           - 否则必须命中 EOS。
        3. 若存在 tool_calls：
           - 调用 parse_tool_calls 提取所有工具调用。
           - tool_calls 结束后必须紧跟 EOS，且其后无多余内容。
        4. 最终校验：
           - 整个文本必须被完全消费（index == len(text)）。
           - content 和 reasoning_content 中不得残留任何特殊 token（BOS/EOS/think/DSML）。

    Args:
        text: 模型原始输出文本（含 EOS 标记）。
        thinking_mode: "chat" 或 "thinking"，决定是否需要提取 reasoning_content。

    Returns:
        {
            "role": "assistant",
            "content": str,           # 正式回复正文（不含 reasoning 和 tool_calls）
            "reasoning_content": str, # <think>...</think> 内的推理文本（thinking 模式）
            "tool_calls": list        # OpenAI 格式的 tool_calls 列表（若有）
        }

    Raises:
        AssertionError / ValueError: 当模型输出格式不符合预期时抛出。
    """
    summary_content, reasoning_content, tool_calls = "", "", []
    index, stop_token = 0, None
    tool_calls_start_token = f"\n\n<{dsml_token}{tool_calls_block_name}"

    is_thinking = thinking_mode == "thinking"
    is_tool_calling = False

    if is_thinking:
        # 提取 reasoning 块：从开头到 </think> 或 tool_calls 起始
        index, content_delta, stop_token = _read_until_stop(index, text, [thinking_end_token, tool_calls_start_token])
        reasoning_content = content_delta
        assert stop_token == thinking_end_token, "Invalid thinking format: missing </think>"

    # 提取正文：从当前位置到 EOS 或 tool_calls 起始
    index, content_delta, stop_token = _read_until_stop(index, text, [eos_token, tool_calls_start_token])
    summary_content = content_delta
    if stop_token == tool_calls_start_token:
        is_tool_calling = True
    else:
        assert stop_token == eos_token, "Invalid format: missing EOS token"

    if is_tool_calling:
        # 解析 DSML tool_calls 块
        index, stop_token, tool_calls = parse_tool_calls(index, text)

        # tool_calls 结束后应紧跟 EOS
        index, tool_ends_text, stop_token = _read_until_stop(index, text, [eos_token])
        assert not tool_ends_text, "Unexpected content after tool calls"

    # 完整性校验：文本必须被完全消费，且最终停在 EOS
    assert len(text) == index and stop_token in [eos_token, None], "Unexpected content at end"

    # 安全校验：正文和推理内容中不得包含任何特殊 token（防止注入或解析残留）
    for sp_token in [bos_token, eos_token, thinking_start_token, thinking_end_token, dsml_token]:
        assert sp_token not in summary_content and sp_token not in reasoning_content, \
            f"Unexpected special token '{sp_token}' in content"

    return {
        "role": "assistant",
        "content": summary_content,
        "reasoning_content": reasoning_content,
        "tool_calls": tool_calls_to_openai_format(tool_calls)
    }
