"""
DeepSeek V4 编码模块测试入口

本文件为 encoding_dsv4.py 的测试套件，覆盖以下场景：
1. thinking 模式 + tool calls 的多轮对话编码与解析
2. thinking 模式无工具，验证 drop_thinking 逻辑
3. 交错 thinking + search（developer + latest_reminder）
4. quick instruction task（chat 模式 + action task）

运行方式：python test_encoding_dsv4.py
"""

import json
import os

from encoding_dsv4 import encode_messages, parse_message_from_completion_text

# 测试数据目录，位于当前文件同级目录下的 tests/ 文件夹
TESTS_DIR = os.path.join(os.path.dirname(__file__), "tests")


def test_case_1():
    """
    测试用例 1：thinking 模式 + tool calls 的多轮对话

    场景说明：
    - 用户询问北京天气，助手需要调用 get_weather 工具
    - 工具执行结果合并回 user 消息，形成多轮交互
    - 最终助手给出带格式的自然语言回复

    验证点：
    - encode_messages 生成的 prompt 与预期 gold 文件完全一致
    - 解析第一轮 assistant 输出：正确提取 reasoning_content、content、tool_calls
    - 解析最后一轮 assistant 输出： reasoning_content 与 content 均正确
    """
    # 加载测试输入：messages 与 tools 定义
    with open(os.path.join(TESTS_DIR, "test_input_1.json")) as f:
        td = json.load(f)
        messages = td["messages"]
        # 将工具定义附加到首条消息（OpenAI 风格）
        messages[0]["tools"] = td["tools"]
    # 加载预期编码结果（gold 文本）
    gold = open(os.path.join(TESTS_DIR, "test_output_1.txt")).read()
    # 执行编码：thinking 模式会包裹 <think> 标签
    prompt = encode_messages(messages, thinking_mode="thinking")
    assert prompt == gold

    # 解析验证：第一轮 assistant 输出（包含 tool call）
    marker = "<｜Assistant｜><think>"
    # 定位第一轮 assistant 思考内容的起始位置
    first_start = prompt.find(marker) + len(marker)
    # 定位第一轮结束位置（下一条 User 消息之前）
    first_end = prompt.find("<｜User｜>", first_start)
    parsed_tc = parse_message_from_completion_text(prompt[first_start:first_end], thinking_mode="thinking")
    # 验证 reasoning_content：助手思考过程
    assert parsed_tc["reasoning_content"] == "The user wants to know the weather in Beijing. I should use the get_weather tool."
    # 验证 content：此时助手尚未输出最终回复，应为空
    assert parsed_tc["content"] == ""
    # 验证 tool_calls：应包含 1 个工具调用
    assert len(parsed_tc["tool_calls"]) == 1
    # 验证工具名称
    assert parsed_tc["tool_calls"][0]["function"]["name"] == "get_weather"
    # 验证工具参数 JSON 解析结果
    assert json.loads(parsed_tc["tool_calls"][0]["function"]["arguments"]) == {"location": "Beijing", "unit": "celsius"}

    # 解析验证：最后一轮 assistant 输出（最终回复）
    last_start = prompt.rfind(marker) + len(marker)
    parsed_final = parse_message_from_completion_text(prompt[last_start:], thinking_mode="thinking")
    # 验证最终 reasoning_content
    assert parsed_final["reasoning_content"] == "Got the weather data. Let me format a nice response."
    # 验证最终 content 包含天气信息
    assert "22°C" in parsed_final["content"]
    # 验证最终输出无 tool_calls
    assert parsed_final["tool_calls"] == []

    print("  [PASS] case 1: thinking with tools (encode + parse)")


def test_case_2():
    """
    测试用例 2：thinking 模式无工具，验证 drop_thinking 逻辑

    场景说明：
    - 多轮对话中，早期 assistant 的思考内容在后续轮次中应被丢弃
    - 仅保留最后一轮 assistant 的 reasoning_content

    验证点：
    - encode_messages 生成的 prompt 与预期 gold 文件一致
    - 解析最后一轮 assistant 输出：reasoning_content 与 content 正确
    - drop_thinking 生效：早期 assistant 的思考文本不应出现在 prompt 中
    """
    # 加载测试输入
    messages = json.load(open(os.path.join(TESTS_DIR, "test_input_2.json")))
    # 加载预期编码结果
    gold = open(os.path.join(TESTS_DIR, "test_output_2.txt")).read()
    # 执行编码：thinking 模式
    prompt = encode_messages(messages, thinking_mode="thinking")
    assert prompt == gold

    # 解析验证：最后一轮 assistant 输出
    marker = "<｜Assistant｜><think>"
    last_start = prompt.rfind(marker) + len(marker)
    parsed = parse_message_from_completion_text(prompt[last_start:], thinking_mode="thinking")
    # 验证 reasoning_content：最后一轮思考内容保留
    assert parsed["reasoning_content"] == "The user asks about the capital of France. It is Paris."
    # 验证 content：最终自然语言回复
    assert parsed["content"] == "The capital of France is Paris."
    # 验证无 tool_calls
    assert parsed["tool_calls"] == []

    # 验证 drop_thinking：第一轮 assistant 的思考内容应已被移除
    assert "The user said hello" not in prompt

    print("  [PASS] case 2: thinking without tools (encode + parse)")


def test_case_3():
    """
    测试用例 3：交错 thinking + search（developer + latest_reminder）

    场景说明：
    - 包含 developer 消息、search 工具结果与 thinking 标签的复杂交错场景
    - 验证 latest_reminder 机制在 thinking 模式下的编码正确性

    验证点：
    - encode_messages 生成的 prompt 与预期 gold 文件完全一致
    """
    # 加载测试输入
    messages = json.load(open(os.path.join(TESTS_DIR, "test_input_3.json")))
    # 加载预期编码结果
    gold = open(os.path.join(TESTS_DIR, "test_output_3.txt")).read()
    # 执行编码并直接比对完整输出
    assert encode_messages(messages, thinking_mode="thinking") == gold
    print("  [PASS] case 3: interleaved thinking + search")


def test_case_4():
    """
    测试用例 4：quick instruction task（chat 模式 + action task）

    场景说明：
    - 使用 chat 模式（非 thinking 模式）处理快速指令类任务
    - 验证 action task 类型消息在 chat 模式下的编码正确性

    验证点：
    - encode_messages 生成的 prompt 与预期 gold 文件完全一致
    """
    # 加载测试输入
    messages = json.load(open(os.path.join(TESTS_DIR, "test_input_4.json")))
    # 加载预期编码结果
    gold = open(os.path.join(TESTS_DIR, "test_output_4.txt")).read()
    # 执行编码：chat 模式不包裹 <think> 标签
    assert encode_messages(messages, thinking_mode="chat") == gold
    print("  [PASS] case 4: quick instruction task")


if __name__ == "__main__":
    print("Running DeepSeek-V4 Encoding Tests...\n")
    test_case_1()
    test_case_2()
    test_case_3()
    test_case_4()
    print("\nAll 4 tests passed!")
