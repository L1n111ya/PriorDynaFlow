import os
import re
import json
import random


from mainagent.llm import get_llm
from mainagent.agent.mainagent import MainAgent
from mainagent.agent.math_agent import MathSolverAgent, MathematicalAnalystAgent, ProgrammingExpertAgent, InspectorAgent
from mainagent.prompt.system_prompt import MATH_SYSTEM_PROMPT
from mainagent.tools.math.math_eq import math_equal
from mainagent.tools.reader import load_jsonl


if __name__ == "__main__":
    LLM = get_llm("qwen-max-latest")

    agent_1 = MathSolverAgent(LLM, MATH_SYSTEM_PROMPT, "MathSolverAgent")
    agent_2 = MathematicalAnalystAgent(LLM, MATH_SYSTEM_PROMPT, "MathematicalAnalystAgent")
    agent_3 = ProgrammingExpertAgent(LLM, MATH_SYSTEM_PROMPT, "ProgrammingExpertAgent")
    agent_4 = InspectorAgent(LLM, MATH_SYSTEM_PROMPT, "InspectorAgent")

    agent = MainAgent()

    agent.register_node("MathSolverAgent", agent_1, is_start=True)
    agent.register_node("MathematicalAnalystAgent", agent_2)
    agent.register_node("ProgrammingExpertAgent", agent_3)
    agent.register_node("InspectorAgent", agent_4)

    data_dir = 'data/gsm8k_eval.jsonl'
    # num_test_sample = 1000

    examples = load_jsonl(data_dir)
    # random.shuffle(examples)
    # examples = examples[:num_test_sample]

    # 创建结果存储目录
    os.makedirs("./results", exist_ok=True)
    # 单独打开文件用于实时写入
    with open("./results/gsm8k_results.jsonl", "w") as f:
        num_success = 0
        for ex in examples:
            # 执行推理
            result = agent.run(
                # f"""The field of this question is {ex['subject']}, and the question is {ex['problem']}"""
                f"""The question is {ex['question']}"""
            )

            ex['pred'] = result
            answer_str = ex['answer'].lstrip().rstrip()
            match = re.search(r'####\s*(\d+)', answer_str)
            clean_answer = match.group(1) if match else answer_str
            ex['answer'] = clean_answer
            # ex['prompt_token'] = prompt_token
            # ex['completion_token'] = completion_token

            # 实时计算准确率
            try:
                if ex['pred'] in ex['answer'] or math_equal(ex['pred'], ex['answer']):
                    ex['result'] = 'True'
                    num_success += 1
                else:
                    ex['result'] = 'False'
            except Exception as e:
                ex['result'] = 'False'
                continue  # 出现异常时跳过当前样本，继续下一个

            # 实时写入单条记录（JSON Lines格式）
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            f.flush()  # 强制刷新缓冲区

        # 最终计算整体准确率
        accuracy = num_success / len(examples)
        print(f"Accuracy: {accuracy:.5f}")
        print(f"Prompt tokens: {agent.prompt_token}")
        print(f"Completion tokens: {agent.completion_token}")

        # 单独保存整体结果
        with open("./results/summary1.json", "w") as sf:
            json.dump({"accuracy": accuracy}, sf, indent=2)