import os
import random


from PriorDynaFlow.llm import get_llm
from PriorDynaFlow.agent.mainflow import PriorDynaFlow
from PriorDynaFlow.agent.general_agent import *
from PriorDynaFlow.prompt.system_prompt import ALL_SYSTEM_PROMPT
from PriorDynaFlow.tools.math.math_eq import math_equal
from PriorDynaFlow.tools.reader import load_jsonl
from PriorDynaFlow.tools.writer import write_jsonl

if __name__ == "__main__":
    LLM = get_llm("qwen-max-latest")

    agent_1 = PlanAgent(LLM, ALL_SYSTEM_PROMPT, "PlanAgent")
    agent_2 = AnalystAgent(LLM, ALL_SYSTEM_PROMPT, "AnalystAgent")
    agent_3 = ProgrammingAgent(LLM, ALL_SYSTEM_PROMPT, "ProgrammingAgent")
    agent_4 = InspectorAgent(LLM, ALL_SYSTEM_PROMPT, "InspectorAgent")
    agent_5 = CodeAuditorAgent(LLM, ALL_SYSTEM_PROMPT, "CodeAuditorAgent")
    agent_6 = TestEngineerAgent(LLM, ALL_SYSTEM_PROMPT, "TestEngineerAgent")

    agent = PriorDynaFlow()

    agent.register_node("PlanAgent", agent_1, is_start=True)
    agent.register_node("AnalystAgent", agent_2)
    agent.register_node("ProgrammingAgent", agent_3)
    agent.register_node("InspectorAgent", agent_4)
    agent.register_node("CodeAuditorAgent", agent_5)
    agent.register_node("TestEngineerAgent", agent_6)

    math_dir = 'data/gsm8k_eval.jsonl'
    code_dir = 'data/humaneval-py.jsonl'
    output_dir = 'results'

    math_set = load_jsonl(math_dir)
    code_set = load_jsonl(code_dir)

    random.shuffle(math_set)
    random.shuffle(code_set)
    math_set = math_set[:len(code_set)]

    math_success = 0
    code_success = 0
    math_count = 0
    code_count = 0

    code_samples = []
    math_samples = []
    for math_data, code_data in zip(math_set, code_set):
        result = agent.run(
            f"""
                用户给你的需要实现的代码是：{code_data["prompt"]}
            """
        )
        sample = {"task_id": code_data["name"], "completion": result}
        test = code_data["test"]
        # test = extract_assert_statements(test)
        is_solved, feedback, _ = PyExecutor().execute(sample["completion"], [test], timeout=100)
        if is_solved:
            code_success += 1

        sample["result"] = feedback
        sample["passed"] = is_solved

        # 更新全局缓存
        code_samples.append(sample)
        code_count += 1
        temp_file = os.path.join(output_dir, f"all_code_eval.jsonl")
        write_jsonl(temp_file, code_samples)
        print(f"code当前整体 pass@1: {code_success / code_count:.5%}")

        result = agent.run(
            # f"""The field of this question is {ex['subject']}, and the question is {ex['problem']}"""
            f"""The math question is {math_data['question']}"""
        )

        math_data['pred'] = result
        # 实时计算准确率
        try:
            if math_data['pred'] in math_data['answer'] or math_equal(math_data['pred'], math_data['answer']):
                math_data['result'] = 'True'
                math_success += 1
            else:
                math_data['result'] = 'False'
        except Exception as e:
            math_data['result'] = 'False'
            continue  # 出现异常时跳过当前样本，继续下一个
        math_count += 1
        math_samples.append(math_data)
        temp_file = os.path.join(output_dir, f"all_math_eval.jsonl")
        write_jsonl(temp_file, math_samples)
        print(f"math当前整体 pass@1: {math_success / math_count:.5%}")