import os
import random

from mainagent.tools.writer import write_jsonl
from mainagent.llm import get_llm
from mainagent.agent.mainagent import MainAgent
from mainagent.agent.code_agent import PlanNode, ResearchNode, GeneratorNode, CodeReviewNode, TestNode
from mainagent.prompt.system_prompt import CODE_SYSTEM_PROMPT
from mainagent.tools.coding.python_executor import PyExecutor
from mainagent.tools.reader import JSONLReader

if __name__ == "__main__":
    LLM = get_llm("qwen-max-latest")

    agent_1 = PlanNode(llm=LLM, system_prompt=CODE_SYSTEM_PROMPT, name="plan_node")
    agent_2 = ResearchNode(llm=LLM, system_prompt=CODE_SYSTEM_PROMPT, name="research_node")
    agent_3 = GeneratorNode(llm=LLM, system_prompt=CODE_SYSTEM_PROMPT, name="code_generate_node")
    agent_4 = CodeReviewNode(llm=LLM, system_prompt=CODE_SYSTEM_PROMPT, name="code_review_node")
    agent_5 = TestNode(llm=LLM, system_prompt=CODE_SYSTEM_PROMPT, name="test_node")

    agent = MainAgent()

    agent.register_node("plan_node", agent_1, is_start=True)
    agent.register_node("research_node", agent_2)
    agent.register_node("code_generate_node", agent_3)
    agent.register_node("code_review_node", agent_4)
    agent.register_node("test_node", agent_5)

    problems = JSONLReader.parse_file(file_path="data/humaneval-py.jsonl")
    random.shuffle(problems)

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # 全局缓存所有已生成的样本
    all_samples = []  # task_id -> sample
    batch_count = 0
    total_solved = 0

    for problem in problems:
        result = agent.run(
            f"""
                用户给你的需要实现的代码是：{problem["prompt"]}
            """
        )
        sample = {"task_id": problem["name"], "completion": result}
        test = problem["test"]
        is_solved, feedback, _ = PyExecutor().execute(sample["completion"], [test], timeout=100)
        if is_solved:
            total_solved += 1

        sample["result"] = feedback
        sample["passed"] = is_solved

        # 更新全局缓存
        all_samples.append(sample)
        batch_count += 1
        temp_file = os.path.join(output_dir, f"cade_eval.jsonl")
        write_jsonl(temp_file, all_samples)
        print(f"当前整体 pass@1: {total_solved / batch_count:.5%}")

    print(f"prompt token: {agent.prompt_token}")
    print(f"completion token: {agent.completion_token}")
    print(f"最终整体 pass@1: {total_solved / batch_count:.5%}")