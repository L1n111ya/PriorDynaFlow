import json
import math
import os.path
import pickle
import logging
from datetime import datetime

import numpy as np
import random
import signal
from contextlib import contextmanager, nullcontext
from collections import defaultdict
from typing import Annotated, Literal, List, Tuple, Dict, Any, Optional
from functools import lru_cache

from mainagent.llm import get_llm
from mainagent.agent.code_agent import BaseNode
from mainagent.config import Config
from mainagent.state import CodeMessageState, HybridMessageState, MathMessageState

from langchain_core.messages import HumanMessage
from langgraph.graph import END

LLM = get_llm("qwen-max-latest")

# 日志配置
logging.basicConfig(
    # filename=Config.LOG_PATH,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s'
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class QLearningDecisionMaker:
    """
    Q-Learning 决策器，用于多节点Agent的节点选择与Q表维护。
    """

    def __init__(self, all_nodes: List[str],
                 learning_rate: float = Config.LEARNING_RATE,
                 discount_factor: float = Config.DISCOUNT_FACTOR,
                 epsilon: float = Config.EPSILON,
                 entropy_weight: float = Config.ENTROPY_WEIGHT,
                 q_table_path: str = Config.Q_TABLE_PATH):
        self.all_nodes = all_nodes
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.entropy_weight = entropy_weight
        self.q_table_path = q_table_path
        self.q_table = defaultdict(dict)  # state -> (action, q_value)

        for state in all_nodes:
            self.q_table[state] = {
                action: 0 for action in all_nodes if action != state
            }

        if os.path.exists(q_table_path):
            self.load_q_table(q_table_path)

    def get_available_nodes(self, state: str) -> List:
        """
        获取当前状态下可用的节点列表。
        """
        if state in self.q_table and self.q_table[state]:
            return [node for node, value in self.q_table[state].items() if value > Config.MIN_ACTION_REWARD]
        return self.all_nodes

    def get_next_avail_nodes(self, state: str, available_nodes: List) -> List:
        """
        使用 epsilon-greedy 策略选择下一个可选节点
        """
        random_node = []
        if random.random() < self.epsilon:
            random_node.append(random.choice(available_nodes))
        # 利用Q值选择最佳节点
        # q_values = {node: self.q_table[state].get(node, 0) for node in available_nodes}
        # max_q = max(q_values.values())
        # best_nodes = [node for node, q in q_values.items() if q == max_q]

        # Sorted by Q value
        sorted_by_q = sorted(self.q_table[state].items(), key=lambda x: x[1], reverse=True)

        #
        q_groups = defaultdict(list)
        for node, q in sorted_by_q:
            q_groups[q].append(node)

        #
        top_two_q_values = list(q_groups.keys())[:2]
        best_nodes = []
        for q in top_two_q_values:
            best_nodes.extend(q_groups[q])

        return (random_node if random_node not in best_nodes else []) + (best_nodes if best_nodes else [])

    def update_episode(self, episode: List[Dict[str, Any]]):
        """
        根据一个 episode 更新Q表
        """
        for t in range(len(episode)):
            state = episode[t]["state"]
            action = episode[t]["action"]
            reward = episode[t]["reward"]
            next_state = episode[t + 1]["state"] if t < len(episode) - 1 else END

            if next_state == END and reward > 0:
                reward *= 1.5

            self.ensure_all_actions(state)
            self.ensure_all_actions(next_state)

            entropy = self.get_entropy(state)
            adjusted_reward = reward + self.entropy_weight * entropy


            self.q_table[state][action] += self.learning_rate * (
                    adjusted_reward + self.discount_factor * max(self.q_table[next_state].values() or [0]) -
                    self.q_table[state][action]
            )

    def update_q_value(self, state: str, action: str, reward: float, next_state: str, done: bool):
        """
        更新Q值
        """
        if done:
            target = reward
        else:
            # r + γ * max_a' Q(s', a')
            next_max_q = max(self.q_table[next_state].values() or [0])
            target = reward + self.discount_factor * next_max_q

        # Q(s,a) = Q(s,a) + α * [target - Q(s,a)]
        self.q_table[state][action] += self.learning_rate * (target - self.q_table[state][action])

    def decay_epsilon(self, decay_rate: float = 0.995, min_epsilon: float = 0.1):
        """衰减epsilon值"""
        self.epsilon = max(self.epsilon * decay_rate, min_epsilon)

    @lru_cache(maxsize=128)
    def get_policy(self, state: str):
        q_values = dict(self.q_table[state])
        if not q_values:
            return {node: 1 / len(self.all_nodes) for node in self.all_nodes}

        actions = list(q_values.keys())
        q_list = np.array([q_values[a] for a in actions])
        exp_q = np.exp(q_list - np.max(q_list))
        probs = exp_q / np.sum(exp_q)
        return dict(zip(actions, probs))

    @lru_cache(maxsize=128)
    def get_entropy(self, state: str):
        policy = self.get_policy(state)
        entropy = -sum(p * math.log(p + 1e-8) for p in policy.values())
        return entropy

    # def save_q_table(self, path: str = "q_table.pkl"):
    #     """保存Q表，先将 defaultdict 转换成普通 dict"""
    #     q_table_regular = {
    #         state: dict(actions) for state, actions in self.q_table.items()
    #     }
    #
    #     with open(path, "wb") as f:
    #         pickle.dump(q_table_regular, f)
    #
    #     logger.info(f"Q-table is saved at {path}")
    # def save_q_table(self, path: str = "q_table.pkl"):
    #     """保存Q表，并记录为JSON格式用于后续可视化"""
    #     # 将 defaultdict 转换为普通 dict 以便序列化
    #     q_table_regular = {
    #         state: dict(actions) for state, actions in self.q_table.items()
    #     }
    #
    #     # 保存原始 Q 表为 pickle
    #     with open(path, "wb") as f:
    #         pickle.dump(q_table_regular, f)
    #
    #     # 保存为 JSON 格式用于可视化
    #     json_path = path.replace(".pkl", ".json")
    #     with open(json_path, "w") as f:
    #         json.dump(q_table_regular, f, indent=4)
    #
    #     logger.info(f"Q-table saved at {path} and exported to {json_path}")
    def save_q_table(self, path: str = "q_table.pkl"):
        """保存Q表，并记录为JSONL格式用于后续可视化"""
        # 将 defaultdict 转换为普通 dict 以便序列化
        q_table_regular = {
            state: dict(actions) for state, actions in self.q_table.items()
        }

        # 保存原始 Q 表为 pickle
        with open(path, "wb") as f:
            pickle.dump(q_table_regular, f)

        # 保存为 JSON 格式用于可视化
        json_path = path.replace(".pkl", ".json")
        with open(json_path, "w") as f:
            json.dump(q_table_regular, f, indent=4)

        # 追加写入 JSONL 文件，一行一个 Q 表快照
        jsonl_path = path.replace(".pkl", "_history.jsonl")
        with open(jsonl_path, "a") as f:
            record = {
                "timestamp": datetime.now().isoformat(),
                "q_table": q_table_regular
            }
            f.write(json.dumps(record) + "\n")

        logger.info(f"Q-table saved at {path}, exported to {json_path} and appended to {jsonl_path}")

    def load_q_table(self, path: str = "q_table.pkl"):
        """加载Q表"""
        path = path or self.q_table_path
        with open(path, "rb") as f:
            loaded = pickle.load(f)
            # Exchange format
            self.q_table = defaultdict(lambda: defaultdict(float), {
                k: defaultdict(float, v) for k, v in loaded.items()
            })
        logger.info(f"Q-table is loaded from {path}")

    def ensure_all_actions(self, state: str):
        if state not in self.q_table:
            self.q_table[state] = {
                action: 0 for action in self.all_nodes if action != state
            }
        else:
            # 添加缺失的 action
            for action in self.all_nodes:
                if action != state and action not in self.q_table[state]:
                    self.q_table[state][action] = 0


class MainAgent:
    """
    多智能体主控类，负责节点注册、Q-learning训练与推理流程。
    """

    def __init__(self, train_num: int = Config.DEFAULT_TRAIN_NUM,
                 q_table_path: str = Config.Q_TABLE_PATH):
        self.nodes: Dict[str, BaseNode] = {END: BaseNode(END)}
        self.start_node: Optional[str] = None
        self.prompt_token = 0
        self.completion_token = 0
        self.q_learning: Optional[QLearningDecisionMaker] = None
        self.run_count = 0
        self.success = 0
        self.train_num = train_num
        self.q_table_path = q_table_path
        self._interrupt_flag = False  # 中断标志

    def register_node(self, node_name: str, node: BaseNode, is_start: bool = False) -> None:
        """注册节点"""
        self.nodes[node_name] = node
        if is_start:
            self.start_node = node_name

    # ----------------------
    # 中断相关工具函数
    # ----------------------

    @contextmanager
    def interrupt_handler(self):
        """上下文管理器：捕获 Ctrl+C 中断"""
        self._interrupt_flag = False

        def signal_handler(sig, frame):
            logger.warning("Interrupt signal received. Will stop after current iteration.")
            self._interrupt_flag = True

        original_handler = signal.signal(signal.SIGINT, signal_handler)
        try:
            yield
        finally:
            signal.signal(signal.SIGINT, original_handler)

    def should_stop(self) -> bool:
        """检查是否需要中断"""
        return self._interrupt_flag

    # ----------------------
    # 主方法 run()
    # ----------------------

    def run(self, input_text: str, test_cases: List = "", enable_retry: bool = True,
            allow_interrupt: bool = True) -> str:
        """
        执行 Agent 的主流程，支持中断和可选重试机制。

        :param input_text: 用户输入的任务描述
        :param max_retry: 最大重试次数，默认为 3 次
        :param enable_retry: 是否启用自动重试机制
        :param allow_interrupt: 是否允许用户通过 Ctrl+C 中断执行
        :return: 生成的代码字符串，若失败则返回空字符串
        """

        if self.q_learning is None:
            self.initialize_q_learning()
            logger.info("Q-Learning initialized.")

        max_retry = Config.MAX_RETRY
        retry_count = 0
        with self.interrupt_handler() if allow_interrupt else nullcontext():
            while enable_retry and retry_count < max_retry or (not enable_retry and retry_count == 0):
                try:
                    if self.run_count > Config.ALL_EXPLORE or retry_count:
                        enable_retry = False
                    retry_count += 1
                    result, state = self._run_single_attempt(input_text, test_cases)
                    if state:
                        return result
                except Exception as e:
                    logging.error(f"Failed to run agent: {e}", exc_info=True)
                    retry_count += 1
                    if not enable_retry or retry_count >= max_retry:
                        logger.error(f"Reached maximum retry count: {max_retry}. Aborting.")
                        return result
                    logger.warning(f"Retrying run... (attempt {retry_count + 1}/{max_retry})")
                    continue

                if self.should_stop():
                    logger.warning("Execution interrupted by user.")
                    return ""

        logger.info("Run completed without generating valid code.")
        return result

    def _run_single_attempt(self, input_text: str, test_cases: List = "") -> [Optional[str], bool]:
        """
        执行一次完整的运行流程，不包括重试逻辑。

        :param input_text: 用户输入的任务描述
        :return: 成功时返回代码字符串，失败时返回 None
        """
        current_state = self._initialize_state(input_text)
        current_state_str = self.start_node
        done = False
        episode = []
        total_reward = 0
        last_executed_node = 'START'

        while not done:
            if self.should_stop():
                logger.warning("Stopping early due to interruption.")
                return None

            available_nodes = self._get_available_nodes(current_state_str, current_state)
            next_avail_nodes = self.q_learning.get_next_avail_nodes(current_state_str, available_nodes)
            if not next_avail_nodes:
                next_avail_nodes = list(self.nodes.keys())

            # 前几轮全探索
            if self.run_count < Config.ALL_EXPLORE:
                next_avail_nodes = list(self.nodes.keys())

            if last_executed_node == current_state_str:
                reward = -Config.REPEATED_PENALTY
            else:
                reward = 0

            # 执行节点
            if current_state.next_node == END:
                logger.info("=====END=====")
                done = True
                self.success += 1
                path_len = len(current_state.executed_nodes)
                path_penalty = max(0, Config.MIN_PATH_LENGTH - path_len)
                reward += Config.SUCCESS_REWARD - path_penalty * Config.PATH_PENALTY
            else:
                # reward += self._calculate_node_reward(current_state)
                current_state = self.nodes[current_state_str].node(current_state, test_cases, next_avail_nodes)
                next_state_str = current_state.next_node or END

                reward += self._calculate_reward(current_state, last_executed_node)
                self.prompt_token += current_state.prompt_token
                self.completion_token += current_state.completion_token

                # 更新状态
                # 更新状态
                current_state_str_old = current_state_str
                current_state_str = next_state_str
                last_executed_node = current_state_str_old

                print(f"reward: {reward}")
                total_reward += reward
                print(f"total_reward: {total_reward}")
                self.run_count += 1
                # next_state_str = current_state.next_node if current_state.next_node else END

                # 记录 Episode
                episode.append({
                    "state": current_state_str_old,
                    "action": current_state_str,
                    "reward": reward,
                })

            # last_executed_node = current_state_str
            # current_state_str = next_state_str

            # print(f"Episode: {episode}")
        # if total_reward < Config.MIN_REWARD and enable_retry:
        #     logger.warning(f"Total reward < {Config.MIN_REWARD}, will retry.")
        #     return current_state.code
            if total_reward < Config.MIN_REWARD:
                logger.warning("=====FORCE END=====")
                return current_state.code, done
            # return current_state.answer

        # 成功完成任务
        if self.run_count <= self.train_num:
            self.q_learning.update_episode(episode)
            self.q_learning.decay_epsilon()

        is_success = (current_state.next_node == END)
        for step in episode:
            node_name = step["action"]
            if node_name in self.nodes:
                self.nodes[node_name].update_success_rate(is_success)

        self.save_q_table()
        # print(f"Final Code: {current_state.code}")
        return current_state.code, done
        # return current_state.answer

    def _get_available_nodes(self, state: str, current_state: CodeMessageState) -> List[str]:
        """
        获取当前状态下可用的节点列表，考虑首次执行不能结束的情况。
        """
        available_nodes = self.q_learning.get_available_nodes(state)
        if len(current_state.executed_nodes) == 0 and END in available_nodes:
            available_nodes.remove(END)
        if not available_nodes and len(current_state.executed_nodes):
            available_nodes = [END]
        elif not available_nodes:
            available_nodes = list(self.nodes.keys())
        return available_nodes

    def _calculate_node_reward(self, current_state: MathMessageState) -> float:
        """
        根据当前节点计算奖励值。
        """
        if self.run_count <= self.train_num:
            current_node = self.nodes[current_state.next_node]
            return -current_node.salaries[current_state.next_node] * 2 + current_node.success_rate * 10
        return 0

    def _calculate_reward(self, state: CodeMessageState, last_node: str) -> float:
        reward = 0
        if state.next_node == END:
            self.success += 1
            path_len = len(state.executed_nodes)
            path_penalty = max(0, Config.MIN_PATH_LENGTH - path_len)
            reward += Config.SUCCESS_REWARD - path_penalty * Config.PATH_PENALTY
        else:
            current_node = self.nodes[state.next_node]
            reward -= current_node.salaries[state.next_node] * Config.BASE_SALARY_MULTIPLIER
            reward += current_node.success_rate * Config.SUCCESS_RATE_MULTIPLIER
        if last_node == state.next_node:
            reward -= Config.NODE_PENALTY
        # elif self.run_count <= self.train_num:

        return reward

    def initialize_q_learning(self):
        """初始化Q-Learning"""
        try:
            all_nodes = list(self.nodes.keys())
            self.q_learning = QLearningDecisionMaker(all_nodes, q_table_path=self.q_table_path)
        except Exception as e:
            logging.error(f"Failed to initialize Q-Learning: {e}")
            raise

    def save_q_table(self):
        """保存当前 Q 表"""
        if self.q_learning:
            self.q_learning.save_q_table(self.q_table_path)

    def load_q_table(self):
        """手动加载 Q 表（可选）"""
        if self.q_learning:
            self.q_learning.load_q_table(self.q_table_path)

    def _initialize_state(self, input_text: str) -> CodeMessageState:
        return CodeMessageState(
            task=input_text,
            messages=HumanMessage(content=input_text),
            code="",
            feedback="",
            executed_nodes=[],
            prompt_token=0,
            completion_token=0,
            next_node=self.start_node
        )
        
    # def _initialize_state(self, input_text: str) -> MathMessageState:
    #     return MathMessageState(
    #         task=input_text,
    #         messages=HumanMessage(content=input_text),
    #         answer="",
    #         code="",
    #         prompt_token=0,
    #         completion_token=0,
    #         next_node=self.start_node,
    #         executed_nodes=[],
    #     )
    
    # def _initialize_state(self, input_text: str) -> HybridMessageState:
    #     return HybridMessageState(
    #         task=input_text,
    #         messages=HumanMessage(content=input_text),
    #         answer="",
    #         code="",
    #         feedback="",
    #         executed_nodes=[],
    #         prompt_token=0,
    #         completion_token=0,
    #         next_node=self.start_node
    #     )



