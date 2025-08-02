"""配置参数管理"""


class Config:
    # MainAgent配置
    MAX_RETRY = 3
    MIN_REWARD = -50
    SUCCESS_REWARD = 100
    PATH_PENALTY = 10
    BASE_SALARY_MULTIPLIER = 2
    SUCCESS_RATE_MULTIPLIER = 5
    MIN_PATH_LENGTH = 5
    NODE_PENALTY = 10
    REPEATED_PENALTY = 10
    ALL_EXPLORE = 50

    # Q-Learning配置
    LEARNING_RATE = 0.1
    DISCOUNT_FACTOR = 0.9
    EPSILON = 0.3
    ENTROPY_WEIGHT = 0.01
    DECAY_RATE = 0.995
    MIN_EPSILON = 0.1
    MIN_ACTION_REWARD = -50

    # 节点薪水配置
    # NODE_SALARIES = {
    #     "plan_node": 5,
    #     "research_node": 2,
    #     "code_generate_node": 5,
    #     "code_review_node": 5,
    #     "test_node": 5
    # }
    # NODE_SALARIES = {
    #     "MathSolverAgent": 5,
    #     "MathematicalAnalystAgent": 5,
    #     "ProgrammingExpertAgent": 5,
    #     "InspectorAgent": 5
    # }
    
    NODE_SALARIES = {
    "PlanAgent": 5,
    "AnalystAgent": 5,
    "ProgrammingAgent": 5,
    "InspectorAgent": 5,
    "CodeAuditorAgent": 5,
    "TestEngineerAgent": 5
}

    # 系统配置
    Q_TABLE_PATH = "../humaneval_q_table_1.pkl"
    DEFAULT_TRAIN_NUM = 1000
    LOG_PATH = '../agent.log'