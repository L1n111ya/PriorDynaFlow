"""自定义异常类"""

class AgentError(Exception):
    """Agent基础异常类"""
    pass

class NodeExecutionError(AgentError):
    """节点执行异常"""
    pass

class QTableError(AgentError):
    """Q表操作异常"""
    pass

class CodeGenerationError(AgentError):
    """代码生成异常"""
    pass