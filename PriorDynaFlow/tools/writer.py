import sys
import subprocess
from typing import Annotated, List
from langchain_core.tools import tool


# @tool
# def python_repl_tool(
#         code: Annotated[str, "生成的Python代码"]
# ):
#     """Use this to execute python code. """
#     try:
#         repl = PythonREPL()
#         # repl.globals = {
#         #     "math": math,
#         #     "re": re,
#         # }
#         result = repl.run(code)
#     except BaseException as e:
#         return f"代码执行出错. ERROR: {e}"
#     result_str = f"执行完毕:\n Stdout: {result}"
#     return (
#             result_str + """
#                 \n\n 如果你完成了所有的任务，在next_node中返回：FINAL ANSWER."""
#     )

@tool
def python_repl_tool(code: Annotated[str, "生成的Python代码"]):
    """Use this to execute python code. If you want to see the output of a value,
        you should print it out with `print(...)`."""
    with open("temp_script.py", "w") as f:
        f.write(code)

    try:
        result = subprocess.run(
            [sys.executable, "temp_script.py"],
            capture_output=True,
            text=True,
            timeout=10
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        output = "ERROR: 代码执行超时"
    except Exception as e:
        output = f"ERROR: {str(e)}"

    return f"执行完毕:\n Stdout: {output}\n\n \`\`\`python \n {code} \n \`\`\` \n\n如果完成任务，请返回：FINAL ANSWER."