# (P)rior(D)yna(F)low--A Priori Dynamic Workflow Construction via Multi-Agent Collaboration

### 🚀Quick Start
#### Install Packages
```bash
conda create -n mainagen python=3.10
conda activate mainagen
pip3 install -e .
```

#### Set Environment Variables
 **OpenAI API Configuration**  
Firstly, set your settings in the ```PriorDynaFlow/llm.py``` file.
```python
base_url = "" # The endpoint of your OPENAI API
api_key = "" # Your OPENAI API KEY
```
**Configuration**  
Then, set your configuration in the ```PriorDynaFlow/config. py``` file.
```python
# YOUR PriorDynaFlow SETTINGS
...
# YOUR Q-Learning SETTINGS
...
YOUR_NODE_SALARIES = {
        "MathSolverAgent": 5,
        "MathematicalAnalystAgent": 5,
        "ProgrammingExpertAgent": 5,
        "InspectorAgent": 5
    }
```
#### Run Experiments
The structure of the experiment in the paper is as follows:
```
experiment
    |-data
        |-humaneval-py.jsonl
        └──gsm8k_eval.jsonl
        ...
    |-results
    |-code_eval.py
    └──mathset_eval.py   
...
```
You can reproduce the experiment of human eval and gsm8k by using the following command:
```bash
python3 code_eval.py  # humaneval or mbpp
python3 mathset_eval.py # gsm8k or math
```
### 🔗Add new agent
The basic agent class is ```PriorDynaFlow/agent/base_agent.py```, which implements the functionality of a universal agent node.

**Implement Your Agent Class** 

Example:
```python
class YourAgent(BaseNode):
    def __init__(self, llm: ChatOpenAI, system_prompt: str, name: str):
        super().__init__(name)
        self.llm = llm
        self.sys_prompt = system_prompt + your_agent_prompt

    def _execute_node(self, state: YourMessageState, next_avail_nodes: List) -> YourMessageState:
        
        # Your Agent Logic
        ...

        goto = self.get_next_node(result, next_avail_nodes)

        new_executed_nodes = state.executed_nodes.copy() if state.executed_nodes else []
        new_executed_nodes.append(goto)

        return YourMessageState(
            next_node=goto,
            ...
        )
```

**Define the Prompt**

```python
YourAgentPrompt = """
...
"""
```
