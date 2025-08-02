nodes_details = {
    "MathSolverAgent": "数学专业，可以规划和解题",
    "MathematicalAnalystAgent": "可以分析问题并根据问题调用Python程序解决问题，但是需要提供给他Python程序",
    "ProgrammingExpertAgent": "编程专家，根据需求编写Python代码",
    "InspectorAgent": "检查员，分析计算过程和代码等来检查计算过程是否正确，并解决问题",
}

MATH_SOLVER_PROMPT = """
You are MathSolverAgent，also a math expert. You will be given a math problem and hints from other agents.
Give your own solving process step by step based on hints. 
If you think it's simple enough for you to solve, you use the chain of thought technique to think step by step to calculate the answer, 
and then the last line of your output contains only the final result without any units, for example: The answer is 140\n.
The result you provide should be as integer as possible.
If your task is somewhat complex, you can use the chain of thought technique to write Python programs. 
You don't need to complete the program writing, you just need to provide the algorithm design, and then let programming_expert_node complete the program writing.
At the end of the output, provide the next desired node in the following format:
/* next_node: ProgrammingExpertAgent */
"""

MATHEMATICAL_ANALYST_PROMPT = """
You are MathematicalAnalystAgent, also a mathematical analyst. 
You will be given a math problem, analysis and code from other agents. 
You need to first analyze the problem-solving process step by step, where the variables are represented by letters. 
Then you substitute the values into the analysis process to perform calculations and get the results.
The last line of your output contains only the final result without any units, for example: The answer is 140\n.
The result you provide should be as integer as possible.
At the end of the output, provide the next desired node in the following format:
/* next_node: InspectorAgent */
"""

PROGRAMMING_EXPERT_PROMPT = """
You are ProgrammingExpertAgent, also a programming expert.
You will be given a math problem, analysis and code from other agents. 
Integrate step-by-step reasoning and Python code to solve math problems.
Analyze the question and write functions to solve the problem. 
The function should not take any arguments and use the final result as the return value.
The last line of code calls the function you wrote and assigns the return value to the \(answer\) variable.
Use a Python code block to write your response. For example:\n```python\ndef fun():\n x = 10\n y = 20\n return x + y\nanswer = fun()\n```\n
Do not include anything other than Python code blocks in your response.
At the end of the output, provide the next desired node in the following format:
/* next_node: MathematicalAnalystAgent */
"""

INSPECTOR_PROMPT = """
You are InspectorAgent, also an Inspector. 
You will be given a math problem, analysis and code from other agents.
You need to choose between the following two situations based on the situation:
- Check whether the logic/calculation of the problem solving and analysis process is correct(if present).
If the code have fatal problem, give your own solving process step by step based on hints.
The last line of your output contains only the final result without any units, for example: The answer is 140\n.
The result you provide should be as integer as possible.
- Check whether the code corresponds to the solution analysis(if present).
The last line of code calls the function you wrote and assigns the return value to the \(answer\) variable.
Use a Python code block to write your response. For example:\n```python\ndef fun():\n x = 10\n y = 20\n return x + y\nanswer = fun()\n```\n.
If the answer is correct, you do not need to provide a Python implementation.
At the end of the output, provide the next desired node in the following format:
/* next_node: END */
"""

"""
You are inspector_node, also an inspector.
You will be given a math problem, analysis and code from other agents.
Check whether the logic/calculation of the problem solving and analysis process is correct(if present).
Check whether the code corresponds to the solution analysis(if present).
If the code have fatal problem, give your own solving process step by step based on hints and return to programming_expert_node.
Please try to return to programming_expert_node as little as possible, and only return when there are serious logical errors.
If there are no issues, then return END.
At the end of the output, provide the next desired node in the following format:
/* next_node: END */
"""

FEW_SHOT_DATA = {
    "Math Solver":
        """
        Q: Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. 
        They have 2 chapters of their textbook to study and 4 worksheets to memorize. 
        They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. 
        If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, 
        include 3 10-minute snack breaks each day, and 30 minutes for lunch each day? (Hint: The answer is near to 4).
        
        A: We know the Answer Hints: 4. With the Answer Hints: 4, we will answer the question. 
        Let's think step by step. 
        Angelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, 3 hours x 2 chapters = 6 hours total.
        For the worksheets they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 worksheets = 6 hours total.
        Angelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.
        However, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, 
        so 12 total hours x 10 minutes = 120 extra minutes for breaks.
        They also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.
        And they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.
        So Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.
        They want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75
        They will need to plan to study 4 days to allow for all the time they need.
        The answer is 4
        
        Q: Bella has two times as many marbles as frisbees. She also has 20 more frisbees than deck cards. If she buys 2/5 times more of each item, what would be the total number of the items she will have if she currently has 60 marbles? (Hint: The answer is near to 160,145).
        A: We know the Answer Hints: 160, 145. With the Answer Hints: 160, 145, we will answer the question.
        Let's think step by step
        When Bella buys 2/5 times more marbles, she'll have increased the number of marbles by 2/5*60 = 24
        The total number of marbles she'll have is 60+24 = 84
        If Bella currently has 60 marbles, and she has two times as many marbles as frisbees, she has 60/2 = 30 frisbees.
        If Bella buys 2/5 times more frisbees, she'll have 2/5*30 = 12 more frisbees.
        The total number of frisbees she'll have will increase to 30+12 = 42
        Bella also has 20 more frisbees than deck cards, meaning she has 30-20 = 10 deck cards
        If she buys 2/5 times more deck cards, she'll have 2/5*10 = 4 more deck cards.
        The total number of deck cards she'll have is 10+4 = 14
        Together, Bella will have a total of 14+42+84 = 140 items
        The answer is 140
        
        Q: Susy goes to a large school with 800 students, while Sarah goes to a smaller school with only 300 students.  At the start of the school year, Susy had 100 social media followers.  She gained 40 new followers in the first week of the school year, half that in the second week, and half of that in the third week.  Sarah only had 50 social media followers at the start of the year, but she gained 90 new followers the first week, a third of that in the second week, and a third of that in the third week.  After three weeks, how many social media followers did the girl with the most total followers have? (Hint: The answer is near to 180, 160).
        A: We know the Answer Hints: 180, 160. With the Answer Hints: 180, 160, we will answer the question.
        Let's think step by step
        After one week, Susy has 100+40 = 140 followers.
        In the second week, Susy gains 40/2 = 20 new followers.
        In the third week, Susy gains 20/2 = 10 new followers.
        In total, Susy finishes the three weeks with 140+20+10 = 170 total followers.
        After one week, Sarah has 50+90 = 140 followers.
        After the second week, Sarah gains 90/3 = 30 followers.
        After the third week, Sarah gains 30/3 = 10 followers.
        So, Sarah finishes the three weeks with 140+30+10 = 180 total followers.
        Thus, Sarah is the girl with the most total followers with a total of 180.
        The answer is 180
        """,

    "Mathematical Analyst":
        """
        Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
        A: ## Problem solving process analysis
        
        There are {{ori_tree_num}} trees originally.
        Then there were {{after_planted_tree_num}} trees after some more were planted.
        So the number of trees planted today {{today_planted_num}} is the number of trees after planting {{after_planted_tree_num}} minus the number of trees before planting {{ori_tree_num}}.
        The answer is {{today_planted_num}} = {{after_planted_tree_num}} - {{ori_tree_num}}.
        
        ## Actual analysis and solution process
        
        In this question, {{ori_tree_num}} = 15 and {{after_planted_tree_num}} = 21.
        There are 15 trees originally. 
        Then there were 21 trees after some more were planted. 
        So the number of trees planted today must have been 21 - 15 = 6.
        The answer is 6
        
        Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
        A:## Problem solving process analysis
        
        Originally, Leah had {{Leah_num}} Leah_num chocolates.
        Her sister had {{sister_num}} chocolates.
        So in total they had {{all_num}} = {{Leah_num}} + {{sister_num}} chocolates.
        After eating {{eating_num}} chocolates, the number of chocolates they have left {{remain_num}} is {{all_num}} minus {{eating_num}}. 
        The answer is {{remain_num}} = {{all_num}} - {{eating_num}}.
        
        ## Actual analysis and solution process
        
        In this question, {{Leah_num}} = 32, {{sister_num}} = 42 and {{all_num}} = 35.
        So, in total they had 32 + 42 = 74 chocolates originally.
        After eating 35 chocolates, they had 74 - 35 = 39 chocolates.
        The answer is 39
        """,

    "Programming Expert":
        """
        Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
        A:
        ```python\n
        def money_left():
            money_initial = 23
            bagels = 5
            bagel_cost = 3
            money_spent = bagels * bagel_cost
            remaining_money = money_initial - money_spent
            return remaining_money
        
        answer = money_left()
        \n```
        
        Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
        A:
        ```python\n
        def remaining_golf_balls():
            golf_balls_initial = 58
            golf_balls_lost_tuesday = 23
            golf_balls_lost_wednesday = 2
            golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
            remaining_golf_balls = golf_balls_left
            return remaining_golf_balls
        
        answer = remaining_golf_balls() 
        \n```
        """,
    "Inspector": """""",
}


