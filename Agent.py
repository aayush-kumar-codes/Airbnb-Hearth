import re
from langchain.tools import Tool
from pydantic.v1 import BaseModel
from typing import List, Dict, Tuple

FINAL_ANSWER_TOKEN = "Final Answer:"
OBSERVATION_TOKEN = "Observation:"
THOUGHT_TOKEN = "Thought:"
PROMPT_TEMPLATE = """
You are an Airbnb Agent who recommend the property based on people preferences and you should use tools:

<tools> {tool_description} </tools>

<instruction> 
1. Your main job is to only select a tool from <tools> tag and don't perform any other analysis. tools will be used further to perform detailed analysis.
2. To call the tool, check <tool_input> for the required parameters.
3. If you don't have enough information to call a tool, ask human.
4. Current User Question is most important
</instruction>

<previous_responses>
{previous_responses}
</previous_responses>

Current User Question: {input}

Think step by step

Use the following format:

Thought: your step by step reasoning based question and <previous_responses> on which tool to select and generate the required input for the tool. check required input for a tool
Action: the action to take always use the exact tools name donot append anything to it, exactly one element of [{tool_names}].
Action Input: the input to the action.

Always reply in above format. Do not repeat above multiple times also provide your final response only.

Thought:
"""
# Observation: the result of the action.
# ... (this Thought/Action/Action Input/Observation repeats N times, use it until you are sure of the answer).


# Thought: I now know the final answer.
# Final Answer: your final answer should always stick to actual input question.


class AirbnbAgent(BaseModel):

    llm: object
    tools: list[Tool]
    prompt_template: str = PROMPT_TEMPLATE
    max_loops: int = 0

    # The stop pattern is used, so the LLM does not hallucinate until the end
    stop_pattern: List[str] = [f'\n{OBSERVATION_TOKEN}', f'\n\t{OBSERVATION_TOKEN}']

    @property
    def tool_description(self) -> str:
        return "\n".join([f"<tool_name>{tool.name}</tool_name>\n<tool_description>{tool.description}</tool_description>" for tool in self.tools])

    @property
    def tool_names(self) -> str:
        return ",".join([tool.name for tool in self.tools])

    @property
    def tool_by_names(self) -> Dict[str, Tool]:
        return {tool.name: tool for tool in self.tools}

    def run(self, question, inputs):
        num_loops = 0
        prompt = self.prompt_template.format(
            tool_description=self.tool_description,
            tool_names=self.tool_names,
            input=question,
            previous_responses='{previous_responses}'
        )

        previous_responses = []
        if (len(inputs) > 0):
            previous_responses = inputs[:]

        self.max_loops = 1
        print("previous_responses", previous_responses)
        print("userinput", question)

        while num_loops < self.max_loops:
            num_loops += 1
            print("********************num loops", num_loops)
            curr_prompt = prompt.format(previous_responses='\n'.join(previous_responses))
            print("curr_prompt", curr_prompt)

            try:
                generated, tool, tool_input = self.decide_next_action(curr_prompt)
                print("-------", generated)
                print("********************tool 77", tool)
                print("****************tool input 77", tool_input)
            except Exception as e:
                print(f"ERROR OCCURED while deciding next action: {e}")
                print("****80")

            print("*************************** got tool", tool)
            if "Final Answer" in generated:
                return generated.split("Final Answer:")[-1].strip()

            if tool == 'InteractHuman':
                if "Action Input:" in generated:
                    return generated.split("Action Input:")[-1].strip()
                return generated

            tool_result = ""
            if tool_input != 'None':
                try:
                    tool_result = self.tool_by_names[tool].run(tool_input)
                    print("tool result", tool_result)
                    print("loop ", num_loops, "************tool result", tool_result)
                except Exception as e:
                    print(f"ERROR OCCURED while calling tool: {e}")

                # return "No data found"
                # break
                generated += f"\n Got Response From Tool {tool} {tool_result}\n"
                previous_responses.append(generated)
                print("***************generated", generated)
                return tool_result

    def decide_next_action(self, prompt: str) -> str:

        generated = self.llm.invoke(prompt, stop=self.stop_pattern).content
        print("llm output ==================================")
        print(generated)
        try:
            tool, tool_input = self._parse(generated)
            return generated, tool, tool_input
        except Exception as e:
            print(e)
            return generated, None, None

    def _parse(self, generated: str) -> Tuple[str, str]:
        if FINAL_ANSWER_TOKEN in generated:
            return "Final Answer", generated.split(FINAL_ANSWER_TOKEN)[-1].strip()
        regex = r"Action: [\[]?(.*?)[\]]?[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, generated, re.DOTALL)
        # print("**************match",match)
        tool = match.group(1).strip()
        tool_input = match.group(2)
        if tool:
            tool = tool.strip()
        if tool_input:
            tool_input = tool_input.strip()
        return tool, tool_input.strip(" ").strip('"')
