from langchain.agents import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
import re

class StrictOutputParser(AgentOutputParser):
    """Parses LLM output strictly according to your system prompt format:
    Thought -> Action/Action Input OR Final Answer
    """

    def parse(self, text: str):
        # Debug: see raw output
        print(f"Parsing output:\n{text}\n{'-'*50}")

        # If both Action and Final Answer exist, prioritize Action
        if "Action:" in text and "Final Answer:" in text:
            print("WARNING: Both Action and Final Answer detected. Prioritizing Action.")
            action_match = re.search(r"Action:\s*(.*?)\nAction Input:\s*(.*?)(?:\n|$)", text, re.DOTALL)
            if action_match:
                action = action_match.group(1).strip()
                action_input = action_match.group(2).strip()
                return AgentAction(tool=action, tool_input=action_input, log=text)

        # Final Answer parsing
        if "Final Answer:" in text:
            final_answer = text.split("Final Answer:")[-1].strip()
            return AgentFinish(return_values={"output": final_answer}, log=text)

        # Action parsing (if only Action exists)
        if "Action:" in text:
            action_match = re.search(r"Action:\s*(.*?)\nAction Input:\s*(.*?)(?:\n|$)", text, re.DOTALL)
            if action_match:
                action = action_match.group(1).strip()
                action_input = action_match.group(2).strip()
                return AgentAction(tool=action, tool_input=action_input, log=text)

        # Fallback: treat whole output as final answer
        return AgentFinish(return_values={"output": text.strip()}, log=text)

    @property
    def _type(self):
        return "strict_output_parser"
