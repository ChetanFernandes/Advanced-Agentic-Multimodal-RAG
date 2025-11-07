from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import GoogleSerperAPIWrapper
import os
from langchain.tools import Tool
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from agentic_rag.backend.parser import StrictOutputParser
from agentic_rag.logger_config import log

class web_agent:
    def __init__(self,llm):
        load_dotenv()
        self.SERPER_API_KEY = os.getenv("SERPER_API_KEY")
        self.search = GoogleSerperAPIWrapper(serper_api_key = self.SERPER_API_KEY)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.llm = llm
        self.system_prompt = """
            You are an intelligent assistant that must follow a strict reasoning format.

            RULES:
            1. You must ALWAYS follow this exact structure:

            Thought: <your reasoning here>
            Action: <tool name>
            Action Input: <the input to the tool>

            OR, if you already know the answer:

            Thought: <your reasoning here>
            Final Answer: <the final response>

            Rules:
            2. Always use InternetSearch for real-time queries before giving Final Answer.
            3. Only produce Final Answer when you have all necessary info. 
            4. If you already know the answer, DO NOT output an Action.
            5. NEVER skip Action if using a tool.
            6. Keep one Thought per step.
            7. Observation lines are auto-handled — do not invent them.
            8. Summarize retrieved info concisely in Final Answer.
            9. If no info is found, say so politely in Final Answer.

            Question: {input}
            """
        self.SUFFIX = """
                Begin!

                Question: {input}
                Thought:{agent_scratchpad} 
                Remember: Only produce Final Answer when ready, following the exact format."""

    def serper_tool(self,query: str) -> str:
        """Your task is to search the information from web for qiven query and share the observation to agent"""
        return self.search.run(query)
    
    def initilze_tool(self):
        tools = [Tool(name="InternetSearch", func = self.serper_tool,
            description="Searches the internet for real-time information. Input should be a query string.")]
        return tools
    

    def initializing_agent(self):
        tools = self.initilze_tool()
        agent = initialize_agent(
                tools = tools,
                llm = self.llm,
                agent = AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, # A type of agent that follows the ReAct (Reason + Act) pattern. It looks at the tool descriptions and decides, without prior training, which one to use.
                memory = self.memory,
                verbose = True,
                handle_parsing_errors = True, 
                agent_executor_kwargs={
                    "prefix": self.system_prompt,  # Regular string
                    "suffix": self.SUFFIX,         # Regular string
                    "max_iterations": 5,
                    "output_parser": StrictOutputParser()
    }
    )
        return agent
    
    async def query_answering_async(self,agent,query,retrived_results):
        try:
            final_response = await agent.ainvoke({"input": f"User Question: {query} \n Retrieved Info: {retrived_results}"})
        except Exception:
            log.info("iteration limit reached, summarize last observation")
            history = self.memory.load_memory_variables({}).get("chat_history", [])
            last_obs = history[-1].content if history else "No observation found."
            summary = await self.llm.ainvoke(
                f"Summarize this observation into a concise Final Answer:\n{last_obs}"
            )
            final_response = {"output": summary}
        return final_response
    



