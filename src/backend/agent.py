from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import GoogleSerperAPIWrapper
import os
from langchain.tools import Tool
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from src.backend.parser import StrictOutputParser
from src.logger_config import log

class web_agent:
    """
    Web-enabled conversational agent using a ReAct-style LangChain agent
    + Serper web search + custom LLM wrapper (Euri or any other LangChain LLM).
    """
    def __init__(self,llm):
        load_dotenv()
        self.SERPER_API_KEY = os.getenv("SERPER_API_KEY")
        self.search = GoogleSerperAPIWrapper(serper_api_key = self.SERPER_API_KEY)
        self.llm = llm
        self.system_prompt = """
            You are an intelligent assistant using ReAct reasoning with tools Follow below rules stricltly while answering. 

            RULES:
            1. You must ALWAYS follow this exact structure:

            Thought: <your reasoning here>
            Action: <tool name>
            Action Input: <the input to the tool>

            OR, if you already know the answer:

            Thought: <your reasoning here>
            Final Answer: <the final response>

            Rules:
            2. Use InternetSearch when the user asks for real-time or factual information.

            3. You may use the InternetSearch tool up to TWO TIMES per user question:
                - First search: perform a broad search.
                - Second search: ONLY IF needed, use a refined and more specific query.

            4. NEVER call the tool more than twice under any circumstance.

            5. Before using InternetSearch a second time, THINK:
                "What exact information was missing from the first search?"
                Then create a better, more precise search query.

            6. After at most two searches, you MUST produce the Final Answer.

            7. Only produce Final Answer when you have enough information.  
               If the first search already gives enough information, do NOT call the tool again.

            8. Never repeat the same search query twice.

            9. If you already know the answer without searching, do NOT use any tool.

            10. If you choose to use a tool, you MUST output an Action block following the format:
                Thought:
                Action: <tool_name>
                Action Input: <your input>

            11. Keep one Thought per step. Never combine multiple thoughts in one line.

            12. Thoughts must be short, clear, logical, and explain your reasoning.

            13. Do NOT fabricate Observation lines. Observations will be auto-generated.

            14. Final answer shouldnt contain repetative points

            15. Final Answer must be a clear, concise summary that combines everything learned. 

            16. If no useful information is found, politely say that no reliable information is available.


            The full conversation so far is available in {chat_history}.
            Question: {input}

            """ 
        self.SUFFIX = """
                Begin!
                Question: {input}
                Thought:{agent_scratchpad} 
                Remember: Only produce Final Answer when ready, following the exact format."""
        
    def create_new_memory(self):
        """Creates and returns a new ConversationBufferMemory instance.
        This is used when a new user starts chatting, to ensure per-user isolation.
        """
        return ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def serper_tool(self,query: str) -> str:
        """Your task is to search the information from web for qiven query and share the observation to agent"""
        raw_result = self.search.run(query)
        summary_prompt = (
        "Summarize the following web search result in a concise and factual way. "
        "Provide bullet points relevant to the original query:\n\n"
        f"User Query: {query}\n\n"
        f"Web Result:\n{raw_result}"
      )
        summary = self.llm.invoke(summary_prompt)
         # Extract text from response
        if isinstance(summary, dict):
            return summary.get("output", "")

        if hasattr(summary, "content"):
            return summary.content

        return str(summary)
      
    
    def initilze_tool(self):
        tools = [Tool(name="InternetSearch", 
                     func = self.serper_tool,
                        description=(
                        "Searches the internet for real-time factual information. "
                        "Use this tool ONLY when necessary. "
                        "The tool may be used up to TWO TIMES per question: "
                        " - First time: broad search\n"
                        " - Second time: refined search ONLY if important info was missing\n"
                        "NEVER repeat the same query, and NEVER use the tool more than twice."
                    ))]
        return tools
    

    def initializing_agent(self):
        try:
            tools = self.initilze_tool()
            memory = self.create_new_memory()
            agent = initialize_agent(
                    tools = tools,
                    llm = self.llm,
                    agent = AgentType.CONVERSATIONAL_REACT_DESCRIPTION, # A type of agent that follows the ReAct (Reason + Act) pattern. It looks at the tool descriptions and decides, without prior training, which one to use.
                    memory = memory,
                    verbose = True,
                    handle_parsing_errors = True, 
                    agent_executor_kwargs={
                        "prefix": self.system_prompt,  # Regular string
                        "suffix": self.SUFFIX,         # Regular string
                        "max_iterations": 5,
                        "early_stopping_method": "generate",
                        "output_parser": StrictOutputParser()})
        
            return agent, memory
        except Exception:
            log.exception("Agent Initilization failed")


    async def query_answering_async(self,agent,query,retrived_results,memory):
        """
            Executes a query against the given agent, using user-specific memory.
            Automatically passes chat history and updates memory with new turns.
        """
        try:

            combined_input = (
                    f"Question: {query}\n\n"
                    f"Retrieved Info:\n{retrived_results}"
                )
            inputs = {
                    "input": combined_input,
                }

            final_response_raw = await agent.ainvoke(inputs)

            final_response = (final_response_raw["output"] if isinstance(final_response_raw, dict) else str(final_response_raw))

            return final_response, memory

        except Exception:
            log.exception("iteration limit reached, summarize last observation")

            history = memory.load_memory_variables({}).get("chat_history", [])
            last_obs = history[-1].content if history else "No observation found."

            summary = await self.llm.ainvoke(
                f"Summarize this observation into a concise Final Answer:\n{last_obs}"
            )
            summary_text = summary.get("output", summary.content) if isinstance(summary, dict) else summary.content

            # Optional: store summary back to memory
            
            return summary_text, memory
    


  
