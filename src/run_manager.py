from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager

class MyCallbackHandler(BaseCallbackHandler):
    def on_retriever_start(self, serialized, inputs, **kwargs):
        print(f"Retriever started with input: {inputs}")

    def on_retriever_end(self, output, **kwargs):
        print(f"Retriever finished, found {len(output)} docs")

# Create a callback manager

callback_manager = CallbackManager([MyCallbackHandler()])


'''

You’d pass a real run_manager if you want to:

Track events happening inside your retriever (e.g., start/end logging, tracing).

Integrate with LangSmith (LangChain’s tracing dashboard).

Stream progress or print debug info at each retrieval step.

Build your own callback system (for logging, metrics, or telemetry)
'''