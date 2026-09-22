from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langfuse import Langfuse, get_client, observe
from langfuse.langchain import CallbackHandler
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

from app.config import get_settings


class AgentState(TypedDict):
    """state for the production agent.
    uses annotated with add_messages reducer for message
    accumulation"""

    messages: Annotated[list[BaseMessage], add_messages]
    error: str | None
    retry_count: int
    model_used: str


class ProductionAgent:
    """Production LangGraph agent with :
    - Retry on failure (model fallback)
    - Graceful error handling
    - Langfuse tracing
    """

    def __init__(self):
        settings = get_settings()

        self.primary_llm = ChatGroq(
            model=settings.PRIMARY_MODEL,
            api_key=settings.GROQ_API_KEY,
            temperature=0,
            reasoning_effort="medium",
            timeout=30,
            max_retries=0,  # we handle the retries ourselves
        )
        self.fallback_llm = ChatGroq(
            model=settings.FALLBACK_MODEL,
            api_key=settings.GROQ_API_KEY,
            temperature=0,
            reasoning_effort="medium",
            timeout=30,
            max_retries=0,  # we handle the retries ourselves
        )
        self.max_retries = settings.MAX_RETRIES
        self.graph = self._build_graph()
        self.langfuse = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_BASE_URL,
        )
        self._langfuse_handler = CallbackHandler()

    def _build_graph(self):
        """Build the LangGraph state machine"""

        def process_message(state: AgentState) -> dict:
            """try to process the message with the primary model"""
            try:
                response = self.primary_llm.invoke(state["messages"])
                return {"messages": [response], "error": None, "model_used": "primary"}
            except Exception as e:
                return {
                    "error": str(e),
                    "retry_count": state["retry_count"] + 1,
                    "model_used": "",
                }

        def try_fallback(state: AgentState) -> dict:
            """fallback to a secondary model"""
            try:
                response = self.fallback_llm.invoke(state["messages"])
                return {"messages": [response], "error": None, "model_used": "fallback"}
            except Exception as e:
                return {"error": str(e), "model_used": ""}

        def handle_error(state: AgentState) -> dict:
            """Gracefully handel errors"""
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "Ï'm sorry. I'm having trouble processing ypur request"
                            "right now. Please try again in a moment."
                        )
                    )
                ],
                "model_used": "error_handler",
            }

        def route_after_process(state: AgentState) -> str:
            """decide what to do after the primary model attempt"""
            if state.get("error") is None:
                return "done"
            elif state["retry_count"] < self.max_retries:
                return "fallback"
            else:
                return "error"

        def route_after_fallback(state: AgentState) -> str:
            """decide what to do after a fallback attempt"""

            if state.get("error") is None:
                return "done"
            else:
                return "error"

        # Build the graph
        graph = StateGraph(AgentState)

        graph.add_node("process", process_message)
        graph.add_node("fallback", try_fallback)
        graph.add_node("error", handle_error)

        graph.add_edge(START, "process")
        graph.add_conditional_edges(
            "process",
            route_after_process,
            {"done": END, "fallback": "fallback", "error": "error"},
        )
        graph.add_conditional_edges(
            "fallback", route_after_fallback, {"done": END, "error": "error"}
        )
        graph.add_edge("error", END)

        return graph.compile()

    @observe(name="production_agent_invoke", capture_input=False, capture_output=False)
    def invoke(self, message: str) -> dict:
        """Invoke the agent with the user message"""

        try:
            langfuse_client = get_client()
            langfuse_client.update_current_span(input=message)

            result = self.graph.invoke(
                {
                    "messages": [HumanMessage(content=message)],
                    "error": None,
                    "retry_count": 0,
                    "model_used": "",
                },
                config={"callbacks": [self._langfuse_handler]},
            )

            langfuse_client.update_current_span(output=result["messages"][-1].content)

            return {
                "response": result["messages"][-1].content,
                "model_used": result.get("model_used", "unknown"),
                "error": result.get("error"),
            }

        except Exception as e:
            raise e


# if __name__ == "__main__":
#     agent = ProductionAgent()
#     print(agent.invoke("How to make money as a software engineer?"))
