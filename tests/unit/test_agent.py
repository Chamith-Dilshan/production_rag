from app.agent import ProductionAgent


class FakeLLM:
    def __init__(self, response: str):
        self.response = response

    def invoke(self, _messages):
        return type("Message", (), {"content": self.response})()


def test_production_agent_init_and_build_graph(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.agent.get_settings",
        lambda: type(
            "Settings",
            (),
            {
                "PRIMARY_MODEL": "primary-model",
                "FALLBACK_MODEL": "fallback-model",
                "GROQ_API_KEY": "key",
                "MAX_RETRIES": 2,
                "LANGFUSE_PUBLIC_KEY": "pub",
                "LANGFUSE_SECRET_KEY": "secret",
                "LANGFUSE_BASE_URL": "http://localhost:3000",
            },
        )(),
    )
    monkeypatch.setattr("app.agent.ChatGroq", lambda **_kwargs: object())
    monkeypatch.setattr("app.agent.Langfuse", lambda **_kwargs: object())
    monkeypatch.setattr("app.agent.CallbackHandler", lambda: object())

    agent = ProductionAgent()

    assert agent.primary_llm is not None
    assert agent.fallback_llm is not None
    assert agent.max_retries == 2
    assert agent.graph is not None


def test_production_agent_invoke_returns_response() -> None:
    agent = ProductionAgent.__new__(ProductionAgent)
    agent.primary_llm = FakeLLM("primary reply")
    agent.fallback_llm = FakeLLM("fallback reply")
    agent.max_retries = 3
    agent._langfuse_handler = object()
    agent.graph = type(
        "Graph",
        (),
        {"invoke": lambda self, state, config=None: {"messages": [type("M", (), {"content": "primary reply"})()], "error": None, "model_used": "primary"}},
    )()

    result = agent.invoke("hello")

    assert result["response"] == "primary reply"
    assert result["model_used"] == "primary"


def test_production_agent_invoke_handles_exception() -> None:
    agent = ProductionAgent.__new__(ProductionAgent)
    agent.primary_llm = FakeLLM("primary reply")
    agent.fallback_llm = FakeLLM("fallback reply")
    agent.max_retries = 3
    agent._langfuse_handler = object()

    def fail_invoke(*args, **kwargs):
        raise RuntimeError("boom")

    agent.graph = type("Graph", (), {"invoke": fail_invoke})()

    try:
        agent.invoke("hello")
        raise AssertionError("Expected RuntimeError")
    except RuntimeError as exc:
        assert str(exc) == "boom"
