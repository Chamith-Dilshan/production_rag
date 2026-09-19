"""Token-budgeted LLM wrapper around ChatGroq, instrumented with Langfuse."""

import logging
import os
from typing import TypedDict

import tiktoken

# Must run before any library builds an SSL context (httpx, Langfuse's OTLP
# exporter, etc.). Makes Python verify TLS via the OS certificate store
# (Windows CryptoAPI) instead of parsing certifi's bundled cafile - works
# around a known CPython bug where ssl.create_default_context(cafile=...)
# does ~143,000 disk operations on Windows and can look like a hang/freeze.
# See: https://github.com/python/cpython/pull/137596
import truststore
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langfuse import get_client, observe
from langfuse.langchain import CallbackHandler

truststore.inject_into_ssl()


logger = logging.getLogger(__name__)


class TokenBudgetError(Exception):
    """Raised when a request would exceed the configured token budget."""


class RequestRecord(TypedDict):
    """One entry in the rolling request-history log."""

    input_tokens: int
    output_tokens: int
    total: int
    request_length: int
    response_length: int


class UsageCounters(TypedDict):
    """Raw, accumulated usage counters."""

    total_input: int
    total_output: int
    total_requests: int
    rejected_requests: int
    request_history: list[RequestRecord]


class UsageStats(UsageCounters):
    """Usage counters plus derived stats, as returned by `get_stats()`."""

    total_tokens: int
    average_request_tokens: float


class TokenBudget:
    """Tracks and enforces a per-request token budget using tiktoken."""

    def __init__(
        self,
        max_tokens_per_request: int = 4000,
        tokenizer_model: str = "gpt-3.5-turbo",
    ) -> None:
        self.max_tokens = max_tokens_per_request
        self.tokenizer_model = tokenizer_model

        # tiktoken only ships encodings for OpenAI model names (this includes
        # the open-weight gpt-oss models, under the bare name "gpt-oss-20b" -
        # NOT under a provider-prefixed name like "openai/gpt-oss-20b"). For
        # any model tiktoken doesn't recognise, fall back to cl100k_base as
        # an approximation.
        try:
            self.encoding = tiktoken.encoding_for_model(tokenizer_model.split("/")[-1])
        except KeyError:
            logger.warning(
                "No tiktoken encoding registered for model %r; falling back to "
                "cl100k_base. Token counts will be an approximation.",
                tokenizer_model,
            )
            self.encoding = tiktoken.get_encoding("cl100k_base")

        self.usage: UsageCounters = {
            "total_input": 0,
            "total_output": 0,
            "total_requests": 0,
            "rejected_requests": 0,
            "request_history": [],
        }

    def estimate_tokens(self, text: str | None) -> int:
        """Estimate token count for `text` using tiktoken."""
        if not text:
            return 0

        try:
            return len(self.encoding.encode(text))
        except Exception:
            # Fallback to rough estimation if tiktoken fails for some input
            logger.warning(
                "tiktoken encoding failed for input of length %d; using word-count "
                "fallback estimation.",
                len(text),
                exc_info=True,
            )
            return int(len(text.split()) * 1.3)

    def check_budget(self, text: str) -> tuple[bool, int]:
        """Check whether `text` fits within the configured token budget."""
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")

        tokens = self.estimate_tokens(text)
        return tokens <= self.max_tokens, tokens

    def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        request_text: str = "",
        response_text: str = "",
    ) -> None:
        """Record token usage for a request that actually reached the LLM.

        Only call this after a real response comes back. Rejected requests
        go through record_rejection() instead and never touch these totals.
        """
        self.usage["total_input"] += input_tokens
        self.usage["total_output"] += output_tokens
        self.usage["total_requests"] += 1

        self.usage["request_history"].append(
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total": input_tokens + output_tokens,
                "request_length": len(request_text),
                "response_length": len(response_text),
            }
        )

        if len(self.usage["request_history"]) > 100:
            self.usage["request_history"] = self.usage["request_history"][-100:]

    def record_rejection(self) -> None:
        """Record a request that was blocked before the LLM was ever called.

        Deliberately does NOT touch total_input/total_output/request_history -
        those represent tokens that were actually sent/received.
        """
        self.usage["rejected_requests"] += 1

    def get_stats(self) -> UsageStats:
        """Return current usage counters plus derived totals/averages."""
        total_tokens = self.usage["total_input"] + self.usage["total_output"]
        total_requests = self.usage["total_requests"]
        return {
            "total_input": self.usage["total_input"],
            "total_output": self.usage["total_output"],
            "total_requests": total_requests,
            "rejected_requests": self.usage["rejected_requests"],
            "request_history": self.usage["request_history"],
            "total_tokens": total_tokens,
            "average_request_tokens": (
                total_tokens / total_requests if total_requests > 0 else 0.0
            ),
        }

    def reset_stats(self) -> None:
        """Reset all usage statistics."""
        self.usage = {
            "total_input": 0,
            "total_output": 0,
            "total_requests": 0,
            "rejected_requests": 0,
            "request_history": [],
        }


class TokenBudgetedLLM:
    """LLM wrapper enforcing a token budget, traced end-to-end with Langfuse.

    The budget check acts as a debouncer: it runs BEFORE the API call and,
    if the request is over budget, the LLM is never invoked - regardless of
    strict_mode. strict_mode only controls how the caller is notified:

    - strict_mode=True  (default): raises TokenBudgetError.
    - strict_mode=False: logs a warning and returns None.

    Either way, a rejected request only increments rejected_requests; it
    never adds to total_input/total_output.
    """

    def __init__(
        self,
        llm,
        max_tokens: int = 4000,
        tokenizer_model: str = "gpt-3.5-turbo",
        strict_mode: bool = True,
    ) -> None:
        self.llm = llm
        self.budget = TokenBudget(
            max_tokens_per_request=max_tokens, tokenizer_model=tokenizer_model
        )
        self.strict_mode = strict_mode
        # One handler per wrapper instance; LangChain callbacks are cheap to
        # construct, and this keeps the wrapper self-contained.
        self._langfuse_handler = CallbackHandler()

    @observe(
        name="token_budgeted_llm.invoke",
        capture_input=False,
        capture_output=False,
    )
    def invoke(self, query: str) -> str | None:
        """Execute a query with a pre-call token budget check.

        Returns the LLM's response text, or None if the request was blocked
        by the budget check in non-strict mode (see class docstring).
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")

        langfuse = get_client()
        is_within_budget, input_tokens = self.budget.check_budget(query)
        langfuse.update_current_span(
            input=query,
            metadata={
                "estimated_input_tokens": input_tokens,
                "max_tokens": self.budget.max_tokens,
                "within_budget": is_within_budget,
                "tokenizer_model": self.budget.tokenizer_model,
            },
        )

        if not is_within_budget:
            # Debounce: stop here, before the API call happens. Nothing is
            # added to total_input/total_output - only the rejection counter.
            self.budget.record_rejection()
            error_msg = (
                f"Request exceeds token budget: {input_tokens} tokens > "
                f"{self.budget.max_tokens} max allowed"
            )
            # Mark the span itself as the reason this call was stopped,
            # rather than relying on an exception to explain it in the trace.
            langfuse.update_current_span(
                output=None,
                level="WARNING",
                status_message=error_msg,
                metadata={
                    "rejected": True,
                    "rejection_reason": "token_budget_exceeded",
                },
            )
            if self.strict_mode:
                raise TokenBudgetError(error_msg)
            logger.warning(
                "%s - request blocked, API call skipped (strict_mode=False)", error_msg
            )
            return None

        # Budget check passed - safe to call the LLM. Attach the Langfuse
        # callback handler so this call is traced as a nested generation
        # (model, tokens, latency, cost).
        try:
            response = self.llm.invoke(
                query, config={"callbacks": [self._langfuse_handler]}
            )
            result = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            raise RuntimeError(f"LLM invocation failed: {str(e)}") from e

        # Prefer the provider's own reported usage (exact) over a tiktoken
        # re-estimate (approximate) when the response exposes it.
        usage_meta = getattr(response, "usage_metadata", None) or {}
        output_tokens = usage_meta.get("output_tokens")
        if output_tokens is None:
            output_tokens = self.budget.estimate_tokens(result)

        self.budget.record_usage(
            input_tokens, output_tokens, request_text=query, response_text=result
        )

        langfuse.update_current_span(
            output=result,
            metadata={"output_tokens": output_tokens},
        )

        return result

    def get_status(self) -> UsageStats:
        """Get current LLM status and statistics."""
        return self.budget.get_stats()

    def reset_stats(self) -> None:
        """Reset statistics (useful for testing)."""
        self.budget.reset_stats()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Loads variables from a .env file in the current directory (or a parent
    # of it) into the process environment. No-op if no .env file exists or
    # the variables are already set some other way.
    load_dotenv()

    if not os.environ.get("GROQ_API_KEY"):
        raise ValueError(
            "GROQ_API_KEY environment variable not set. If you're using a "
            ".env file, make sure it's in this script's working directory "
            "(or a parent of it)."
        )

    # Langfuse reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL
    # from the environment; get_client() returns a singleton.
    langfuse = get_client()
    if not langfuse.auth_check():
        logger.warning(
            "Langfuse credentials missing or invalid - traces will not be sent. "
            "Set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL."
        )

    # api_key intentionally omitted below: ChatGroq reads GROQ_API_KEY from
    # the environment itself as a proper pydantic SecretStr. Passing a plain
    # `str` here is what triggers "str is not assignable to SecretStr | None"
    # from static type checkers (Pyright/mypy) - it works at runtime via
    # Pydantic coercion, but the declared field type doesn't match.
    groq_model = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.7,
    )

    # tokenizer_model picks a tiktoken encoding for estimating token counts -
    # gpt-oss models have an exact tiktoken encoding (o200k_harmony), found
    # via the bare model name (see TokenBudget.__init__).
    llm = TokenBudgetedLLM(groq_model, max_tokens=100, tokenizer_model="gpt-oss-20b")

    queries = [
        "What is the capital of France?",  # Within budget
        "Write a detailed essay about the history of artificial intelligence and its impact on modern society.",  # Likely within budget
        "What is the capital of Germany?" * 100,  # Over budget
    ]

    for i, query in enumerate(queries, 1):
        try:
            print(f"\n--- Query {i} ---")
            print(f"Input length: {len(query)} characters")

            result = llm.invoke(query=query)
            if result is None:
                print("Result: (blocked by token budget - no API call made)")
            else:
                print(f"Result: {result[:200]}...")  # Print first 200 chars

            stats = llm.get_status()
            print(
                f"Total usage: {stats['total_input']} input, "
                f"{stats['total_output']} output tokens"
            )

        except TokenBudgetError as e:
            print(f"❌ Budget error: {e}")
        except Exception as e:
            print(f"❌ Error: {e}")

    # Final statistics
    print("\n--- Final Statistics ---")
    final_stats = llm.get_status()
    print(f"Total requests: {final_stats['total_requests']}")
    print(f"Rejected requests: {final_stats['rejected_requests']}")
    print(f"Total tokens used: {final_stats['total_tokens']}")
    print(f"Average per request: {final_stats['average_request_tokens']:.1f} tokens")

    # Short-lived scripts must flush explicitly or buffered traces are lost.
    langfuse.flush()


if __name__ == "__main__":
    main()
