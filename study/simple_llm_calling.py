import json

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()


def simple_rag():
    # Initialize Groq LLM
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.7,
        top_p=1,
        reasoning_effort="medium",
    )

    # Define the expected JSON structure
    parser = JsonOutputParser(
        pydantic_object={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "price": {"type": "number"},
                "features": {"type": "array", "items": {"type": "string"}},
            },
        }
    )

    # Create a simple prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Extract product details into JSON with this structure:
            {{
                "name": "product name here",
                "price": number_here_without_currency_symbol,
                "features": ["feature1", "feature2", "feature3"]
            }}""",
            ),
            ("user", "{input}"),
        ]
    )

    # Create the chain that guarantees JSON output
    chain = prompt | llm | parser

    def parse_product(description: str) -> dict:
        result = chain.invoke({"input": description})
        print(json.dumps(result, indent=2))

    # Example usage
    description = """The Kees Van Der Westen Speedster is a high-end, single-group espresso machine known for its precision, performance, 
    and industrial design. Handcrafted in the Netherlands, it features dual boilers for brewing and steaming, PID temperature control for 
    consistency, and a unique pre-infusion system to enhance flavor extraction. Designed for enthusiasts and professionals, it offers 
    customizable aesthetics, exceptional thermal stability, and intuitive operation via a lever system. The pricing is approximatelyt $18,499 
    depending on the retailer and customization options."""

    parse_product(description)


if __name__ == "__main__":
    simple_rag()
