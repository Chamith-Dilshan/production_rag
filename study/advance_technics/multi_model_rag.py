"""
Lesson 6.6: Multimodal RAG with ColPali

Traditional RAG pipeline:
PDF → Extract text → Chunk → Embed text → Search

Problem: Text extraction DESTROYS visual information!
- Tables become jumbled text
- Charts lose all meaning
- Diagrams are completely lost
- Layout context disappears

ColPali solution:
PDF → Convert to images → Embed images directly → Search

The retrieved page IMAGES go to a vision-capable LLM (GPT-4V, Claude)
which can "see" tables, charts, and diagrams properly.
"""

from textwrap import dedent

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

# ============================================================
# THE PROBLEM: TEXT EXTRACTION DESTROYS INFORMATION
# ============================================================


def demonstrate_extraction_problem():
    """
    Show how text extraction fails for visual content.
    """

    print("=" * 60)
    print("THE PROBLEM: Text Extraction Destroys Information")
    print("=" * 60)

    # Simulated table in a PDF
    original_table = """
    ┌─────────────────────────────────────────────────────────┐
    │              Q1 2025 Sales by Region                    │
    ├─────────────┬─────────────┬─────────────┬───────────────┤
    │ Region      │ Q1 Target   │ Q1 Actual   │ Variance      │
    ├─────────────┼─────────────┼─────────────┼───────────────┤
    │ North       │ $2.5M       │ $2.8M       │ +12% ✓        │
    │ South       │ $1.8M       │ $1.5M       │ -17% ✗        │
    │ East        │ $3.2M       │ $3.4M       │ +6% ✓         │
    │ West        │ $2.1M       │ $2.0M       │ -5%           │
    ├─────────────┼─────────────┼─────────────┼───────────────┤
    │ TOTAL       │ $9.6M       │ $9.7M       │ +1%           │
    └─────────────┴─────────────┴─────────────┴───────────────┘
    """

    # What text extraction often produces
    extracted_text = """
    Q1 2025 Sales by Region Region Q1 Target Q1 Actual Variance
    North $2.5M $2.8M +12% South $1.8M $1.5M -17% East $3.2M $3.4M +6%
    West $2.1M $2.0M -5% TOTAL $9.6M $9.7M +1%
    """

    print("\nOriginal table in PDF:")
    print(original_table)

    print("\nAfter text extraction (PyPDF, pdfminer, etc.):")
    print("-" * 60)
    print(extracted_text)

    print("\n" + "-" * 60)
    print("WHAT'S LOST:")
    print("-" * 60)
    print(
        """
    1. TABLE STRUCTURE - Rows and columns are jumbled
    2. VISUAL INDICATORS - The ✓ and ✗ symbols may be lost
    3. ALIGNMENT - Can't tell which number belongs to which region
    4. CONTEXT - "South is underperforming" is visually obvious but not in text

    Now imagine this with:
    - Complex financial reports with nested tables
    - Charts showing trends
    - Diagrams with arrows and connections
    - Flowcharts and organizational charts

    TEXT EXTRACTION IS A LOSSY OPERATION!
    """
    )


# ============================================================
# THE SOLUTION: VISION-BASED DOCUMENT UNDERSTANDING
# ============================================================


def explain_colpali_approach():
    """
    Explain the ColPali/vision-based approach to document RAG.
    """

    print("\n" + "=" * 60)
    print("THE SOLUTION: Vision-Based Document RAG")
    print("=" * 60)

    approach = """
    TRADITIONAL RAG PIPELINE:
    ┌─────────┐    ┌──────────────┐    ┌─────────────┐    ┌───────────┐
    │   PDF   │───▶│ Extract Text │───▶│ Chunk Text  │───▶│ Embed     │
    │         │    │ (PyPDF)      │    │             │    │ (OpenAI)  │
    └─────────┘    └──────────────┘    └─────────────┘    └───────────┘
                          │
                    ❌ INFORMATION LOST HERE


    COLPALI / VISION RAG PIPELINE:
    ┌─────────┐    ┌──────────────┐    ┌─────────────┐    ┌───────────┐
    │   PDF   │───▶│ Convert to   │───▶│ Embed Images│───▶│ Store in  │
    │         │    │ Images       │    │ (ColPali)   │    │ Vector DB │
    └─────────┘    └──────────────┘    └─────────────┘    └───────────┘
                          │
                    ✓ VISUAL INFO PRESERVED


    AT QUERY TIME:
    ┌─────────┐    ┌──────────────┐    ┌─────────────┐    ┌───────────┐
    │  Query  │───▶│ Embed Query  │───▶│ Find Similar│───▶│ Return    │
    │         │    │ (ColPali)    │    │ Pages       │    │ PAGE IMAGES│
    └─────────┘    └──────────────┘    └─────────────┘    └───────────┘
                                                                │
                                                                ▼
                                                         ┌───────────┐
                                                         │ Vision LLM│
                                                         │ (GPT-4V)  │
                                                         │ "Sees" the│
                                                         │ table!    │
                                                         └───────────┘


    COLPALI KEY INSIGHT:
    ─────────────────────────────────────────────────────────────────
    ColPali (Contextual Late-interaction for Pali) creates embeddings
    that capture BOTH text AND visual layout in a single vector.

    - Based on PaliGemma (Google's vision-language model)
    - One embedding per document image
    - Query embedding matches against document images
    - No text extraction needed!

    The retrieved document IMAGE is then sent to a vision-capable LLM
    which can actually "see" and understand the table/chart/diagram.
    """

    print(approach)


# ============================================================
# DEMO: VISION LLM WITH DOCUMENT IMAGE
# ============================================================


def demo_vision_llm_analysis():
    """
    Demonstrate using GPT-4V to analyze a document with visual elements.

    Note: This demo uses a text description since we don't have actual
    PDF images. In production, you'd send the actual page image.
    """

    print("\n" + "=" * 60)
    print("DEMO: Vision LLM Document Analysis")
    print("=" * 60)

    # In a real implementation, you would:
    # 1. Convert PDF page to image
    # 2. Encode image as base64
    # 3. Send to GPT-4V with the query

    # For this demo, we'll simulate the process
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.0,
    )  # GPT-4o has vision

    # Simulated document description (in real use, this would be an image)
    document_description = """
    [This is a description of a PDF page image that would be sent to GPT-4V]

    The page contains a financial report with:
    - A table titled "Q1 2025 Sales by Region"
    - Four regions: North, South, East, West
    - Columns: Region, Q1 Target, Q1 Actual, Variance
    - North: $2.5M target, $2.8M actual, +12% (green checkmark)
    - South: $1.8M target, $1.5M actual, -17% (red X)
    - East: $3.2M target, $3.4M actual, +6% (green checkmark)
    - West: $2.1M target, $2.0M actual, -5%
    - Total: $9.6M target, $9.7M actual, +1%
    - A pie chart showing regional contribution percentages
    - A trend line showing quarterly progression
    """

    query = "Which region is underperforming and by how much?"

    print(f'\nQuery: "{query}"')
    print("\nDocument contains a sales table with visual indicators...")

    # In production with actual image:
    # response = llm.invoke([
    #     {"type": "text", "text": query},
    #     {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
    # ])

    # For demo, we'll use text description
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are analyzing a financial document. The user will describe
what's in the document image, and you should answer their question based on
the visual information described.

Focus on:
- Specific numbers and percentages
- Visual indicators (colors, checkmarks, X marks)
- Trends shown in charts
- Relationships between data points""",
            ),
            (
                "human",
                """Document contents:
{document}

Question: {query}

Provide a clear, specific answer based on the document:""",
            ),
        ]
    )

    chain = prompt | llm
    response = chain.invoke({"document": document_description, "query": query})

    print("\nVision LLM Response:")
    print("-" * 60)
    print(response.content)

    print("\n" + "-" * 60)
    print("KEY ADVANTAGE:")
    print("-" * 60)
    print(
        dedent("""
                 With text extraction, we'd have:
            "South $1.8M $1.5M -17%"

                 With vision, the LLM can see:
                     - The red X indicating failure
                     - The context of other regions performing well
                     - The visual hierarchy showing South as the outlier
                     - Any trend lines or charts supporting the analysis
                 """).strip()
    )


# ============================================================
# COLPALI IMPLEMENTATION CODE
# ============================================================


def show_colpali_implementation():
    """
    Show how to implement ColPali-based multimodal RAG.
    """

    print("\n" + "=" * 60)
    print("COLPALI IMPLEMENTATION")
    print("=" * 60)

    implementation = '''
    # Installation (requires GPU for optimal performance)
    # pip install colpali-engine pdf2image pillow

    from colpali_engine import ColPali
    from pdf2image import convert_from_path
    import torch

    # 1. INITIALIZE COLPALI
    # ─────────────────────────────────────────────────────────────
    model = ColPali.from_pretrained(
        "vidore/colpali-v1.2",
        torch_dtype=torch.bfloat16,
        device_map="cuda"  # Requires GPU
    )
    processor = model.processor

    # 2. CONVERT PDF TO IMAGES
    # ─────────────────────────────────────────────────────────────
    def pdf_to_images(pdf_path: str) -> list:
        """Convert each PDF page to an image."""
        images = convert_from_path(pdf_path, dpi=150)
        return images

    # 3. CREATE IMAGE EMBEDDINGS
    # ─────────────────────────────────────────────────────────────
    def embed_document_images(images: list) -> list:
        """Create ColPali embeddings for each page image."""
        embeddings = []
        for image in images:
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                embedding = model(**inputs).last_hidden_state.mean(dim=1)

            embeddings.append(embedding.cpu().numpy())
        return embeddings

    # 4. EMBED QUERY
    # ─────────────────────────────────────────────────────────────
    def embed_query(query: str):
        """Create ColPali embedding for text query."""
        inputs = processor(text=query, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            embedding = model(**inputs).last_hidden_state.mean(dim=1)

        return embedding.cpu().numpy()

    # 5. SEARCH AND RETRIEVE
    # ─────────────────────────────────────────────────────────────
    def search_documents(query: str, document_embeddings: list, images: list, k: int = 3):
        """Find most relevant page images for the query."""
        query_embedding = embed_query(query)

        # Calculate similarities
        similarities = []
        for i, doc_emb in enumerate(document_embeddings):
            sim = np.dot(query_embedding.flatten(), doc_emb.flatten())
            similarities.append((i, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top-k page images
        results = []
        for idx, score in similarities[:k]:
            results.append({
                "page_number": idx + 1,
                "score": score,
                "image": images[idx]
            })
        return results

    # 6. ANSWER WITH VISION LLM
    # ─────────────────────────────────────────────────────────────
    def answer_with_vision(query: str, page_images: list):
        """Send retrieved page images to GPT-4V for analysis."""
        import base64
        from io import BytesIO

        # Convert PIL images to base64
        image_contents = []
        for result in page_images:
            buffered = BytesIO()
            result["image"].save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            image_contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_str}"}
            })

        # Call GPT-4V
        from openai import OpenAI
        client = OpenAI()

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Based on these document pages, answer: {query}"},
                        *image_contents
                    ]
                }
            ]
        )

        return response.choices[0].message.content

    # FULL PIPELINE
    # ─────────────────────────────────────────────────────────────
    def multimodal_rag_query(pdf_path: str, query: str):
        """Complete multimodal RAG pipeline."""
        # 1. Convert PDF to images
        images = pdf_to_images(pdf_path)

        # 2. Create embeddings (do once, store in vector DB)
        embeddings = embed_document_images(images)

        # 3. Search for relevant pages
        results = search_documents(query, embeddings, images, k=3)

        # 4. Answer with vision LLM
        answer = answer_with_vision(query, results)

        return answer
    '''

    print(implementation)


# ============================================================
# USE CASES
# ============================================================


def show_use_cases():
    """
    Show when to use multimodal RAG.
    """

    print("\n" + "=" * 60)
    print("WHEN TO USE MULTIMODAL RAG")
    print("=" * 60)

    use_cases = """
    PERFECT FOR:
    ─────────────────────────────────────────────────────────────
    ✅ Financial Reports
       - Complex tables with nested headers
       - Charts showing trends
       - Footnotes and annotations

    ✅ Technical Documentation
       - Architecture diagrams
       - Flowcharts
       - Code with syntax highlighting

    ✅ Scientific Papers
       - Figures and graphs
       - Mathematical equations
       - Data visualizations

    ✅ Legal Documents
       - Formatted contracts
       - Tables of terms
       - Signature pages

    ✅ Medical Records
       - Diagnostic images
       - Lab result tables
       - Clinical charts


    NOT NECESSARY FOR:
    ─────────────────────────────────────────────────────────────
    ❌ Plain text documents (novels, articles)
    ❌ Simple structured data (CSV-like)
    ❌ Documents where text extraction works well
    ❌ Real-time applications (vision models are slower)
    ❌ Cost-sensitive applications (GPT-4V costs more)


    COST COMPARISON:
    ─────────────────────────────────────────────────────────────
    Text RAG:
    - Embedding: ~$0.0001 per page
    - Query: ~$0.01 per query (GPT-4o-mini)

    Multimodal RAG:
    - Embedding: ~$0.001 per page (ColPali + GPU)
    - Query: ~$0.10 per query (GPT-4o with images)

    ~10x more expensive, but necessary for visual content!
    """

    print(use_cases)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LESSON 6.6: MULTIMODAL RAG WITH COLPALI")
    print("=" * 60)
    print("\nWhen text extraction isn't enough")
    print("=" * 60)

    # 1. Show the problem
    demonstrate_extraction_problem()

    # 2. Explain ColPali approach
    explain_colpali_approach()

    # 3. Demo vision LLM
    demo_vision_llm_analysis()

    # 4. Show implementation
    show_colpali_implementation()

    # 5. Use cases
    show_use_cases()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print(
        """
    1. Text extraction loses tables, charts, and diagrams
    2. ColPali embeds document IMAGES, not extracted text
    3. Vision LLMs (GPT-4o, Claude) can "see" the documents
    4. Best for: financial reports, technical docs, anything visual
    5. Trade-off: ~10x more expensive, but preserves information
    6. The future: documents are images, not text

    IMPLEMENTATION CHECKLIST:
    □ GPU for ColPali embeddings (or use cloud GPU)
    □ pdf2image for PDF → image conversion
    □ Vector DB that supports image embeddings
    □ GPT-4o or Claude for vision-based answering
    □ Fallback to text RAG for plain documents
    """
    )
