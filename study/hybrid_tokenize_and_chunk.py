import tiktoken
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

"""
Using HybridChunker to chunk a PDF document
and enrich the chunked text with contextual information.

In a RAG / retrieval context, it is important to make sure 
that the chunker and embedding model are using the same tokenizer.
"""

FILE_PATH = "../docs/doc1.pdf"
EMBED_MODEL_ID = "gpt-4o"
MAX_TOKENS = 128 * 1024  # set to a small number for illustrative purposes

tokenizer = OpenAITokenizer(
    tokenizer=tiktoken.encoding_for_model(EMBED_MODEL_ID),
    max_tokens=MAX_TOKENS,  # context window length required for OpenAI tokenizers
)


def hybrid_tokenize_and_chunk():
    converter = DocumentConverter()
    doc = converter.convert(source=FILE_PATH).document

    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True,  # optional, defaults to True
    )
    chunk_iter = chunker.chunk(dl_doc=doc)
    chunks = list(chunk_iter)

    for i, chunk in enumerate(chunks):
        print(f"=== {i} ===")
        txt_tokens = tokenizer.count_tokens(chunk.text)
        print(f"chunk.text ({txt_tokens} tokens):\n{chunk.text!r}")

        ser_txt = chunker.contextualize(chunk=chunk)
        ser_tokens = tokenizer.count_tokens(ser_txt)
        print(f"chunker.contextualize(chunk) ({ser_tokens} tokens):\n{ser_txt!r}")

        print()


if __name__ == "__main__":
    hybrid_tokenize_and_chunk()
