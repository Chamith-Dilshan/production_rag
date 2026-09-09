from langchain_docling.loader import DoclingLoader

FILE_PATH = "../docs/doc1.pdf"


def document_loader():
    loader = DoclingLoader(file_path=FILE_PATH)

    docs = loader.load()

    print(f"Loaded {len(docs)} documents.")
    print(f"Content preview: {docs[0].page_content[:100]}...")
    print(f"Metadata: {docs[0].metadata}")

    # for d in docs:
    #     print(f"=== {d.metadata['source']} ===")
    #     print(f"- {d.page_content=}")


if __name__ == "__main__":
    document_loader()
