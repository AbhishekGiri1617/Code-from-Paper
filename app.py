import os
from dotenv import load_dotenv

from rag.loader import load_pdf
from rag.splitter import split_docs
from rag.vectorstore import create_vectorstore

from graph.workflow import build_graph


load_dotenv()


def read_full_text(docs):
    return "\n".join([doc.page_content for doc in docs])


def main():

    print("📄 Loading paper...")

    docs = load_pdf("Paper.pdf")

    print("✂️ Splitting...")

    chunks = split_docs(docs)

    print("📦 Creating vector store...")

    db = create_vectorstore(chunks)

    paper_text = read_full_text(docs)

    print("🤖 Starting Multi-Agent System...")

    app = build_graph()

    result = app.invoke({

    "paper_text": paper_text,

    "summary": None,
    "analysis": None,
    "design": None,
    "code": None,
    "final_code": None
})


    final_code = result["final_code"]

    with open("generated_code.py", "w", encoding="utf-8") as f:
        f.write(final_code)

    print("✅ Code generated: generated_code.py")


if __name__ == "__main__":
    main()
