import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from rag.loader import load_pdf
from rag.splitter import split_docs
from rag.vectorstore import create_vectorstore
from graph.workflow import build_graph

load_dotenv()
st.set_page_config(
    page_title="Code From Paper",
    layout="centered"
)


st.title("📄➡️💻 Code From Paper")
st.subheader("Convert Research Papers into Executable Code")


# File Upload
uploaded_file = st.file_uploader(
    "Upload Research Paper (PDF)",
    type=["pdf"]
)


if uploaded_file:

    st.success("PDF Uploaded Successfully!")

    if st.button("Generate Code"):

        with st.spinner("Processing paper... Please wait ⏳"):

            # Save file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                pdf_path = tmp.name


            # Load & Process
            docs = load_pdf(pdf_path)
            chunks = split_docs(docs)
            db = create_vectorstore(chunks)

            paper_text = "\n".join(
                [doc.page_content for doc in docs]
            )


            # Build graph
            app = build_graph()


            # Initialize State
            state = {
                "paper_text": paper_text,

                "summary": None,
                "analysis": None,
                "design": None,
                "code": None,
                "final_code": None
            }


            # Run pipeline
            result = app.invoke(state)

            final_code = result["final_code"]


            # Save output
            with open("generated_code.py", "w", encoding="utf-8") as f:
                f.write(final_code)


            # Display output
            st.success("Code Generated Successfully!")

            st.subheader("Generated Code")

            st.code(final_code, language="python")


            # Download Button
            st.download_button(
                label="Download Code",
                data=final_code,
                file_name="generated_code.py",
                mime="text/plain"
            )


            # Cleanup
            os.remove(pdf_path)
