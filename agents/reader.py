from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import os


def reader_agent(state):

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = PromptTemplate(
        input_variables=["paper"],
        template="""
        Read this research paper and extract:

        - Problem Statement
        - Dataset
        - Methodology
        - Evaluation Metrics

        Paper:
        {paper}
        """
    )

    chain = prompt | llm

    result = chain.invoke({
        "paper": state["paper_text"]
    })

    state["summary"] = result.content

    return state
