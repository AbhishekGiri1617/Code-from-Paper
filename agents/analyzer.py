from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import os


def analyzer_agent(state):

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = PromptTemplate(
        input_variables=["summary"],
        template="""
        Analyze this research summary and extract:

        - Model architecture
        - Algorithms
        - Hyperparameters
        - Training pipeline

        Summary:
        {summary}
        """
    )

    chain = prompt | llm

    result = chain.invoke({
        "summary": state["summary"]
    })

    state["analysis"] = result.content

    return state
