from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import os


def architect_agent(state):

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = PromptTemplate(
        input_variables=["analysis"],
        template="""
        Design a monolithic software architecture for
        implementing this research.

        Include:
        - Libraries
        - Modules
        - Main pipeline
        - Config structure

        Analysis:
        {analysis}
        """
    )

    chain = prompt | llm

    result = chain.invoke({
        "analysis": state["analysis"]
    })

    state["design"] = result.content

    return state
