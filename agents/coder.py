from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import os


def coder_agent(state):

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = PromptTemplate(
        input_variables=["design"],
        template="""
        Generate a complete MONOLITHIC Python file.

        Requirements:
        - Data loading
        - Model
        - Training
        - Evaluation
        - CLI
        - Comments

        Architecture:
        {design}

        Output ONLY valid Python code.
        """
    )

    chain = prompt | llm

    result = chain.invoke({
        "design": state["design"]
    })

    state["code"] = result.content

    return state
