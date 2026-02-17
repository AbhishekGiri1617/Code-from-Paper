from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import os


def validator_agent(state):

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = PromptTemplate(
        input_variables=["code"],
        template="""
        Review and fix this Python code.

        Ensure:
        - No syntax errors
        - All imports present
        - Runs correctly
        - Best practices

        Code:
        {code}

        Output only final corrected code.
        """
    )

    chain = prompt | llm

    result = chain.invoke({
        "code": state["code"]
    })

    state["final_code"] = result.content

    return state
