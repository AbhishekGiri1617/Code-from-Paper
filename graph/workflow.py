from langgraph.graph import StateGraph

from agents.reader import reader_agent
from agents.analyzer import analyzer_agent
from agents.architect import architect_agent
from agents.coder import coder_agent
from agents.validator import validator_agent
from typing import TypedDict, Optional


class PaperState(TypedDict):
    paper_text: str

    # Agent Outputs
    summary: Optional[str]
    analysis: Optional[str]
    design: Optional[str]
    code: Optional[str]
    final_code: Optional[str]


def build_graph():

    graph = StateGraph(PaperState)

    graph.add_node("reader", reader_agent)
    graph.add_node("analyzer", analyzer_agent)
    graph.add_node("architect", architect_agent)
    graph.add_node("coder", coder_agent)
    graph.add_node("validator", validator_agent)

    graph.set_entry_point("reader")

    graph.add_edge("reader", "analyzer")
    graph.add_edge("analyzer", "architect")
    graph.add_edge("architect", "coder")
    graph.add_edge("coder", "validator")

    graph.set_finish_point("validator")

    return graph.compile()
