
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

from .state import AgentState
from .nodes import instructor_node, agent_node, finalize_node, handoff_node
from .tools import BANKING_TOOLS

def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("instructor", instructor_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(BANKING_TOOLS))
    workflow.add_node("finalize", finalize_node)
    workflow.add_node("handoff", handoff_node)

    def post_instructor_router(state: AgentState):
        if not state.get("is_banking_related", True):
            return "handoff"
        return "agent"

    
    def should_continue(state: AgentState):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return "finalize"

    workflow.set_entry_point("instructor")
    workflow.add_conditional_edges("instructor", post_instructor_router)
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")       
    workflow.add_edge("finalize", END)
    workflow.add_edge("handoff", END)

    return workflow.compile()


app_graph = build_graph()
