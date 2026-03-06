
import os
from typing import List

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from langchain_core.runnables import RunnableConfig
from openai import BadRequestError
from .state import AgentState
from .models import QueryClassification
from .tools import BANKING_TOOLS, search_knowledgebase, calculate_emi, user_info_lookup, get_user_details, create_ticket


AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

MAX_MODEL_TURNS = 10
CONFIRM_TOKENS = {"yes", "yep", "yeah", "y", "sure", "please", "go ahead", "do it", "create a ticket", "please create a ticket", "ok", "okay"}

def _is_short_confirmation(text: str) -> bool:
    t = text.strip().lower()
    
    return len(t) <= 16 or any(tok in t for tok in CONFIRM_TOKENS)


def _azure_llm(temperature: float = 0.0):
    if AZURE_API_KEY and AZURE_ENDPOINT and AZURE_DEPLOYMENT and AZURE_API_VERSION:
        return AzureChatOpenAI(
            azure_deployment=AZURE_DEPLOYMENT,
            azure_endpoint=AZURE_ENDPOINT,
            api_version=AZURE_API_VERSION,
            api_key=AZURE_API_KEY,
            temperature=temperature,
        )
    
    return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)



def _sanitize_history_for_openai(messages: List):
    """
    OpenAI strict rule: a ToolMessage must be *immediately* preceded by an AIMessage containing tool_calls.
    This sanitizer drops:
      - Leading ToolMessages
      - Any ToolMessage that doesn't directly follow an AI tool-calling message within the cleaned list
    """
    cleaned: List = []
    for m in messages:
        if isinstance(m, ToolMessage):
            if cleaned and isinstance(cleaned[-1], AIMessage) and getattr(cleaned[-1], "tool_calls", None):
                cleaned.append(m)
            else:
                
                continue
        else:
            cleaned.append(m)

    
    while cleaned and isinstance(cleaned[0], ToolMessage):
        cleaned.pop(0)

    return cleaned



instructor_llm = _azure_llm(temperature=0.0)

agent_base_llm = _azure_llm(temperature=0.1)


def _tools_for_category(category: str):
    """
    Limit tool surface to reduce hallucinations and improve accuracy.
    """
    if category == "general":
        return [search_knowledgebase]  
    
    return [user_info_lookup, calculate_emi, get_user_details, create_ticket, search_knowledgebase]


def instructor_node(state: AgentState, config: RunnableConfig = None):
    """
    Classify the latest user message into banking categories and decide route.
    Also handles short follow-ups by inheriting previous context.
    """
    user_msg_obj = state["messages"][-1]
    user_msg = user_msg_obj.content
    prev_category = (state.get("category") or "").lower()
    prev_is_bank = state.get("is_banking_related")

    
    if _is_short_confirmation(user_msg) and prev_category:
        
        prev_ai_text = ""
        for m in reversed(state["messages"][:-1]):
            if isinstance(m, AIMessage) and m.content:
                prev_ai_text = m.content
                break

        refined = user_msg
        if prev_ai_text:
            refined = f"Follow-up confirmation from user: '{user_msg}'. Prior assistant message: '{prev_ai_text}'. Act accordingly (e.g., proceed if assistant asked for confirmation)."

        
        out = {
            "category": prev_category,
            "is_banking_related": True if prev_is_bank is None else bool(prev_is_bank),
            "messages": state["messages"][:-1] + [HumanMessage(content=refined)],
        }
        return out

    
    structured = instructor_llm.with_structured_output(QueryClassification)
    sys = (
        "You are an expert banking query classifier. "
        "Classify if the query is banking-related and assign a category from: "
        "[credit card, bank account,'kyc', 'fraud', 'dispute', 'general', 'loan', 'others']. "
        "Return a refined_query that is concise, unambiguous, and uses banking terminology."
    )
    res = structured.invoke(f"{sys}\n\nUser: {user_msg}")

    out = {
        "category": res.category,
        "is_banking_related": bool(res.is_banking_related),
        "messages": state["messages"][:-1] + [HumanMessage(content=res.refined_query)],
    }
    return out



def agent_node(state: AgentState, config: RunnableConfig = None):
    """
    Tool-enabled agent with Azure-friendly prompting and policy-safe retry.
    """
    category = (state.get("category") or "others").lower()
    tools = _tools_for_category(category)

    
    llm = agent_base_llm.bind_tools(tools)

   
    sys = SystemMessage(content=(
        "You are a helpful, precise banking assistant.\n"
        f"- Detected category: {category}\n"
        "- Use the provided tools when necessary to answer accurately.\n"
        "- For loan calculations, use `calculate_emi` (requires principal, years, credit_score), get credit score using user_info_lookup.\n"
        "- For account details, use `user_info_lookup` (extract customer_id like CUST00001).\n"
        "- For general product/policy info, use `search_knowledgebase`.\n"
        "- If the prior assistant turn asked for confirmation to proceed (e.g., create a ticket) and the user confirms, "
        "  proceed to call the appropriate tool with concise, sensible defaults.\n"
        "- always ask for customer confirmation When creating a ticket and show all the details related to ticket to customer in responce after creating the ticket .\n"
        "- Always produce a user-ready answer after you have all necessary info.\n"
        "- If key inputs are missing, ask a concise follow-up question."
    ))

    
    sanitized_history = _sanitize_history_for_openai(state["messages"])
    if MAX_MODEL_TURNS and len(sanitized_history) > MAX_MODEL_TURNS:
        sanitized_history = sanitized_history[-MAX_MODEL_TURNS:]

    msgs: List = [sys] + sanitized_history

   
    try:
        ai = llm.invoke(msgs)
        return {"messages": [ai]}
    except Exception as e:
        msg = str(e)
        if ("ResponsibleAIPolicyViolation" in msg) or ("content_filter" in msg.lower()) or ("jailbreak" in msg.lower()):
            
            fallback_sys = SystemMessage(
                content="You are a banking assistant. Answer clearly and concisely, using tools if needed."
            )
            short_window = sanitized_history[-1:] if len(sanitized_history) > 6 else sanitized_history
            fallback_msgs = [fallback_sys] + short_window
            ai = llm.invoke(fallback_msgs)
            return {"messages": [ai]}
        
        raise



def finalize_node(state: AgentState, config: RunnableConfig = None):
    """
    Emit the final assistant message content as final_response.
    """
    
    last_ai = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and (msg.content and not getattr(msg, "tool_calls", None)):
            last_ai = msg
            break

    content = (last_ai.content if last_ai else "I couldn't form a final response.")
    return {"final_response": content, "is_satisfactory": True}


def handoff_node(state: AgentState, config: RunnableConfig = None):
    """
    Graceful exit when query is not banking-related.
    """
    return {
        "final_response": (
            "I’m optimized for banking-related queries (loans, KYC, disputes, fraud, products). "
            "Please share a banking question, and I’ll help right away."
        ),
        "is_satisfactory": False
    }
