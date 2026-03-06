
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
from .graph import app_graph
from langchain_core.messages import HumanMessage, BaseMessage

from dotenv import load_dotenv
load_dotenv()
app = FastAPI(title="Banking Multi-Agent System")


SESSION_STORE: Dict[str, Dict[str, Any]] = {}

MAX_HISTORY = 30  

class UserRequest(BaseModel):
    session_id: str
    user_id: str
    user_name: str
    query: str

@app.get("/health")
def health_check():
    return {"status": "active"}

def _trim_history(state: Dict[str, Any]) -> Dict[str, Any]:
    msgs = state.get("messages", [])
    if len(msgs) > MAX_HISTORY:
        state["messages"] = msgs[-MAX_HISTORY:]
    return state

@app.post("/chat")
async def chat_endpoint(req: UserRequest):
    sid = req.session_id

    # 1) Load existing session state or initialize
    state = SESSION_STORE.get(sid)
    if not state:
        state = {
            "session_id": req.session_id,
            "user_id": req.user_id,
            "user_name": req.user_name,
            "messages": [],
            
            # "is_banking_related": True,
        }

   
    state["messages"] = state.get("messages", []) + [HumanMessage(content=req.query)]

    run_config = {
        # tags show up on the root run and propagate to children (nodes/tools)
        "tags": ["banking-assistant", "fastapi", "chat"],
        # metadata is searchable and visible in the run sidebar
        "metadata": {
            "session_id": req.session_id,
            "user_id": req.user_id,
            "user_name": req.user_name,
            "route": "/chat",
        },
        # thread_id groups runs by session in LangSmith
        "configurable": {"thread_id": req.session_id},
        # optional name for the root run in LangSmith
        "run_name": "chat_session",
    }

    result = await app_graph.ainvoke(state,config=run_config)

    
    SESSION_STORE[sid] = _trim_history(result)

    
    return {
        "response": result.get("final_response", "Processing complete."),
        "category": result.get("category"),
        "is_banking_related": result.get("is_banking_related"),
        "meta": {
            "session": req.session_id,
            "is_satisfactory": result.get("is_satisfactory", False),
        },
    }
