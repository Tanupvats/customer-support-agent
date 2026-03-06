
# state.py
import operator
from typing import Annotated, List, Optional, Dict, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage

class AgentState(TypedDict, total=False):
    
    session_id: str
    user_id: str
    user_name: str

    
    messages: Annotated[List[BaseMessage], operator.add]

    
    category: Optional[str]
    is_banking_related: Optional[bool]

   
    current_plan: Optional[Dict[str, Any]]
    final_response: Optional[str]
    is_satisfactory: Optional[bool]

    
    turn_count: Optional[int]
