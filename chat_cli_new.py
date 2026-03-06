
import os
import sys
import uuid
import requests


from dotenv import load_dotenv
load_dotenv()
BASE_URL = os.environ.get("BANKING_AGENT_BASE_URL", "http://127.0.0.1:8000")
TIMEOUT = float(os.environ.get("BANKING_AGENT_TIMEOUT", "30"))

DEFAULT_USER_ID = os.environ.get("BANKING_AGENT_USER_ID", "CUST00001")
DEFAULT_USER_NAME = os.environ.get("BANKING_AGENT_USER_NAME", "Riya Mehta")

def get_json(resp):
    try:
        return resp.json()
    except Exception:
        return {"raw_text": resp.text}

def main():
    # Health check
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        print(f"[health] {health.status_code} -> {get_json(health)}")
    except Exception as e:
        print(f"Failed to reach {BASE_URL}/health: {e}")
        sys.exit(1)

    # Keep session fixed for the whole CLI run
    session_id = os.environ.get("BANKING_AGENT_SESSION_ID") or f"cli-{uuid.uuid4().hex[:8]}"

    # (Optional) print LangSmith context so you know which project is receiving traces
    ls_project = os.environ.get("LANGCHAIN_PROJECT", "(default)")
    ls_on = os.environ.get("LANGCHAIN_TRACING_V2", "false")
    print("\n=== Banking Assistant CLI ===")
    print(f"LangSmith tracing: {ls_on} | Project: {ls_project}")
    print("Type your message and press Enter. Type 'exit' to quit.")
    print(f"Base URL: {BASE_URL} | session_id: {session_id} | user_id: {DEFAULT_USER_ID}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {'goodbye',"bye","exit", "quit",'q',}:
            print("Goodbye!")
            break

        payload = {
            "session_id": session_id,
            "user_id": DEFAULT_USER_ID,
            "user_name": DEFAULT_USER_NAME,
            "query": user_input,
        }

        try:
            resp = requests.post(f"{BASE_URL}/chat", json=payload, timeout=TIMEOUT)
        except Exception as e:
            print(f"[error] POST /chat failed: {e}")
            continue

        data = get_json(resp)
        status = resp.status_code

        print(f"\n[status] {status}")
        print(f"Assistant: {data.get('response')}")
        meta = data.get("meta", {})
        category = data.get("category")
        is_bank = data.get("is_banking_related")
        is_sat = meta.get("is_satisfactory")
        if category is not None:
            print(f"(category: {category}, banking: {is_bank}, satisfactory: {is_sat})")
        print("")

if __name__ == "__main__":
    main()
