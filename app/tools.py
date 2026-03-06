from langchain.tools import tool
import pandas as pd
from typing import Optional, Dict, Any


try:
    from app.retriver.app.main import retrieve
except Exception:
    retrieve = None  

@tool
def search_knowledgebase(query: str) -> str:
    """Useful for finding general information about bank products and policies."""
    if retrieve is None:
        return "Knowledgebase retriever is not configured on this server."
    result = retrieve(query)
    try:
        return ". ".join([getattr(x, "content", str(x)) for x in result if x])
    except Exception:
        return str(result)

@tool
def calculate_emi(principal: float, years: int, credit_score: int) -> dict:
    """
    Calculate EMI by deriving interest rate from credit score.

    Returns:
      {
        "interest_rate": float,       # annual %
        "monthly_emi": float,         # EMI amount
        "total_interest": float,      # interest over loan life
        "total_payable": float        # principal + interest
      }
    """
    if principal <= 0:
        raise ValueError("Principal amount must be greater than 0")
    if years <= 0:
        raise ValueError("Tenure (years) must be greater than 0")
    if not 300 <= credit_score <= 900:
        raise ValueError("Credit score must be between 300 and 900")

    if credit_score >= 800:
        annual_rate = 8.25
    elif credit_score >= 750:
        annual_rate = 9.00
    elif credit_score >= 700:
        annual_rate = 10.50
    elif credit_score >= 650:
        annual_rate = 12.50
    elif credit_score >= 600:
        annual_rate = 15.00
    else:
        annual_rate = 18.50

    months = years * 12
    monthly_rate = annual_rate / 12 / 100

    emi = (principal * monthly_rate * (1 + monthly_rate) ** months) / ((1 + monthly_rate) ** months - 1)
    emi = round(emi, 2)
    total_payable = round(emi * months, 2)
    total_interest = round(total_payable - principal, 2)

    return {
        "interest_rate": annual_rate,
        "monthly_emi": emi,
        "total_interest": total_interest,
        "total_payable": total_payable
    }

@tool
def get_user_details(user_id: str) -> dict:
    """Retrieves active products for the user from 'user_data_db.csv'."""
    df = pd.read_csv("user_data_db.csv")
    data = df[df["customer_id"] == user_id].to_dict(orient="records")
    return {"records": data, "count": len(data)}

@tool
def create_ticket(summary: str, queue: str, severity: str, customer_id: Optional[str] = None) -> Dict[str, Any]:
    """create ticket for unresolved query"""
    import uuid
    return {
        "ticket_id": f"TCK-{uuid.uuid4().hex[:8].upper()}",
        "queue": queue,
        "severity": severity,
        "summary": summary,
        "customer_id": customer_id,
    }

@tool
def user_info_lookup(query: str) -> dict:
    """
    Free-text lookup for user loan/profile data from 'user_data_db.csv'.
    Extracts customer_id from free text and returns compact JSON fields.
    """
    import re
    from difflib import get_close_matches

    try:
        CANONICAL_FIELDS = [
            "Name","customer_id","gender","age","city","state","pincode","occupation","employer_type",
            "monthly_income","Loan_type","loan_account_number","loan_open_date","disbursal_date",
            "disbursal_channel","sanction_amount","interest_rate","monthly_emi_amount","total_emi_count",
            "emi_paid","months_past_due","dpd_days","last_payment_date","prepayment_flag","foreclosure_flag",
            "outstanding_principal","principal_paid","interest_paid_to_date","cibil_score",
            "total_active_loans","total_closed_loans","enquiries_last_6m","account_status",
            "risk_grade","branch_code","ifsc_code"
        ]
        CANONICAL_SET = {c.lower(): c for c in CANONICAL_FIELDS}
        FIELD_ALIASES = {
            "customerid": "customer_id", "cust_id": "customer_id", "custid": "customer_id",
            "user_id": "customer_id", "userid": "customer_id", "name": "Name", "user": "Name",
            "loan type": "Loan_type", "loan_type": "Loan_type",
            "loan account number": "loan_account_number", "account number": "loan_account_number",
            "loan number": "loan_account_number",
            "emi": "monthly_emi_amount", "monthly emi": "monthly_emi_amount", "emi amount": "monthly_emi_amount",
            "interest": "interest_rate", "interest rate": "interest_rate", "rate": "interest_rate",
            "sanction": "sanction_amount", "sanction amount": "sanction_amount",
            "disbursal": "disbursal_date", "disbursal date": "disbursal_date",
            "loan open": "loan_open_date", "loan open date": "loan_open_date",
            "outstanding": "outstanding_principal",
            "outstanding principal": "outstanding_principal",
            "principal left": "outstanding_principal",
            "principal balance": "outstanding_principal",
            "balance left": "outstanding_principal",
            "remaining principal": "outstanding_principal",
            "amount left": "outstanding_principal",
            "principal paid": "principal_paid",
            "interest paid": "interest_paid_to_date",
            "emi paid": "emi_paid",
            "total emis": "total_emi_count", "total emi count": "total_emi_count",
            "dpd": "dpd_days", "dpd days": "dpd_days",
            "mpd": "months_past_due", "months past due": "months_past_due",
            "last payment": "last_payment_date", "last payment date": "last_payment_date",
            "prepayment": "prepayment_flag", "foreclosure": "foreclosure_flag",
            "status": "account_status", "risk": "risk_grade",
            "cibil": "cibil_score", "cibil score": "cibil_score",
            "income": "monthly_income", "monthly income": "monthly_income",
            "active loans": "total_active_loans",
            "closed loans": "total_closed_loans",
            "enquiries": "enquiries_last_6m",
            "branch": "branch_code",
            "ifsc": "ifsc_code",
        }
        KEYWORD_FIELD_HINTS = {
            "balance": ["outstanding_principal"],
            "outstanding": ["outstanding_principal"],
            "principal": ["outstanding_principal", "principal_paid"],
            "emi": ["monthly_emi_amount", "emi_paid", "total_emi_count"],
            "payment": ["last_payment_date", "emi_paid", "months_past_due", "dpd_days"],
            "rate": ["interest_rate"],
            "interest": ["interest_rate", "interest_paid_to_date"],
            "status": ["account_status", "risk_grade"],
            "cibil": ["cibil_score"],
            "income": ["monthly_income"],
            "sanction": ["sanction_amount"],
            "loan": ["Loan_type", "loan_account_number", "loan_open_date", "disbursal_date"],
            "branch": ["branch_code", "ifsc_code"],
        }
        CUSTOMER_ID_PATTERNS = [r"\bCUST\d{5,}\b", r"\bCUS\d{5,}\b", r"\bCUSTOMER\d{4,}\b"]

        def extract_customer_id(text: str):
            for pat in CUSTOMER_ID_PATTERNS:
                m = re.search(pat, text, flags=re.IGNORECASE)
                if m:
                    return m.group(0).upper()
            m = re.search(r"(customer[\s_-]*id[:\s]+)([A-Z0-9]+)", text, flags=re.IGNORECASE)
            return m.group(2).upper() if m else None

        def normalize_field_name(raw: str):
            key = raw.strip().lower()
            if key in FIELD_ALIASES:
                return FIELD_ALIASES[key]
            if key in CANONICAL_SET:
                return CANONICAL_SET[key]
            candidates = list(CANONICAL_SET.keys()) + list(FIELD_ALIASES.keys())
            close = get_close_matches(key, candidates, n=1, cutoff=0.83)
            if not close:
                return None
            c = close[0]
            return FIELD_ALIASES.get(c) or CANONICAL_SET.get(c)

        def infer_requested_fields(text: str):
            q = text.strip()
            requested = []
            # Quoted phrases
            for a, b in re.findall(r'"([^"]+)"|\'([^\']+)\'', q):
                token = (a or b).strip()
                cand = normalize_field_name(token)
                if cand and cand not in requested:
                    requested.append(cand)
            # Tokens after verbs
            m = re.search(r"(?:show|get|need|give|tell|share|fetch|want|see)\s+([a-zA-Z0-9_\-\s,]+)",
                          q, flags=re.IGNORECASE)
            if m:
                segment = m.group(1)
                for token in re.split(r"[,\|/]+|\band\b", segment):
                    token = token.strip(" .?;:").lower()
                    if not token or len(token) > 64:
                        continue
                    cand = normalize_field_name(token)
                    if cand and cand not in requested:
                        requested.append(cand)
            # Keyword hints
            ql = q.lower()
            for kw, fields in KEYWORD_FIELD_HINTS.items():
                if kw in ql:
                    for f in fields:
                        if f not in requested:
                            requested.append(f)
            # Default compact set
            if not requested:
                requested = [
                    "Loan_type", "loan_account_number", "account_status",
                    "outstanding_principal", "monthly_emi_amount",
                    "interest_rate", "emi_paid", "total_emi_count",
                    "last_payment_date"
                ]
            return [f for f in requested if f in CANONICAL_FIELDS]

        def format_value(field: str, value):
            import pandas as _pd
            if _pd.isna(value):
                return None
            import re as _re
            if isinstance(value, str) and _re.fullmatch(r"[\d,]+(\.\d+)?", value):
                try:
                    value = float(value.replace(",", ""))
                except Exception:
                    pass
            if field in {"sanction_amount","monthly_emi_amount","outstanding_principal",
                         "principal_paid","interest_paid_to_date","monthly_income"}:
                try: return round(float(value), 2)
                except Exception: return value
            if field == "interest_rate":
                try: return round(float(value), 3)
                except Exception: return value
            if field in {"total_emi_count","emi_paid","months_past_due","dpd_days",
                         "total_active_loans","total_closed_loans","enquiries_last_6m",
                         "age","cibil_score"}:
                try: return int(float(value))
                except Exception: return value
            if field in {"loan_open_date","disbursal_date","last_payment_date"}:
                return str(value)
            if field in {"prepayment_flag","foreclosure_flag"}:
                s = str(value).strip().lower()
                if s in {"y","yes","true","1"}: return True
                if s in {"n","no","false","0"}: return False
                return bool(value)
            return value

        def safe_select_row(df: pd.DataFrame, customer_id: str):
            if "customer_id" not in df.columns:
                return None
            mask = df["customer_id"].astype(str).str.upper() == str(customer_id).upper()
            rows = df[mask]
            return None if rows.empty else rows.iloc[0].to_dict()

        df = pd.read_csv("user_data_db.csv")
        cust_id = extract_customer_id(query)
        if not cust_id:
            return {"customer_id": None, "fields": {}, "message": "Customer ID not found. Include e.g. CUST00001."}
        row = safe_select_row(df, cust_id)
        if not row:
            return {"customer_id": cust_id, "fields": {}, "message": f"No record found for {cust_id}."}
        requested_fields = infer_requested_fields(query)
        result_fields = {f: format_value(f, row.get(f)) for f in requested_fields}
        return {"customer_id": cust_id, "fields": result_fields, "message": "OK"}

    except FileNotFoundError:
        return {"customer_id": None, "fields": {}, "message": "Data file 'user_data_db.csv' not found on server."}
    except Exception as e:
        return {"customer_id": None, "fields": {}, "message": f"Lookup error: {e.__class__.__name__}: {str(e)}"}


BANKING_TOOLS = [search_knowledgebase, calculate_emi, user_info_lookup, get_user_details, create_ticket]
