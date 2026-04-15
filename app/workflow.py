import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

from .llm import classify_pages_with_llm, extract_structured_with_llm
from .pdf_utils import (
    extract_pdf_pages_text,
    first_match,
    save_selected_pages_as_pdf,
    split_lines,
)
from .schemas import DOC_TYPES


class ClaimState(TypedDict, total=False):
    claim_id: str
    pdf_path: str
    pages_text: Dict[int, str]
    pages_by_type: Dict[str, List[int]]
    routed_pdfs: Dict[str, str]
    id_data: Dict[str, Any]
    discharge_data: Dict[str, Any]
    itemized_bill_data: Dict[str, Any]
    final_result: Dict[str, Any]


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _compact_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _extract_field_value(text: str, labels: List[str]) -> str:
    lines = split_lines(text.replace("|", "\n"))
    for idx, line in enumerate(lines):
        lower_line = line.lower()
        for label in labels:
            if label in lower_line:
                if ":" in line:
                    value = line.split(":", 1)[1].strip()
                    if value:
                        return value
                if idx + 1 < len(lines):
                    nxt = lines[idx + 1].strip()
                    if nxt and ":" not in nxt:
                        return nxt
    return ""


def _heuristic_classify_page(text: str) -> str:
    t = _normalized_text(text)
    c = _compact_text(text)
    strong_rules = [
        ("claim_forms", ["medicalclaimform", "claimreference", "claimdetails"]),
        ("cheque_or_bank_details", ["globaltrustbank", "chequedetails", "bankaccountdetails", "ifscroutingnumber"]),
        ("identity_document", ["governmentidcard", "idnumber", "insuranceverificationform"]),
        ("itemized_bill", ["itemizedhospitalbill", "itemizedcharges"]),
        ("discharge_summary", ["dischargesummary", "admissiondate", "dischargedate"]),
        ("cash_receipt", ["cashreceipt", "receiptno", "totalamountpaid", "paymentmethodcash"]),
        ("prescription", ["prescriptionsummary", "prescription", "rxdr", "dosage"]),
        ("investigation_report", ["pathologylaboratory", "laboratoryreport", "completebloodcount", "comprehensivemetabolicpanel", "lipidpanel"]),
    ]
    for doc_type, needles in strong_rules:
        if any(n in c for n in needles):
            return doc_type

    weak_rules = [
        ("claim_forms", ["claim form", "claim details"]),
        ("cheque_or_bank_details", ["cheque details", "bank account details", "ifsc", "routing number"]),
        ("identity_document", ["government id card", "id number", "insurance verification"]),
        ("itemized_bill", ["itemized hospital bill", "itemized charges"]),
        ("discharge_summary", ["discharge summary", "admission date", "discharge date"]),
        ("cash_receipt", ["cash receipt", "receipt no", "total amount paid"]),
        ("prescription", ["prescription", "rx", "dosage"]),
        ("investigation_report", ["pathology laboratory", "laboratory report", "complete blood count", "metabolic panel", "lipid panel"]),
    ]
    for doc_type, keywords in weak_rules:
        if any(keyword in t for keyword in keywords):
            return doc_type
    return "other"


def segregator_node(state: ClaimState) -> ClaimState:
    pdf_path = Path(state["pdf_path"])
    pages_text = extract_pdf_pages_text(pdf_path)

    # Deterministic heading-based routing is primary for scanned docs.
    pages_by_type = {doc_type: [] for doc_type in DOC_TYPES}
    for page_num, text in pages_text.items():
        pages_by_type[_heuristic_classify_page(text)].append(page_num)

    # Keep classification deterministic for scanned docs.
    _ = classify_pages_with_llm  # Intentional no-op placeholder for optional future use.

    for doc_type in DOC_TYPES:
        pages_by_type[doc_type] = sorted(set(pages_by_type.get(doc_type, [])))

    tmp_dir = Path(tempfile.mkdtemp(prefix="claim_pages_"))
    routed_pdfs: Dict[str, str] = {}
    route_targets = {
        "id_agent": pages_by_type["identity_document"],
        "discharge_summary_agent": pages_by_type["discharge_summary"],
        "itemized_bill_agent": pages_by_type["itemized_bill"],
    }
    for route_name, pages in route_targets.items():
        if pages:
            out_pdf = tmp_dir / f"{route_name}.pdf"
            save_selected_pages_as_pdf(pdf_path, pages, out_pdf)
            routed_pdfs[route_name] = str(out_pdf)

    return {
        "pages_text": pages_text,
        "pages_by_type": pages_by_type,
        "routed_pdfs": routed_pdfs,
    }


def id_agent_node(state: ClaimState) -> ClaimState:
    pages = state.get("pages_by_type", {}).get("identity_document", [])
    pages_text = state.get("pages_text", {})
    text = "\n".join(pages_text.get(p, "") for p in pages)

    llm_data = extract_structured_with_llm(
        "Extract identity details",
        text,
        {
            "patient_name": "",
            "date_of_birth": "",
            "id_numbers": [],
            "policy_number": "",
            "member_id": "",
        },
    )
    if llm_data:
        return {"id_data": llm_data}

    id_numbers = []
    for pattern in [r"\bID-[0-9\-]+\b", r"\bPAT-[0-9]+\b", r"\bMRN-[0-9]+\b", r"\bPOL-[A-Z0-9\-]+\b"]:
        id_numbers.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    id_numbers = list(dict.fromkeys(id_numbers))
    id_data = {
        "patient_name": _extract_field_value(text, ["full name", "patient name", "name"]),
        "date_of_birth": _extract_field_value(text, ["date of birth", "dob"]),
        "id_numbers": id_numbers[:10],
        "policy_number": _extract_field_value(text, ["policy number", "policy no"]),
        "member_id": _extract_field_value(text, ["member id", "patient id", "member no"]),
    }
    return {"id_data": id_data}


def discharge_summary_agent_node(state: ClaimState) -> ClaimState:
    pages = state.get("pages_by_type", {}).get("discharge_summary", [])
    pages_text = state.get("pages_text", {})
    text = "\n".join(pages_text.get(p, "") for p in pages)

    llm_data = extract_structured_with_llm(
        "Extract discharge summary",
        text,
        {
            "diagnosis": "",
            "admission_date": "",
            "discharge_date": "",
            "physician_name": "",
            "hospital_name": "",
        },
    )
    if llm_data:
        return {"discharge_data": llm_data}

    hospital_name = first_match(r"([A-Z][A-Za-z ]+MEDICAL CENTER)", text, flags=0)
    discharge_data = {
        "diagnosis": _extract_field_value(text, ["admission diagnosis", "final diagnosis", "diagnosis"]),
        "admission_date": _extract_field_value(text, ["admission date", "date of admission"]),
        "discharge_date": _extract_field_value(text, ["discharge date", "date of discharge"]),
        "physician_name": _extract_field_value(text, ["attending physician", "physician", "doctor", "consultant"]),
        "hospital_name": hospital_name or _extract_field_value(text, ["hospital name", "medical center", "hospital"]),
    }
    return {"discharge_data": discharge_data}


def itemized_bill_agent_node(state: ClaimState) -> ClaimState:
    pages = state.get("pages_by_type", {}).get("itemized_bill", [])
    pages_text = state.get("pages_text", {})
    # Restrict parsing to page(s) explicitly headed "ITEMIZED HOSPITAL BILL".
    table_page_text = ""
    for p in pages:
        page_text = pages_text.get(p, "")
        compact = _compact_text(page_text)
        if "itemizedhospitalbill" in compact and "itemizedcharges" in compact:
            table_page_text = page_text
            break

    if not table_page_text:
        return {"itemized_bill_data": {"items": [], "total_amount": 0.0}}

    lines = split_lines(table_page_text.replace("|", "\n"))
    items: List[Dict[str, Any]] = []

    # Focus only on the "ITEMIZED CHARGES" table section.
    start_idx = 0
    end_idx = len(lines)
    for i, line in enumerate(lines):
        compact_line = _compact_text(line)
        if compact_line == "itemizedcharges":
            start_idx = i + 1
            continue
        if i > start_idx and compact_line in {"subtotal", "grandtotal", "totalamount", "netpayable"}:
            end_idx = i
            break
    table_lines = lines[start_idx:end_idx]

    i = 0
    while i < len(table_lines):
        line = table_lines[i].strip()
        if not re.fullmatch(r"\d{2}/\d{2}/\d{2,4}", line):
            i += 1
            continue

        j = i + 1
        desc_parts: List[str] = []
        qty = 1.0
        money_values: List[float] = []

        while j < len(table_lines):
            cur = table_lines[j].strip()
            if not cur:
                j += 1
                continue
            if re.fullmatch(r"\d{2}/\d{2}/\d{2,4}", cur):
                break

            money = re.search(r"\$\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]{1,2})?)", cur)
            if money:
                money_values.append(float(money.group(1).replace(",", "")))
            elif re.fullmatch(r"\d+(?:\.\d+)?", cur) and not desc_parts:
                qty = float(cur)
            else:
                desc_parts.append(cur)
            j += 1

        description = re.sub(r"\s+", " ", " ".join(desc_parts)).strip(" -:")
        tail_qty = re.search(r"^(.*)\s(\d+(?:\.\d+)?)$", description)
        if tail_qty and qty == 1.0:
            description = tail_qty.group(1).strip()
            qty = float(tail_qty.group(2))
        if description and money_values:
            unit_price = money_values[0]
            amount = money_values[1] if len(money_values) > 1 else money_values[0]
            items.append(
                {
                    "description": description[:160],
                    "quantity": qty,
                    "unit_price": unit_price,
                    "amount": amount,
                }
            )
        i = j

    text = "\n".join(lines)
    explicit_total = first_match(r"(?:grand total|total amount|net payable)\s*[:\-]?\s*\$?\s*([0-9,]+(?:\.[0-9]{1,2})?)", text)
    if explicit_total:
        total = float(explicit_total.replace(",", ""))
    else:
        total = round(sum(item["amount"] for item in items), 2)
    return {"itemized_bill_data": {"items": items, "total_amount": total}}


def aggregator_node(state: ClaimState) -> ClaimState:
    final = {
        "claim_id": state.get("claim_id", ""),
        "segregation": state.get("pages_by_type", {}),
        "identity_document": state.get("id_data", {}),
        "discharge_summary": state.get("discharge_data", {}),
        "itemized_bill": state.get("itemized_bill_data", {}),
    }
    return {"final_result": final}


def build_claim_graph():
    graph = StateGraph(ClaimState)
    graph.add_node("segregator_agent", segregator_node)
    graph.add_node("id_agent", id_agent_node)
    graph.add_node("discharge_summary_agent", discharge_summary_agent_node)
    graph.add_node("itemized_bill_agent", itemized_bill_agent_node)
    graph.add_node("aggregator", aggregator_node)

    graph.add_edge(START, "segregator_agent")
    graph.add_edge("segregator_agent", "id_agent")
    graph.add_edge("segregator_agent", "discharge_summary_agent")
    graph.add_edge("segregator_agent", "itemized_bill_agent")
    graph.add_edge("id_agent", "aggregator")
    graph.add_edge("discharge_summary_agent", "aggregator")
    graph.add_edge("itemized_bill_agent", "aggregator")
    graph.add_edge("aggregator", END)
    return graph.compile()

