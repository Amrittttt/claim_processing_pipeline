import json
import os
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover
    ChatOpenAI = None


def _get_chat_model():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or ChatOpenAI is None:
        return None
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)


def classify_pages_with_llm(pages_text: Dict[int, str], doc_types: List[str]) -> Dict[str, List[int]]:
    llm = _get_chat_model()
    if llm is None:
        return {}

    page_summaries = []
    for page_num, text in pages_text.items():
        page_summaries.append(
            {
                "page": page_num,
                "text_preview": text[:2500],
            }
        )

    prompt = (
        "Classify each PDF page into exactly one document type from the allowed list. "
        "Return strict JSON in this shape: "
        '{"pages_by_type":{"identity_document":[1], "itemized_bill":[2], "...":[]}}'
    )
    messages = [
        SystemMessage(
            content=(
                "You classify healthcare claim pages. Only output valid JSON. "
                f"Allowed document types: {doc_types}"
            )
        ),
        HumanMessage(content=f"{prompt}\n\nPages:\n{json.dumps(page_summaries)}"),
    ]
    response = llm.invoke(messages)
    try:
        data = json.loads(response.content)
    except Exception:
        return {}
    raw = data.get("pages_by_type", {})
    result: Dict[str, List[int]] = {k: [] for k in doc_types}
    for key, value in raw.items():
        if key in result and isinstance(value, list):
            result[key] = [int(v) for v in value if isinstance(v, int) or (isinstance(v, str) and v.isdigit())]
    return result


def extract_structured_with_llm(task_name: str, text: str, output_schema: Dict[str, Any]) -> Dict[str, Any]:
    llm = _get_chat_model()
    if llm is None:
        return {}
    messages = [
        SystemMessage(
            content=(
                "Extract structured data for healthcare claims. Return strict JSON only."
            )
        ),
        HumanMessage(
            content=(
                f"Task: {task_name}\n"
                f"JSON schema keys expected: {list(output_schema.keys())}\n"
                f"Document text:\n{text[:12000]}"
            )
        ),
    ]
    response = llm.invoke(messages)
    try:
        data = json.loads(response.content)
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}

