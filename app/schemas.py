from typing import Dict, List

from pydantic import BaseModel, Field


DOC_TYPES = [
    "claim_forms",
    "cheque_or_bank_details",
    "identity_document",
    "itemized_bill",
    "discharge_summary",
    "prescription",
    "investigation_report",
    "cash_receipt",
    "other",
]


class SegregationResult(BaseModel):
    pages_by_type: Dict[str, List[int]] = Field(default_factory=dict)


class IdentityExtraction(BaseModel):
    patient_name: str = ""
    date_of_birth: str = ""
    id_numbers: List[str] = Field(default_factory=list)
    policy_number: str = ""
    member_id: str = ""


class DischargeSummaryExtraction(BaseModel):
    diagnosis: str = ""
    admission_date: str = ""
    discharge_date: str = ""
    physician_name: str = ""
    hospital_name: str = ""


class BillItem(BaseModel):
    description: str
    quantity: float = 1.0
    unit_price: float = 0.0
    amount: float = 0.0


class ItemizedBillExtraction(BaseModel):
    items: List[BillItem] = Field(default_factory=list)
    total_amount: float = 0.0

