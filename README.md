# Claim Processing Pipeline Assignment

FastAPI + LangGraph service for claim PDF processing with:
- AI-powered page segregation
- Three extraction agents (ID, Discharge Summary, Itemized Bill)
- Aggregated JSON output

## Workflow

The graph follows your required flow:

`START -> segregator_agent -> (id_agent, discharge_summary_agent, itemized_bill_agent) -> aggregator -> END`

### Segregator Agent
- Reads PDF page by page
- Classifies each page into one of:
  - `claim_forms`
  - `cheque_or_bank_details`
  - `identity_document`
  - `itemized_bill`
  - `discharge_summary`
  - `prescription`
  - `investigation_report`
  - `cash_receipt`
  - `other`
- Routes only relevant pages to each extraction agent

### Extraction Agents
- `id_agent`: patient identity, DOB, IDs, policy/member details
- `discharge_summary_agent`: diagnosis, admission/discharge dates, physician/hospital
- `itemized_bill_agent`: line items and computed total

### Aggregator
- Combines all agent outputs into a final JSON payload

## Tech Stack
- FastAPI
- LangGraph
- LangChain
- pypdf

## Setup

```bash
cd /home/piyush/repo/pipeline_project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional (for LLM-powered segregation/extraction):

```bash
export OPENAI_API_KEY="your_key_here"
export OPENAI_MODEL="gpt-4o-mini"
```

Without an API key, the app falls back to deterministic parsing heuristics.

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

## API

### Health
`GET /health`

### Process Claim
`POST /api/process`

Form data:
- `claim_id` (string)
- `file` (PDF)

### Test via Swagger UI (recommended for GitHub users)

If you cloned this repository and do not have the original sample PDF, use your own claim PDF:

1. Start server:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
2. Open Swagger:
   - `http://127.0.0.1:8000/docs`
3. Expand `POST /api/process` and click **Try it out**
4. Enter any `claim_id`
5. Upload any `.pdf` using the `file` input
6. Click **Execute** to get JSON output

### Test via cURL

```bash
curl -X POST "http://localhost:8000/api/process" \
  -F "claim_id=claim-001" \
  -F "file=@/absolute/path/to/your-claim.pdf"
```

> Note: The assignment sample PDF is not included in this public repo. Use any compatible claim PDF for testing.

Example response shape:

```json
{
  "claim_id": "claim-001",
  "segregation": {
    "identity_document": [1, 2],
    "itemized_bill": [4, 5],
    "discharge_summary": [3]
  },
  "identity_document": {
    "patient_name": "..."
  },
  "discharge_summary": {
    "diagnosis": "..."
  },
  "itemized_bill": {
    "items": [],
    "total_amount": 0
  }
}
```

## Notes for Submission
- Record a short Loom/video showing:
  - LangGraph node flow
  - Segregator behavior
  - Page-level routing to agents
  - Final aggregated output
- Push this folder to your GitHub repo and share it with the assignment email.

