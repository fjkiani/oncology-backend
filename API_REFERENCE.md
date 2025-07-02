# Clinical Trials API Reference

## Overview

The Clinical Trials API provides endpoints for searching clinical trials, assessing patient eligibility, and managing follow-up tasks.

**Base URL:** `http://localhost:8000` (development)

## Core Endpoints

### 1. Search Clinical Trials

**POST** `/api/search-trials`

Search for relevant clinical trials with optional patient eligibility assessment.

```json
// Request
{
  "query": "lung cancer phase 2 trials",
  "patient_context": {
    "diagnosis": { "primary": "NSCLC", "stage": "IIIA" },
    "demographics": { "age": 65, "gender": "Female" }
  }
}

// Response
{
  "success": true,
  "data": {
    "found_trials": [
      {
        "nct_id": "NCT12345678",
        "title": "Phase 2 Study of Drug X in NSCLC",
        "status": "Recruiting",
        "phase": "Phase 2",
        "llm_assessment": {
          "summary": "Patient appears eligible...",
          "eligibility_status": "Likely Eligible",
          "met_criteria": [...],
          "unmet_criteria": [...],
          "unclear_criteria": [...]
        },
        "action_suggestions": [...]
      }
    ]
  }
}
```

### 2. Get Patient Data

**GET** `/api/patients/{patient_id}`

Retrieve comprehensive patient information.

```json
// Response
{
  "success": true,
  "data": {
    "demographics": { "name": "Jane Doe", "age": 65 },
    "diagnosis": { "primary": "NSCLC", "stage": "IIIA" },
    "mutations": [{ "gene": "EGFR", "mutation": "L858R" }],
    "recentLabs": [...]
  }
}
```

### 3. Get Trial Details

**GET** `/api/trial-details/{trial_id}?patient_id={patient_id}`

Get detailed trial information with optional patient-specific assessment.

### 4. Request Deep Dive Analysis

**POST** `/api/request-deep-dive`

Detailed analysis of unclear or unmet eligibility criteria.

```json
// Request
{
  "unmet_criteria": [...],
  "unclear_criteria": [...],
  "patient_data": {...},
  "trial_data": {...}
}

// Response
{
  "summary": "Detailed analysis...",
  "recommendations": [...],
  "clarifications": [...]
}
```

### 5. Plan Follow-ups

**POST** `/api/plan-followups`

Convert action suggestions into structured tasks.

```json
// Request
{
  "action_suggestions": [...],
  "patient_id": "PATIENT001",
  "trial_id": "NCT12345678"
}

// Response
{
  "success": true,
  "planned_tasks": [...]
}
```

### 6. Task Management

**GET** `/api/tasks`

Retrieve all Kanban tasks.

### 7. Agent Proxy

**POST** `/api/agent-proxy`

Execute agent commands via HTTP.

```json
// Request
{
  "patientId": "PATIENT001",
  "intent": "summarize_deep_dive",
  "prompt": "Summarize eligibility assessment"
}
```

## WebSocket API

**WebSocket** `/ws?token={auth_token}`

Real-time communication for collaborative features.

### Message Types

- `join_room` - Join a consultation room
- `chat_message` - Send chat messages
- `agent_command` - Execute agent commands
- `analyze_initiator_note` - Analyze consultation notes

```json
// Agent command example
{
  "type": "agent_command",
  "command": "/compare-therapy",
  "text": "/compare-therapy current=\"Drug A\" vs=\"Drug B\" focus=\"efficacy\"",
  "patientId": "PATIENT001"
}
```

## Data Models

### Patient Context
```typescript
interface PatientContext {
  diagnosis?: { primary?: string; stage?: string; };
  demographics?: { age?: number; gender?: string; };
  labs?: Array<{ test: string; result: string; date: string; }>;
  mutations?: Array<{ gene: string; mutation: string; status: string; }>;
}
```

### Eligibility Assessment
```typescript
interface EligibilityAssessment {
  summary: string;
  eligibility_status: "Likely Eligible" | "Likely Ineligible" | "Eligibility Unclear due to missing info";
  met_criteria: Array<{ criterion: string; trial_snippet: string; }>;
  unmet_criteria: Array<{ criterion: string; trial_snippet: string; reasoning: string; }>;
  unclear_criteria: Array<{ criterion: string; trial_snippet: string; reasoning: string; }>;
}
```

### Action Suggestion
```typescript
interface ActionSuggestion {
  action_type: "TASK" | "PATIENT_MESSAGE_SUGGESTION" | "CLINICAL_QUESTION";
  suggestion: string;
  draft_text?: string;
  criterion?: string;
  missing_info?: string;
}
```

## Error Handling

### HTTP Status Codes
- `200` - Success
- `400` - Bad Request
- `404` - Not Found
- `500` - Internal Server Error
- `422` - Validation Error

### Error Response Format
```json
{
  "detail": "Error description",
  "status_code": 400
}
```

## Authentication

Currently using placeholder authentication for WebSocket connections:
```
Token format: "valid_token_USER_ID"
```

## Testing

Run the test suite to verify functionality:

```bash
# Basic tests (no external API calls)
python test_complete_pipeline.py --skip-etl

# Test API endpoints (requires server running)
uvicorn main:app --reload  # In one terminal
python test_complete_pipeline.py --test-api --skip-etl  # In another

# Full pipeline test with small sample
python test_complete_pipeline.py --small-sample
```

## Example Workflow

```javascript
// 1. Search trials
const response = await fetch('/api/search-trials', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: 'EGFR positive lung cancer',
    patient_context: { diagnosis: { primary: 'NSCLC' } }
  })
});

// 2. Process results
const result = await response.json();
const trials = result.data.found_trials;

// 3. Plan follow-ups
await fetch('/api/plan-followups', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    action_suggestions: trials[0].action_suggestions,
    patient_id: 'PATIENT001',
    trial_id: trials[0].nct_id
  })
});
```

## Database Architecture

The system uses:
- **SQLite** for structured trial metadata
- **AstraDB** for vector similarity search
- **SentenceTransformer** for embedding generation

See [DEPLOYMENT.md](DEPLOYMENT.md) for setup instructions.

---

For complete documentation and deployment instructions, see the full documentation files in the project repository. 