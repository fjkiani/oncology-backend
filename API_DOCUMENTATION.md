# Clinical Trials API Documentation

This document provides comprehensive documentation for the Clinical Trials Search and Patient Eligibility Assessment API.

## Overview

The Clinical Trials API provides endpoints for:
- Searching clinical trials using vector similarity and traditional search
- Assessing patient eligibility for clinical trials using AI
- Managing patient data and trial information
- Planning follow-up tasks based on trial criteria

## Base URL

```
http://localhost:8000  # Development
https://your-domain.com  # Production
```

## Authentication

Currently using placeholder authentication for WebSocket connections. REST endpoints are open for development.

```javascript
// WebSocket authentication token format
const token = "valid_token_USER_ID";
```

## API Endpoints

### 1. Root Endpoint

**GET** `/`

Simple health check endpoint.

**Response:**
```json
{
  "message": "Beat Cancer AI Backend is running"
}
```

### 2. Clinical Trial Search

**POST** `/api/search-trials`

Search for clinical trials using vector similarity search and optional patient context for eligibility assessment.

**Request Body:**
```json
{
  "query": "lung cancer phase 2 trials",
  "patient_context": {
    "diagnosis": {
      "primary": "Non-small cell lung cancer",
      "stage": "IIIA"
    },
    "demographics": {
      "age": 65,
      "gender": "Female"
    },
    "labs": [
      {
        "test": "EGFR mutation",
        "result": "Positive",
        "date": "2024-01-15"
      }
    ]
  }
}
```

**Request Parameters:**
- `query` (string, required): Search query for clinical trials
- `patient_context` (object, optional): Patient data for eligibility assessment

**Response:**
```json
{
  "success": true,
  "data": {
    "found_trials": [
      {
        "nct_id": "NCT12345678",
        "title": "Phase 2 Study of Drug X in NSCLC Patients",
        "status": "Recruiting",
        "phase": "Phase 2",
        "source_url": "https://clinicaltrials.gov/ct2/show/NCT12345678",
        "llm_assessment": {
          "summary": "Patient appears eligible based on diagnosis and biomarkers.",
          "eligibility_status": "Likely Eligible",
          "met_criteria": [
            {
              "criterion": "EGFR positive NSCLC",
              "trial_snippet": "Patients with EGFR-positive non-small cell lung cancer",
              "reasoning": null
            }
          ],
          "unmet_criteria": [],
          "unclear_criteria": [
            {
              "criterion": "Performance status",
              "trial_snippet": "ECOG performance status 0-1",
              "reasoning": "Performance status not specified in patient data"
            }
          ]
        },
        "action_suggestions": [
          {
            "action_type": "TASK",
            "suggestion": "Verify ECOG performance status",
            "draft_text": "Please assess patient's ECOG performance status",
            "criterion": "ECOG performance status 0-1",
            "missing_info": "Performance status not documented"
          }
        ]
      }
    ]
  }
}
```

### 3. Patient Data Retrieval

**GET** `/api/patients/{patient_id}`

Retrieve patient data including demographics, medical history, and mutations.

**Path Parameters:**
- `patient_id` (string): Patient identifier (case-insensitive)

**Response:**
```json
{
  "success": true,
  "data": {
    "demographics": {
      "name": "Jane Doe",
      "age": 65,
      "gender": "Female",
      "dateOfBirth": "1958-05-15"
    },
    "diagnosis": {
      "primary": "Non-small cell lung cancer",
      "stage": "IIIA",
      "histology": "Adenocarcinoma"
    },
    "mutations": [
      {
        "gene": "EGFR",
        "mutation": "L858R",
        "status": "Positive"
      }
    ],
    "recentLabs": [
      {
        "test": "Complete Blood Count",
        "date": "2024-01-10",
        "results": {
          "hemoglobin": "12.5 g/dL",
          "platelets": "250,000/μL"
        }
      }
    ]
  }
}
```

### 4. Trial Details

**GET** `/api/trial-details/{trial_id}?patient_id={patient_id}`

Get detailed information about a specific clinical trial with optional patient-specific eligibility assessment.

**Path Parameters:**
- `trial_id` (string): NCT ID of the trial

**Query Parameters:**
- `patient_id` (string, optional): Patient ID for eligibility assessment

**Response:**
```json
{
  "trial_data": {
    "nct_id": "NCT12345678",
    "title": "Phase 2 Study of Drug X in NSCLC Patients",
    "status": "Recruiting",
    "phase": "Phase 2",
    "sponsor": "Pharmaceutical Company Inc",
    "brief_summary": "This study evaluates the efficacy of Drug X...",
    "eligibility_criteria": "Inclusion: 1) Confirmed NSCLC diagnosis...",
    "locations": ["Hospital A", "Medical Center B"],
    "enrollment_count": 150
  },
  "eligibility_assessment": {
    "summary": "Patient assessment summary...",
    "eligibility_status": "Likely Eligible",
    "met_criteria": [...],
    "unmet_criteria": [...],
    "unclear_criteria": [...]
  }
}
```

### 5. Deep Dive Analysis

**POST** `/api/request-deep-dive`

Request detailed eligibility analysis for specific criteria that were initially marked as unmet or unclear.

**Request Body:**
```json
{
  "unmet_criteria": [
    {
      "criterion": "Prior treatment requirement",
      "trial_snippet": "Must have received at least one prior systemic therapy",
      "reasoning": "Patient treatment history unclear"
    }
  ],
  "unclear_criteria": [
    {
      "criterion": "Performance status",
      "trial_snippet": "ECOG performance status 0-1",
      "reasoning": "Performance status not documented"
    }
  ],
  "patient_data": {
    "diagnosis": {...},
    "medicalHistory": {...}
  },
  "trial_data": {
    "nct_id": "NCT12345678",
    "title": "Trial Title"
  }
}
```

**Response:**
```json
{
  "summary": "Detailed analysis of eligibility criteria...",
  "recommendations": [
    {
      "criterion": "Performance status",
      "recommendation": "Assess ECOG performance status during next visit",
      "priority": "high",
      "rationale": "Required for trial eligibility determination"
    }
  ],
  "clarifications": [
    {
      "criterion": "Prior treatment",
      "clarification": "Patient's chemotherapy history suggests eligibility",
      "confidence": "medium"
    }
  ]
}
```

### 6. Plan Follow-ups

**POST** `/api/plan-followups`

Convert action suggestions into structured tasks for follow-up planning.

**Request Body:**
```json
{
  "action_suggestions": [
    {
      "action_type": "TASK",
      "suggestion": "Verify ECOG performance status",
      "draft_text": "Please assess patient's ECOG performance status",
      "criterion": "ECOG performance status 0-1",
      "missing_info": "Performance status not documented"
    }
  ],
  "patient_id": "PATIENT001",
  "trial_id": "NCT12345678",
  "trial_title": "Phase 2 Study of Drug X"
}
```

**Response:**
```json
{
  "success": true,
  "planned_tasks": [
    {
      "id": "task_PATIENT001_NCT12345678_1640995200_0",
      "columnId": "followUpNeeded",
      "content": "Clarify: ECOG performance status 0-1",
      "patientId": "PATIENT001",
      "suggestion_type": "TASK",
      "related_criterion": "ECOG performance status 0-1",
      "trial_id": "NCT12345678",
      "trial_title": "Phase 2 Study of Drug X"
    }
  ]
}
```

### 7. Task Management

**GET** `/api/tasks`

Retrieve all tasks in the Kanban task store.

**Response:**
```json
[
  {
    "id": "task_PATIENT001_NCT12345678_1640995200_0",
    "columnId": "followUpNeeded",
    "content": "Clarify: ECOG performance status 0-1",
    "patientId": "PATIENT001",
    "suggestion_type": "TASK",
    "related_criterion": "ECOG performance status 0-1",
    "trial_id": "NCT12345678",
    "trial_title": "Phase 2 Study of Drug X"
  }
]
```

### 8. Agent Activity

**GET** `/api/agent_activity`

Get the current activity status of all registered agents.

**Response:**
```json
[
  {
    "agent_key": "clinical_trial_finder",
    "agent_name": "Clinical Trial Agent",
    "status": "idle",
    "last_activity": "2024-01-15T10:30:00Z",
    "current_task": null
  }
]
```

### 9. Prompt Processing

**POST** `/api/prompt/{patient_id}`

Process a natural language prompt using the orchestrator system.

**Path Parameters:**
- `patient_id` (string): Patient identifier

**Request Body:**
```json
{
  "prompt": "Find clinical trials for this lung cancer patient"
}
```

**Response:**
```json
{
  "status": "success",
  "agent_used": "clinical_trial_finder",
  "result": {
    "message": "Found 5 relevant clinical trials",
    "trials": [...]
  }
}
```

### 10. Feedback Submission

**POST** `/api/feedback/{patient_id}`

Submit feedback on AI-generated content with blockchain logging.

**Path Parameters:**
- `patient_id` (string): Patient identifier

**Request Body:**
```json
{
  "feedback_text": "The eligibility assessment was very helpful and accurate",
  "ai_output_context": "NCT12345678_eligibility_assessment"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Feedback received and metadata logged to blockchain.",
  "blockchain_tx_hash": "0x1234567890abcdef..."
}
```

### 11. Agent Proxy

**POST** `/api/agent-proxy`

Execute agent commands via HTTP interface.

**Request Body:**
```json
{
  "patientId": "PATIENT001",
  "intent": "summarize_deep_dive",
  "prompt": "Summarize the eligibility assessment",
  "payload": {
    "trial_id": "NCT12345678",
    "focus": "eligibility_criteria"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "agent_name": "data_analyzer",
  "output": {
    "summary": "Deep dive analysis summary...",
    "key_findings": [...],
    "recommendations": [...]
  }
}
```

## WebSocket API

The WebSocket endpoint provides real-time communication for collaborative features.

### Connection

**WebSocket** `/ws?token={auth_token}`

**Authentication:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws?token=valid_token_USER123');
```

### Message Types

#### Join Room
```json
{
  "type": "join_room",
  "roomId": "consultation_room_123"
}
```

#### Chat Message
```json
{
  "type": "chat_message",
  "roomId": "consultation_room_123",
  "content": "Patient has shown good response to current treatment",
  "sender": {
    "id": "USER123",
    "name": "Dr. Smith"
  }
}
```

#### Agent Command
```json
{
  "type": "agent_command",
  "command": "/compare-therapy",
  "text": "/compare-therapy current=\"Carboplatin + Paclitaxel\" vs=\"Pembrolizumab\" focus=\"efficacy,toxicity\"",
  "patientId": "PATIENT001"
}
```

#### Analyze Initiator Note
```json
{
  "type": "analyze_initiator_note",
  "roomId": "consultation_room_123",
  "note_text": "Patient presented with progressive disease after first-line therapy...",
  "sender": {
    "id": "USER123",
    "name": "Dr. Smith"
  }
}
```

## Error Handling

### Standard Error Response

```json
{
  "detail": "Error message describing what went wrong",
  "status_code": 400
}
```

### Common HTTP Status Codes

- `200` - Success
- `400` - Bad Request (invalid input)
- `404` - Not Found (patient/trial not found)
- `500` - Internal Server Error
- `422` - Validation Error (invalid request format)

### WebSocket Error Messages

```json
{
  "type": "error",
  "message": "Error description",
  "timestamp": 1640995200.123
}
```

## Rate Limiting

Currently no rate limiting implemented. In production, consider:
- 100 requests per minute per IP for search endpoints
- 10 requests per minute for deep dive analysis
- WebSocket connection limits per user

## Data Models

### Patient Context

```typescript
interface PatientContext {
  diagnosis?: {
    primary?: string;
    stage?: string;
    histology?: string;
  };
  demographics?: {
    age?: number;
    gender?: string;
    name?: string;
  };
  labs?: Array<{
    test: string;
    result: string;
    date: string;
  }>;
  mutations?: Array<{
    gene: string;
    mutation: string;
    status: string;
  }>;
}
```

### Trial Data

```typescript
interface TrialData {
  nct_id: string;
  title: string;
  status: string;
  phase: string;
  sponsor: string;
  brief_summary: string;
  detailed_description?: string;
  eligibility_criteria: string;
  locations: string[];
  enrollment_count: number;
  last_updated_date: string;
}
```

### Eligibility Assessment

```typescript
interface EligibilityAssessment {
  summary: string;
  eligibility_status: "Likely Eligible" | "Likely Ineligible" | "Eligibility Unclear due to missing info";
  met_criteria: Array<{
    criterion: string;
    trial_snippet: string;
    reasoning?: string;
  }>;
  unmet_criteria: Array<{
    criterion: string;
    trial_snippet: string;
    reasoning: string;
  }>;
  unclear_criteria: Array<{
    criterion: string;
    trial_snippet: string;
    reasoning: string;
  }>;
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

## Examples

### Complete Trial Search Workflow

```javascript
// 1. Search for trials
const searchResponse = await fetch('/api/search-trials', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: 'EGFR positive lung cancer',
    patient_context: {
      diagnosis: { primary: 'NSCLC', stage: 'IIIA' },
      demographics: { age: 65, gender: 'Female' }
    }
  })
});

const searchResult = await searchResponse.json();
const trials = searchResult.data.found_trials;

// 2. Get detailed information for a specific trial
const trialId = trials[0].nct_id;
const detailsResponse = await fetch(`/api/trial-details/${trialId}?patient_id=PATIENT001`);
const trialDetails = await detailsResponse.json();

// 3. Plan follow-ups based on action suggestions
const actionSuggestions = trials[0].action_suggestions;
const followupResponse = await fetch('/api/plan-followups', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    action_suggestions: actionSuggestions,
    patient_id: 'PATIENT001',
    trial_id: trialId,
    trial_title: trials[0].title
  })
});

const plannedTasks = await followupResponse.json();
```

## Testing

Use the provided test script to verify API functionality:

```bash
# Basic tests (no external API calls)
python test_complete_pipeline.py --skip-etl

# Full tests including ETL pipeline
python test_complete_pipeline.py --small-sample

# Test API endpoints (requires server running)
python test_complete_pipeline.py --test-api --skip-etl
```

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for production deployment instructions including:
- Environment setup
- Database configuration
- API security considerations
- Monitoring and logging

---

**Last Updated:** December 2024  
**API Version:** 1.0  
**Environment:** Development 