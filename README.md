# AI Cancer Care CoPilot

A smart, HIPAA-compliant AI application designed to act as a true CoPilot for oncologists, leveraging an AI agent-based architecture to analyze patient records, automate tasks, and facilitate real-time collaboration.

## Overview

This application aims to significantly reduce administrative burden and help overcome obstacles in cancer care through intelligent automation, analysis, and transparent collaboration. It uses specialized AI agents (built with Python and potentially frameworks like LangChain/AutoGen) coordinated by an orchestrator backend (FastAPI) to handle tasks like:

*   Patient record summarization and Q&A (using models like Google Gemini).
*   Drafting notifications, referrals, and potentially orders.
*   Assisting with scheduling.
*   Providing information on medication side effects.
*   Identifying potential clinical trials.

The frontend is built with React (Vite) and TailwindCSS, providing a user interface for interacting with the CoPilot and viewing patient data (currently mocked).

## Core Concepts

*   **Agent-Based Architecture:** Specialized agents handle specific tasks, orchestrated by a central component.
*   **Role-Centric Workflow:** The orchestrator identifies needs and routes tasks to agents that initiate workflows relevant to specific clinical roles (Clinician, Nurse, Admin, Research, Pharmacy).
*   **Human-in-the-Loop:** Clinician oversight and approval remain paramount for critical actions suggested or drafted by the AI.
*   **HIPAA Compliance:** Designed with security and privacy first, including considerations for BAAs, data encryption, access controls, and keeping PHI off any potential blockchain layer.

## Project Structure

*   `backend/`: Contains the FastAPI application, agent implementations (`agents/`), orchestrator logic (`core/`), and blockchain utilities (`core/`).
*   `frontend/`: Contains the React frontend application (`src/`).
*   `blockchain/`: (Planned) Will contain Hardhat project for smart contracts.
*   `.env`: For environment variables (API keys, etc. - **DO NOT COMMIT**).
*   `cursorRules.md`: Internal notes, project plan, and scratchpad for AI assistant.
*   `README.md`: This file.

## Setup and Running

**(Instructions to be added here - e.g., Python venv setup, pip install, npm install, running backend/frontend)**

## Blockchain POC: Contribution Tracking (Feedback Logging)

### Goal

This Proof-of-Concept (POC) demonstrates the use of blockchain technology to create an immutable, auditable log of specific **contributions** related to the AI CoPilot's usage and improvement, starting with **clinician feedback on AI-generated outputs**.

### Core Concept: Off-Chain Data, On-Chain Metadata

The fundamental principle guiding this implementation is ensuring **no Protected Health Information (PHI) or sensitive data is ever stored on the blockchain.**

*   **Off-Chain:** The actual content of the contribution (e.g., the text of the feedback provided by a clinician) is stored securely within our conventional, HIPAA-compliant backend systems (database, secure logs).
*   **On-Chain:** Only non-sensitive **metadata** is recorded on the blockchain ledger. This includes:
    *   An identifier for the contributor (e.g., the backend service's address logging the event).
    *   The type of contribution (e.g., `"AI_Feedback"`).
    *   The timestamp of the contribution.
    *   A **cryptographic hash** (a unique, fixed-size fingerprint) of the off-chain data. This hash acts as a verifiable link to the original data without revealing its content.

### Benefits (Why Use Blockchain for This?)

1.  **Immutable Audit Trail & Provenance:** Creates a permanent, tamper-proof record confirming *that* feedback was given, *when*, and by *whom* (or which system component logged it). This enhances trust and provides a verifiable history of interactions.
2.  **Foundation for Contribution Tracking:** Establishes the technical pattern for reliably logging various contributions (feedback, participation in Federated Learning, etc.) needed for potential future incentive models or collaborative improvement efforts.
3.  **Enhanced Trust & Transparency:** While the POC uses a local network, the architecture allows for future scenarios where a shared, immutable ledger could increase trust between users, the system, and potentially different participating institutions.
4.  **Data Integrity & Security:** Securely links the immutable on-chain metadata to the sensitive, off-chain data via cryptographic hashes, ensuring we can prove the existence and timing of a contribution without exposing sensitive details on the public ledger, reinforcing HIPAA compliance.

*In Simple Terms for Clinicians:* This system acts like a highly secure digital logbook. When you provide feedback, it permanently records *that* you gave feedback and when, ensuring your input is formally acknowledged and contributes to improving the AI CoPilot, all while keeping patient data completely separate and safe.

### How it Works (POC Workflow)

1.  **Feedback Submission (Frontend - Optional):** A user provides feedback on an AI output via the UI.
2.  **API Call (Frontend -> Backend):** The frontend sends the feedback data (text, context) to a dedicated backend API endpoint: `POST /api/feedback/{patient_id}`.
3.  **Secure Storage (Backend - Off-Chain):** The backend stores the full feedback content securely in a standard database or secure logging system (this part is assumed/not fully implemented in the POC).
4.  **Hashing (Backend):** The backend (`feedback` router) calculates a cryptographic hash (SHA-256) of the feedback data.
5.  **Blockchain Interaction (Backend -> Blockchain):** The backend calls the utility function `blockchain_utils.record_contribution`.
6.  **Smart Contract Execution (On-Chain):** `record_contribution` builds and signs a transaction to call the `logContribution` function on the `ContributionTracker` smart contract, passing the contribution type (`"AI_Feedback"`) and the calculated hash.
7.  **Logging (On-Chain):** The smart contract validates the call (e.g., checks if the caller is authorized, typically the owner account specified by `BLOCKCHAIN_PRIVATE_KEY`), records the metadata (contributor address, type, hash, timestamp) immutably on the blockchain ledger, and emits an event (`ContributionLogged`).
8.  **API Response (Backend -> Frontend):** The backend API responds to the frontend, indicating success or failure and including the blockchain transaction hash if successful (e.g., `{"status":"success", "blockchain_tx_hash":"0x..."}`).

### Technology Stack (POC)

*   **Blockchain Network:** Local Hardhat Network (Simulated Ethereum environment for development).
*   **Smart Contract Language:** Solidity.
*   **Development Framework:** Hardhat (for compiling, testing, deploying contracts).
*   **Backend Library:** Web3.py (for Python backend interaction with the blockchain).

### Smart Contract (`ContributionTracker.sol`)

A simple Solidity contract acting as a registry. Its primary functions are:
*   Define the data structure for a `Contribution`.
*   Provide a `logContribution` function to add new entries (metadata + hash).
*   Store entries immutably.
*   Allow retrieval of logged entries.
*   Emit an event upon successful logging.

### Current Status

This is currently implemented as a **Proof-of-Concept** using a **local development network**. It demonstrates the mechanism but is not connected to a persistent or shared ledger.

### Future Considerations

*   Deployment to a private/consortium blockchain for shared, controlled access.
*   Expanding tracked contributions (e.g., Federated Learning updates).
*   **Extending Logging:** Applying the same metadata logging mechanism to track key **system actions** (e.g., scheduling initiated/confirmed, referral drafted/approved, notification sent) for a more complete, immutable audit trail of CoPilot activity.
*   Integration with potential incentive systems (tokenization).

## Feature Focus: Advanced Consultation Agents

Building upon the real-time consultation feature, this work focuses on implementing more sophisticated AI agents that can be invoked directly within the chat flow to assist clinicians during discussions or case reviews.

### Goal

Enhance the utility of the Doctor-to-Doctor consultation feature by providing on-demand, context-aware information synthesis and drafting capabilities through specialized AI agents.

### Agents Being Developed:

1.  **`ComparativeTherapyAgent`**
    *   **Purpose:** Provides clinicians with a structured comparison of specified treatment regimens based on requested criteria (e.g., efficacy, side effects).
    *   **Trigger:** Invoked via a chat command (e.g., `/compare-therapy current:[RegimenA] vs alternative:[RegimenB] focus:criteria`).
    *   **Workflow:** Retrieves relevant patient context, queries knowledge sources (LLM, potentially integrated databases), synthesizes findings based on focus criteria.
    *   **Output:** Posts a structured comparison summary directly into the consultation chat.
    *   **Disclaimer:** Output is AI-generated for informational purposes and requires clinical verification. Does not constitute a treatment recommendation.

2.  **`PatientEducationDraftAgent`**
    *   **Purpose:** Assists clinicians by drafting patient-friendly explanations of medical topics discussed during the consultation.
    *   **Trigger:** Invoked via a chat command (e.g., `/draft-patient-info topic:"Explanation of Treatment X"`), potentially using chat context.
    *   **Workflow:** Uses an LLM with tailored prompts to generate clear, simple text explaining the specified topic at an appropriate reading level.
    *   **Output:** Posts text clearly marked as a **DRAFT** into the consultation chat, intended solely for clinician review and editing *before* any potential use with a patient.
    *   **Safety:** Clinician review and editing of the draft are mandatory before sharing any information derived from it with patients.

### Integration

Both agents will receive triggers from the consultation chat interface, be processed by the backend orchestrator, and post their results back into the same chat session for immediate visibility to the participants.


