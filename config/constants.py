# Agent Intents
SUMMARIZE = "summarize"
CHECK_INTERACTIONS = "check_interactions"
ANALYZE_GENOMIC_PROFILE = "analyze_genomic_profile"
SUMMARIZE_DEEP_DIVE = "summarize_deep_dive"
ANALYZE_INITIATOR_NOTE = "analyze_initiator_note"
SYNTHESIZE_CONSULTATION = "synthesize_consultation"
# New intents that might be used by agents added to activity tracker
NOTIFY = "notify"
SCHEDULE = "schedule"
REFERRAL = "referral"
FIND_TRIALS = "find_trials"
MANAGE_SIDE_EFFECTS = "manage_side_effects"

# Agent Identifiers
DATA_ANALYZER = "data_analyzer"
INTERACTION_CHECKER = "interaction_checker"
GENOMIC_ANALYST = "genomic_analyst"
CONSULT_NOTE_ANALYZER = "consult_note_analyzer"
CONSULTATION_SYNTHESIZER = "consultation_synthesizer"
# Add new agent identifiers used in AGENT_METADATA
NOTIFIER = "notifier"
SCHEDULER = "scheduler"
REFERRAL_DRAFTER = "referral_drafter"
CLINICAL_TRIAL_FINDER = "clinical_trial_finder" # Corresponds to ClinicalTrialAgent
SIDE_EFFECT_MANAGER = "side_effect_manager" # Corresponds to SideEffectAgent
COMPARATIVE_THERAPIST = "comparative_therapist" # Corresponds to ComparativeTherapyAgent
PATIENT_EDUCATOR = "patient_educator" # Corresponds to PatientEducationDraftAgent
COMPARATIVE_THERAPY_AGENT = "comparative_therapy_agent" # Conceptual
INTEGRATIVE_MEDICINE_AGENT = "integrative_medicine_agent" # Conceptual
CRISPR_AGENT = "crispr_agent" # Conceptual/Future
EVO2_AGENT = "evo2_agent" # Conceptual VEP simulation

# New Agent for Lab Orders
DRAFT_LAB_ORDER_COMMAND = "draft_lab_order" # Command name
LAB_ORDER_AGENT = "lab_order_agent" # Agent key

# New Command for Clinical Trial Matching
MATCH_ELIGIBLE_TRIALS_COMMAND = "match_eligible_trials"
CLINICAL_TRIAL_AGENT = "clinical_trial_agent" # Agent key for ClinicalTrialAgent

# Agent Statuses
AGENT_STATUS_IDLE = "idle"

# Supported Intents Mapping (Example, adjust as needed)
# This set should ideally include all intents the system might try to parse.
SUPPORTED_INTENTS = {
    SUMMARIZE,
    CHECK_INTERACTIONS,
    ANALYZE_GENOMIC_PROFILE,
    SUMMARIZE_DEEP_DIVE,
    ANALYZE_INITIATOR_NOTE,
    SYNTHESIZE_CONSULTATION,
    NOTIFY,
    SCHEDULE,
    REFERRAL,
    FIND_TRIALS,
    MANAGE_SIDE_EFFECTS
} 