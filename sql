-- This would be executed within a Python script using the sqlite3 module
-- to create the table in a file named, e.g., 'backend/db/trials.db'

CREATE TABLE IF NOT EXISTS clinical_trials (
    source_url TEXT PRIMARY KEY,             -- Unique URL from metadata, used as primary key
    nct_id TEXT,                             -- ClinicalTrials.gov ID (parsed)
    primary_id TEXT,                         -- Primary Sponsor ID (parsed)
    title TEXT,                              -- Trial Title (from metadata or markdown)
    status TEXT,                             -- Trial Status (parsed from markdown)
    phase TEXT,                              -- Trial Phase (parsed from markdown)
    description_text TEXT,                   -- Parsed 'Description' section
    inclusion_criteria_text TEXT,            -- Parsed 'Inclusion Criteria' section
    exclusion_criteria_text TEXT,            -- Parsed 'Exclusion Criteria' section
    objectives_text TEXT,                    -- Parsed 'Trial Objectives and Outline' section
    eligibility_text TEXT,                   -- Combined Inclusion/Exclusion text that gets embedded
    raw_markdown TEXT,                       -- Store the original markdown content
    metadata_json TEXT                       -- Store the original metadata as a JSON string
);

-- Optional: Create indices for faster lookups on commonly filtered fields
CREATE INDEX IF NOT EXISTS idx_status ON clinical_trials (status);
CREATE INDEX IF NOT EXISTS idx_phase ON clinical_trials (phase);
CREATE INDEX IF NOT EXISTS idx_nct_id ON clinical_trials (nct_id);
