# Example usage:
# from mock_patient_data_dict import mock_patient_data_dict
# patient_data = mock_patient_data_dict.get("PAT12345", None)

mock_patient_data_dict = {
    "PAT12345": {
        "id": "PAT12345",
        "demographics": {
            "first_name": "John",
            "last_name": "Doe",
            "dob": "1978-05-15",
            "gender": "Male",
            "race": "White",
            "ethnicity": "Non-Hispanic",
            "insurance": "Medicare"
        },
        "diagnosis": {
            "primary": "Metastatic Melanoma",
            "stage": "IV",
            "date_of_diagnosis": "2022-10-05",
            "histology": "Malignant melanoma, not otherwise specified",
            "grade": "3",
            "sites": ["Skin - Back", "Liver", "Lung"]
        },
        "mutations": [
            {
                "hugo_gene_symbol": "BRAF",
                "protein_change": "V600E",
                "variant_type": "Missense_Mutation",
                "genomic_coordinate_hg38": "chr7:140753336A>T",
                "allele_frequency": 0.45,
                "mutation_id": "MU100001"
            },
            {
                "hugo_gene_symbol": "EGFR",
                "protein_change": "T790M",
                "variant_type": "Missense_Mutation",
                "genomic_coordinate_hg38": "chr7:55249071C>T",
                "allele_frequency": 0.28,
                "mutation_id": "MU100002"
            },
            {
                "hugo_gene_symbol": "TP53",
                "protein_change": "R248Q",
                "variant_type": "Missense_Mutation",
                "genomic_coordinate_hg38": "chr17:7577547G>A",
                "allele_frequency": 0.39,
                "mutation_id": "MU100003"
            }
        ],
        "treatments": [
            {
                "name": "Dabrafenib",
                "type": "Targeted Therapy",
                "start_date": "2022-11-01",
                "end_date": "2023-07-01",
                "response": "Partial Response",
                "notes": "Initial response followed by progression"
            },
            {
                "name": "Trametinib",
                "type": "Targeted Therapy",
                "start_date": "2022-11-01",
                "end_date": "2023-07-01",
                "response": "Partial Response",
                "notes": "Given in combination with Dabrafenib"
            },
            {
                "name": "Pembrolizumab",
                "type": "Immunotherapy",
                "start_date": "2023-07-15",
                "end_date": None,
                "response": "Stable Disease",
                "notes": "Ongoing treatment"
            }
        ],
        "currentMedications": [
            {
                "name": "Pembrolizumab",
                "type": "Immunotherapy",
                "status": "active"
            }
        ],
        "vitals": {
            "latest": {
                "date": "2023-08-01",
                "height_cm": 180,
                "weight_kg": 75.5,
                "bmi": 23.3,
                "bp_systolic": 128,
                "bp_diastolic": 82,
                "temperature_c": 36.8,
                "heart_rate": 72,
                "respiratory_rate": 16,
                "pain_score": 2
            },
            "history": [
                {
                    "date": "2023-07-01",
                    "weight_kg": 77.1,
                    "bmi": 23.8,
                    "bp_systolic": 135,
                    "bp_diastolic": 84,
                    "heart_rate": 78
                },
                {
                    "date": "2023-06-01",
                    "weight_kg": 78.2,
                    "bmi": 24.1,
                    "bp_systolic": 138,
                    "bp_diastolic": 86,
                    "heart_rate": 76
                }
            ]
        },
        "labs": [
            {
                "date": "2023-08-01",
                "test": "CBC",
                "results": {
                    "WBC": {"value": 5.2, "unit": "10^9/L", "reference_range": "4.0-11.0", "status": "normal"},
                    "RBC": {"value": 4.5, "unit": "10^12/L", "reference_range": "4.5-5.9", "status": "normal"},
                    "Hgb": {"value": 13.8, "unit": "g/dL", "reference_range": "13.5-17.5", "status": "normal"},
                    "Hct": {"value": 41.2, "unit": "%", "reference_range": "41.0-53.0", "status": "normal"},
                    "Platelets": {"value": 210, "unit": "10^9/L", "reference_range": "150-450", "status": "normal"}
                }
            },
            {
                "date": "2023-08-01",
                "test": "CMP",
                "results": {
                    "Glucose": {"value": 95, "unit": "mg/dL", "reference_range": "70-99", "status": "normal"},
                    "BUN": {"value": 15, "unit": "mg/dL", "reference_range": "7-20", "status": "normal"},
                    "Creatinine": {"value": 0.9, "unit": "mg/dL", "reference_range": "0.6-1.2", "status": "normal"},
                    "ALT": {"value": 35, "unit": "U/L", "reference_range": "7-56", "status": "normal"},
                    "AST": {"value": 29, "unit": "U/L", "reference_range": "10-40", "status": "normal"}
                }
            },
            {
                "date": "2023-08-01",
                "test": "LDH",
                "results": {
                    "LDH": {"value": 225, "unit": "U/L", "reference_range": "140-280", "status": "normal"}
                }
            }
        ],
        "notes": [
            {
                "date": "2023-08-01",
                "author": "Dr. Johnson",
                "type": "Progress Note",
                "content": "Patient reports fatigue (grade 1) but otherwise tolerating pembrolizumab well. Imaging shows stable disease with no new lesions. Continue current treatment plan and monitor."
            },
            {
                "date": "2023-07-15",
                "author": "Dr. Johnson",
                "type": "Treatment Note",
                "content": "After progression on BRAF/MEK inhibitor therapy, starting pembrolizumab today. Discussed potential immune-related adverse events with patient. Biopsy of new liver lesion reveals EGFR T790M mutation, which may contribute to BRAF inhibitor resistance."
            },
            {
                "date": "2023-07-01",
                "author": "Dr. Johnson",
                "type": "Imaging Note",
                "content": "CT scan shows new liver lesions concerning for progression. Discontinuing dabrafenib/trametinib. Will perform biopsy of liver lesion to assess for resistance mechanisms and consider second-line options."
            }
        ],
        "clinical_trials": {
            "eligible": [
                {
                    "nct_id": "NCT12345678",
                    "title": "Pembrolizumab + Novel Agent XR-768 for Advanced Melanoma",
                    "phase": "Phase 2",
                    "status": "Recruiting",
                    "match_reason": "BRAF V600 mutation, prior progression on BRAF/MEK therapy"
                },
                {
                    "nct_id": "NCT87654321",
                    "title": "Targeting EGFR-Mediated Resistance in BRAF-Mutant Melanoma",
                    "phase": "Phase 1/2",
                    "status": "Recruiting",
                    "match_reason": "BRAF V600E + EGFR T790M mutations"
                }
            ]
        }
    },
    "PAT12344": {
        "id": "PAT12344",
        "demographics": {
            "first_name": "Jane",
            "last_name": "Smith",
            "dob": "1980-01-20",
            "gender": "Female",
            "race": "Asian",
            "ethnicity": "Hispanic",
            "insurance": "PPO"
        },
    "diagnosis": {
            "primary": "Metastatic Melanoma",
            "stage": "IV",
            "date_of_diagnosis": "2022-10-05",
            "histology": "Malignant melanoma, not otherwise specified",
            "grade": "3",
            "sites": ["Skin - Back", "Liver", "Lung"]
        },
        "mutations": [
            {
                "hugo_gene_symbol": "BRAF",
                "protein_change": "V600E",
                "variant_type": "Missense_Mutation",
                "genomic_coordinate_hg38": "chr7:140753336A>T",
                "allele_frequency": 0.45,
                "mutation_id": "MU100001"
            },
            {
                "hugo_gene_symbol": "EGFR",
                "protein_change": "T790M",
                "variant_type": "Missense_Mutation",
                "genomic_coordinate_hg38": "chr7:55249071C>T",
                "allele_frequency": 0.28,
                "mutation_id": "MU100002"
            },
            {
                "hugo_gene_symbol": "TP53",
                "protein_change": "R248Q",
                "variant_type": "Missense_Mutation",
                "genomic_coordinate_hg38": "chr17:7577547G>A",
                "allele_frequency": 0.39,
                "mutation_id": "MU100003"
            }
        ],
        "treatments": [
            {
                "name": "Dabrafenib",
                "type": "Targeted Therapy",
                "start_date": "2022-11-01",
                "end_date": "2023-07-01",
                "response": "Partial Response",
                "notes": "Initial response followed by progression"
            },
            {
                "name": "Trametinib",
                "type": "Targeted Therapy",
                "start_date": "2022-11-01",
                "end_date": "2023-07-01",
                "response": "Partial Response",
                "notes": "Given in combination with Dabrafenib"
            },
            {
                "name": "Pembrolizumab",
                "type": "Immunotherapy",
                "start_date": "2023-07-15",
                "end_date": None,
                "response": "Stable Disease",
                "notes": "Ongoing treatment"
            }
        ],
        "currentMedications": [
            {
                "name": "Pembrolizumab",
                "type": "Immunotherapy",
                "status": "active"
            }
        ],
        "vitals": {
            "latest": {
                "date": "2023-08-01",
                "height_cm": 180,
                "weight_kg": 75.5,
                "bmi": 23.3,
                "bp_systolic": 128,
                "bp_diastolic": 82,
                "temperature_c": 36.8,
                "heart_rate": 72,
                "respiratory_rate": 16,
                "pain_score": 2
            },
            "history": [
                {
                    "date": "2023-07-01",
                    "weight_kg": 77.1,
                    "bmi": 23.8,
                    "bp_systolic": 135,
                    "bp_diastolic": 84,
                    "heart_rate": 78
                },
                {
                    "date": "2023-06-01",
                    "weight_kg": 78.2,
                    "bmi": 24.1,
                    "bp_systolic": 138,
                    "bp_diastolic": 86,
                    "heart_rate": 76
                }
            ]
        },
        "labs": [
            {
                "date": "2023-08-01",
                "test": "CBC",
                "results": {
                    "WBC": {"value": 5.2, "unit": "10^9/L", "reference_range": "4.0-11.0", "status": "normal"},
                    "RBC": {"value": 4.5, "unit": "10^12/L", "reference_range": "4.5-5.9", "status": "normal"},
                    "Hgb": {"value": 13.8, "unit": "g/dL", "reference_range": "13.5-17.5", "status": "normal"},
                    "Hct": {"value": 41.2, "unit": "%", "reference_range": "41.0-53.0", "status": "normal"},
                    "Platelets": {"value": 210, "unit": "10^9/L", "reference_range": "150-450", "status": "normal"}
                }
            },
            {
                "date": "2023-08-01",
                "test": "CMP",
                "results": {
                    "Glucose": {"value": 95, "unit": "mg/dL", "reference_range": "70-99", "status": "normal"},
                    "BUN": {"value": 15, "unit": "mg/dL", "reference_range": "7-20", "status": "normal"},
                    "Creatinine": {"value": 0.9, "unit": "mg/dL", "reference_range": "0.6-1.2", "status": "normal"},
                    "ALT": {"value": 35, "unit": "U/L", "reference_range": "7-56", "status": "normal"},
                    "AST": {"value": 29, "unit": "U/L", "reference_range": "10-40", "status": "normal"}
                }
            },
            {
                "date": "2023-08-01",
                "test": "LDH",
                "results": {
                    "LDH": {"value": 225, "unit": "U/L", "reference_range": "140-280", "status": "normal"}
                }
            }
        ],
        "notes": [
            {
                "date": "2023-08-01",
                "author": "Dr. Johnson",
                "type": "Progress Note",
                "content": "Patient reports fatigue (grade 1) but otherwise tolerating pembrolizumab well. Imaging shows stable disease with no new lesions. Continue current treatment plan and monitor."
            },
            {
                "date": "2023-07-15",
                "author": "Dr. Johnson",
                "type": "Treatment Note",
                "content": "After progression on BRAF/MEK inhibitor therapy, starting pembrolizumab today. Discussed potential immune-related adverse events with patient. Biopsy of new liver lesion reveals EGFR T790M mutation, which may contribute to BRAF inhibitor resistance."
            },
            {
                "date": "2023-07-01",
                "author": "Dr. Johnson",
                "type": "Imaging Note",
                "content": "CT scan shows new liver lesions concerning for progression. Discontinuing dabrafenib/trametinib. Will perform biopsy of liver lesion to assess for resistance mechanisms and consider second-line options."
            }
        ],
        "clinical_trials": {
            "eligible": [
                {
                    "nct_id": "NCT12345678",
                    "title": "Pembrolizumab + Novel Agent XR-768 for Advanced Melanoma",
                    "phase": "Phase 2",
                    "status": "Recruiting",
                    "match_reason": "BRAF V600 mutation, prior progression on BRAF/MEK therapy"
                },
                {
                    "nct_id": "NCT87654321",
                    "title": "Targeting EGFR-Mediated Resistance in BRAF-Mutant Melanoma",
                    "phase": "Phase 1/2",
                    "status": "Recruiting",
                    "match_reason": "BRAF V600E + EGFR T790M mutations"
                }
            ]
        }
    },
    # ... other patient data ...
} 