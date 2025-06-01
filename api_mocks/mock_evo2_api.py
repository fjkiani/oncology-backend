"""
Mock API for simulating responses from an Evo2-like Variant Effect Predictor.
For V1.5 development of the GenomicAnalystAgent.
"""
from typing import Dict, Any, Optional
import random # For generating mock scores
import re

# More structured known variant classifications
# Now includes mock predictive scores and knowledgebase entries
KNOWN_VARIANT_CLASSIFICATIONS = {
    "BRAF": {
        "V600E": {
            "classification": "PATHOGENIC_BY_RULE",
            "reasoning": "BRAF V600E is a well-known activating mutation targeted by multiple therapies.",
            "consequence": "missense_variant",
            "sift": "deleterious (mock)",
            "polyphen": "probably_damaging (mock)",
            "clinvar": "Pathogenic (mock)",
            "oncokb": "Level 1 (mock)",
            "genomic_coordinate_hg38": "chr7:140753336A>T",
        },
        "V600K": {
            "classification": "PATHOGENIC_BY_RULE",
            "reasoning": "BRAF V600K is a known activating mutation.",
            "consequence": "missense_variant",
            "sift": "deleterious (mock)",
            "polyphen": "probably_damaging (mock)",
            "clinvar": "Pathogenic (mock)",
            "oncokb": "Level 1 (mock)",
            "genomic_coordinate_hg38": "chr7:140753336A>C",
        },
        "K601E": {
            "classification": "PATHOGENIC_BY_RULE",
            "reasoning": "BRAF K601E is a less common but known activating mutation in the kinase domain.",
            "consequence": "missense_variant",
            "sift": "deleterious (mock)",
            "polyphen": "probably_damaging (mock)",
            "clinvar": "Pathogenic (mock)",
            "oncokb": "Level 3A (mock)",
            "genomic_coordinate_hg38": "chr7:140753341A>G",
        },
        "G469A": {
            "classification": "PATHOGENIC_BY_RULE",
            "reasoning": "BRAF G469A occurs in the P-loop of the kinase domain and is associated with intermediate kinase activity.",
            "consequence": "missense_variant",
            "sift": "deleterious (mock)",
            "polyphen": "probably_damaging (mock)",
            "clinvar": "Pathogenic (mock)",
            "oncokb": "Level 3A (mock)",
            "genomic_coordinate_hg38": "chr7:140754175G>C",
        }
    },
    "EGFR": {
        "L858R": {
            "classification": "PATHOGENIC_BY_RULE",
            "reasoning": "EGFR L858R is a common activating mutation in lung cancer.",
            "consequence": "missense_variant",
            "sift": "deleterious (mock)",
            "polyphen": "probably_damaging (mock)",
            "clinvar": "Pathogenic (mock)",
            "oncokb": "Level A2 (mock)", # Example, might differ
            "genomic_coordinate_hg38": "chr7:55259515T>G",
        },
        "T790M": {
            "classification": "RESISTANCE_BY_RULE",
            "reasoning": "EGFR T790M is a common resistance mutation to EGFR inhibitors.",
            "consequence": "missense_variant",
            "sift": "deleterious (mock)", # Assuming it's deleterious to drug binding
            "polyphen": "possibly_damaging (mock)",
            "clinvar": "Pathogenic (mock)", # In the context of resistance
            "oncokb": "Resistance (mock)",
            "genomic_coordinate_hg38": "chr7:55249071C>T",
        },
        "exon19del": {
            "classification": "PATHOGENIC_BY_RULE",
            "reasoning": "EGFR exon 19 deletions are common activating mutations in lung cancer and predict response to EGFR TKIs.",
            "consequence": "inframe_deletion",
            "sift": "deleterious (mock)",
            "polyphen": "probably_damaging (mock)",
            "clinvar": "Pathogenic (mock)",
            "oncokb": "Level 1 (mock)",
            "genomic_coordinate_hg38": "chr7:55242465_55242479del",
        },
        "G719A": {
            "classification": "PATHOGENIC_BY_RULE",
            "reasoning": "EGFR G719A is an activating mutation in the kinase domain that predicts sensitivity to EGFR TKIs.",
            "consequence": "missense_variant",
            "sift": "deleterious (mock)",
            "polyphen": "probably_damaging (mock)",
            "clinvar": "Pathogenic (mock)",
            "oncokb": "Level 1 (mock)",
            "genomic_coordinate_hg38": "chr7:55241708G>C",
        },
        "C797S": {
            "classification": "RESISTANCE_BY_RULE",
            "reasoning": "EGFR C797S confers resistance to third-generation EGFR TKIs like osimertinib.",
            "consequence": "missense_variant",
            "sift": "deleterious (mock)",
            "polyphen": "probably_damaging (mock)",
            "clinvar": "Pathogenic (mock)",
            "oncokb": "Resistance (mock)",
            "genomic_coordinate_hg38": "chr7:55249092G>C",
        }
    },
    "TP53": {
        "R248Q": {
            "classification": "PATHOGENIC_BY_RULE",
            "reasoning": "TP53 R248Q is a known hotspot oncogenic mutation.",
            "consequence": "missense_variant",
            "sift": "deleterious (mock)",
            "polyphen": "probably_damaging (mock)",
            "clinvar": "Pathogenic (mock)",
            "oncokb": "Likely Oncogenic (mock)",
            "genomic_coordinate_hg38": "chr17:7577547G>A",
        },
        "P72R": { # Example of a potentially more common, less impactful variant
            "classification": "UNCLEAR_BY_RULE", # Or VUS_BY_RULE
            "reasoning": "TP53 P72R is a common polymorphism with unclear clinical significance in many contexts.",
            "consequence": "missense_variant",
            "sift": "tolerated (mock)",
            "polyphen": "benign (mock)",
            "clinvar": "Benign/Likely_Benign (mock)",
            "oncokb": "N/A (mock)",
            "genomic_coordinate_hg38": "chr17:7579472G>C",
        },
        "R175H": {
            "classification": "PATHOGENIC_BY_RULE",
            "reasoning": "TP53 R175H is a common oncogenic mutation that disrupts the DNA binding domain and abolishes p53 function.",
            "consequence": "missense_variant",
            "sift": "deleterious (mock)",
            "polyphen": "probably_damaging (mock)",
            "clinvar": "Pathogenic (mock)",
            "oncokb": "Oncogenic (mock)",
            "genomic_coordinate_hg38": "chr17:7578406G>A",
        },
        "Y220C": {
            "classification": "PATHOGENIC_BY_RULE",
            "reasoning": "TP53 Y220C destabilizes the protein structure and is a potential target for small molecule reactivation.",
            "consequence": "missense_variant",
            "sift": "deleterious (mock)",
            "polyphen": "probably_damaging (mock)",
            "clinvar": "Pathogenic (mock)",
            "oncokb": "Likely Oncogenic (mock)",
            "genomic_coordinate_hg38": "chr17:7578190A>G",
        }
    },
    "KRAS": {
        "G12C": {
            "classification": "PATHOGENIC_BY_RULE",
            "reasoning": "KRAS G12C is an oncogenic mutation with targeted therapies available.",
            "consequence": "missense_variant",
            "sift": "deleterious (mock)",
            "polyphen": "probably_damaging (mock)",
            "clinvar": "Pathogenic (mock)",
            "oncokb": "Level 2B (mock)",
            "genomic_coordinate_hg38": "chr12:25398284C>A",
        },
         "G12D": {
            "classification": "PATHOGENIC_BY_RULE",
            "reasoning": "KRAS G12D is a common oncogenic mutation.",
            "consequence": "missense_variant",
            "sift": "deleterious (mock)",
            "polyphen": "probably_damaging (mock)",
            "clinvar": "Pathogenic (mock)",
            "oncokb": "Likely Oncogenic (mock)",
            "genomic_coordinate_hg38": "chr12:25398284C>T",
        },
        "G12V": {
            "classification": "PATHOGENIC_BY_RULE",
            "reasoning": "KRAS G12V is a common oncogenic mutation that locks KRAS in the active state.",
            "consequence": "missense_variant",
            "sift": "deleterious (mock)",
            "polyphen": "probably_damaging (mock)",
            "clinvar": "Pathogenic (mock)",
            "oncokb": "Oncogenic (mock)",
            "genomic_coordinate_hg38": "chr12:25398284C>A",
        },
        "Q61H": {
            "classification": "PATHOGENIC_BY_RULE",
            "reasoning": "KRAS Q61H impairs GTP hydrolysis, leading to constitutive activation.",
            "consequence": "missense_variant",
            "sift": "deleterious (mock)",
            "polyphen": "probably_damaging (mock)",
            "clinvar": "Pathogenic (mock)",
            "oncokb": "Oncogenic (mock)",
            "genomic_coordinate_hg38": "chr12:25380275T>G",
        }
    },
    "PIK3CA": {
        "E545K": {
            "classification": "PATHOGENIC_BY_RULE",
            "reasoning": "PIK3CA E545K is a common activating mutation in the helical domain seen in multiple cancer types.",
            "consequence": "missense_variant",
            "sift": "deleterious (mock)",
            "polyphen": "probably_damaging (mock)",
            "clinvar": "Pathogenic (mock)",
            "oncokb": "Oncogenic (mock)",
            "genomic_coordinate_hg38": "chr3:178936091G>A",
        },
        "H1047R": {
            "classification": "PATHOGENIC_BY_RULE",
            "reasoning": "PIK3CA H1047R is a common activating mutation in the kinase domain and may predict sensitivity to PI3K inhibitors.",
            "consequence": "missense_variant",
            "sift": "deleterious (mock)",
            "polyphen": "probably_damaging (mock)",
            "clinvar": "Pathogenic (mock)",
            "oncokb": "Oncogenic (mock)",
            "genomic_coordinate_hg38": "chr3:178952085A>G",
        }
    },
    "BRCA1": {
        "185delAG": {
            "classification": "PATHOGENIC_BY_RULE",
            "reasoning": "BRCA1 185delAG (c.68_69delAG) is a founder mutation that leads to a truncated, non-functional protein.",
            "consequence": "frameshift_variant",
            "sift": "deleterious (mock)",
            "polyphen": "probably_damaging (mock)",
            "clinvar": "Pathogenic (mock)",
            "oncokb": "Oncogenic (mock)",
            "genomic_coordinate_hg38": "chr17:43124027_43124028delAG",
        },
        "C61G": {
            "classification": "PATHOGENIC_BY_RULE",
            "reasoning": "BRCA1 C61G disrupts the RING domain, impairing E3 ubiquitin ligase activity.",
            "consequence": "missense_variant",
            "sift": "deleterious (mock)",
            "polyphen": "probably_damaging (mock)",
            "clinvar": "Pathogenic (mock)",
            "oncokb": "Oncogenic (mock)",
            "genomic_coordinate_hg38": "chr17:43104911T>G",
        }
    },
    "BRCA2": {
        "6174delT": {
            "classification": "PATHOGENIC_BY_RULE",
            "reasoning": "BRCA2 6174delT is a founder mutation that results in a truncated, non-functional protein.",
            "consequence": "frameshift_variant",
            "sift": "deleterious (mock)",
            "polyphen": "probably_damaging (mock)",
            "clinvar": "Pathogenic (mock)",
            "oncokb": "Oncogenic (mock)",
            "genomic_coordinate_hg38": "chr13:32340301delT",
        }
    },
    "IDH1": {
        "R132H": {
            "classification": "PATHOGENIC_BY_RULE",
            "reasoning": "IDH1 R132H is a common mutation in gliomas that alters enzyme function to produce the oncometabolite 2-HG.",
            "consequence": "missense_variant",
            "sift": "deleterious (mock)",
            "polyphen": "probably_damaging (mock)",
            "clinvar": "Pathogenic (mock)",
            "oncokb": "Oncogenic (mock)",
            "genomic_coordinate_hg38": "chr2:208248389C>T",
        }
    },
    "DEFAULT_PATHOGENIC": { # Generic response for "pathogenic" queries if specific variant unknown
        "classification": "PREDICTED_PATHOGENIC_BY_MOCK_EVO2",
        "reasoning": "Mock Evo2 simulation based on query intent (pathogenic). Specific evidence not found in mock DB.",
        "consequence": "unknown",
        "sift": "unknown (mock)",
        "polyphen": "unknown (mock)",
        "clinvar": "Uncertain significance (mock)",
        "oncokb": "N/A (mock)",
    },
    "DEFAULT_BENIGN": {
        "classification": "PREDICTED_BENIGN_BY_MOCK_EVO2",
        "reasoning": "Mock Evo2 simulation based on query intent (benign). Specific evidence not found in mock DB.",
        "consequence": "unknown",
        "sift": "tolerated (mock)", # Assuming default benign aligns with tolerated
        "polyphen": "benign (mock)",
        "clinvar": "Benign (mock)",
        "oncokb": "N/A (mock)",
    },
    "DEFAULT_VUS": {
        "classification": "UNCLEAR_BY_MOCK_EVO2", # Changed from PREDICTED_VUS
        "reasoning": "Mock Evo2 simulation: variant significance is unclear or not found in mock DB.",
        "consequence": "unknown",
        "sift": "unknown (mock)",
        "polyphen": "unknown (mock)",
        "clinvar": "Uncertain significance (mock)",
        "oncokb": "N/A (mock)",
    }
}

# More detailed general variant type effects for the mock API
GENERAL_VARIANT_TYPE_EFFECTS_MOCK = {
    "Nonsense_Mutation": {
        "classification": "LIKELY_PATHOGENIC_BY_MOCK_EVO2", # Often disruptive
        "reasoning": "Nonsense mutations typically lead to truncated proteins, often causing loss of function.",
        "consequence": "nonsense",
        "sift": "deleterious (mock)",
        "polyphen": "probably_damaging (mock)",
        "clinvar": "Likely Pathogenic (mock general)",
        "oncokb": "Likely Oncogenic (mock general)",
    },
    "Frameshift_Variant": { # Covers Frame_Shift_Del and Frame_Shift_Ins
        "classification": "LIKELY_PATHOGENIC_BY_MOCK_EVO2", # Often highly disruptive
        "reasoning": "Frameshift mutations alter the reading frame, usually resulting in a non-functional protein.",
        "consequence": "frameshift_variant",
        "sift": "deleterious (mock)",
        "polyphen": "probably_damaging (mock)",
        "clinvar": "Likely Pathogenic (mock general)",
        "oncokb": "Likely Oncogenic (mock general)",
    },
    "Splice_Site": { # Covers Splice_Site_SNP, Splice_Site_Del, Splice_Site_Ins
        "classification": "LIKELY_PATHOGENIC_BY_MOCK_EVO2", # Often disruptive
        "reasoning": "Splice site mutations can lead to aberrant splicing and non-functional proteins.",
        "consequence": "splice_acceptor_variant or splice_donor_variant", # Could be more specific
        "sift": "deleterious (mock)",
        "polyphen": "probably_damaging (mock)",
        "clinvar": "Likely Pathogenic (mock general)",
        "oncokb": "Likely Oncogenic (mock general)",
    },
    "Missense_Mutation": { # Default for missense if not specifically known
        "classification": "UNCLEAR_BY_MOCK_EVO2", # Missense effects are highly variable
        "reasoning": "The impact of a missense mutation is variable and requires specific evidence for classification.",
        "consequence": "missense_variant",
        "sift": "unknown (mock)",
        "polyphen": "unknown (mock)",
        "clinvar": "Uncertain significance (mock general)",
        "oncokb": "N/A (mock general)",
    },
    "In_Frame_Del": { # In-frame deletions
        "classification": "UNCLEAR_BY_MOCK_EVO2", # Effect can vary
        "reasoning": "In-frame deletions remove amino acids without changing the reading frame; impact is variable.",
        "consequence": "inframe_deletion",
        "sift": "unknown (mock)",
        "polyphen": "unknown (mock)",
        "clinvar": "Uncertain significance (mock general)",
        "oncokb": "N/A (mock general)",
    },
     "In_Frame_Ins": { # In-frame insertions
        "classification": "UNCLEAR_BY_MOCK_EVO2", # Effect can vary
        "reasoning": "In-frame insertions add amino acids without changing the reading frame; impact is variable.",
        "consequence": "inframe_insertion",
        "sift": "unknown (mock)",
        "polyphen": "unknown (mock)",
        "clinvar": "Uncertain significance (mock general)",
        "oncokb": "N/A (mock general)",
    },
    # Add other general types if needed, e.g., Silent, 5'UTR, 3'UTR
    "Silent": {
        "classification": "LIKELY_BENIGN_BY_MOCK_EVO2",
        "reasoning": "Silent mutations do not change the amino acid sequence, generally considered benign.",
        "consequence": "synonymous_variant",
        "sift": "tolerated (mock)",
        "polyphen": "benign (mock)",
        "clinvar": "Benign (mock general)",
        "oncokb": "N/A (mock general)",
    },
     "DEFAULT": { # Fallback for unknown variant types
        "classification": "UNCLEAR_BY_MOCK_EVO2",
        "reasoning": "Variant type effect not specifically modeled in Mock Evo2.",
        "consequence": "unknown",
        "sift": "unknown (mock)",
        "polyphen": "unknown (mock)",
        "clinvar": "Uncertain significance (mock general)",
        "oncokb": "N/A (mock general)",
    }
}

def get_variant_effect_mock(gene_symbol: str, variant_query: str, variant_type: Optional[str] = None, query_intent: Optional[str] = None):
    """
    Simulates a call to an Evo2-like API for variant effect prediction.
    Returns a more structured dictionary with detailed mock information.

    Args:
        gene_symbol: The gene symbol (e.g., "BRAF").
        variant_query: The variant information (e.g., "V600E", "p.V600E", "c.1799T>A").
                       Can also be a general type like "Nonsense_Mutation".
        variant_type: The type of variant from MAF/VCF if available (e.g., "Missense_Mutation", "Nonsense_Mutation").
                      This helps in classifying generic types if specific variant is not in KNOWN_VARIANT_CLASSIFICATIONS.
        query_intent: An optional parameter indicating the desired outcome from the query
                      (e.g., "pathogenic", "benign"). Used for fallback if variant is unknown.

    Returns:
        A dictionary containing the simulated VEP details.
    """
    # Normalize protein change format if present (e.g., p.V600E -> V600E)
    protein_change_match = re.match(r"^[pP]\\.([A-Za-z0-9*]+)$", variant_query)
    normalized_variant_query = protein_change_match.group(1) if protein_change_match else variant_query
    
    data_source = "BeatCancer_MockEvo2_v1.5.1" # Versioning this mock

    base_response = {
        "input_variant_query": variant_query,
        "gene_symbol": gene_symbol,
        "protein_change": normalized_variant_query if not variant_query.endswith(("_Mutation", "_Variant")) and not variant_query == "Splice_Site" else None, # Store normalized p. change
        "canonical_variant_id": f"{gene_symbol}:p.{normalized_variant_query}" if protein_change_match else f"{gene_symbol}:{variant_query}",
        "variant_type_from_input": variant_type,
        "data_source": data_source,
        # Default values, to be overridden
        "simulated_classification": "UNCLEAR_BY_MOCK_EVO2",
        "classification_reasoning": "Variant not found in mock DB and no clear type/intent match.",
        "predicted_consequence": "unknown",
        "simulated_tools": {"sift": "unknown (mock)", "polyphen": "unknown (mock)"},
        "mock_knowledgebases": {"clinvar_significance": "Uncertain significance (mock)", "oncokb_level": "N/A (mock)"},
        "genomic_coordinate_hg38": None,  # Default to None, will be populated if available
        # --- New Evo2-like fields ---
        "delta_likelihood_score": None,
        "evo2_prediction": None,
        "evo2_confidence": None
    }

    # Helper function to generate Evo2-like scores based on classification
    def _generate_mock_evo2_scores(classification: str) -> Dict[str, Any]:
        score_output = {
            "delta_likelihood_score": None,
            "evo2_prediction": None,
            "evo2_confidence": None
        }
        if "PATHOGENIC" in classification or "RESISTANCE_BY_RULE" in classification:
            score_output["delta_likelihood_score"] = round(random.uniform(-0.005, -0.0005), 7)
            score_output["evo2_confidence"] = round(random.uniform(0.65, 0.98), 2)
        elif "BENIGN" in classification:
            score_output["delta_likelihood_score"] = round(random.uniform(-0.0001, 0.0001), 7)
            score_output["evo2_confidence"] = round(random.uniform(0.65, 0.98), 2)
        elif "UNCLEAR" in classification or "VUS" in classification:
            score_output["delta_likelihood_score"] = round(random.uniform(-0.0005, 0.0005), 7)
            score_output["evo2_confidence"] = round(random.uniform(0.3, 0.6), 2)
        
        if score_output["delta_likelihood_score"] is not None:
            if "RESISTANCE_BY_RULE" in classification:
                 score_output["evo2_prediction"] = "Resistance predicted (mock Evo2)"
            elif score_output["delta_likelihood_score"] < -0.0003:
                score_output["evo2_prediction"] = "Likely pathogenic (mock Evo2)"
            else:
                score_output["evo2_prediction"] = "Likely benign (mock Evo2)"
        return score_output

    # 1. Check known specific variants
    if gene_symbol in KNOWN_VARIANT_CLASSIFICATIONS and \
       normalized_variant_query in KNOWN_VARIANT_CLASSIFICATIONS[gene_symbol]:
        details = KNOWN_VARIANT_CLASSIFICATIONS[gene_symbol][normalized_variant_query]
        base_response.update({
            "simulated_classification": details["classification"],
            "classification_reasoning": details["reasoning"],
            "predicted_consequence": details.get("consequence", "unknown"),
            "simulated_tools": {
                "sift": details.get("sift", "unknown (mock)"),
                "polyphen": details.get("polyphen", "unknown (mock)")
            },
            "mock_knowledgebases": {
                "clinvar_significance": details.get("clinvar", "Uncertain significance (mock)"),
                "oncokb_level": details.get("oncokb", "N/A (mock)")
            }
        })
        # Add genomic coordinates if they exist in the details
        if "genomic_coordinate_hg38" in details:
            base_response["genomic_coordinate_hg38"] = details["genomic_coordinate_hg38"]
            
        # Ensure protein_change is correctly set for specific variant matches
        base_response["protein_change"] = normalized_variant_query
        base_response["canonical_variant_id"] = f"{gene_symbol}:p.{normalized_variant_query}"
        
        # Add Evo2 scores
        evo2_scores = _generate_mock_evo2_scores(base_response["simulated_classification"])
        base_response.update(evo2_scores)
        return base_response

    # 2. Check general variant types (if specific variant not found or query is a type)
    type_to_check = None
    # Handle if variant_query itself is a general type name (e.g. "Nonsense_Mutation")
    # or if variant_type (from patient data) indicates a general type.
    # Prioritize explicit query if it matches a general type name.
    if variant_query in GENERAL_VARIANT_TYPE_EFFECTS_MOCK:
        type_to_check = variant_query
    elif variant_type:
        if variant_type in GENERAL_VARIANT_TYPE_EFFECTS_MOCK:
            type_to_check = variant_type
        elif "Frame_Shift" in variant_type:
            type_to_check = "Frameshift_Variant"
        elif "Splice_Site" in variant_type: # Covers Splice_Site_SNP, Splice_Site_Del, Splice_Site_Ins etc.
            type_to_check = "Splice_Site"
        # Add more general mappings if necessary, e.g. for In_Frame_Ins, In_Frame_Del if they are not direct keys

    if type_to_check and type_to_check in GENERAL_VARIANT_TYPE_EFFECTS_MOCK:
        details = GENERAL_VARIANT_TYPE_EFFECTS_MOCK[type_to_check]
        base_response.update({
            "simulated_classification": details["classification"],
            "classification_reasoning": details["reasoning"],
            "predicted_consequence": details.get("consequence", base_response["predicted_consequence"]), # Keep "unknown" if not specified
            "simulated_tools": {
                "sift": details.get("sift", base_response["simulated_tools"]["sift"]),
                "polyphen": details.get("polyphen", base_response["simulated_tools"]["polyphen"])
            },
            "mock_knowledgebases": {
                "clinvar_significance": details.get("clinvar", base_response["mock_knowledgebases"]["clinvar_significance"]),
                "oncokb_level": details.get("oncokb", base_response["mock_knowledgebases"]["oncokb_level"])
            }
        })
        # If the original query was just a type, or if it's a general classification
        # protein_change should be None and canonical_variant_id should reflect the type.
        base_response["protein_change"] = None
        base_response["canonical_variant_id"] = f"{gene_symbol}:{type_to_check}"
        
        # Add Evo2 scores
        evo2_scores = _generate_mock_evo2_scores(base_response["simulated_classification"])
        base_response.update(evo2_scores)
        return base_response
    
    # 3. Use query_intent as a fallback if no specific match and not a general type match
    intent_reasoning_prefix = "Mock Evo2 simulation based on query intent"
    if query_intent:
        default_details_key = None
        if query_intent == "pathogenic":
            default_details_key = "DEFAULT_PATHOGENIC"
        elif query_intent == "benign":
            default_details_key = "DEFAULT_BENIGN"
        # Add other intents like "resistance", "activating" if needed and map to a default VUS or specific default.
        
        if default_details_key and default_details_key in KNOWN_VARIANT_CLASSIFICATIONS:
            details = KNOWN_VARIANT_CLASSIFICATIONS[default_details_key]
            base_response.update({
                "simulated_classification": details["classification"],
                "classification_reasoning": f"{intent_reasoning_prefix} ('{query_intent}'). {details['reasoning']}",
                "predicted_consequence": details.get("consequence", base_response["predicted_consequence"]),
                "simulated_tools": {
                    "sift": details.get("sift", base_response["simulated_tools"]["sift"]),
                    "polyphen": details.get("polyphen", base_response["simulated_tools"]["polyphen"])
                },
                "mock_knowledgebases": {
                    "clinvar_significance": details.get("clinvar", base_response["mock_knowledgebases"]["clinvar_significance"]),
                    "oncokb_level": details.get("oncokb", base_response["mock_knowledgebases"]["oncokb_level"])
                }
            })
            # For intent-based, protein_change is not applicable from the query itself
            base_response["protein_change"] = None 
            base_response["canonical_variant_id"] = f"{gene_symbol}:IntentBasedQuery-{query_intent}"
            
            # Add Evo2 scores
            evo2_scores = _generate_mock_evo2_scores(base_response["simulated_classification"])
            base_response.update(evo2_scores)
            return base_response

    # 4. If still no match, use the default VUS / Unclear
    # Decide if it's an unknown specific variant (use DEFAULT_VUS from KNOWN_VARIANT_CLASSIFICATIONS)
    # or an unrecognized general type (use DEFAULT from GENERAL_VARIANT_TYPE_EFFECTS_MOCK)
    
    # Heuristic: if variant_query contains typical protein change patterns (letters and numbers)
    # or was normalized from p. notation, it's likely an attempt at a specific variant.
    is_likely_specific_variant_attempt = bool(protein_change_match or re.search(r"[A-Za-z]\d+[A-Za-z*]$", normalized_variant_query))

    if is_likely_specific_variant_attempt:
        final_default_details = KNOWN_VARIANT_CLASSIFICATIONS["DEFAULT_VUS"]
        # protein_change and canonical_variant_id are already set from initial base_response for specific variants
    else: # Likely an unrecognized general type query or other non-specific query
        final_default_details = GENERAL_VARIANT_TYPE_EFFECTS_MOCK["DEFAULT"]
        base_response["protein_change"] = None # Not a specific protein change
        base_response["canonical_variant_id"] = f"{gene_symbol}:{variant_query}" # Use original query as part of ID


    base_response.update({
        "simulated_classification": final_default_details["classification"],
        "classification_reasoning": final_default_details["reasoning"],
        "predicted_consequence": final_default_details.get("consequence", base_response["predicted_consequence"]),
        "simulated_tools": {
            "sift": final_default_details.get("sift", base_response["simulated_tools"]["sift"]),
            "polyphen": final_default_details.get("polyphen", base_response["simulated_tools"]["polyphen"])
        },
        "mock_knowledgebases": {
            "clinvar_significance": final_default_details.get("clinvar", base_response["mock_knowledgebases"]["clinvar_significance"]),
            "oncokb_level": final_default_details.get("oncokb", base_response["mock_knowledgebases"]["oncokb_level"])
        }
    })
    # Add Evo2 scores for the final default case
    evo2_scores = _generate_mock_evo2_scores(base_response["simulated_classification"])
    base_response.update(evo2_scores)
    return base_response

# Example Usage:
if __name__ == '__main__':
    print("--- Specific Known Variant ---")
    print(get_variant_effect_mock("BRAF", "V600E", "Missense_Mutation"))
    print("\n--- Specific Known Variant (p. notation) ---")
    print(get_variant_effect_mock("TP53", "p.R248Q", "Missense_Mutation"))
    print("\n--- Unknown Specific Missense Variant (likely specific attempt) ---")
    print(get_variant_effect_mock("ABC", "X123Y", "Missense_Mutation"))
    print("\n--- Known General Variant Type (query is the type) ---")
    print(get_variant_effect_mock("TP53", "Nonsense_Mutation", "Nonsense_Mutation"))
    print("\n--- General Variant Type from patient data (novel specific variant, falls back to type) ---")
    print(get_variant_effect_mock("BRCA1", "some_novel_variant_text", "Nonsense_Mutation"))
    print("\n--- Frameshift from patient data (compound type) ---")
    print(get_variant_effect_mock("BRCA2", "c.123delA_someframeshift", "Frame_Shift_Del"))
    print("\n--- Silent Mutation from patient data (known general type) ---")
    print(get_variant_effect_mock("XYZ", "A10A", "Silent")) # Assuming A10A is a specific variant query
    print("\n--- Query with Intent (Pathogenic) ---")
    print(get_variant_effect_mock("SOMEGENE", "SomeUnknownVar", "Missense_Mutation", query_intent="pathogenic"))
    print("\n--- Query with Intent (Benign) ---")
    print(get_variant_effect_mock("OTHERGENE", "AnotherUnknown", "Missense_Mutation", query_intent="benign"))
    print("\n--- Completely Unknown Type/Query (not matching specific patterns or general types) ---")
    print(get_variant_effect_mock("ANO1", "MysteriousChangeType", "Intronic_Variant")) # Intronic_Variant not in GENERAL_VARIANT_TYPE_EFFECTS_MOCK
    print("\n--- EGFR T790M (Resistance by rule) ---")
    print(get_variant_effect_mock("EGFR", "T790M", "Missense_Mutation"))
    print("\n--- TP53 P72R (Unclear by rule) ---")
    print(get_variant_effect_mock("TP53", "p.P72R", "Missense_Mutation")) 