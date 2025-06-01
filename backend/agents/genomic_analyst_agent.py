from typing import Dict, Any, List, Tuple, Optional, Set
import logging
import re # Import regex
from pydantic import BaseModel, Field

# Use Optional for < Python 3.10 compatibility in mock_evo2_api imports
from backend.api_mocks.mock_evo2_api import get_variant_effect_mock 
# Removed KNOWN_VARIANT_CLASSIFICATIONS and call_mock_evo2_variant_analyzer as they are not directly used by this agent after V1.5 refactor for get_variant_effect_mock

# Attempt to import the interface, handle if not found for now
try:
    from ..core.agent_interface import AgentInterface
    from ..core.llm_clients import GeminiClient # <--- Add GeminiClient import
except ImportError:
    logging.warning("AgentInterface or GeminiClient not found. Using dummy classes.")
    class AgentInterface: # type: ignore
        pass
    class GeminiClient: # type: ignore
        async def get_text_response(self, prompt: str) -> str:
            return "LLM Client not available. Summary could not be generated."

# Define Pydantic models for structured output

class SimulatedTools(BaseModel):
    sift: Optional[str] = None
    polyphen: Optional[str] = None

class MockKnowledgebases(BaseModel):
    clinvar_significance: Optional[str] = None
    oncokb_level: Optional[str] = None

class SimulatedVEPDetail(BaseModel):
    input_variant_query: str = Field(..., description="The original variant query string used for the VEP lookup.")
    gene_symbol: str
    protein_change: Optional[str] = Field(None, description="Normalized protein change, e.g., V600E.")
    canonical_variant_id: Optional[str] = Field(None, description="A canonical representation of the variant, e.g., GENE:p.Change.")
    simulated_classification: str = Field(..., description="The classification assigned by the mock VEP.")
    classification_reasoning: str = Field(..., description="The reasoning behind the mock VEP classification.")
    predicted_consequence: Optional[str] = Field(None, description="The predicted molecular consequence (e.g., missense_variant).")
    simulated_tools: Optional[SimulatedTools] = Field(None, description="Mock scores from bioinformatics tools like SIFT, PolyPhen.")
    mock_knowledgebases: Optional[MockKnowledgebases] = Field(None, description="Mock interpretations from knowledgebases like ClinVar, OncoKB.")
    variant_type_from_input: Optional[str] = Field(None, description="The variant type provided as input to the VEP lookup (e.g., from MAF).")
    data_source: Optional[str] = Field(None, description="Identifier for the data source and version of the mock VEP.")
    # --- Added for Evo2-like mock output ---
    delta_likelihood_score: Optional[float] = Field(None, description="Simulated delta likelihood score from Evo2-like analysis.")
    evo2_prediction: Optional[str] = Field(None, description="Simulated prediction from Evo2-like analysis (e.g., 'Likely pathogenic', 'Likely benign').")
    evo2_confidence: Optional[float] = Field(None, description="Simulated confidence score (0.0-1.0) for the Evo2-like prediction.")


class GeneSummaryStatus(BaseModel):
    status: str = Field(..., description="The summary status for the gene regarding the criterion (e.g., MET, NOT_MET, ACTIVATING_FOUND, PATHOGENIC_FOUND, WILD_TYPE, VUS_PRESENT, RESISTANCE_FOUND, UNCLEAR).")
    details: str = Field(default="", description="Additional details or reasoning for the gene summary status.")
    # We could add a list of relevant VEP details here if needed per gene summary,
    # but for now, all VEP details are aggregated in the main result.

class GenomicAnalysisResult(BaseModel):
    criterion_id: str
    criterion_query: str
    status: str # MET, NOT_MET, UNCLEAR, ERROR
    evidence: str # This will be a string summary for now.
    gene_summary_statuses: Dict[str, GeneSummaryStatus] = Field(default_factory=dict, description="Detailed summary status for each gene involved in the criterion.")
    simulated_vep_details: List[SimulatedVEPDetail] = Field(default_factory=list, description="Detailed results from the (mock) Variant Effect Predictor for each relevant variant.")
    clinical_significance_context: Optional[str] = Field(None, description="Contextual information about the clinical significance of findings.")
    crispr_recommendations: List[Dict[str, Any]] = Field(default_factory=list, description="List of conceptual CRISPR therapeutic recommendations based on analysis.")
    errors: List[str] = Field(default_factory=list)


class GenomicAnalystAgent(AgentInterface):
    """Agent specializing in analyzing genomic criteria using simulated VEP logic."""

    @property
    def name(self) -> str:
        return "GenomicAnalystAgent"

    @property
    def description(self) -> str:
        return "Analyzes genomic criteria by simulating variant effect prediction based on mock patient data and clinical interpretations."

    def __init__(self):
        logging.info(f"[{self.name}] Initialized (V1.5 - Enhanced Mock Evo2 Simulation).")
        
        self.known_genes = ["PIK3CA", "KRAS", "TP53", "BRCA1", "BRCA2", "AKT", "AKT1", "AKT2", "AKT3", "EGFR", "BRAF", "ERBB2", "FGFR1", "FGFR2", "FGFR3", "IDH1", "IDH2", "MET", "ALK", "ROS1", "RET", "NTRK1", "NTRK2", "NTRK3"]
        # ERBB2 is HER2
        
        self.intent_patterns = {
            'ACTIVATING_PRESENCE': [r'activating', r'oncogenic', r'gain[- ]of[- ]function', r'gof'],
            'PATHOGENIC_PRESENCE': [r'pathogenic', r'deleterious', r'loss[- ]of[- ]function', r'lof'],
            'RESISTANCE_PRESENCE': [r'resistance mutation'],
            'WILD_TYPE': [r'wild[- ]?type', r'wt', r'absence of mutation', r'no mutation', r'negative for mutation', r'unmutated', r'negative'],
            'MUTATION_PRESENCE': [r'mutation', r'variant', r'alteration', r'mutated', r'change'], # General mutation presence
            # It's tricky to have RESISTANCE_ABSENCE here as _determine_criterion_intent handles general negation.
            # We rely on the combination of a RESISTANCE keyword and overall negation.
        }
        
        self.three_letter_aa_map = {
            'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
            'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
            'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
            'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M', 'TER': '*'
        }

        self.clinical_context_map: Dict[str, Dict[str, str]] = {
            "DEFAULT": {
                "MET": "The genomic criterion was met based on the patient's mutation profile and simulated analysis.",
                "NOT_MET": "The patient's mutation profile did not meet the specific genomic criterion based on simulated analysis.",
                "UNCLEAR": "The analysis for this genomic criterion was unclear, potentially due to ambiguous query, missing data, or variants of unknown significance.",
                "ERROR": "An error occurred during the genomic analysis."
            },
            "EGFR": {
                "WILD_TYPE_TRUE": "EGFR wild-type status confirmed. This is relevant for therapies where EGFR mutations are contraindications or predict lack of response (e.g., some anti-EGFR antibodies in CRC if KRAS is also WT).",
                "ACTIVATING_MUTATION_TRUE": "Presence of an activating EGFR mutation (e.g., L858R, Exon 19 del) makes the patient potentially eligible for EGFR-targeted therapies (e.g., Osimertinib, Gefitinib) in NSCLC.",
                "RESISTANCE_MUTATION_TRUE": "Detection of an EGFR resistance mutation (like T790M) indicates likely resistance to earlier-generation EGFR inhibitors, influencing subsequent treatment choices (e.g., consider Osimertinib).",
                "PATHOGENIC_MUTATION_TRUE": "A pathogenic EGFR mutation was identified. Depending on its specific nature (activating, resistance, other), this could influence trial eligibility or therapeutic options.",
                "DEFAULT_CONTEXT": "EGFR is a key oncogene in several cancers, notably NSCLC. Its mutational status (wild-type, specific activating mutations, or resistance mutations) dictates eligibility for various targeted therapies."
            },
            "BRAF": {
                "ACTIVATING_MUTATION_TRUE": "Activating BRAF mutations (esp. V600E/K) are key biomarkers in melanoma, thyroid, NSCLC, and other cancers, indicating eligibility for BRAF/MEK inhibitor therapies.",
                "WILD_TYPE_TRUE": "BRAF wild-type status confirmed. BRAF-targeted therapies are typically not indicated based on this gene alone.",
                "DEFAULT_CONTEXT": "BRAF is a proto-oncogene; specific mutations like V600E make it a therapeutic target."
            },
            "KRAS": {
                "ACTIVATING_MUTATION_TRUE": "Specific activating KRAS mutations (e.g., G12C, G12D) are common oncogenic drivers. KRAS G12C is now targetable in some cancers (e.g., NSCLC, CRC), opening new treatment avenues and trial options.",
                "WILD_TYPE_TRUE": "KRAS wild-type status is often required for therapies targeting the EGFR pathway (like cetuximab/panitumumab) in colorectal cancer, as KRAS mutations predict resistance.",
                "DEFAULT_CONTEXT": "KRAS is one of the most frequently mutated oncogenes. Its status is critical for treatment decisions in CRC and NSCLC, among others."
            },
            "TP53": {
                "PATHOGENIC_MUTATION_TRUE": "Pathogenic TP53 mutations are very common across cancer types and are generally associated with poorer prognosis and altered response to some therapies. Their presence is a frequent factor in clinical trial design and interpretation, though rarely a direct target itself.",
                "WILD_TYPE_TRUE": "TP53 wild-type status is generally associated with better prognosis and response to certain DNA-damaging therapies compared to mutated TP53.",
                "DEFAULT_CONTEXT": "TP53 is a critical tumor suppressor gene. Mutations are common and have broad implications for cancer development and treatment response."
            },
            "ERBB2": { # HER2
                "ACTIVATING_MUTATION_TRUE": "Activating ERBB2 (HER2) mutations (distinct from amplification) can occur in lung, breast, and other cancers, making patients eligible for HER2-targeted therapies like trastuzumab deruxtecan.",
                "AMPLIFICATION_TRUE": "ERBB2 (HER2) amplification (not typically found in MAF files directly as a 'mutation', but this context is for general ERBB2 status) is a key biomarker in breast and gastric cancers for anti-HER2 therapies (trastuzumab, pertuzumab).", # Note: Agent currently doesn't detect 'amplification' from text query
                "WILD_TYPE_TRUE": "ERBB2 (HER2) wild-type status (no activating mutation or amplification) generally means HER2-targeted therapies are not indicated.",
                "DEFAULT_CONTEXT": "ERBB2 (HER2) is an important oncogene. Its status (amplification or specific mutations) is critical for targeted therapy in breast, gastric, lung, and other cancers."
            }
            # Add more genes (BRCA1/2, PIK3CA, ALK, ROS1 etc.) as needed
        }
        self.clinical_significance_context: Optional[str] = None # Instance variable to store context

        # --- Added for Phase 2.5.1 ---
        self.BASIC_GENE_INFO = {
            "BRAF": "BRAF is a gene that makes a protein involved in cell growth and signaling. Certain mutations can cause it to become overactive, driving cancer growth.",
            "EGFR": "EGFR (Epidermal Growth Factor Receptor) is a protein on the surface of cells that helps them grow and divide. Mutations can lead to uncontrolled cell growth in cancer.",
            "TP53": "TP53 is a tumor suppressor gene. It plays a critical role in preventing cancer by controlling cell division and cell death. Mutations can impair its function."
        }
        self.SPECIFIC_VARIANT_IMPLICATIONS = {
            "BRAF_V600E": "V600E is a common activating mutation in melanoma and other cancers, making tumors sensitive to BRAF/MEK inhibitors.",
            "EGFR_T790M": "T790M is a common resistance mutation in EGFR-mutated NSCLC, typically arising after treatment with first or second-generation EGFR inhibitors.",
            "TP53_R248Q": "R248Q is a common oncogenic mutation in TP53, associated with loss of tumor suppressor function and genomic instability."
        }
        self.CONCEPTUAL_EVO2_INSIGHTS = {
            "PAT12345_BRAF_V600E": "ConceptualEvo2Agent Insight: Simulation suggests BRAF V600E is a strong activating mutation with high oncogenic potential.",
            "PAT12345_EGFR_T790M": "ConceptualEvo2Agent Insight: Simulation identifies EGFR T790M as a common resistance mechanism to earlier generation EGFR inhibitors."
        }
        self.CONCEPTUAL_CRISPR_INSIGHTS = {
            "PAT12345_TP53_R248Q": "ConceptualCRISPRAgent Insight: Simulation suggests TP53 R248Q, a common oncogenic mutation, could be a theoretical target for future gene editing therapies, though this remains highly experimental."
        }
        # --- End Added for Phase 2.5.1 ---

        self.llm_client = GeminiClient() # <--- Initialize LLM client

    def _extract_genes(self, query_text: str) -> List[str]:
        found_genes = set()
        query_upper = query_text.upper()
        # Sort known_genes by length descending to match longer names first (e.g., "NTRK1" before "TRK")
        # Though in current list, this is not an issue. Good practice if aliases were less distinct.
        sorted_known_genes = sorted(self.known_genes, key=len, reverse=True)

        for gene in sorted_known_genes:
            # Regex to match whole gene names, not as substrings of other words.
            # Allows gene to be followed by non-alphanumeric or end of string.
            # Allows gene to be preceded by non-alphanumeric or start of string.
            pattern = r'(?<![A-Z0-9])' + re.escape(gene) + r'(?![A-Z0-9])'
            if re.search(pattern, query_upper):
                # Special handling for HER2 -> ERBB2
                if gene == "HER2":
                    found_genes.add("ERBB2")
                else:
                    found_genes.add(gene)
        return sorted(list(found_genes))

    def _normalize_variant_name(self, variant_name: str) -> str:
        """Normalizes variant name, primarily for protein changes."""
        # Standard p. notation: p.Val600Glu -> V600E
        match_p_dot = re.match(r"^[pP]\.(?:([A-Z][a-z]{2}))?([A-Z*])(\d+)(?:([A-Z][a-z]{2}))?([A-Z*]|fs\*?\d*)$", variant_name)
        if match_p_dot:
            aa1_3l, aa1_1l_direct, pos, aa2_3l, aa2_ext = match_p_dot.groups()
            
            aa1_final = ""
            if aa1_1l_direct: # e.g. p.V600E
                aa1_final = aa1_1l_direct
            elif aa1_3l: # e.g. p.Val600Glu
                aa1_final = self.three_letter_aa_map.get(aa1_3l.upper(), "?")
            
            aa2_final = ""
            if len(aa2_ext) == 1 and aa2_ext.isalpha(): # Single letter like E in V600E
                 aa2_final = aa2_ext.upper()
            elif aa2_3l : # e.g. p.Val600Glu
                aa2_final = self.three_letter_aa_map.get(aa2_3l.upper(), "?")
            else: # fs, del, ins, *, etc.
                aa2_final = aa2_ext

            if aa1_final and pos and aa2_final:
                return f"{aa1_final}{pos}{aa2_final}"

        # Simpler V600E, EXON19DEL (already somewhat normalized by _extract_specific_variants)
        # This function is more for ensuring a consistent format if input is messy.
        # For now, return uppercase if no specific p.dot match
        return variant_name.upper()


    def _extract_specific_variants(self, query_text: str) -> List[str]:
        variants_found = set()
        
        # Pattern for p. notation: p. (optional 3-letter AA1) (1-letter AA1) (Position) (optional 3-letter AA2) (1-letter AA2 or * or fs)
        # Captures parts for normalization.
        protein_pattern_p_dot = r'[pP]\.(?:([A-Z][a-z]{2}))?([A-Z])(\d+)(?:([A-Z][a-z]{2}))?([A-Z*]|fs\*?\d*|\*|[Dd][Ee][Ll]|[Ii][Nn][Ss]|[Dd][Uu][Pp])'
        
        # Pattern for simple V600E notation (no p.)
        short_protein_pattern = r'(?<![a-zA-Z\d])([A-Z])(\d+)([A-Z*]|fs\*?\d*|[Dd][Ee][Ll]|[Ii][Nn][Ss]|[Dd][Uu][Pp])(?![a-zA-Z\d])'
        
        exon_pattern = r'(?:exon|ex)\s*(\d+)\s*(?:deletion|del|insertion|ins|mutation|variant|alteration|mut|var|alt)\b'
        
        slash_pattern = r'(?<![a-zA-Z\d])([A-Z])(\d+)([A-Z*])\/([A-Z*])(?![a-zA-Z\d])'

        # General "mutation" type that could be a variant type query like "Nonsense_Mutation"
        # or "Missense", "Frameshift" etc. from GENERAL_VARIANT_TYPE_EFFECTS_MOCK in mock API
        # Ensure it captures "Nonsense_Mutation", "Missense Mutation", etc.
        general_type_pattern = r'\b([A-Za-z]+(?:[-_][A-Za-z]+)*)\s*(?:mutation|variant|alteration)\b'


        for match in re.finditer(protein_pattern_p_dot, query_text):
            aa1_3l, aa1_1l_direct, pos, aa2_3l, aa2_ext = match.groups()
            aa1 = aa1_1l_direct if aa1_1l_direct else self.three_letter_aa_map.get(aa1_3l.upper(), "") if aa1_3l else ""
            
            aa2_norm = aa2_ext # Default
            if aa2_3l: # If full 3-letter like "Glu"
                aa2_norm = self.three_letter_aa_map.get(aa2_3l.upper(), aa2_ext)
            elif aa2_ext.isalpha() and len(aa2_ext) == 1: # V600E -> E
                aa2_norm = aa2_ext.upper()
            # For "del", "ins", "fs", "*" keep as is (already upper by regex or doesn't matter)
            
            if aa1 and pos and aa2_norm:
                variants_found.add(f"{aa1}{pos}{aa2_norm}")
            elif pos and aa2_norm : # For cases like p.T790M where first aa is implied or not stated.
                 variants_found.add(f"{pos}{aa2_norm}")


        for match in re.finditer(short_protein_pattern, query_text):
            variant_str = f"{match.group(1).upper()}{match.group(2)}{match.group(3).upper()}"
            variants_found.add(variant_str)
            
        for match in re.finditer(exon_pattern, query_text, re.IGNORECASE):
            exon_num = match.group(1)
            change_type = ""
            if "del" in match.group(0).lower(): change_type = "DEL"
            elif "ins" in match.group(0).lower(): change_type = "INS"
            else: change_type = "MUT" # generic mutation/variant in exon
            variants_found.add(f"EXON{exon_num}{change_type}")

        for match in re.finditer(slash_pattern, query_text, re.IGNORECASE):
            aa1, pos, aa2_1, aa2_2 = match.groups()
            variants_found.add(f"{aa1.upper()}{pos}{aa2_1.upper()}")
            variants_found.add(f"{aa1.upper()}{pos}{aa2_2.upper()}")

        for match in re.finditer(general_type_pattern, query_text, re.IGNORECASE):
            # This might capture things like "Missense_Mutation" or "Nonsense mutation"
            # We should check if this matches keys in GENERAL_VARIANT_TYPE_EFFECTS_MOCK from the mock API later
            # For now, just add it. It will be passed as variant_query to mock API.
            type_name = match.group(1).replace(" ", "_") # e.g. "Missense Mutation" -> "Missense_Mutation"
            variants_found.add(type_name)


        # Fallback for 3-letter codes not perfectly caught by p.dot if they exist standalone
        three_letter_standalone_pattern = r'(?<![a-zA-Z\d])([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})(?!\d?[a-zA-Z])'
        for match in re.finditer(three_letter_standalone_pattern, query_text):
            aa1_3l, pos, aa2_3l = match.groups()
            aa1_1l = self.three_letter_aa_map.get(aa1_3l.upper())
            aa2_1l = self.three_letter_aa_map.get(aa2_3l.upper())
            if aa1_1l and aa2_1l:
                 variants_found.add(f"{aa1_1l}{pos}{aa2_1l}")
        
        # Filter out generic terms that are too broad unless they are specific types like "Nonsense_Mutation"
        # This is a bit heuristic.
        generic_filter_terms = {"MUTATION", "VARIANT", "ALTERATION"}
        final_variants = {v for v in variants_found if v.upper() not in generic_filter_terms or "_" in v}


        return sorted(list(final_variants))


    def _determine_criterion_intent(self, query_text: str) -> Dict[str, Any]:
        """
        Determines the intent of the genomic criterion.
        Output: Dict with 'primary_intent' (e.g., 'CHECK_PRESENCE', 'CHECK_ABSENCE', 'GET_EFFECT')
                     'status_keyword' (e.g., 'ACTIVATING', 'PATHOGENIC/LOF', 'RESISTANCE', 'WILD_TYPE', 'ANY_MUTATION')
                     'is_negated' (boolean for overall query negation)
        """
        query_lower = query_text.lower()
        
        intent_details = {
            "primary_intent": "GET_EFFECT", # Default if no other strong signals
            "status_keyword": "ANY_MUTATION", # Default status to look for if not specified
            "is_negated": False
        }

        # 1. Detect overall negation (Absence of X, No X, Not X)
        # More comprehensive negation/absence detection
        absence_keywords = [
            r'\babsenc(?:e|y)\s+of\b', r'\bno\s+(?:known\s+)?', r'\bwithout\b', 
            r'\bnegative\s+for\b', r'\bnot\s+have\b', r'\bnon-mutated\b',
            r'\bshould\s+not\s+have\b', r'\bmust\s+not\s+be\b', r'\bexcludes?\s+if\b',
            r'^(?!.*presence of.*)' # Heuristic: if "presence of" is not in query, lean towards absence if other keywords match
        ]
        # Check for explicit presence keywords that might override an "absence" vibe
        presence_keywords = [r'\bpresence\s+of\b', r'\bhas\b', r'\bwith\b', r'\bpositive\s+for\b']

        # If "effect of" or "impact of" is present, it's likely a GET_EFFECT query, not presence/absence.
        if re.search(r'\b(?:effect|impact)\s+of\b', query_lower):
            intent_details["primary_intent"] = "GET_EFFECT"
        else: # Check for presence/absence signals
            is_explicitly_present = any(re.search(pk, query_lower) for pk in presence_keywords)
            is_explicitly_absent = any(re.search(ak, query_lower) for ak in absence_keywords)

            if is_explicitly_absent and not is_explicitly_present:
                intent_details["is_negated"] = True
                intent_details["primary_intent"] = "CHECK_ABSENCE"
            elif is_explicitly_present:
                intent_details["primary_intent"] = "CHECK_PRESENCE"
            # If neither, it might be an implicit presence check or effect query.
            # For now, if not "effect of" and not clearly "absence", assume "CHECK_PRESENCE" if status keywords are found.
            elif not re.search(r'\b(?:effect|impact)\s+of\b', query_lower):
                 intent_details["primary_intent"] = "CHECK_PRESENCE"


        # 2. Detect most specific status keyword
        # Priority: Activating/Pathogenic/Resistance > Wild-Type > General Mutation
        status_priority_map = {
            'ACTIVATING': self.intent_patterns['ACTIVATING_PRESENCE'],
            'PATHOGENIC/LOF': self.intent_patterns['PATHOGENIC_PRESENCE'],
            'RESISTANCE': self.intent_patterns['RESISTANCE_PRESENCE'],
            'WILD_TYPE': self.intent_patterns['WILD_TYPE'],
            'ANY_MUTATION': self.intent_patterns['MUTATION_PRESENCE'] 
        }

        for status_key, patterns in status_priority_map.items():
            if any(re.search(r'\b' + pattern + r'\b', query_lower) for pattern in patterns):
                intent_details["status_keyword"] = status_key
                # If we find a specific status, and primary_intent was default GET_EFFECT,
                # it's more likely a CHECK_PRESENCE/CHECK_ABSENCE of this status.
                if intent_details["primary_intent"] == "GET_EFFECT" and status_key != "ANY_MUTATION":
                    intent_details["primary_intent"] = "CHECK_ABSENCE" if intent_details["is_negated"] else "CHECK_PRESENCE"
                break # Found highest priority status

        # Refine primary_intent based on status_keyword if still ambiguous
        if intent_details["primary_intent"] == "GET_EFFECT" and intent_details["status_keyword"] != "ANY_MUTATION":
             # "Pathogenic BRAF V600E" is more a check for presence of pathogenic status than just "effect"
            intent_details["primary_intent"] = "CHECK_ABSENCE" if intent_details["is_negated"] else "CHECK_PRESENCE"
        
        # If query is "HER2-negative", it's WILD_TYPE and CHECK_PRESENCE (of wild-type)
        if "her2-negative" in query_lower.replace(" ", ""):
            intent_details["status_keyword"] = "WILD_TYPE"
            intent_details["primary_intent"] = "CHECK_PRESENCE" # Presence of WT
            intent_details["is_negated"] = False # Not negated overall, it's a positive statement about negativity


        logging.info(f"Determined Intent for '{query_text[:70]}...': {intent_details}")
        return intent_details


    def _classify_variant_simulated(
        self, 
        gene_symbol: str, 
        patient_variants_for_gene: List[Dict[str, Any]], 
        specific_variants_from_query: List[str], # Variants explicitly mentioned in the query text
        intent_details: Dict[str, Any]
    ) -> Tuple[List[SimulatedVEPDetail], List[str], Set[str]]:
        """
        Simulates variant classification using the mock Evo2 API.
        Processes variants based on query intent and patient data.
        Returns a list of SimulatedVEPDetail objects, errors, and found patient variants that matched query specifics.
        """
        processed_vep_details: List[SimulatedVEPDetail] = []
        errors: List[str] = []
        # Tracks specific protein changes from patient data that match a variant mentioned in the query
        matched_patient_variants_to_query_specifics: Set[str] = set()

        # Normalize variants from query for matching purposes
        normalized_query_variants = {self._normalize_variant_name(v) for v in specific_variants_from_query}
        
        query_intent_for_mock_api = intent_details.get("status_keyword")
        # If status is ANY_MUTATION, pass None or a generic intent to mock API unless it's specifically for "effect of"
        if query_intent_for_mock_api == "ANY_MUTATION" and intent_details["primary_intent"] != "GET_EFFECT":
            query_intent_for_mock_api = None # Let mock API decide based on variant type if no specific status desired
        elif intent_details["primary_intent"] == "GET_EFFECT":
            query_intent_for_mock_api = None # For "effect of", we want the raw interpretation

        # Scenario 1: Query explicitly mentions specific variants (e.g., "BRAF V600E", "effect of TP53 R248Q")
        if specific_variants_from_query:
            for query_variant_norm in normalized_query_variants:
                # This query_variant_norm is what we ask the mock API about.
                # We also check if this variant exists in the patient's data for this gene.
                patient_variant_match_data = None
                for pv_data in patient_variants_for_gene:
                    pv_protein_change = pv_data.get("protein_change")
                    if pv_protein_change and self._normalize_variant_name(pv_protein_change) == query_variant_norm:
                        patient_variant_match_data = pv_data
                        matched_patient_variants_to_query_specifics.add(query_variant_norm)
                        break
                
                variant_type_for_api = patient_variant_match_data.get("variant_type") if patient_variant_match_data else None
                
                # Call mock API for each specific variant from the query
                try:
                    mock_response = get_variant_effect_mock(
                        gene_symbol=gene_symbol,
                        variant_query=query_variant_norm, # Use the (normalized) variant from the query
                        variant_type=variant_type_for_api,
                        query_intent=query_intent_for_mock_api
                    )
                    vep_detail = SimulatedVEPDetail(**mock_response)
                    processed_vep_details.append(vep_detail)
                except Exception as e:
                    logging.error(f"Mock API error for query variant {gene_symbol} {query_variant_norm}: {e}")
                    errors.append(f"Mock API Error (query variant {query_variant_norm}): {str(e)}")
                    processed_vep_details.append(SimulatedVEPDetail(
                        input_variant_query=query_variant_norm, gene_symbol=gene_symbol,
                        simulated_classification="ERROR_MOCK_API", classification_reasoning=str(e)
                    ))

        # Scenario 2: Query is general about a gene/status (e.g., "pathogenic KRAS mutation", "any mutation in BRCA1")
        # In this case, we iterate through all of the patient's variants for that gene.
        # This also covers "effect of GENE" (no specific variant) -> interpret all patient variants in GENE.
        elif not specific_variants_from_query and patient_variants_for_gene:
            for pv_data in patient_variants_for_gene:
                pv_protein_change = pv_data.get("protein_change")
                pv_variant_type = pv_data.get("variant_type")

                # If protein_change is missing, but variant_type is informative (e.g. "Nonsense_Mutation") use that for query
                # Otherwise, if protein_change is present, it's the primary identifier for the query.
                variant_query_for_api = pv_protein_change if pv_protein_change else pv_variant_type
                if not variant_query_for_api:
                    errors.append(f"Skipping patient variant in {gene_symbol} due to missing protein_change and variant_type: {pv_data}")
                    continue

                try:
                    mock_response = get_variant_effect_mock(
                        gene_symbol=gene_symbol,
                        variant_query=variant_query_for_api,
                        variant_type=pv_variant_type, # Always pass patient's variant_type
                        query_intent=query_intent_for_mock_api 
                    )
                    vep_detail = SimulatedVEPDetail(**mock_response)
                    processed_vep_details.append(vep_detail)
                    if pv_protein_change: # If we processed based on a specific patient protein change
                         matched_patient_variants_to_query_specifics.add(self._normalize_variant_name(pv_protein_change))
                except Exception as e:
                    logging.error(f"Mock API error for patient variant {gene_symbol} {variant_query_for_api}: {e}")
                    errors.append(f"Mock API Error (patient variant {variant_query_for_api}): {str(e)}")
                    processed_vep_details.append(SimulatedVEPDetail(
                        input_variant_query=variant_query_for_api, gene_symbol=gene_symbol, protein_change=pv_protein_change,
                        simulated_classification="ERROR_MOCK_API", classification_reasoning=str(e),
                        variant_type_from_input=pv_variant_type
                    ))
        
        # Scenario 3: No specific variants in query, and patient has no variants for this gene.
        # This is relevant for "absence of mutation in GENE" or "GENE wild-type".
        # The mock API might still be called with the gene name and intent (e.g. "WILD_TYPE")
        # to get a "canonical" wild-type representation if desired.
        elif not specific_variants_from_query and not patient_variants_for_gene:
             if intent_details["primary_intent"] == "CHECK_PRESENCE" and intent_details["status_keyword"] == "WILD_TYPE":
                try: # Get a canonical "wild-type" entry for the gene
                    mock_response = get_variant_effect_mock(
                        gene_symbol=gene_symbol, variant_query="Wild_Type", # Special query
                        query_intent="benign" # Assuming WT implies benign for mock API's default
                    )
                    # Modify response to clearly state it's a WT record
                    mock_response["simulated_classification"] = "WILD_TYPE_CONFIRMED"
                    mock_response["classification_reasoning"] = f"Patient has no mutations in {gene_symbol}. Confirmed Wild-Type status."
                    mock_response["protein_change"] = None
                    mock_response["canonical_variant_id"] = f"{gene_symbol}:Wild_Type"
                    vep_detail = SimulatedVEPDetail(**mock_response)
                    processed_vep_details.append(vep_detail)
                except Exception as e:
                    errors.append(f"Mock API Error (gene wild-type call for {gene_symbol}): {str(e)}")
                    # Add placeholder if error
                    processed_vep_details.append(SimulatedVEPDetail(
                        input_variant_query="Wild_Type", gene_symbol=gene_symbol,
                        simulated_classification="ERROR_MOCK_API", classification_reasoning=str(e)
                    ))


        return processed_vep_details, errors, matched_patient_variants_to_query_specifics
        
    def _get_gene_summary_status(self, gene_symbol: str, gene_vep_details: List[SimulatedVEPDetail], intent_details: Dict) -> GeneSummaryStatus:
        """
        Determines the summary status for a single gene based on its VEP details and query intent.
        Example statuses: MET, NOT_MET, ACTIVATING_FOUND, PATHOGENIC_FOUND, WILD_TYPE, VUS_PRESENT, RESISTANCE_FOUND, etc.
        """
        
        # Priority of statuses found in patient's variants for this gene
        # (irrespective of query intent for now, just what's in the patient)
        highest_priority_classification_found = "WILD_TYPE" # Default if no variants or only benign
        classification_priorities = [
            "RESISTANCE_BY_RULE", "PREDICTED_RESISTANCE_BY_MOCK_EVO2",
            "PATHOGENIC_BY_RULE", "PREDICTED_PATHOGENIC_BY_MOCK_EVO2", "LIKELY_PATHOGENIC_BY_MOCK_EVO2",
            "ACTIVATING_BY_RULE", "PREDICTED_ACTIVATING_BY_MOCK_EVO2", # Note: Activating might be a subset of Pathogenic for some tools
            "UNCLEAR_BY_RULE", "UNCLEAR_BY_MOCK_EVO2", "PREDICTED_VUS", # VUS/Unclear
            "LIKELY_BENIGN_BY_MOCK_EVO2", "PREDICTED_BENIGN_BY_MOCK_EVO2",
            "WILD_TYPE_CONFIRMED" # Explicit WT confirmation
        ]
        
        if not gene_vep_details: # No variants processed for this gene (e.g. patient has no mutations in it)
            highest_priority_classification_found = "WILD_TYPE_CONFIRMED"
        else:
            for classification_level in classification_priorities:
                if any(detail.simulated_classification.upper().startswith(classification_level) for detail in gene_vep_details):
                    highest_priority_classification_found = classification_level
                    break
        
        # Map this highest patient variant status to a gene summary term
        gene_actual_status_term = "UNCLEAR" # Default
        if "RESISTANCE" in highest_priority_classification_found: gene_actual_status_term = "RESISTANCE_MUTATION_DETECTED"
        elif "PATHOGENIC" in highest_priority_classification_found: gene_actual_status_term = "PATHOGENIC_MUTATION_DETECTED"
        elif "ACTIVATING" in highest_priority_classification_found: gene_actual_status_term = "ACTIVATING_MUTATION_DETECTED"
        elif "UNCLEAR" in highest_priority_classification_found or "VUS" in highest_priority_classification_found: gene_actual_status_term = "VUS_DETECTED"
        elif "BENIGN" in highest_priority_classification_found: gene_actual_status_term = "BENIGN_VARIANT_DETECTED"
        elif "WILD_TYPE" in highest_priority_classification_found: gene_actual_status_term = "WILD_TYPE_CONFIRMED"

        # Now, evaluate against the query intent
        query_status_keyword = intent_details["status_keyword"] # e.g. "ACTIVATING", "PATHOGENIC/LOF", "WILD_TYPE"
        primary_intent = intent_details["primary_intent"]       # e.g. "CHECK_PRESENCE", "CHECK_ABSENCE", "GET_EFFECT"

        met_status = "UNCLEAR"
        summary_details = f"Gene {gene_symbol}: Patient's most significant variant status is '{gene_actual_status_term}'. Query intent: {primary_intent} of '{query_status_keyword}'."

        if primary_intent == "GET_EFFECT":
            # For "effect of", determine MET/NOT_MET based on what was found
            if gene_actual_status_term in ["PATHOGENIC_MUTATION_DETECTED", "ACTIVATING_MUTATION_DETECTED", "RESISTANCE_MUTATION_DETECTED"]:
                 met_status = "MET" # We found the kind of effect implied by these statuses
                 summary_details += f" Reporting observed effect ({gene_actual_status_term})."
            elif gene_actual_status_term == "WILD_TYPE_CONFIRMED":
                 met_status = "NOT_MET" # Effect sought (mutation) was not found
                 summary_details += " Reporting wild-type status found."
            elif gene_actual_status_term == "BENIGN_VARIANT_DETECTED":
                 met_status = "NOT_MET" # Effect sought (significant mutation) was not found
                 summary_details += f" Reporting observed effect ({gene_actual_status_term})."
            elif gene_actual_status_term == "VUS_DETECTED":
                 met_status = "UNCLEAR" # Effect is uncertain
                 summary_details += f" Reporting observed effect ({gene_actual_status_term})."
            else: # Default fallback
                 met_status = "UNCLEAR"
                 summary_details += " Reporting observed effect (status unclear)."
        else: # CHECK_PRESENCE or CHECK_ABSENCE
            # Mapping query keywords to patient's actual status terms
            # This needs to be robust. e.g. query "pathogenic" should match patient "PATHOGENIC_MUTATION_DETECTED"
            keyword_to_actual_term_map = {
                "ACTIVATING": "ACTIVATING_MUTATION_DETECTED",
                "PATHOGENIC/LOF": "PATHOGENIC_MUTATION_DETECTED",
                "RESISTANCE": "RESISTANCE_MUTATION_DETECTED",
                "WILD_TYPE": "WILD_TYPE_CONFIRMED",
                "ANY_MUTATION": ["ACTIVATING_MUTATION_DETECTED", "PATHOGENIC_MUTATION_DETECTED", "RESISTANCE_MUTATION_DETECTED", "VUS_DETECTED"] # A list of "mutated" states
            }
            
            target_actual_term = keyword_to_actual_term_map.get(query_status_keyword)

            if target_actual_term:
                condition_met = False
                if isinstance(target_actual_term, list): # For ANY_MUTATION
                    condition_met = gene_actual_status_term in target_actual_term
                else: # For specific statuses
                    condition_met = gene_actual_status_term == target_actual_term

                if primary_intent == "CHECK_PRESENCE":
                    met_status = "MET" if condition_met else "NOT_MET"
                elif primary_intent == "CHECK_ABSENCE":
                    met_status = "MET" if not condition_met else "NOT_MET"
                summary_details += f" Criterion for {query_status_keyword} was {met_status}."
            else:
                summary_details += " Could not map query status keyword to patient's variant status for MET/NOT_MET evaluation."
                met_status = "UNCLEAR" # If query status keyword is unmappable

        return GeneSummaryStatus(status=met_status, details=summary_details)


    def _generate_clinical_significance_context(self, target_genes: List[str], final_gene_summaries: Dict[str, GeneSummaryStatus], overall_status: str, intent_details: Dict[str, Any]) -> str:
        # This method might be too high-level for specific variant context.
        # Let's enhance context generation within _analyze_single_genomic_query or its callers.
        # For now, it uses self.clinical_context_map based on gene and overall status.
        # We will augment the 'clinical_significance_context' field of GenomicAnalysisResult directly
        # in _analyze_single_genomic_query for more granular, variant-specific context.
        
        # Fallback or general context if no specific gene map matches.
        contexts = []
        for gene in target_genes:
            gene_specific_map = self.clinical_context_map.get(gene.upper())
            
            if gene_specific_map:
                # Determine a key for context based on overall status and intent
                # This logic is simplified; real mapping might be more complex
                context_key = "DEFAULT_CONTEXT" # Fallback within gene specific map
                if overall_status == "MET" and intent_details.get("presence_required") == True:
                    if intent_details.get("required_status") == "ACTIVATING":
                        context_key = "ACTIVATING_MUTATION_TRUE"
                    elif intent_details.get("required_status") == "PATHOGENIC/LOF":
                        context_key = "PATHOGENIC_MUTATION_TRUE"
                    elif intent_details.get("required_status") == "RESISTANCE":
                        context_key = "RESISTANCE_MUTATION_TRUE"
                elif overall_status == "MET" and intent_details.get("presence_required") == False and intent_details.get("required_status") == "ANY_MUTATION": # Implies WT
                     context_key = "WILD_TYPE_TRUE"
                elif overall_status == "NOT_MET" and intent_details.get("required_status") == "WILD_TYPE" and intent_details.get("presence_required") == True : # WT required but not found
                     # Could add a "NOT_WILD_TYPE" or similar key
                     pass


                contexts.append(gene_specific_map.get(context_key, gene_specific_map.get("DEFAULT_CONTEXT", "")))
            else: # No specific map for the gene, use general default
                contexts.append(self.clinical_context_map.get("DEFAULT", {}).get(overall_status, "No specific context available."))
        
        return " ".join(filter(None, contexts))


    async def run(self, patient_data: Dict[str, Any], prompt_details: Dict[str, Any]) -> Dict[str, Any]:
        # patient_mutations = patient_data.get("mutations", [])
        # patient_id = patient_data.get("patientId", "Unknown_Patient_ID") # Use patientId from patient_data
        # user_prompt = prompt_details.get("prompt", "") # Original prompt text
        intent = prompt_details.get("intent")
        entities = prompt_details.get("entities", {})
        criterion_id = prompt_details.get("criterion_id", "genomic_query_default")


        if intent == "analyze_genomic_profile":
            # Pass the full patient_data to _perform_holistic_genomic_analysis
            return await self._perform_holistic_genomic_analysis(patient_data=patient_data, criterion_id_prefix="holistic_mutation")
        elif intent == "analyze_genomic_criterion": # Assuming this intent for single criterion
            # For single criterion analysis, we still need patient_mutations and patient_id
            patient_mutations = patient_data.get("mutations", [])
            patient_id = patient_data.get("patientId", "Unknown_Patient_ID_Run") # Ensure patientId is available
            genomic_query = entities.get("genomic_criterion_text") or prompt_details.get("prompt") # Get query from entities or fallback
            
            if not genomic_query:
                return GenomicAnalysisResult(
                    criterion_id=criterion_id,
                    criterion_query="N/A",
                    status="ERROR",
                    evidence="No genomic criterion query provided.",
                    errors=["Missing genomic_criterion_text in entities or prompt."]
                ).model_dump()

            return (await self._analyze_single_genomic_query(
                genomic_query=genomic_query, 
                patient_mutations=patient_mutations,
                patient_id=patient_id, # Pass patient_id
                criterion_id=criterion_id,
                patient_data_for_context=patient_data # Pass full patient_data for context generation
            )).model_dump()
        else: # Fallback or direct query if no specific intent mapping
            # This path might be deprecated if intents are always used.
            patient_mutations = patient_data.get("mutations", [])
            patient_id = patient_data.get("patientId", "Unknown_Patient_ID_Fallback")
            genomic_query = prompt_details.get("prompt", "")
            if not genomic_query:
                 return {"status": "ERROR", "message": "No query provided for GenomicAnalystAgent."}
            
            analysis_result_obj = await self._analyze_single_genomic_query(
                genomic_query=genomic_query, 
                patient_mutations=patient_mutations,
                patient_id=patient_id, # Pass patient_id
                criterion_id=criterion_id,
                patient_data_for_context=patient_data # Pass full patient_data for context generation
            )
            return analysis_result_obj.model_dump()
        

    async def _analyze_single_genomic_query(
        self, 
        genomic_query: str, 
        patient_mutations: List[Dict[str, Any]], 
        patient_id: str, # Added patient_id parameter
        criterion_id: str,
        patient_data_for_context: Optional[Dict[str, Any]] = None # Added for richer context
    ) -> GenomicAnalysisResult:
        logging.info(f"[{self.name}] Analyzing genomic query: '{genomic_query}' for patient '{patient_id}' with criterion ID '{criterion_id}'")
        
        target_genes = self._extract_genes(genomic_query)
        specific_variants_from_query = self._extract_specific_variants(genomic_query)
        intent_details = self._determine_criterion_intent(genomic_query)

        logging.debug(f"Extracted genes: {target_genes}, specific variants: {specific_variants_from_query}, intent: {intent_details}")

        if not target_genes and not specific_variants_from_query: # If query is too vague
             # If the query IS a general variant type (like "Missense_Mutation") it will be in specific_variants_from_query
            if not any(vq in get_variant_effect_mock("", vq, "").simulated_classification for vq in specific_variants_from_query): # Check if it's a known general type
                return GenomicAnalysisResult(
                    criterion_id=criterion_id,
                    criterion_query=genomic_query,
                    status="UNCLEAR",
                    evidence=f"Could not identify specific genes or recognizable variant types in the query: '{genomic_query}'.",
                    errors=["Query too vague or does not match known gene/variant patterns."]
                )

        # Initialize with default values
        overall_status = "UNCLEAR" 
        final_evidence_parts = [
            f"Genomic Query: {genomic_query}",
            f"Determined Intent: Required Status = {intent_details.get('required_status', 'ANY')}, Presence Required = {intent_details.get('presence_required', True)}"
        ]
        all_simulated_vep_details: List[SimulatedVEPDetail] = []
        all_errors: List[str] = []
        
        # Store per-gene summary statuses
        final_gene_summaries: Dict[str, GeneSummaryStatus] = {}
        
        # Filter patient mutations to only those in target_genes if target_genes is not empty
        relevant_patient_mutations = []
        if target_genes:
            for mut in patient_mutations:
                if mut.get("hugo_gene_symbol", "").upper() in [g.upper() for g in target_genes]:
                    relevant_patient_mutations.append(mut)
        else: # If no specific genes, consider all patient mutations (e.g. for general variant type query)
            relevant_patient_mutations = patient_mutations


        # This part simulates VEP for relevant patient mutations and variants from query
        # It should return classifications for actual patient variants found.
        # The logic for specific_variants_from_query is tricky if they are general types vs specific alterations
        # Let's assume _classify_variant_simulated handles this.
        
        # Re-thinking: _classify_variant_simulated takes patient_variants_for_gene.
        # We need to iterate through target_genes OR through patient_mutations if no target_genes (e.g. for general "Missense_Mutation" query)

        if not target_genes and specific_variants_from_query:
            # Handle cases like "is there any activating mutation?" or "Missense_Mutation"
            # Here, we might need to iterate all patient mutations and check against the general variant query.
            # For now, this path is less defined. _classify_variant_simulated might handle variant_query as a type.
             logging.warning(f"Query with specific variants but no target genes: {specific_variants_from_query}. Analysis may be limited.")
             # This case is complex, as `_classify_variant_simulated` is gene-centric.
             # Let's assume the first variant in `specific_variants_from_query` might be a general type for now.
             # This part needs more robust logic if we expect queries like "any activating mutation".
             # For the holistic view, this path is not taken.

        gene_centric_analysis_conducted = False
        if target_genes:
            gene_centric_analysis_conducted = True
            for gene_symbol_orig in target_genes:
                gene_symbol = gene_symbol_orig.upper()
                # Get patient's mutations for THIS specific gene
                patient_variants_for_this_gene = [
                    m for m in relevant_patient_mutations if m.get("hugo_gene_symbol", "").upper() == gene_symbol
                ]

                gene_vep_details, gene_errors, _ = self._classify_variant_simulated(
                gene_symbol=gene_symbol,
                    patient_variants_for_gene=patient_variants_for_this_gene,
                    specific_variants_from_query=specific_variants_from_query, # Pass query variants
                intent_details=intent_details
            )
                all_simulated_vep_details.extend(gene_vep_details)
                all_errors.extend(gene_errors)

                gene_summary = self._get_gene_summary_status(gene_symbol, gene_vep_details, intent_details)
                final_gene_summaries[gene_symbol_orig] = gene_summary # Use original casing for output key
        else: # No target genes, but there might be specific_variants_from_query (like a general type)
            # This is for queries like "any missense mutation" - requires iterating all patient mutations.
            # The current _classify_variant_simulated is gene-centric.
            # For now, if no target_genes, we'll rely on the initial check against specific_variants_from_query.
            # This path will likely result in UNCLEAR for complex general queries without a gene.
            pass


        # Determine overall status based on gene summaries
        if final_gene_summaries:
            met_count = 0
            not_met_count = 0
            unclear_count = 0
            for gene_sum_stat in final_gene_summaries.values():
                if gene_sum_stat.status == "MET": met_count +=1
                elif gene_sum_stat.status == "NOT_MET": not_met_count +=1
                else: unclear_count +=1
            
            # Logic for overall status based on combined gene statuses:
            # If presence is required: all target genes must be MET.
            # If absence is required (presence_required=False): all target genes must be MET (meaning the 'not present' condition is met).
            if intent_details.get("presence_required", True): # Default to presence required
                if met_count == len(target_genes) and met_count > 0 :
                    overall_status = "MET"
                elif not_met_count > 0: # If any is explicitly NOT_MET
                    overall_status = "NOT_MET"
                else: # Mix of MET and UNCLEAR, or only UNCLEAR
                    overall_status = "UNCLEAR"
            else: # Absence of the condition is required
                if met_count == len(target_genes) and met_count > 0: # All genes met the "absence" condition
                    overall_status = "MET"
                elif not_met_count > 0: # If any gene FAILED the "absence" condition (i.e., the thing was found)
                    overall_status = "NOT_MET"
                else: # No genes met the "absence" condition
                    overall_status = "UNCLEAR"
        
            if not target_genes and met_count == 0 and not_met_count == 0 and unclear_count == 0: # No genes processed, means UNCLEAR
                    overall_status = "UNCLEAR"


        final_evidence_parts.append(f"Overall Status: {overall_status}")
        for gene, summary in final_gene_summaries.items():
            final_evidence_parts.append(f" - Gene {gene}: Status {summary.status}. Details: {summary.details}")
        
        # --- Context Enhancement for Phase 2.5.1 ---
        generated_context_parts = []
        # Use the original criterion query to provide context for the specific mutation being analyzed in holistic view.
        # For holistic analysis, genomic_query will be like "Effect of BRAF V600E"
        
        # Attempt to extract gene and variant from the query if it's for holistic single mutation analysis
        holistic_match = re.match(r"Effect of ([A-Z0-9]+) ([A-Za-z0-9*]+)", genomic_query)
        context_gene = ""
        context_variant = ""

        if holistic_match:
            context_gene = holistic_match.group(1).upper()
            context_variant = holistic_match.group(2).upper() # Normalize case for matching
        elif target_genes: # Fallback to first target gene if holistic_match fails
            context_gene = target_genes[0].upper()
            if specific_variants_from_query:
                 context_variant = specific_variants_from_query[0].upper()


        if context_gene:
            gene_info = self.BASIC_GENE_INFO.get(context_gene)
            if gene_info:
                generated_context_parts.append(f"General Info for {context_gene}: {gene_info}")

            variant_key = f"{context_gene}_{context_variant}"
            variant_implication = self.SPECIFIC_VARIANT_IMPLICATIONS.get(variant_key)
            if variant_implication:
                generated_context_parts.append(f"Implication of {context_variant}: {variant_implication}")

            # --- Add Evo2 prediction to context if available ---
            # Find the VEP detail that matches the main gene/variant of the query
            # This is a simplified approach; a more robust match might be needed if multiple VEP details exist for a single query.
            primary_vep_detail = None
            if all_simulated_vep_details:
                for vep_detail in all_simulated_vep_details:
                    # Check if this vep_detail corresponds to the main gene and variant of the query
                    is_gene_match = vep_detail.gene_symbol.upper() == context_gene
                    # Check variant match: vep_detail.protein_change or vep_detail.input_variant_query
                    is_variant_match = False
                    if context_variant:
                        if vep_detail.protein_change and vep_detail.protein_change.upper() == context_variant:
                            is_variant_match = True
                        elif vep_detail.input_variant_query and vep_detail.input_variant_query.upper() == context_variant:
                            is_variant_match = True
                    
                    if is_gene_match and (not context_variant or is_variant_match): # If no specific variant in query, gene match is enough
                        primary_vep_detail = vep_detail
                        break 
            
            if primary_vep_detail and primary_vep_detail.evo2_prediction:
                evo2_pred_text = primary_vep_detail.evo2_prediction
                evo2_conf_text = ""
                if primary_vep_detail.evo2_confidence is not None:
                    evo2_conf_text = f" (Confidence: {primary_vep_detail.evo2_confidence:.0%})"
                generated_context_parts.append(f"Mock Evo2 Analysis: {evo2_pred_text}{evo2_conf_text}.")
            # --- End Evo2 prediction in context ---

            # Add conceptual agent insights for PAT12345
            current_patient_id = patient_data_for_context.get("patientId") if patient_data_for_context else patient_id
            if current_patient_id == "PAT12345":
                evo2_insight_key = f"PAT12345_{variant_key}"
                evo2_insight = self.CONCEPTUAL_EVO2_INSIGHTS.get(evo2_insight_key)
                if evo2_insight:
                    generated_context_parts.append(evo2_insight)
                
                crispr_insight_key = f"PAT12345_{variant_key}"
                crispr_insight = self.CONCEPTUAL_CRISPR_INSIGHTS.get(crispr_insight_key)
                if crispr_insight:
                    generated_context_parts.append(crispr_insight)
        
        current_clinical_significance_context = " ".join(generated_context_parts)
        if not current_clinical_significance_context and target_genes: # Fallback to old method if new one yields nothing
            current_clinical_significance_context = self._generate_clinical_significance_context(target_genes, final_gene_summaries, overall_status, intent_details)

        # --- Logging before Mock CRISPR Check ---
        logging.info(f"DEBUG Crispr Check: Query='{genomic_query}', Gene='{context_gene}', Variant='{context_variant}', Status='{overall_status}'")

        # --- Mock CRISPR Recommendation Generation (Placeholder) ---
        generated_crispr_recs = []
        # Example: Add a mock recommendation if the query is for BRAF V600E and it's considered MET
        # Note: `genomic_query` is the raw query string. `context_gene` and `context_variant` are extracted.
        if context_gene == "BRAF" and context_variant == "V600E" and overall_status == "MET":
            logging.info("DEBUG Crispr Check: Conditions MET for BRAF V600E mock CRISPR rec.") # Log if condition is met
            mock_braf_v600e_rec = {
                "target_gene": "BRAF",
                "target_variant": "V600E (c.1799T>A)", # Include c. notation for display
                "editing_type": "Base Editing (A->G)", 
                "recommended_approach": "Utilize an Adenine Base Editor (ABE) to directly revert the pathogenic c.1799T>A mutation to the wild-type sequence (c.1799T). This is a conceptual recommendation based on mock data.",
                "rationale": "BRAF V600E is a well-characterized oncogenic driver. Direct correction via base editing is feasible and offers a precise therapeutic strategy, potentially minimizing off-target indels compared to nuclease-based HDR.",
                "potential_tools": ["ABEmax", "ABE8e"],
                "confidence_score": 0.90, # Example confidence
                "supporting_evidence": ["PMID:28800903", "PMID:31775477"], # Example relevant PMIDs
                "source": "GenomicAnalystAgent_MockRec_v0.1"
            }
            generated_crispr_recs.append(mock_braf_v600e_rec)
        # Add more conditions for other mock recommendations if needed.
        # For example, if TP53 R248Q is analyzed:
        elif context_gene == "TP53" and context_variant == "R248Q" and overall_status == "MET":
            mock_tp53_r248q_rec = {
                "target_gene": "TP53",
                "target_variant": "R248Q (c.743G>A)",
                "editing_type": "HDR Correction (Conceptual)",
                "recommended_approach": "Conceptually, explore precise gene editing via Homology-Directed Repair (HDR) to restore the wild-type TP53 sequence at c.743G>A. This remains highly experimental for TP53 hotspot mutations.",
                "rationale": "TP53 R248Q is a common oncogenic mutation. Restoring wild-type function is a theoretical goal, but challenges in efficiency and delivery for TP53 are significant.",
                "potential_tools": ["SpCas9 + HDR template", "High-fidelity Cas9 variants"],
                "confidence_score": 0.65,
                "source": "GenomicAnalystAgent_MockRec_v0.1"
            }
            generated_crispr_recs.append(mock_tp53_r248q_rec)

        return GenomicAnalysisResult(
            criterion_id=criterion_id,
            criterion_query=genomic_query,
            status=overall_status,
            evidence="\n".join(final_evidence_parts),
            gene_summary_statuses=final_gene_summaries,
            simulated_vep_details=all_simulated_vep_details,
            clinical_significance_context=current_clinical_significance_context if current_clinical_significance_context else "No specific clinical significance context generated.",
            crispr_recommendations=generated_crispr_recs, # Use the generated list
            errors=all_errors
        )

    async def _generate_llm_summary_of_holistic_analysis(self, analyses: List[GenomicAnalysisResult], patient_name: str) -> str:
        """
        Generates a natural language summary of the holistic genomic analysis results using an LLM.
        """
        if not analyses:
            return "No genomic findings were processed for summary."

        # Construct a detailed prompt for the LLM
        prompt_context_parts = []
        for i, analysis in enumerate(analyses):
            gene = "Unknown Gene"
            if analysis.simulated_vep_details and analysis.simulated_vep_details[0].gene_symbol:
                gene = analysis.simulated_vep_details[0].gene_symbol
            elif analysis.gene_summary_statuses:
                gene = list(analysis.gene_summary_statuses.keys())[0]
            
            variant = analysis.criterion_query.replace(f"Effect of {gene} ", "").split(" (")[0] if analysis.criterion_query else "N/A"
            status = analysis.status
            
            # Extract VEP and CRISPR details more thoroughly
            vep_detail = analysis.simulated_vep_details[0] if analysis.simulated_vep_details else None
            classification = vep_detail.simulated_classification if vep_detail else "N/A"
            evo2_pred = vep_detail.evo2_prediction if vep_detail else "N/A"
            evo2_conf = f"{vep_detail.evo2_confidence:.0%}" if vep_detail and vep_detail.evo2_confidence is not None else "N/A"
            
            crispr_rec = analysis.crispr_recommendations[0] if analysis.crispr_recommendations else None
            crispr_summary = f"CRISPR Potential: {crispr_rec['editing_type']}" if crispr_rec else "CRISPR Potential: None"

            prompt_context_parts.append(
                # Add Evo2 Confidence and CRISPR Summary
                f"{i+1}. Gene: {gene}, Variant: {variant}, Status: {status}, Classification: {classification}, Evo2 Prediction: {evo2_pred} (Confidence: {evo2_conf}), {crispr_summary}"
            )
        
        findings_str = "\n".join(prompt_context_parts)

        prompt = f"""
Act as a clinical genomics expert. Based on the following summarized genomic findings for patient {patient_name}, provide a concise (2-4 sentences) natural language summary.
Explain the key implications in simple terms. Do not list the findings one by one, but synthesize them.
Focus on whether mutations are pathogenic, represent resistance, or other key clinical takeaways.
**Incorporate the Evo2 confidence levels and any noted CRISPR editing potential into your assessment.**

Summarized Findings:
{findings_str}

Natural Language Summary:
"""
        try:
            logging.info(f"[{self.name}] Generating natural language summary with LLM. Prompt: {prompt[:300]}...")
            # Change method call from get_text_response to generate
            summary_text = await self.llm_client.generate(prompt) 
            logging.info(f"[{self.name}] LLM summary generated: {summary_text[:100]}...")
            return summary_text.strip()
        except Exception as e:
            logging.error(f"[{self.name}] Error generating LLM summary: {e}")
            return "Could not generate a natural language summary due to an error."


    async def _perform_holistic_genomic_analysis(
        self, 
        patient_data: Dict[str, Any],
        criterion_id_prefix: str = "holistic_mutation"
    ) -> Dict[str, Any]:
        """
        Performs a holistic genomic analysis by iterating through patient mutations
        and calling _analyze_single_genomic_query for each.
        Then, generates a natural language summary of the findings.
        """
        patient_mutations = patient_data.get("mutations", [])
        # Correct patient identification (Phase 2.5.1)
        patient_id = patient_data.get("patientId", "Unknown_Patient_ID")
        patient_name = patient_data.get("demographics", {}).get("name", "this patient")


        if not patient_mutations:
            logging.warning(f"[{self.name}] No mutations found for patient {patient_id} in _perform_holistic_genomic_analysis.")
            return {
                "analysis_summary": f"No mutations found for patient {patient_name} to analyze.",
                "natural_language_summary": "No mutations were available in the patient's record to perform a genomic analysis.",
                "details": [],
                "errors": ["No mutations provided for holistic analysis."],
                "parsed_intent": "analyze_genomic_profile", 
                "parsed_entities": {}
            }

        analysis_results: List[GenomicAnalysisResult] = []
        analysis_errors: List[str] = []
        
        # For demo, limit to first N mutations to avoid excessive processing time.
        # In a real scenario, you might have more sophisticated filtering or batching.
        mutations_to_analyze = patient_mutations[:5] 
        logging.info(f"[{self.name}] Performing holistic analysis for patient {patient_id} on {len(mutations_to_analyze)} mutations (out of {len(patient_mutations)} total).")

        for index, mutation_record in enumerate(mutations_to_analyze):
            gene_symbol = mutation_record.get("hugo_gene_symbol")
            protein_change = mutation_record.get("protein_change")
            variant_type = mutation_record.get("variant_type") # e.g., Missense_Mutation

            if not gene_symbol or not protein_change:
                logging.warning(f"[{self.name}] Skipping mutation due to missing gene/protein info: {mutation_record}")
                analysis_errors.append(f"Skipped mutation due to missing gene/protein info: {str(mutation_record)[:100]}")
                continue

            # Construct a query string, similar to what might come from a criterion.
            # Example: "Effect of BRAF V600E (Variant Type: Missense_Mutation)"
            variant_type_str = f" (Variant Type: {variant_type})" if variant_type else ""
            query_str = f"Effect of {gene_symbol} {protein_change}{variant_type_str}"
            
            criterion_id = f"{criterion_id_prefix}_{gene_symbol}_{protein_change}_{index}"

            try:
                logging.debug(f"[{self.name}] Analyzing query for holistic analysis: {query_str}")
                single_analysis_result: GenomicAnalysisResult = await self._analyze_single_genomic_query(
                    genomic_query=query_str,
                    patient_mutations=patient_mutations, # Pass all patient mutations for context
                    patient_id=patient_id,
                    criterion_id=criterion_id,
                    patient_data_for_context=patient_data # Pass full patient data for context generation
                )
                analysis_results.append(single_analysis_result)
            except Exception as e:
                logging.error(f"[{self.name}] Error analyzing mutation {gene_symbol} {protein_change} for patient {patient_id}: {e}", exc_info=True)
                analysis_errors.append(f"Error analyzing {gene_symbol} {protein_change}: {str(e)[:100]}")
                # Create a basic error entry for this mutation
                analysis_results.append(GenomicAnalysisResult(
                    criterion_id=criterion_id,
                    criterion_query=query_str,
                    status="ERROR",
                    evidence=f"Failed to analyze: {e}",
                    errors=[f"Failed to analyze: {str(e)[:100]}"]
                ))
        
        # --- Generate Old Summary (for backward compatibility / fallback) ---
        # This summary is quite generic.
        summary_parts = []
        for res in analysis_results:
            if res.status == "MET" and res.clinical_significance_context:
                 # Try to find the gene name from the query or VEP details.
                gene_name_for_summary = res.criterion_query.split(' ')[2] if len(res.criterion_query.split(' ')) > 2 else "Unknown Gene"
                if res.simulated_vep_details and res.simulated_vep_details[0].gene_symbol:
                     gene_name_for_summary = res.simulated_vep_details[0].gene_symbol

                # Extract only the first part of the context for brevity in this old summary
                brief_context = res.clinical_significance_context.split('.')[0] + '.' if res.clinical_significance_context else ""
                summary_parts.append(f"  {gene_name_for_summary} {res.criterion_query.split(' ')[3] if len(res.criterion_query.split(' ')) > 3 else 'variant'}: {brief_context}")


        old_analysis_summary = f"Holistic Genomic Analysis for Patient {patient_name}:\\n" + "\n".join(summary_parts)
        if not summary_parts:
            old_analysis_summary += "  No specific findings with detailed clinical context were identified from the analyzed mutations."
        old_analysis_summary += f"\\n\\nHolistic genomic analysis process completed for {patient_name}."

        # --- Generate New Natural Language Summary using LLM ---
        natural_language_summary_text = await self._generate_llm_summary_of_holistic_analysis(analysis_results, patient_name)

        final_result = {
            "analysis_summary": old_analysis_summary, # Keep old summary for now
            "natural_language_summary": natural_language_summary_text, # New LLM summary
            "details": [res.model_dump() for res in analysis_results],
            "errors": analysis_errors,
            "parsed_intent": "analyze_genomic_profile", # Hardcoded for this specialized function
            "parsed_entities": {} # No specific entities parsed at this level
        }
        return final_result

# Example usage (for testing, not part of the agent typically)
if __name__ == '__main__':
    # This section would need asyncio setup to run async methods
    # agent = GenomicAnalystAgent()
    # result = agent.run(patient_data=..., prompt_details=...)
    # print(result)
    pass

    # Mock patient data
    mock_patient_mutations_1 = [
        {"hugo_gene_symbol": "BRAF", "protein_change": "V600E", "variant_type": "Missense_Mutation"},
        {"hugo_gene_symbol": "TP53", "protein_change": "R248Q", "variant_type": "Missense_Mutation"},
        {"hugo_gene_symbol": "EGFR", "protein_change": "L858R", "variant_type": "Missense_Mutation"},
    ]
    mock_patient_mutations_2 = [
        {"hugo_gene_symbol": "KRAS", "protein_change": "G12D", "variant_type": "Missense_Mutation"},
        {"hugo_gene_symbol": "EGFR", "protein_change": "T790M", "variant_type": "Missense_Mutation"} 
    ]
    mock_patient_mutations_3_wt_kras = [
         {"hugo_gene_symbol": "TP53", "protein_change": "P72R", "variant_type": "Missense_Mutation"},
    ]


    queries = [
        "Effect of BRAF V600E",
        "Presence of activating KRAS mutation",
        "Absence of EGFR T790M resistance mutation",
        "EGFR wild-type",
        "Pathogenic mutation in TP53",
        "Nonsense_Mutation in BRCA1", # Patient has no BRCA1
        "Impact of PIK3CA H1047R", # Patient has no PIK3CA
        "ERBB2 negative", # HER2-negative
        "ALK mutation" # Patient no ALK
    ]

    print("\\n--- Testing with Patient 1 (BRAF V600E, TP53 R248Q, EGFR L858R) ---")
    for q in queries:
        print(f"\\n--- Query: {q} ---")
        result = agent.run(genomic_query=q, patient_id="PAT123", patient_mutations=mock_patient_mutations_1)
        print(f"Status: {result.status}")
        print(f"Gene Summaries: {result.gene_summary_statuses}")
        print(f"Clinical Context: {result.clinical_significance_context}")
        if result.errors: print(f"Errors: {result.errors}")
        # print(f"VEP Details: {result.simulated_vep_details}")
        # print(f"Evidence: \\n{result.evidence}")


    print("\\n\\n--- Testing with Patient 2 (KRAS G12D, EGFR T790M) ---")
    for q in queries:
        print(f"\\n--- Query: {q} ---")
        result = agent.run(genomic_query=q, patient_id="PAT456", patient_mutations=mock_patient_mutations_2)
        print(f"Status: {result.status}")
        print(f"Gene Summaries: {result.gene_summary_statuses}")
        print(f"Clinical Context: {result.clinical_significance_context}")
        if result.errors: print(f"Errors: {result.errors}")

    print("\\n\\n--- Testing with Patient 3 (TP53 P72R - effectively WT for KRAS) ---")
    kras_queries = [
        "KRAS wild-type",
        "Absence of KRAS G12C",
        "Activating KRAS mutation"
    ]
    for q in kras_queries:
        print(f"\\n--- Query: {q} ---")
        result = agent.run(genomic_query=q, patient_id="PAT789", patient_mutations=mock_patient_mutations_3_wt_kras)
        print(f"Status: {result.status}")
        print(f"Gene Summaries: {result.gene_summary_statuses}")
        print(f"Clinical Context: {result.clinical_significance_context}")
        if result.errors: print(f"Errors: {result.errors}") 