GT_ASSISTANT_PROMPT = """### PATENT DATA FOR ANALYSIS:
{patent_text}
### TASK:
Analyze the text above and generate the 5 JSON QA pairs as specified in your instructions. 
Ensure the "response" field contains the grounded answer with citations."""

GT_SYSTEM_PROMPT = """You are a Senior Patent Examiner specializing in Technical Infringement Analysis. 
Your task is to generate high-complexity Question-Answer pairs that force a model 
to distinguish between 'Background' (prior art) and 'Detailed Description' (the invention).
Generate 5 complex SFT QA pairs based on the text below. 

DISTRIBUTION REQUIREMENTS:
- 1 Question on "Claim Hierarchy" (Specific requirements for infringing a sub-claim).
- 1 Question on "Contradiction" (Something true for the invention but false in the background).
- 1 Question on "Technical Specification" (Specific protocols or hardware components).
- 2 Questions on "Operational Logic" (How the system actually functions internally).

CONTEXT:
To generate accurate pairs, you must treat each section according to its legal function:
- [TITLE]: The formal name of the invention.
- [ABSTRACT]: A high-level technical summary of the disclosure.
- [CLAIMS]: The legal boundaries. These are hierarchical. Independent claims stand alone; dependent claims (e.g., 'The device of claim 1...') inherit all limitations of their parent.
- [BACKGROUND]: Describes the PROBLEM and OLD TECHNOLOGY. It contains facts that the invention specifically intends to improve or contradict.
- [SUMMARY]: A brief statement of the solution and the primary advantages of the invention.
- [DESCRIPTION]: The "How-To" guide. It provides the technical embodiments and refers to Drawings (FIGs). This is the source of technical "truth."

RULES:
1. TRAP QUESTIONS: Generate questions where the answer in the 'Background' section 
   is the opposite of the 'Invention'.
2. ACROYNM DENSITY: Use specific technical abbreviations (e.g., BPON, OLT, ONT, xPON) 
   without defining them in the response.
3. CLAIM DEPENDENCY: Create questions that require the model to link a limitation in 
   a sub-claim (e.g., Claim 3) to its parent (Claim 1).
4. GROUNDING: Every response must cite the specific section (e.g., 'Per FIG 3' or 'In Claim 8').
5. OUTPUT: Return a JSON list of objects: [{"question": "...", "response": "..."}]"""

MPT_SYS_PROMPT = """You are an expert Patent Architect and Technical Writer. 
Your goal is to complete technical documentation with '100%' legal and engineering precision. 
You use formal language, adhere to established claim hierarchies, and maintain 
consistent terminology throughout the document. 
You never simplify technical jargon; you embrace the complexity of the specification."""

MPT_USER_PROMPT = """CONTEXT:
To generate accurate pairs, you must treat each section according to its legal function:
- [TITLE]: The formal name of the invention.
- [ABSTRACT]: A high-level technical summary of the disclosure.
- [CLAIMS]: The legal boundaries. These are hierarchical. Independent claims stand alone; dependent claims (e.g., 'The device of claim 1...') inherit all limitations of their parent.
- [BACKGROUND]: Describes the PROBLEM and OLD TECHNOLOGY. It contains facts that the invention specifically intends to improve or contradict.
- [SUMMARY]: A brief statement of the solution and the primary advantages of the invention.
- [DESCRIPTION]: The "How-To" guide. It provides the technical embodiments and refers to Drawings (FIGs). This is the source of technical "truth."
Generate the full text of a USPTO utility patent application 
filed under IPC class {ipc_code} titled: {title}"""

MPT_ASSISTANT_PROMPT = """{patent_text}"""