🧠 Event–Entity Graph-Based Summarization for Manipuri News

A structure-aware, linguistically grounded summarization framework for low-resource languages, specifically designed for Manipuri (Meiteilon).

This work introduces a deterministic, interpretable pipeline that avoids hallucination by grounding summaries in an event–entity graph.

⸻

🚀 Overview

Traditional summarization models struggle in low-resource settings due to:
	•	Limited training data
	•	Rich morphology
	•	Lack of reliable linguistic tools

This project addresses these challenges using a structured approach:

💡 First decide what to say (planning), then how to say it (realization).

⸻

🏗️ Architecture
Document → Feature Extraction → Event–Entity Graph → Planner → Realizer → Verified Summary

Key Components:
	•	Feature Extraction → Named entities + events
	•	Graph Construction → Semantic representation of document
	•	Planner → Selects salient events and entities
	•	Realizer → Generates summary using templates
	•	Verification → Ensures factual consistency

⸻

✨ Key Features
	•	✅ Hallucination-free summarization
	•	✅ Fully interpretable pipeline
	•	✅ Graph-based semantic modeling
	•	✅ Evidence-grounded outputs ([s1], [s2])
	•	✅ Works without large training data

⸻
 Repository Structure
.
├── dataset/
│   └── cleaned_dataset.json
├── planner_ready_outputs/
├── planner_outputs/
├── realizer_outputs/
├── verification_outputs/
├── main.py
├── ablation.py
└── README.md
