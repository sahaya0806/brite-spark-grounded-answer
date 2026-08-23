import json
import os
import sys
from pathlib import Path

# Ensure UTF-8 output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

from src.pipeline import PolicyQAPipeline

questions = [
    ("Q1", "What information must an applicant provide?"),
    ("Q2", "What evidence is required to establish an applicant's identity, residence, income, and resources?"),
    ("Q3", "What are the recipient's obligations to report changes in circumstances?"),
    ("Q4", "What income threshold is used when assessing eligibility?"),
    ("Q5", "What income can be disregarded when calculating entitlement?"),
    ("Q6", "How many days does a recipient have to report a change?"),
    ("Q7", "What is the policy for full-time students?"),
    ("Q8", "What is the policy for a household that owns three electric vehicles?"),
    ("Q9", "Does the program provide a special benefit for households affected by flooding?"),
    ("Q10", "What rule applies to full-time students under the policy?"),
]

corpus_path = Path("data/raw/policy_manual.md")
print(f"Building pipeline on {corpus_path}...")
pipeline = PolicyQAPipeline.build_from_corpus(corpus_path)

eval_records = []

for qid, q in questions:
    print(f"\n==================================================")
    print(f"{qid}: {q}")
    print(f"==================================================")
    answer = pipeline.ask(q)
    print(f"Status: [{answer.status.value}]")
    print(f"Answer: {answer.answer_text}")
    print(f"Citations:")
    for c in answer.citations:
        print(f"  - {c}")
    print(f"Supporting Clause IDs: {answer.supporting_clause_ids}")
    print(f"Refusal: {answer.refusal}")
    if answer.conflicts:
        print("Conflicts:")
        for cf in answer.conflicts:
            print(f"  - §{cf.clause_a.clause_id} ({cf.value_a}) vs §{cf.clause_b.clause_id} ({cf.value_b})")

    eval_records.append({
        "qid": qid,
        "question": q,
        "status": answer.status.value,
        "answer": answer.answer_text,
        "citations": list(answer.citations),
        "supporting_clause_ids": list(answer.supporting_clause_ids),
        "refusal": answer.refusal,
        "rationale": answer.rationale,
        "conflicts": [
            {
                "clause_a": cf.clause_a.clause_id,
                "value_a": cf.value_a,
                "clause_b": cf.clause_b.clause_id,
                "value_b": cf.value_b,
                "explanation": cf.explanation,
            }
            for cf in answer.conflicts
        ]
    })

with open("evaluation_10_questions.json", "w", encoding="utf-8") as f:
    json.dump(eval_records, f, indent=2, ensure_ascii=False)

print("\nEvaluation completed successfully! Results saved to evaluation_10_questions.json")
