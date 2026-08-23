# Evidence evaluation package
from src.evidence.models import (
    DecisionStatus,
    EvidenceDecision,
    EvidenceItem,
    ConflictDetail,
)
from src.evidence.evaluator import EvidenceEvaluator, EvidenceConfig
from src.evidence.scoring import (
    extract_numeric_facts,
    extract_support_signals,
    extract_gap_signals,
    compute_relevance_score,
    NumericFact,
)
from src.evidence.contradiction import detect_conflicts

__all__ = [
    "DecisionStatus",
    "EvidenceDecision",
    "EvidenceItem",
    "ConflictDetail",
    "EvidenceEvaluator",
    "EvidenceConfig",
    "extract_numeric_facts",
    "extract_support_signals",
    "extract_gap_signals",
    "compute_relevance_score",
    "NumericFact",
    "detect_conflicts",
]
