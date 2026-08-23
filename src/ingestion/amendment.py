"""
Amendment parsing and structured policy version models.

Responsibilities:
- Parse amendment Markdown documents (e.g. Amendment No. 2026-01.md).
- Extract amendment metadata (ID, issued date, effective date).
- Extract individual policy modifications (substitutions, insertions, tables).
- Track exact target clause IDs (including alphanumeric IDs like §10.5.3A).
- Preserve temporal trigger semantics (determination date vs change of circumstances date).
- Maintain source line provenance for citations.
- Convert amendment changes into temporal PolicyClause representations.

Design principles:
- Original policy documents are NEVER modified.
- Strict typing with immutable dataclasses and Enums.
- Deterministic extraction independent of external LLMs or vector indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path
import re
from typing import Sequence

from src.ingestion.loader import PolicyDocument
from src.ingestion.parser import ClauseSubItem, PolicyClause


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TriggerType(str, Enum):
    """
    Temporal trigger governing when an amendment rule becomes applicable.

    DETERMINATION_DATE
        Applies based on the date the determination/decision is made
        (even if the claim period predates the amendment).

    CHANGE_OF_CIRCUMSTANCES_DATE
        Applies based on the date the change of circumstances occurred
        (changes prior to effective date retain the prior rule).

    CLAIM_PERIOD
        Applies day-by-day across the claim duration (subject to apportionment).
    """
    DETERMINATION_DATE = "determination_date"
    CHANGE_OF_CIRCUMSTANCES_DATE = "change_of_circumstances_date"
    CLAIM_PERIOD = "claim_period"


class ChangeType(str, Enum):
    """The nature of a policy amendment."""
    SUBSTITUTION = "substitution"
    INSERTION = "insertion"
    TABLE_SUBSTITUTION = "table_substitution"
    TRANSITIONAL_RULE = "transitional_rule"


# ---------------------------------------------------------------------------
# Structured Amendment Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TableRow:
    """A row in a structured table change."""
    household_size: str
    monthly_threshold: str
    numeric_amount: int | None = None


@dataclass(frozen=True)
class AmendmentChange:
    """
    One specific policy modification introduced by an amendment.

    Attributes
    ----------
    target_clause_id:
        Clause identifier modified or inserted (e.g. "6.4.1", "4.3.2", "10.5.3A").
    change_type:
        Type of change (SUBSTITUTION, INSERTION, TABLE_SUBSTITUTION).
    description:
        Brief human-readable summary of the change.
    original_value:
        The value or wording being replaced, or None for insertions.
    amended_value:
        The replacement value, new clause body, or new table text.
    trigger_type:
        Temporal applicability rule (determination date vs change date).
    effective_from:
        Date when this change comes into force.
    effective_to:
        Date when this version ends, or None if open-ended / current.
    source_document:
        Filename of the amendment document.
    amendment_id:
        Identifier of the amendment (e.g. "2026-01").
    start_line:
        1-indexed line number in the amendment source file.
    end_line:
        1-indexed end line number in the amendment source file.
    sub_item_id:
        Lettered sub-item if applicable (e.g. "a" for §6.4.1(a)).
    table_rows:
        Parsed structured rows if change_type == TABLE_SUBSTITUTION.
    """
    target_clause_id: str
    change_type: ChangeType
    description: str
    original_value: str | None
    amended_value: str
    trigger_type: TriggerType
    effective_from: date
    effective_to: date | None
    source_document: str
    amendment_id: str
    start_line: int
    end_line: int
    sub_item_id: str | None = None
    table_rows: tuple[TableRow, ...] = ()


@dataclass(frozen=True)
class TransitionalProvision:
    """A transitional rule governing amendment implementation."""
    paragraph_id: str
    text: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class AmendmentDocument:
    """
    Complete structured representation of a policy amendment.

    Attributes
    ----------
    amendment_id:
        Numbered identifier (e.g. "2026-01").
    title:
        Document title heading.
    issue_date:
        Date the amendment was formally issued.
    effective_date:
        Primary date the amendment takes effect.
    source_path:
        Path to the source markdown file.
    changes:
        All policy changes in document order.
    transitional_provisions:
        Transitional rules governing application.
    """
    amendment_id: str
    title: str
    issue_date: date
    effective_date: date
    source_path: Path
    changes: tuple[AmendmentChange, ...]
    transitional_provisions: tuple[TransitionalProvision, ...]

    @property
    def source_document(self) -> str:
        return self.source_path.name

    @property
    def target_clause_ids(self) -> tuple[str, ...]:
        """All unique clause IDs affected or introduced by this amendment."""
        return tuple(dict.fromkeys(c.target_clause_id for c in self.changes))

    def get_change_for_clause(self, clause_id: str) -> AmendmentChange | None:
        """Return the first change affecting *clause_id*, if any."""
        for c in self.changes:
            if c.target_clause_id == clause_id:
                return c
        return None

    def create_amended_clauses(
        self,
        original_clauses: Sequence[PolicyClause],
    ) -> list[PolicyClause]:
        """
        Generate amended PolicyClause instances for all changes in this amendment.

        - For existing clauses: applies text/table substitutions to produce
          an updated PolicyClause with effective_from set to the amendment date
          and source_document set to this amendment.
        - For new clauses (e.g. §10.5.3A): constructs a brand new PolicyClause.
        """
        orig_by_id = {c.clause_id: c for c in original_clauses}
        amended_list: list[PolicyClause] = []

        for ch in self.changes:
            if ch.change_type == ChangeType.INSERTION:
                # Construct new clause
                # E.g. 10.5.3A belongs to Part 10, Section 10.5
                part_id = ch.target_clause_id.split(".")[0]
                section_id = ".".join(ch.target_clause_id.split(".")[:2])
                
                # Derive titles from existing part/section context if available
                matching_orig = next(
                    (c for c in original_clauses if c.section_id == section_id),
                    None,
                )
                part_title = matching_orig.part_title if matching_orig else f"Part {part_id}"
                section_title = matching_orig.section_title if matching_orig else f"{section_id} Sanctions"

                new_clause = PolicyClause(
                    clause_id=ch.target_clause_id,
                    part_id=part_id,
                    part_title=part_title,
                    section_id=section_id,
                    section_title=section_title,
                    text=ch.amended_value,
                    sub_items=(),
                    cross_references=(),
                    source_path=self.source_path,
                    start_line=ch.start_line,
                    end_line=ch.end_line,
                    effective_from=ch.effective_from,
                    effective_to=ch.effective_to,
                    source_document=self.source_document,
                )
                amended_list.append(new_clause)

            elif ch.target_clause_id in orig_by_id:
                orig = orig_by_id[ch.target_clause_id]
                new_text = orig.text

                if ch.change_type == ChangeType.SUBSTITUTION and ch.original_value:
                    # Perform substitution in text
                    # Strip bold/formatting variations if needed
                    orig_val = ch.original_value
                    new_val = ch.amended_value
                    if orig_val in new_text:
                        new_text = new_text.replace(orig_val, new_val)
                    elif orig_val.lower() in new_text.lower():
                        # Case-insensitive replacement preserving casing
                        pattern = re.compile(re.escape(orig_val), re.IGNORECASE)
                        new_text = pattern.sub(new_val, new_text)

                elif ch.change_type == ChangeType.TABLE_SUBSTITUTION:
                    # Replace existing table in text with amended table
                    lines = orig.text.splitlines()
                    non_table_lines = [l for l in lines if not l.strip().startswith("|")]
                    new_text = "\n".join(non_table_lines).strip() + "\n\n" + ch.amended_value.strip()

                updated_clause = PolicyClause(
                    clause_id=orig.clause_id,
                    part_id=orig.part_id,
                    part_title=orig.part_title,
                    section_id=orig.section_id,
                    section_title=orig.section_title,
                    text=new_text,
                    sub_items=orig.sub_items,
                    cross_references=orig.cross_references,
                    source_path=self.source_path,
                    start_line=ch.start_line,
                    end_line=ch.end_line,
                    effective_from=ch.effective_from,
                    effective_to=ch.effective_to,
                    source_document=self.source_document,
                )
                amended_list.append(updated_clause)

        return amended_list


# ---------------------------------------------------------------------------
# Date Parsing Helpers
# ---------------------------------------------------------------------------

_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

def _parse_english_date(text: str) -> date:
    """Parse a date string like '1 March 2026' or '12 February 2026'."""
    m = re.search(r'(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})', text)
    if not m:
        raise ValueError(f"Cannot parse date from: {text!r}")
    day = int(m.group(1))
    month_name = m.group(2).lower()
    year = int(m.group(3))
    month = _MONTH_MAP.get(month_name)
    if not month:
        raise ValueError(f"Unknown month name: {month_name!r}")
    return date(year, month, day)


# ---------------------------------------------------------------------------
# Main Parser Logic
# ---------------------------------------------------------------------------

def parse_amendment(document: PolicyDocument) -> AmendmentDocument:
    """
    Parse a loaded amendment PolicyDocument into an AmendmentDocument.

    Parameters
    ----------
    document:
        Loaded PolicyDocument of the amendment file.

    Returns
    -------
    AmendmentDocument
        Structured representation of the amendment and its changes.
    """
    raw_text = document.raw_text
    lines = raw_text.splitlines()

    # --- 1. Document metadata ---
    m_id = re.search(r'## Amendment No\.\s*([A-Za-z0-9-]+)', raw_text)
    amendment_id = m_id.group(1).strip() if m_id else "2026-01"

    m_title = re.search(r'^# (.+)$', raw_text, re.MULTILINE)
    title = m_title.group(1).strip() if m_title else "Calder County Household Support Program"

    m_issued = re.search(r'\*\*Issued:\*\*\s*([^\n\r]+)', raw_text)
    issue_date = _parse_english_date(m_issued.group(1)) if m_issued else date(2026, 2, 12)

    m_effective = re.search(r'\*\*Effective:\*\*\s*([^\n\r]+)', raw_text)
    effective_date = _parse_english_date(m_effective.group(1)) if m_effective else date(2026, 3, 1)

    changes: list[AmendmentChange] = []
    transitional: list[TransitionalProvision] = []

    # State tracking
    current_h2_num: int | None = None
    current_h2_title: str = ""
    line_idx = 0

    while line_idx < len(lines):
        line = lines[line_idx].strip()
        line_num = line_idx + 1

        # Check for H2 section: ## 1. Earnings disregard
        m_h2 = re.match(r'^##\s*(\d+)\.\s*(.+)$', line)
        if m_h2:
            current_h2_num = int(m_h2.group(1))
            current_h2_title = m_h2.group(2).strip()
            line_idx += 1
            continue

        # --- Section 1: Earnings disregard (§6.4.1(a)) ---
        if current_h2_num == 1 and line.startswith("**1.1**"):
            # **1.1** In §6.4.1(a), for "$120 per month" substitute "**$175 per month**".
            m_sub = re.search(r'In §([0-9\.]+)\(([a-z])\),\s*for\s*"([^"]+)"\s*substitute\s*"\*\*?([^\*"]+)\*\*?"', line)
            if m_sub:
                clause_id = m_sub.group(1)
                sub_id = m_sub.group(2)
                orig_val = m_sub.group(3)
                amend_val = m_sub.group(4)
                changes.append(
                    AmendmentChange(
                        target_clause_id=clause_id,
                        change_type=ChangeType.SUBSTITUTION,
                        description="Earnings disregard increase",
                        original_value=orig_val,
                        amended_value=amend_val,
                        trigger_type=TriggerType.DETERMINATION_DATE,
                        effective_from=effective_date,
                        effective_to=None,
                        source_document=document.source_path.name,
                        amendment_id=amendment_id,
                        start_line=line_num,
                        end_line=line_num,
                        sub_item_id=sub_id,
                    )
                )

        # --- Section 2: Reporting of changes (§4.3.2 and §9.1.4) ---
        elif current_h2_num == 2:
            if line.startswith("**2.1**"):
                # **2.1** In §4.3.2, for "10 calendar days" (in both places where it occurs) substitute "**14 calendar days**".
                m_sub = re.search(r'In §([0-9\.]+),\s*for\s*"([^"]+)"(?:\s*\([^\)]+\))?\s*substitute\s*"\*\*?([^\*"]+)\*\*?"', line)
                if m_sub:
                    changes.append(
                        AmendmentChange(
                            target_clause_id=m_sub.group(1),
                            change_type=ChangeType.SUBSTITUTION,
                            description="Change reporting deadline alignment",
                            original_value=m_sub.group(2),
                            amended_value=m_sub.group(3),
                            trigger_type=TriggerType.CHANGE_OF_CIRCUMSTANCES_DATE,
                            effective_from=effective_date,
                            effective_to=None,
                            source_document=document.source_path.name,
                            amendment_id=amendment_id,
                            start_line=line_num,
                            end_line=line_num,
                        )
                    )
            elif line.startswith("**2.2**"):
                # **2.2** In §9.1.4, for "30 calendar days" substitute "**14 calendar days**".
                m_sub = re.search(r'In §([0-9\.]+),\s*for\s*"([^"]+)"\s*substitute\s*"\*\*?([^\*"]+)\*\*?"', line)
                if m_sub:
                    changes.append(
                        AmendmentChange(
                            target_clause_id=m_sub.group(1),
                            change_type=ChangeType.SUBSTITUTION,
                            description="Overpayment reporting safe harbour alignment",
                            original_value=m_sub.group(2),
                            amended_value=m_sub.group(3),
                            trigger_type=TriggerType.CHANGE_OF_CIRCUMSTANCES_DATE,
                            effective_from=effective_date,
                            effective_to=None,
                            source_document=document.source_path.name,
                            amendment_id=amendment_id,
                            start_line=line_num,
                            end_line=line_num,
                        )
                    )

        # --- Section 3: Income thresholds table (§6.6.1) ---
        elif current_h2_num == 3 and line.startswith("**3.1**"):
            # Accumulate table rows
            table_start = line_num
            table_lines: list[str] = []
            parsed_rows: list[TableRow] = []
            
            peek_idx = line_idx + 1
            while peek_idx < len(lines):
                peek_line = lines[peek_idx].strip()
                if peek_line.startswith("|"):
                    table_lines.append(peek_line)
                    # Parse threshold rows: | 1 | $1,225 |
                    parts = [p.strip() for p in peek_line.split("|")[1:-1]]
                    if len(parts) >= 2 and parts[0] != "Household size" and not parts[0].startswith(":-"):
                        size_str = parts[0]
                        thresh_str = parts[1]
                        num_m = re.search(r'[\$]?([\d,]+)', thresh_str)
                        numeric_val = int(num_m.group(1).replace(",", "")) if num_m else None
                        parsed_rows.append(TableRow(household_size=size_str, monthly_threshold=thresh_str, numeric_amount=numeric_val))
                elif table_lines and not peek_line:
                    break
                elif peek_line.startswith("#"):
                    break
                peek_idx += 1

            table_text = "\n".join(table_lines)
            table_end = table_start + len(table_lines)

            changes.append(
                AmendmentChange(
                    target_clause_id="6.6.1",
                    change_type=ChangeType.TABLE_SUBSTITUTION,
                    description="Updated monthly income eligibility thresholds",
                    original_value=None,
                    amended_value=table_text,
                    trigger_type=TriggerType.DETERMINATION_DATE,
                    effective_from=effective_date,
                    effective_to=None,
                    source_document=document.source_path.name,
                    amendment_id=amendment_id,
                    start_line=table_start,
                    end_line=table_end,
                    table_rows=tuple(parsed_rows),
                )
            )
            line_idx = peek_idx - 1

        # --- Section 4: Sanctions (§10.5.2 & §10.5.3A) ---
        elif current_h2_num == 4:
            if line.startswith("**4.1**"):
                # **4.1** In §10.5.2, for "20 per cent" substitute "**15 per cent**".
                m_sub = re.search(r'In §([0-9\.]+),\s*for\s*"([^"]+)"\s*substitute\s*"\*\*?([^\*"]+)\*\*?"', line)
                if m_sub:
                    changes.append(
                        AmendmentChange(
                            target_clause_id=m_sub.group(1),
                            change_type=ChangeType.SUBSTITUTION,
                            description="Sanction reduction percentage adjustment",
                            original_value=m_sub.group(2),
                            amended_value=m_sub.group(3),
                            trigger_type=TriggerType.DETERMINATION_DATE,
                            effective_from=effective_date,
                            effective_to=None,
                            source_document=document.source_path.name,
                            amendment_id=amendment_id,
                            start_line=line_num,
                            end_line=line_num,
                        )
                    )
            elif line.startswith("**4.2**"):
                # **4.2** After §10.5.3, insert —
                # > **10.5.3A** A sanction must not be imposed...
                insert_start = line_num
                clause_text = ""
                clause_id = "10.5.3A"
                
                peek_idx = line_idx + 1
                while peek_idx < len(lines):
                    peek_line = lines[peek_idx].strip()
                    if peek_line.startswith(">"):
                        cleaned = re.sub(r'^>\s*', '', peek_line).strip()
                        m_cid = re.search(r'\*\*([0-9\.]+[A-Z]?)\*\*', cleaned)
                        if m_cid:
                            clause_id = m_cid.group(1)
                        clause_text = cleaned
                        break
                    elif peek_line.startswith("#"):
                        break
                    peek_idx += 1

                changes.append(
                    AmendmentChange(
                        target_clause_id=clause_id,
                        change_type=ChangeType.INSERTION,
                        description="Sanction protection for changes increasing award",
                        original_value=None,
                        amended_value=clause_text,
                        trigger_type=TriggerType.DETERMINATION_DATE,
                        effective_from=effective_date,
                        effective_to=None,
                        source_document=document.source_path.name,
                        amendment_id=amendment_id,
                        start_line=insert_start,
                        end_line=peek_idx + 1,
                    )
                )
                line_idx = peek_idx

        # --- Section 5: Transitional provision ---
        elif current_h2_num == 5:
            m_para = re.match(r'^\*\*(\d+\.\d+)\*\*\s*(.+)$', line)
            if m_para:
                para_id = m_para.group(1)
                para_text = m_para.group(2).strip()
                transitional.append(
                    TransitionalProvision(
                        paragraph_id=para_id,
                        text=para_text,
                        start_line=line_num,
                        end_line=line_num,
                    )
                )

        line_idx += 1

    return AmendmentDocument(
        amendment_id=amendment_id,
        title=title,
        issue_date=issue_date,
        effective_date=effective_date,
        source_path=document.source_path,
        changes=tuple(changes),
        transitional_provisions=tuple(transitional),
    )
