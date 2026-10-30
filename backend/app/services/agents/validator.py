from __future__ import annotations

import re
from typing import Any

from app.services.agent_tools import NOT_PROVIDED, extract_user_leave_details, parse_leave_days
from app.services.rag import INSUFFICIENT_ANSWER, parse_citation_ids

CITATION_CHECK_NOTE = (
    "Citation ID checks confirm retrieved source labels only; they do not verify factual correctness."
)
ANNUAL_LIMIT_PATTERNS = (
    re.compile(
        r"up to\s+(\d{1,2})\s+days?\s+of\s+(?:paid\s+)?(?:vacation|leave|pto)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:receive|receives|entitled to|allotment of|allowance of|maximum of)\s+"
        r"(\d{1,2})\s+(?:vacation\s+|paid\s+|calendar\s+)?days",
        re.IGNORECASE,
    ),
    re.compile(r"(\d{1,2})\s+vacation days", re.IGNORECASE),
    re.compile(
        r"(\d{1,2})\s+days?\s+of\s+(?:paid\s+)?(?:vacation|leave)(?:\s+(?:annually|per year|each year|each calendar year))?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:annually|per year|each year|each calendar year)[^\d]{0,40}(\d{1,2})\s+days",
        re.IGNORECASE,
    ),
    re.compile(
        r"(\d{1,2})\s+days?[^\n.]{0,40}(?:annually|per year|each year|each calendar year)",
        re.IGNORECASE,
    ),
)
ADVANCE_NOTICE_RE = re.compile(
    r"at least\s+(\d{1,2})\s+days?\s+before",
    re.IGNORECASE,
)
CONSECUTIVE_RE = re.compile(
    r"(\d{1,2})\s+or more consecutive days",
    re.IGNORECASE,
)
OR_MORE_DAYS_RE = re.compile(
    r"(\d{1,2})\s+or more(\s+consecutive)?\s+days",
    re.IGNORECASE,
)
REASON_REQUIRED_RE = re.compile(
    r"(?:must specify|must provide|required to (?:give|provide|specify)|must include)"
    r".{0,80}\breason\b|\breason\b.{0,80}(?:must be specified|is required|must specify)",
    re.IGNORECASE,
)
DATE_REQUIRED_RE = re.compile(
    r"before planned leave|planned leave|start date|leave dates|date of leave",
    re.IGNORECASE,
)
EXCEEDING_DAYS_RE = re.compile(
    r"\b(?:exceeding|more than|over)\s+(\d{1,2})(?:\s+consecutive)?\s+days\b",
    re.IGNORECASE,
)
LATE_NOTICE_RE = re.compile(
    r"not\s+submitted|submitted\s+(?:fewer|less)\s+than|"
    r"late(?:r)?\s+(?:request|submission)|without\s+(?:enough\s+|sufficient\s+)?notice|"
    r"in\s+advance",
    re.IGNORECASE,
)
APPROVAL_RE = re.compile(
    r"manager\s+approval|require[sd]?\s+(?:manager\s+)?approval",
    re.IGNORECASE,
)
UNSUPPORTED_LATE_APPROVAL_CLAUSE_RE = re.compile(
    r"\s*,?\s*and\s+(?:for\s+)?(?:all\s+|any\s+)?"
    r"requests?\s+that\s+are\s+not\s+submitted[^\[.]*",
    re.IGNORECASE,
)


def validate_workflow(state: dict[str, Any]) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    intent = state.get("intent") or "informational"
    sources = state.get("sources") or []
    excerpts = state.get("excerpts") or []
    valid_ids = {item.get("source_id") for item in sources if item.get("source_id")}
    request = state.get("request") or ""
    answer = state.get("answer") or ""
    draft = state.get("draft")
    rules = extract_vacation_policy_rules(excerpts)
    corrected = answer
    abstain = False
    extra: dict[str, Any] = {}
    if intent != "draft_action" and answer:
        grounded = ground_informational_answer(answer, excerpts)
        corrected = grounded["answer"]
        issues.extend(grounded["issues"])
        abstain = grounded["abstain"]
        if abstain:
            extra = {
                "abstained": True,
                "grounded": False,
                "citation_valid": True,
                "status": "completed",
            }

    if intent == "draft_action":
        _validate_leave(request, draft, excerpts, valid_ids, rules, issues)
    else:
        _validate_informational(corrected, valid_ids, issues)

    errors = [item for item in issues if item["severity"] == "error"]
    passed = not errors
    citation_valid = extra.get("citation_valid", state.get("citation_valid"))
    if any(item["code"] == "unknown_citation" for item in errors):
        citation_valid = False
    return {
        "answer": corrected if intent != "draft_action" else state.get("answer"),
        "validation": {
            "passed": passed,
            "issues": _dedupe_issues(issues),
            "citation_check": "retrieved_id_match",
            "semantic_verification": False,
            "note": CITATION_CHECK_NOTE,
            "policy_rules": rules,
        },
        "citation_valid": citation_valid,
        **extra,
    }


def extract_vacation_policy_rules(excerpts: list[dict[str, Any]]) -> dict[str, Any]:
    text = _excerpt_text(excerpts)
    advance = ADVANCE_NOTICE_RE.search(text)
    consecutive = CONSECUTIVE_RE.search(text)
    return {
        "annual_days": _parse_annual_limit(text),
        "advance_notice_days": int(advance.group(1)) if advance else None,
        "consecutive_manager_threshold": int(consecutive.group(1)) if consecutive else None,
        "reason_required": bool(REASON_REQUIRED_RE.search(text)),
        "dates_required": bool(advance or DATE_REQUIRED_RE.search(text)),
        "late_submission_requires_approval": _late_submission_requires_approval(text),
        "balance_verified": False,
    }


def ground_informational_answer(
    answer: str,
    excerpts: list[dict[str, Any]],
) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    updated, threshold_changed = correct_threshold_wording(answer, excerpts)
    if threshold_changed:
        issues.append(
            {
                "code": "threshold_wording_corrected",
                "message": "Replaced 'exceeding N days' style wording with the policy's 'N or more' threshold.",
                "severity": "warning",
            }
        )
    if answer_claims_late_submission_requires_approval(updated) and not (
        extract_vacation_policy_rules(excerpts).get("late_submission_requires_approval")
    ):
        stripped = UNSUPPORTED_LATE_APPROVAL_CLAUSE_RE.sub("", updated)
        stripped = re.sub(r"[ \t]{2,}", " ", stripped)
        stripped = re.sub(r" +([.,;:])", r"\1", stripped).strip()
        if answer_claims_late_submission_requires_approval(stripped) or len(stripped) < 20:
            issues.append(
                {
                    "code": "unsupported_approval_trigger",
                    "message": (
                        "Answer treated a submission deadline as a manager-approval trigger, "
                        "which retrieved evidence does not support."
                    ),
                    "severity": "error",
                }
            )
            return {
                "answer": INSUFFICIENT_ANSWER,
                "issues": issues,
                "abstain": True,
            }
        issues.append(
            {
                "code": "unsupported_approval_trigger",
                "message": (
                    "Removed an unsupported claim that late submission requires manager approval. "
                    "Advance-notice rules are not approval triggers unless the policy says so."
                ),
                "severity": "warning",
            }
        )
        updated = stripped
    return {"answer": updated, "issues": issues, "abstain": False}


def answer_claims_late_submission_requires_approval(answer: str) -> bool:
    for part in _answer_parts(answer):
        if APPROVAL_RE.search(part) and LATE_NOTICE_RE.search(part):
            return True
    return False


def correct_threshold_wording(
    answer: str,
    excerpts: list[dict[str, Any]],
) -> tuple[str, bool]:
    text = _excerpt_text(excerpts)
    updated = answer
    for match in OR_MORE_DAYS_RE.finditer(text):
        number = match.group(1)
        consecutive = match.group(2) or ""
        replacement = f"{number} or more{consecutive} days"
        updated = re.sub(
            rf"\b(?:exceeding|more than|over)\s+{number}(?:\s+consecutive)?\s+days\b",
            replacement,
            updated,
            flags=re.IGNORECASE,
        )
    return updated, updated != answer


def _validate_informational(answer: str, valid_ids: set[str], issues: list[dict[str, str]]) -> None:
    cited = parse_citation_ids(answer)
    if any(label not in valid_ids for label in cited):
        issues.append(
            {
                "code": "unknown_citation",
                "message": "Informational answer cites IDs that are not in retrieved evidence.",
                "severity": "error",
            }
        )
    if answer and not cited:
        issues.append(
            {
                "code": "missing_citations",
                "message": "Factual answer has no retrieved-source citations.",
                "severity": "warning",
            }
        )
    if EXCEEDING_DAYS_RE.search(answer):
        issues.append(
            {
                "code": "threshold_wording_unverified",
                "message": "Answer uses 'exceeding N days' and could not be aligned to an 'N or more' policy phrase.",
                "severity": "warning",
            }
        )


def _validate_leave(
    request: str,
    draft: dict[str, Any] | None,
    excerpts: list[dict[str, Any]],
    valid_ids: set[str],
    rules: dict[str, Any],
    issues: list[dict[str, str]],
) -> None:
    provided = extract_user_leave_details(request)
    days = provided["days"]
    if days is None and isinstance((draft or {}).get("days"), int):
        days = draft.get("days")
    if days is None:
        issues.append(
            {
                "code": "missing_required_days",
                "message": "Leave request does not specify how many days are needed.",
                "severity": "error",
            }
        )
    if not excerpts:
        issues.append(
            {
                "code": "missing_policy_evidence",
                "message": "No retrieved policy evidence is available for this leave request.",
                "severity": "error",
            }
        )
        return
    allotment = rules.get("annual_days")
    if allotment is None:
        issues.append(
            {
                "code": "policy_rule_unverified",
                "message": "No annual vacation allowance could be parsed from retrieved evidence.",
                "severity": "error",
            }
        )
    elif days is not None and days > allotment:
        issues.append(
            {
                "code": "days_exceed_policy",
                "message": (
                    f"Requested {days} day(s) exceed the retrieved annual vacation allowance of "
                    f"{allotment} day(s)."
                ),
                "severity": "error",
            }
        )
    issues.append(
        {
            "code": "employee_balance_unknown",
            "message": "Annual allowance is not a remaining balance; eligibility was not verified.",
            "severity": "warning",
        }
    )
    dates_required = bool(rules.get("dates_required"))
    threshold = rules.get("consecutive_manager_threshold")
    if isinstance(days, int) and isinstance(threshold, int) and days >= threshold:
        dates_required = True
    reason_value = _draft_or_request_value(draft, "reason")
    if _is_missing(reason_value):
        reason_value = provided["reason"]
    start_date = _draft_or_request_value(draft, "start_date")
    if _is_missing(start_date):
        start_date = provided["start_date"]
    end_date = _draft_or_request_value(draft, "end_date")
    if _is_missing(end_date):
        end_date = provided["end_date"]
    if rules.get("reason_required") and _is_missing(reason_value):
        issues.append(
            {
                "code": "missing_reason",
                "message": "Retrieved policy requires a leave reason, and the user did not provide one.",
                "severity": "error",
            }
        )
    if dates_required and _is_missing(start_date) and _is_missing(end_date):
        issues.append(
            {
                "code": "missing_dates",
                "message": "Retrieved policy requires planned leave date information, which was not provided.",
                "severity": "error",
            }
        )
    if draft is None:
        return
    if draft.get("had_invalid_citations"):
        issues.append(
            {
                "code": "invalid_citations_removed",
                "message": "The action agent cited IDs that were not retrieved; those IDs were removed.",
                "severity": "warning",
            }
        )
    excerpt_text = _excerpt_text(excerpts).lower()
    for field in ("reason", "start_date", "end_date", "employee_name"):
        value = draft.get(field)
        if not isinstance(value, str) or _is_missing(value):
            continue
        if value.lower() in excerpt_text and value.lower() not in request.lower():
            issues.append(
                {
                    "code": "policy_used_as_personal_detail",
                    "message": f"{field} appears to be copied from policy text rather than the user request.",
                    "severity": "error",
                }
            )
    for label in draft.get("source_ids") or []:
        if label not in valid_ids:
            issues.append(
                {
                    "code": "unknown_citation",
                    "message": "Draft source_ids include a label that was not retrieved.",
                    "severity": "error",
                }
            )


def _parse_annual_limit(text: str) -> int | None:
    found: list[int] = []
    for pattern in ANNUAL_LIMIT_PATTERNS:
        for match in pattern.finditer(text):
            window = text[max(0, match.start() - 48) : min(len(text), match.end() + 48)].lower()
            if "before" in window or "consecutive" in window or "notice" in window:
                continue
            found.append(int(match.group(1)))
    if not found:
        return None
    return min(found)


def _late_submission_requires_approval(text: str) -> bool:
    for sentence in _sentences(text):
        if APPROVAL_RE.search(sentence) and LATE_NOTICE_RE.search(sentence):
            return True
    return False


def _answer_parts(answer: str) -> list[str]:
    parts: list[str] = []
    for sentence in _sentences(answer):
        parts.append(sentence)
        parts.extend(
            piece.strip()
            for piece in re.split(r"\s+and\s+for\s+", sentence, flags=re.IGNORECASE)
            if piece.strip()
        )
    return parts


def _sentences(text: str) -> list[str]:
    return [piece.strip() for piece in re.split(r"(?<=[.!?])\s+", text) if piece.strip()]


def _excerpt_text(excerpts: list[dict[str, Any]]) -> str:
    return " ".join(item.get("content") or "" for item in excerpts)


def _draft_or_request_value(draft: dict[str, Any] | None, field: str) -> str | None:
    if not draft:
        return None
    value = draft.get(field)
    return value if isinstance(value, str) else None


def _is_missing(value: str | None) -> bool:
    if value is None:
        return True
    return value.strip() in {"", NOT_PROVIDED, "none", "null", "n/a"}


def _dedupe_issues(issues: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, str]] = []
    for issue in issues:
        key = (issue["code"], issue["message"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(issue)
    return unique
