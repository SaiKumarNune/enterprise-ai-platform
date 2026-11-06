"""Small isolated corpus used only by the benchmark runner."""

from __future__ import annotations

DOCUMENTS = [
    {
        "id": "leave",
        "filename": "leave_policy.txt",
        "text": (
            "Full-time employees receive 15 vacation days each calendar year. "
            "Unused vacation does not carry over. Employees may take 10 paid sick days each year. "
            "A doctor's note is required after three consecutive sick days."
        ),
        "chunks": [
            "Full-time employees receive 15 vacation days each calendar year. Unused vacation does not carry over.",
            "Employees may take 10 paid sick days each year. A doctor's note is required after three consecutive sick days.",
        ],
    },
    {
        "id": "badge",
        "filename": "badge_policy.txt",
        "text": "Lost badges must be reported to security within 2 hours. Replacement badges require manager approval.",
        "chunks": [
            "Lost badges must be reported to security within 2 hours.",
            "Replacement badges require manager approval.",
        ],
    },
    {
        "id": "hours",
        "filename": "office_hours.txt",
        "text": "The office opens at 09:00 on weekdays and closes at 18:00. The cafeteria menu changes on Fridays.",
        "chunks": [
            "The office opens at 09:00 on weekdays and closes at 18:00.",
            "The cafeteria menu changes on Fridays.",
        ],
    },
]

QUERIES = [
    "How many vacation days do full-time employees receive?",
    "When is a doctor's note required for sick leave?",
    "How must lost badges be reported?",
    "When does the office open?",
]
