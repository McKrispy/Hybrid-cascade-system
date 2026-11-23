"""Task schemas, label definitions, criteria, and canonical descriptions.

Timestamp: 2026-11-09 13:16 (Asia/Shanghai)

Provides canonical fixed label order, human-readable descriptions, and task schema registry
for Banking77, CLINC150 (with explicit REJECT rule), ANLI R2/R3, and StrategyQA.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.infra.paths import get_data_dir


class TaskLabelSchema(BaseModel):
    label_schema_id: str
    task_type: str  # "choice" or "noul"
    allowed_labels: List[str]
    label_descriptions: Dict[str, str]
    reject_allowed: bool = False
    reject_label: Optional[str] = None
    reject_description: Optional[str] = None


def _format_intent_description(intent: str, domain: str = "") -> str:
    """Format snake_case intent into a clear, grammatical English description."""
    words = intent.replace("_", " ").strip()
    if domain:
        return f"Customer inquiries or requests concerning {words} in {domain}."
    return f"User requests or inquiries concerning {words}."


def build_banking77_schema(raw_dir: Path) -> TaskLabelSchema:
    """Build canonical label schema for Banking77 (77 fine-grained classes)."""
    cat_file = raw_dir / "banking77" / "categories.json"
    with open(cat_file, "r", encoding="utf-8") as f:
        categories = json.load(f)

    # Maintain deterministic sorted canonical order
    categories = sorted(categories)
    descriptions = {}
    for cat in categories:
        descriptions[cat] = _format_intent_description(cat, "online banking and card services")

    return TaskLabelSchema(
        label_schema_id="banking77",
        task_type="choice",
        allowed_labels=categories,
        label_descriptions=descriptions,
        reject_allowed=False,
    )


def build_clinc150_schema(raw_dir: Path) -> TaskLabelSchema:
    """Build canonical label schema for CLINC150 (150 ID classes + REJECT)."""
    full_file = raw_dir / "clinc150" / "data_full.json"
    with open(full_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    test_samples = data.get("test", [])
    intents = sorted(list(set(intent for _, intent in test_samples)))
    if len(intents) != 150:
        raise ValueError(f"Expected 150 unique intents in CLINC test set, found {len(intents)}")

    descriptions = {}
    for intent in intents:
        descriptions[intent] = _format_intent_description(intent, "virtual assistant commands")

    # Add standard REJECT definition required by plan.md line 161
    reject_desc = (
        "Out-of-scope query that cannot be answered or does not match any of the supported 150 assistant intent categories."
    )
    descriptions["REJECT"] = reject_desc

    return TaskLabelSchema(
        label_schema_id="clinc150",
        task_type="choice",
        allowed_labels=intents,
        label_descriptions=descriptions,
        reject_allowed=True,
        reject_label="REJECT",
        reject_description=reject_desc,
    )


def build_anli_schema() -> TaskLabelSchema:
    """Build canonical label schema for ANLI R2/R3 (3-class NLI)."""
    labels = ["entailment", "neutral", "contradiction"]
    descriptions = {
        "entailment": "The premise provides sufficient factual evidence that the hypothesis is definitely true.",
        "neutral": "The premise does not provide enough information to confirm or deny the hypothesis.",
        "contradiction": "The premise directly contradicts or refutes the hypothesis, making the hypothesis false.",
    }
    return TaskLabelSchema(
        label_schema_id="anli_r2r3",
        task_type="choice",
        allowed_labels=labels,
        label_descriptions=descriptions,
        reject_allowed=False,
    )


def build_strategyqa_schema() -> TaskLabelSchema:
    """Build canonical label schema for StrategyQA (2-class yes/no Noul)."""
    labels = ["yes", "no"]
    descriptions = {
        "yes": "The answer to the multi-step strategy reasoning question is affirmative / True.",
        "no": "The answer to the multi-step strategy reasoning question is negative / False.",
    }
    return TaskLabelSchema(
        label_schema_id="strategyqa",
        task_type="noul",
        allowed_labels=labels,
        label_descriptions=descriptions,
        reject_allowed=False,
    )


def generate_and_save_label_schemas(raw_dir: Path, out_path: Optional[Path] = None) -> Dict[str, Any]:
    """Generate all four task schemas and persist to label_schemas.json."""
    if out_path is None:
        out_path = get_data_dir() / "label_schemas.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    schemas = {
        "banking77": build_banking77_schema(raw_dir).model_dump(),
        "clinc150": build_clinc150_schema(raw_dir).model_dump(),
        "anli_r2r3": build_anli_schema().model_dump(),
        "strategyqa": build_strategyqa_schema().model_dump(),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(schemas, f, indent=2, ensure_ascii=False)

    return schemas


def load_label_schemas(schema_path: Optional[Path] = None) -> Dict[str, TaskLabelSchema]:
    """Load persisted label schemas."""
    if schema_path is None:
        schema_path = get_data_dir() / "label_schemas.json"
    with open(schema_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: TaskLabelSchema(**v) for k, v in data.items()}

