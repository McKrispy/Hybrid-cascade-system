"""Data adapters for reading and standardizing the four benchmark datasets.

Timestamp: 2026-11-09 13:16 (Asia/Shanghai)

Transforms raw data into canonical intermediate sample records:
- Banking77: 3,080 test queries across 77 fine-grained intents
- CLINC150: 1,500 ID queries (10 per class) + 500 OOS queries from oos_test
- ANLI: 1,000 R2 + 1,200 R3 test queries, premise family grouping
- StrategyQA: 2,000 questions sampled deterministically from public train pool
"""

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class IntermediateSample(BaseModel):
    dataset_id: str
    source_split: str
    task_type: str  # "choice" or "noul"
    text_fields: Dict[str, str]
    label_schema_id: str
    gold_label: str
    is_oos: bool = False
    group_key: str  # raw text or premise used for deterministic clustering
    analysis_annotations: Dict[str, Any] = Field(default_factory=dict)
    source_id: Optional[str] = None


def normalize_text(text: str) -> str:
    """Normalize text for consistent deduplication and grouping."""
    return " ".join(text.strip().split())


def hash_text(text: str) -> str:
    """Create short SHA-256 hash for group identification."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_raw_banking77(raw_dir: Path) -> List[IntermediateSample]:
    """Load Banking77 test set (3,080 rows)."""
    csv_path = raw_dir / "banking77" / "test.csv"
    samples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            text = row["text"].strip()
            category = row["category"].strip()
            norm = normalize_text(text).lower()
            samples.append(
                IntermediateSample(
                    dataset_id="banking77",
                    source_split="test",
                    task_type="choice",
                    text_fields={"text": text},
                    label_schema_id="banking77",
                    gold_label=category,
                    is_oos=False,
                    group_key=norm,
                    source_id=f"banking77_raw_{idx:04d}",
                    analysis_annotations={"original_category": category},
                )
            )
    return samples


def load_raw_clinc150(raw_dir: Path, seed: int = 42) -> List[IntermediateSample]:
    """Load CLINC150: 1,500 ID (10 per class) + 500 OOS queries."""
    full_path = raw_dir / "clinc150" / "data_full.json"
    with open(full_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Sample 10 items per ID class from test (150 * 10 = 1,500)
    by_intent: Dict[str, List[str]] = {}
    for text, intent in data.get("test", []):
        by_intent.setdefault(intent, []).append(text.strip())

    samples: List[IntermediateSample] = []
    rng = random.Random(seed)

    for intent in sorted(by_intent.keys()):
        queries = sorted(list(set(by_intent[intent])))
        # Select 10 queries with seed
        shuffled = list(queries)
        rng.shuffle(shuffled)
        selected_id = shuffled[:10]
        # Keep deterministic order
        selected_id.sort()

        for q_idx, q in enumerate(selected_id):
            norm = normalize_text(q).lower()
            samples.append(
                IntermediateSample(
                    dataset_id="clinc150",
                    source_split="test",
                    task_type="choice",
                    text_fields={"text": q},
                    label_schema_id="clinc150",
                    gold_label=intent,
                    is_oos=False,
                    group_key=norm,
                    source_id=f"clinc_id_{intent}_{q_idx:02d}",
                    analysis_annotations={"original_intent": intent},
                )
            )

    # 2. Sample 500 OOS queries from oos_test
    oos_queries = sorted(list(set(text.strip() for text, _ in data.get("oos_test", []))))
    rng_oos = random.Random(seed)
    shuffled_oos = list(oos_queries)
    rng_oos.shuffle(shuffled_oos)
    selected_oos = shuffled_oos[:500]
    selected_oos.sort()

    for o_idx, q in enumerate(selected_oos):
        norm = normalize_text(q).lower()
        samples.append(
            IntermediateSample(
                dataset_id="clinc150",
                source_split="oos_test",
                task_type="choice",
                text_fields={"text": q},
                label_schema_id="clinc150",
                gold_label="REJECT",
                is_oos=True,
                group_key=norm,
                source_id=f"clinc_oos_{o_idx:04d}",
                analysis_annotations={"original_intent": "oos"},
            )
        )

    return samples


def load_raw_anli(raw_dir: Path) -> List[IntermediateSample]:
    """Load ANLI R2 (1,000) and R3 (1,200) test sets, grouping by premise."""
    samples: List[IntermediateSample] = []
    label_map = {
        "e": "entailment",
        "n": "neutral",
        "c": "contradiction",
        "entailment": "entailment",
        "neutral": "neutral",
        "contradiction": "contradiction",
    }

    for round_name, filename in [("r2", "r2_test.jsonl"), ("r3", "r3_test.jsonl")]:
        file_path = raw_dir / "anli" / filename
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                raw_label = record["label"].strip().lower()
                canonical_label = label_map[raw_label]
                premise = record["context"].strip()
                hypothesis = record["hypothesis"].strip()
                uid = record.get("uid", "")
                genre = record.get("genre", "")
                reason = record.get("reason", "")

                # Group by normalized premise text to prevent cross-role context leakage
                premise_norm = normalize_text(premise)

                samples.append(
                    IntermediateSample(
                        dataset_id="anli_r2r3",
                        source_split=f"{round_name}_test",
                        task_type="choice",
                        text_fields={"premise": premise, "hypothesis": hypothesis},
                        label_schema_id="anli_r2r3",
                        gold_label=canonical_label,
                        is_oos=False,
                        group_key=premise_norm,
                        source_id=uid or f"anli_{round_name}_{len(samples):04d}",
                        analysis_annotations={
                            "round": round_name,
                            "genre": genre,
                            "reason": reason,
                            "uid": uid,
                        },
                    )
                )

    return samples


def load_raw_strategyqa(raw_dir: Path, seed: int = 42) -> List[IntermediateSample]:
    """Load StrategyQA: sample 2,000 questions deterministically from train pool."""
    train_path = raw_dir / "strategyqa" / "train.json"
    with open(train_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    # Sort deterministically by qid
    records.sort(key=lambda r: str(r.get("qid", "")))
    rng = random.Random(seed)
    selected = rng.sample(records, 2000)
    selected.sort(key=lambda r: str(r.get("qid", "")))

    samples: List[IntermediateSample] = []
    for item in selected:
        qid = str(item["qid"])
        question = item["question"].strip()
        ans_bool = bool(item["answer"])
        gold_label = "yes" if ans_bool else "no"
        norm_q = normalize_text(question).lower()

        samples.append(
            IntermediateSample(
                dataset_id="strategyqa",
                source_split="train",
                task_type="noul",
                text_fields={"question": question},
                label_schema_id="strategyqa",
                gold_label=gold_label,
                is_oos=False,
                group_key=norm_q,
                source_id=qid,
                analysis_annotations={
                    "qid": qid,
                    "term": item.get("term", ""),
                    "facts": item.get("facts", []),
                    "decomposition": item.get("decomposition", []),
                },
            )
        )

    return samples


def load_all_raw_samples(raw_dir: Path, seed: int = 42) -> Dict[str, List[IntermediateSample]]:
    """Load all four datasets into intermediate representations."""
    return {
        "banking77": load_raw_banking77(raw_dir),
        "clinc150": load_raw_clinc150(raw_dir, seed=seed),
        "anli_r2r3": load_raw_anli(raw_dir),
        "strategyqa": load_raw_strategyqa(raw_dir, seed=seed),
    }

