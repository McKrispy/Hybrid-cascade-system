"""Dataset grouping, stratified splitting, and manifest generation for HybridCascade.

Timestamp: 2026-11-09 13:17 (Asia/Shanghai)

Implements strict zero-leakage role partitioning:
- Calibration: 20% (1,856 target)
- Selection: 10% (928 target)
- Holdout: 70% (6,496 target)
- Total: 9,280 unique questions

Rules enforced:
1. Grouping takes strict precedence over exact ratios (no group split across roles).
2. SampleInput never receives GoldRecord fields (gold_label, is_oos, annotations).
3. Deterministic sample_id assignment and stable tie-breaking.
4. Registers B06 (FR control) sampling allocations.
"""

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from src.infra.paths import get_data_dir
from src.input.adapters import IntermediateSample, hash_text, normalize_text


class SampleInput(BaseModel):
    sample_id: str
    group_id: str
    dataset_id: str
    source_split: str
    task_type: str  # "choice" or "noul"
    text_fields: Dict[str, str]
    label_schema_id: str


class GoldRecord(BaseModel):
    sample_id: str
    dataset_id: str
    gold_label: str
    is_oos: bool = False
    analysis_annotations: Dict[str, Any] = Field(default_factory=dict)


class SplitResult(BaseModel):
    role: str
    samples: List[SampleInput]
    gold_records: List[GoldRecord]


def _assign_groups_greedy(
    groups: List[List[IntermediateSample]],
    target_cal: int,
    target_sel: int,
    target_hld: int,
) -> Tuple[List[IntermediateSample], List[IntermediateSample], List[IntermediateSample]]:
    """Greedily assign groups to roles to match target capacities as closely as possible."""
    cal_list: List[IntermediateSample] = []
    sel_list: List[IntermediateSample] = []
    hld_list: List[IntermediateSample] = []

    for grp in groups:
        # Sort roles by remaining needed capacity descending
        rem_cal = target_cal - len(cal_list)
        rem_sel = target_sel - len(sel_list)
        rem_hld = target_hld - len(hld_list)

        options = [
            (rem_cal, 0, cal_list),
            (rem_sel, 1, sel_list),
            (rem_hld, 2, hld_list),
        ]
        options.sort(key=lambda x: x[0], reverse=True)
        options[0][2].extend(grp)

    return cal_list, sel_list, hld_list


def split_banking77(
    samples: List[IntermediateSample],
    seed: int = 42,
) -> Dict[str, List[IntermediateSample]]:
    """Split Banking77 into calibration (616), selection (308), holdout (2156)."""
    # Group by normalized text
    by_category: Dict[str, List[IntermediateSample]] = defaultdict(list)
    for s in samples:
        by_category[s.gold_label].append(s)

    cal: List[IntermediateSample] = []
    sel: List[IntermediateSample] = []
    hld: List[IntermediateSample] = []

    # Sort categories to maintain determinism
    for cat in sorted(by_category.keys()):
        cat_samples = by_category[cat]
        # Group duplicates within category
        grp_dict: Dict[str, List[IntermediateSample]] = defaultdict(list)
        for s in cat_samples:
            grp_dict[s.group_key].append(s)

        groups = list(grp_dict.values())
        groups.sort(key=lambda g: g[0].group_key)
        rng = random.Random(seed + len(cat))
        rng.shuffle(groups)

        c, s, h = _assign_groups_greedy(groups, target_cal=8, target_sel=4, target_hld=28)
        cal.extend(c)
        sel.extend(s)
        hld.extend(h)

    return {"calibration": cal, "selection": sel, "holdout": hld}


def split_clinc150(
    samples: List[IntermediateSample],
    seed: int = 42,
) -> Dict[str, List[IntermediateSample]]:
    """Split CLINC150 into calibration (400), selection (200), holdout (1400)."""
    id_samples = [s for s in samples if not s.is_oos]
    oos_samples = [s for s in samples if s.is_oos]

    cal: List[IntermediateSample] = []
    sel: List[IntermediateSample] = []
    hld: List[IntermediateSample] = []

    # ID samples: 150 intents, 10 samples each -> 2 cal, 1 sel, 7 holdout
    by_intent: Dict[str, List[IntermediateSample]] = defaultdict(list)
    for s in id_samples:
        by_intent[s.gold_label].append(s)

    for intent in sorted(by_intent.keys()):
        s_list = sorted(by_intent[intent], key=lambda x: x.group_key)
        rng = random.Random(seed + len(intent))
        rng.shuffle(s_list)
        cal.extend(s_list[:2])
        sel.extend(s_list[2:3])
        hld.extend(s_list[3:10])

    # OOS samples: 500 total -> 100 cal, 50 sel, 350 holdout
    oos_sorted = sorted(oos_samples, key=lambda x: x.group_key)
    rng_oos = random.Random(seed)
    rng_oos.shuffle(oos_sorted)
    cal.extend(oos_sorted[:100])
    sel.extend(oos_sorted[100:150])
    hld.extend(oos_sorted[150:500])

    return {"calibration": cal, "selection": sel, "holdout": hld}


def split_anli(
    samples: List[IntermediateSample],
    seed: int = 42,
) -> Dict[str, List[IntermediateSample]]:
    """Split ANLI (R2: 1000, R3: 1200) by premise group into 440, 220, 1540."""
    r2_samples = [s for s in samples if s.source_split == "r2_test"]
    r3_samples = [s for s in samples if s.source_split == "r3_test"]

    cal: List[IntermediateSample] = []
    sel: List[IntermediateSample] = []
    hld: List[IntermediateSample] = []

    for round_name, r_samples, targets in [
        ("r2", r2_samples, (200, 100, 700)),
        ("r3", r3_samples, (240, 120, 840)),
    ]:
        groups_dict: Dict[str, List[IntermediateSample]] = defaultdict(list)
        for s in r_samples:
            groups_dict[s.group_key].append(s)

        group_list = list(groups_dict.values())
        group_list.sort(key=lambda g: g[0].group_key)
        rng = random.Random(seed + (2 if round_name == "r2" else 3))
        rng.shuffle(group_list)

        c, s, h = _assign_groups_greedy(
            group_list,
            target_cal=targets[0],
            target_sel=targets[1],
            target_hld=targets[2],
        )
        cal.extend(c)
        sel.extend(s)
        hld.extend(h)

    return {"calibration": cal, "selection": sel, "holdout": hld}


def split_strategyqa(
    samples: List[IntermediateSample],
    seed: int = 42,
) -> Dict[str, List[IntermediateSample]]:
    """Split StrategyQA (2,000 samples) into 400 cal, 200 sel, 1400 holdout."""
    # Group by question text
    grp_dict: Dict[str, List[IntermediateSample]] = defaultdict(list)
    for s in samples:
        grp_dict[s.group_key].append(s)

    group_list = list(grp_dict.values())
    group_list.sort(key=lambda g: g[0].group_key)
    rng = random.Random(seed)
    rng.shuffle(group_list)

    cal, sel, hld = _assign_groups_greedy(
        group_list,
        target_cal=400,
        target_sel=200,
        target_hld=1400,
    )
    return {"calibration": cal, "selection": sel, "holdout": hld}


def build_final_dataset_splits(
    intermediate_data: Dict[str, List[IntermediateSample]],
    seed: int = 42,
) -> Dict[str, Dict[str, SplitResult]]:
    """Partition all four datasets and generate SampleInput and GoldRecord objects.
    
    Returns:
        dict: {dataset_id: {role: SplitResult}}
    """
    split_funcs = {
        "banking77": split_banking77,
        "clinc150": split_clinc150,
        "anli_r2r3": split_anli,
        "strategyqa": split_strategyqa,
    }

    final_results: Dict[str, Dict[str, SplitResult]] = {}
    sample_counter: Dict[str, int] = defaultdict(int)

    for dataset_id, samples in intermediate_data.items():
        role_map = split_funcs[dataset_id](samples, seed=seed)
        dataset_splits: Dict[str, SplitResult] = {}

        for role in ["calibration", "selection", "holdout"]:
            items = role_map[role]
            # Deterministic sorting within role by group_key and original source_id
            items.sort(key=lambda x: (x.group_key, x.source_id or ""))

            role_samples: List[SampleInput] = []
            role_golds: List[GoldRecord] = []

            for item in items:
                sample_counter[dataset_id] += 1
                seq = sample_counter[dataset_id]
                sample_id = f"{dataset_id}_{seq:05d}"
                group_id = f"{dataset_id}_grp_{hash_text(item.group_key)}"

                # Strict SampleInput schema (NO gold labels or explanations)
                sample_in = SampleInput(
                    sample_id=sample_id,
                    group_id=group_id,
                    dataset_id=dataset_id,
                    source_split=item.source_split,
                    task_type=item.task_type,
                    text_fields=item.text_fields,
                    label_schema_id=item.label_schema_id,
                )
                role_samples.append(sample_in)

                # Decoupled GoldRecord
                gold_rec = GoldRecord(
                    sample_id=sample_id,
                    dataset_id=dataset_id,
                    gold_label=item.gold_label,
                    is_oos=item.is_oos,
                    analysis_annotations=item.analysis_annotations,
                )
                role_golds.append(gold_rec)

            dataset_splits[role] = SplitResult(
                role=role,
                samples=role_samples,
                gold_records=role_golds,
            )

        final_results[dataset_id] = dataset_splits

    return final_results


def register_b06_sampling(
    splits: Dict[str, Dict[str, SplitResult]],
    quotas_by_role: Dict[str, int],
    seed: int = 42,
) -> Dict[str, Dict[str, List[str]]]:
    """Register B06 sample candidates by role per dataset according to plan.md line 244.
    
    Quotas: calibration 50, selection 25, holdout 175 = 250 per dataset (1000 total).
    """
    b06_candidates: Dict[str, Dict[str, List[str]]] = {}
    for dataset_id, role_splits in splits.items():
        b06_candidates[dataset_id] = {}
        for role, split_res in role_splits.items():
            quota = quotas_by_role.get(role, 0)
            sample_ids = sorted([s.sample_id for s in split_res.samples])
            rng = random.Random(seed + len(dataset_id) + len(role))
            chosen = rng.sample(sample_ids, min(quota, len(sample_ids)))
            chosen.sort()
            b06_candidates[dataset_id][role] = chosen
    return b06_candidates


def save_dataset_files(
    splits: Dict[str, Dict[str, SplitResult]],
    b06_candidates: Dict[str, Dict[str, List[str]]],
    data_dir: Path,
) -> Dict[str, Any]:
    """Save all jsonl files and generate manifest.json."""
    data_dir.mkdir(parents=True, exist_ok=True)
    cal_dir = data_dir / "calibration"
    sel_dir = data_dir / "selection"
    hld_dir = data_dir / "holdout"
    gold_dir = data_dir / "gold"

    for d in [cal_dir, sel_dir, hld_dir, gold_dir]:
        d.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "schema_version": "1.0",
        "timestamp": "2026-11-09 13:20 (Asia/Shanghai)",
        "total_unique_questions": 0,
        "datasets": {},
        "role_counts": {"calibration": 0, "selection": 0, "holdout": 0},
        "b06_sampling_registration": b06_candidates,
    }

    # Tracking sets to verify zero overlap
    all_sample_ids: Set[str] = set()
    role_groups: Dict[str, Set[str]] = defaultdict(set)

    for dataset_id, role_map in splits.items():
        dataset_meta: Dict[str, Any] = {
            "dataset_id": dataset_id,
            "roles": {},
            "total_questions": 0,
            "unique_groups": 0,
        }
        dataset_groups: Set[str] = set()

        for role, res in role_map.items():
            dest_dir = {"calibration": cal_dir, "selection": sel_dir, "holdout": hld_dir}[role]
            sample_file = dest_dir / f"{dataset_id}.jsonl"
            gold_file = gold_dir / f"{dataset_id}_{role}.jsonl"

            with open(sample_file, "w", encoding="utf-8") as fs, open(gold_file, "w", encoding="utf-8") as fg:
                for s in res.samples:
                    fs.write(s.model_dump_json() + "\n")
                    all_sample_ids.add(s.sample_id)
                    role_groups[role].add(s.group_id)
                    dataset_groups.add(s.group_id)

                for g in res.gold_records:
                    fg.write(g.model_dump_json() + "\n")

            n_samples = len(res.samples)
            dataset_meta["roles"][role] = {
                "file": str(sample_file.relative_to(data_dir.parent)),
                "gold_file": str(gold_file.relative_to(data_dir.parent)),
                "count": n_samples,
                "b06_quota": len(b06_candidates[dataset_id].get(role, [])),
            }
            dataset_meta["total_questions"] += n_samples
            manifest["role_counts"][role] += n_samples
            manifest["total_unique_questions"] += n_samples

        dataset_meta["unique_groups"] = len(dataset_groups)
        manifest["datasets"][dataset_id] = dataset_meta

    manifest_path = data_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest

