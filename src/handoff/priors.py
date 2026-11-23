"""Prior extraction and formatting for System-1 to System-2 handoffs in HybridCascade.

Timestamp: 2026-11-10 13:22 (Asia/Shanghai)

Extracts Top-2 candidate identity, ranking, and raw probabilities from Jev System-1 outputs:
1. Top-2 determined by raw Jev probability descending
2. Ties broken deterministically by schema canonical order
3. Raw probabilities preserved as-is without re-normalization (p1, p2 are raw)
4. Asserts error risk e_hat is never passed into the prior payload
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple


def extract_top2_prior(
    probabilities: Optional[Dict[str, float]],
    allowed_labels: Sequence[str],
) -> Tuple[List[str], Optional[Dict[str, Any]]]:
    """Extract Top-2 prior candidates and payload from Jev raw probabilities.
    
    Args:
        probabilities: Dictionary of raw Jev category probabilities.
        allowed_labels: Canonical sequence of allowed labels from TaskSchema.
        
    Returns:
        Tuple of:
            - top2_labels: List of the 2 selected labels in probability rank order [Top-1, Top-2]
            - prior_payload: Dictionary containing structured candidate records with rank, label, and raw probability
    """
    if not probabilities or not allowed_labels:
        # Fallback if upstream Jev prediction is missing or empty
        fallback_top2 = list(allowed_labels[:2]) if len(allowed_labels) >= 2 else list(allowed_labels)
        return fallback_top2, None

    # Map each label to its canonical index for deterministic tie-breaking
    canonical_indices = {lbl: i for i, lbl in enumerate(allowed_labels)}

    # Collect available candidates with probabilities
    items = []
    for lbl in allowed_labels:
        p = float(probabilities.get(lbl, 0.0))
        items.append((lbl, p, canonical_indices.get(lbl, 999999)))

    # Sort primarily by probability descending (-p), then by canonical order ascending (+idx)
    items.sort(key=lambda x: (-x[1], x[2]))

    top2_items = items[:2]
    top2_labels = [item[0] for item in top2_items]

    prior_payload = {
        "candidates": [
            {
                "rank": idx + 1,
                "label": item[0],
                "probability": float(item[1]),  # Raw probability preserved as-is
            }
            for idx, item in enumerate(top2_items)
        ]
    }

    return top2_labels, prior_payload

