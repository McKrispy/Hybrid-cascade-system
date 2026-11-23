"""Controlled 200-tick Clean 3-Field F0 (zero prior_stance_evaluation placeholder) ablation.

Runs 200 escalated ticks x 1 arm (F0_clean) x 2 models (gpt-6.1-sol, deepseek-v4.1-flash) = 400 calls.
Keeps Sections 1-3 of FinancialHandoffBuilder identical to F0; removes prior_stance_evaluation completely
from both the system prompt and Section 5 JSON schema.
"""

import concurrent.futures
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SANDBOX_ROOT = PROJECT_ROOT / "sandbox"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SANDBOX_ROOT))

from daemon.clients import _coerce_decision, _coerce_str  # noqa: E402
from daemon.fetcher import StreamTick  # noqa: E402
from daemon.handoff import FinancialHandoffBuilder  # noqa: E402
from scripts.track2.run_alpaca_short_ablation import compute_arm_financial_metrics, load_sandbox_env  # noqa: E402

CLEAN_3FIELD_SYSTEM_PROMPT = (
    "You are a senior quantitative risk officer. Analyze financial data and respond strictly in JSON format "
    "with quant_vs_news_synthesis, reasoning_chain, and decision (one of LONG_ACCUMULATE, NEUTRAL_HOLD, SHORT_LIQUIDATE)."
)

CLEAN_3FIELD_SCHEMA_BLOCK = """\n## 5. Decision Output Schema
You must output strict JSON with the following 3 fields:
```json
{
  "quant_vs_news_synthesis": "1-2 sentences synthesizing technical indicators against breaking news headlines, explicitly identifying any divergence.",
  "reasoning_chain": "Step-by-step causal multi-factor deduction weighing momentum sustainability and risk-reward asymmetry.",
  "decision": "<ONE_OF_ALLOWED_CANDIDATE_ACTIONS>"
}
```"""


def build_f0_clean_prompt(tick: StreamTick) -> str:
    full_f0 = FinancialHandoffBuilder.build_prompt(tick, "F0", None)
    marker = "\n## 5. Decision Output Schema"
    assert marker in full_f0, f"Missing Section 5 marker in F0 prompt for {tick.tick_id}"
    prefix = full_f0.split(marker, 1)[0]
    prompt = prefix + CLEAN_3FIELD_SCHEMA_BLOCK
    assert "prior_stance_evaluation" not in prompt, "Unexpected prior_stance_evaluation in F0-Clean prompt!"
    return prompt


def parse_3field_json(content: str) -> Tuple[str, bool, Dict[str, str]]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = [ln for ln in cleaned.splitlines() if not ln.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "decision" in parsed:
            dec = _coerce_decision(parsed["decision"])
            fields = {
                "quant_vs_news_synthesis": _coerce_str(parsed.get("quant_vs_news_synthesis"), ""),
                "reasoning_chain": _coerce_str(parsed.get("reasoning_chain"), ""),
            }
            return dec, True, fields
    except Exception:
        pass
    dec = _coerce_decision(content)
    return dec, False, {}


def call_model_f0_clean(
    model_key: str,
    api_key: str,
    prompt: str,
    tick_id: str,
    max_retries: int = 5,
) -> Dict[str, Any]:
    if model_key == "gpt-6.1-sol":
        endpoint = os.environ.get("OPENAI_BASE_URL", "").strip() or "https://api.openai.com/v1/chat/completions"
        if endpoint.endswith("/v1") or endpoint.endswith("/v1/"):
            endpoint = endpoint.rstrip("/") + "/chat/completions"
        api_model = "gpt-6.1-sol"
        in_rate, out_rate = 2.50, 10.00
        timeout = 45.0
    elif model_key == "deepseek-v4.1-flash":
        endpoint = os.environ.get("DEEPSEEK_BASE_URL", "").strip() or "https://api.deepseek.com/chat/completions"
        if endpoint.endswith("/v1") or endpoint.endswith("/v1/"):
            endpoint = endpoint.rstrip("/") + "/chat/completions"
        api_model = "deepseek-flash"
        in_rate, out_rate = 0.14, 0.28
        timeout = 35.0
    else:
        raise ValueError(f"Unknown model_key: {model_key}")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": api_model,
        "messages": [
            {"role": "system", "content": CLEAN_3FIELD_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
        "max_completion_tokens": 800,
    }

    last_err: Optional[str] = None
    t0 = time.perf_counter()
    for attempt in range(max_retries + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(endpoint, json=payload, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                usage = data.get("usage", {})
                pt = int(usage.get("prompt_tokens", 0))
                ct = int(usage.get("completion_tokens", 0))
                details = usage.get("completion_tokens_details") or {}
                rt = int(details.get("reasoning_tokens", 0))
                cost = round((pt / 1_000_000.0) * in_rate + (ct / 1_000_000.0) * out_rate, 6)
                content = data["choices"][0]["message"]["content"]
                dec, parsed_ok, fields = parse_3field_json(content)
                return {
                    "tick_id": tick_id,
                    "model_id": model_key,
                    "arm_id": "F0_clean",
                    "returned_model": data.get("model", api_model),
                    "decision": dec,
                    "parsed_ok": parsed_ok,
                    "fields": fields,
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "reasoning_tokens": rt,
                    "total_tokens": pt + ct,
                    "cost_usd": cost,
                    "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                    "status": "valid" if parsed_ok else "parse_fallback",
                    "raw_response": content,
                }
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                time.sleep(min(1.5 * (2 ** attempt), 15.0))
                continue
            last_err = f"HTTP_{resp.status_code}: {resp.text[:300]}"
            break
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError, httpx.ConnectTimeout) as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < max_retries:
                time.sleep(min(1.5 * (2 ** attempt), 12.0))
                continue
            break
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            break

    raise RuntimeError(f"Failed call {tick_id}/{model_key}/F0_clean: {last_err}")


def main() -> None:
    load_sandbox_env()
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not openai_key or not deepseek_key:
        raise RuntimeError("Missing OPENAI_API_KEY or DEEPSEEK_API_KEY in dev/sandbox/.env")

    real_dir = PROJECT_ROOT / "archives" / "track2_alpaca_real"
    all_rows = [json.loads(ln) for ln in (real_dir / "records.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    esc_rows = [r for r in all_rows if r.get("was_escalated")]
    assert len(esc_rows) == 200

    stream_by_id = {
        json.loads(ln)["tick_id"]: StreamTick(**json.loads(ln))
        for ln in (PROJECT_ROOT / "archives" / "track2_alpaca_real" / "alpaca_real_30m_stream.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    }

    ckpt_path = PROJECT_ROOT / "archives" / "track2_alpaca_real" / "alpaca_f0_clean_200.jsonl"
    done_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if ckpt_path.exists():
        for ln in ckpt_path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                rec = json.loads(ln)
                if rec.get("status") == "valid":
                    done_by_key[(rec["tick_id"], rec["model_id"])] = rec
        print(f"Loaded {len(done_by_key)} completed F0-Clean rows from checkpoint.", flush=True)

    tasks: List[Tuple[str, str, str]] = []
    for r in esc_rows:
        tid = r["tick_id"]
        prompt = build_f0_clean_prompt(stream_by_id[tid])
        for mkey in ("gpt-6.1-sol", "deepseek-v4.1-flash"):
            if (tid, mkey) not in done_by_key:
                tasks.append((tid, mkey, prompt))

    print(f"Pending F0-Clean ablation calls: {len(tasks)} / 400", flush=True)
    file_lock = threading.Lock()
    t0 = time.time()

    if tasks:
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
            futs = {}
            for tid, mkey, prompt in tasks:
                key = openai_key if mkey == "gpt-6.1-sol" else deepseek_key
                fut = pool.submit(call_model_f0_clean, mkey, key, prompt, tid)
                futs[fut] = (tid, mkey)

            done_cnt = 0
            for fut in concurrent.futures.as_completed(futs):
                rec = fut.result()
                done_by_key[(rec["tick_id"], rec["model_id"])] = rec
                with file_lock:
                    with ckpt_path.open("a", encoding="utf-8") as fw:
                        fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done_cnt += 1
                if done_cnt % 50 == 0 or done_cnt == len(tasks):
                    print(f"[{time.time()-t0:6.1f}s] {done_cnt}/{len(tasks)} completed ({len(done_by_key)}/400)", flush=True)

    assert len(done_by_key) == 400, f"Expected 400, got {len(done_by_key)}"
    parse_failures = sum(1 for v in done_by_key.values() if not v.get("parsed_ok"))

    cost_by_model = {}
    for mkey in ("gpt-6.1-sol", "deepseek-v4.1-flash"):
        m_recs = [v for v in done_by_key.values() if v["model_id"] == mkey]
        cost_by_model[mkey] = {
            "requests": len(m_recs),
            "prompt_tokens": sum(r["prompt_tokens"] for r in m_recs),
            "completion_tokens": sum(r["completion_tokens"] for r in m_recs),
            "reasoning_tokens": sum(r["reasoning_tokens"] for r in m_recs),
            "total_tokens": sum(r["total_tokens"] for r in m_recs),
            "cost_usd": round(sum(r["cost_usd"] for r in m_recs), 6),
        }
    total_cost = round(sum(v["cost_usd"] for v in cost_by_model.values()), 6)

    clean_decs = {
        mkey: {r["tick_id"]: done_by_key[(r["tick_id"], mkey)]["decision"] for r in esc_rows}
        for mkey in ("gpt-6.1-sol", "deepseek-v4.1-flash")
    }

    # Load existing 4-field F0/F1 error_follow for direct comparison
    cot_ab = json.loads((real_dir / "cot_ablation_200.json").read_text(encoding="utf-8"))

    metrics_by_model = {}
    for mkey in ("gpt-6.1-sol", "deepseek-v4.1-flash"):
        m = compute_arm_financial_metrics(esc_rows, lambda r, mk=mkey: clean_decs[mk][r["tick_id"]])
        # Agreement with original F0 (4-Field N/A)
        orig_key = "s2_decisions_gpt" if mkey == "gpt-6.1-sol" else "s2_decisions_deepseek"
        agree = sum(1 for r in esc_rows if clean_decs[mkey][r["tick_id"]] == r[orig_key]["F0"])
        f1_ef = cot_ab["cot_4field"][mkey]["F1"]["error_follow_pct"]
        m["agreement_with_f0_4field_pct"] = round(100.0 * agree / len(esc_rows), 2)
        m["f1_minus_f0_clean_anchor_pp"] = round(f1_ef - m["error_follow_pct"], 2)
        metrics_by_model[mkey] = m

    out = {
        "metadata": {
            "description": "Clean 3-field F0 ablation (removing prior_stance_evaluation entirely)",
            "total_escalated_ticks": len(esc_rows),
            "requests": 400,
            "strict_json_parse_failures": parse_failures,
            "cost_usd": total_cost,
            "cost_by_model": cost_by_model,
        },
        "f0_clean_3field": metrics_by_model,
    }
    out_path = real_dir / "f0_clean_ablation_200.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    (PROJECT_ROOT / "archives" / "track2_alpaca_real" / "f0_clean_ablation_200.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
