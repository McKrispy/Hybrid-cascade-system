"""Controlled 200-tick Short-Direct JSON vs. 4-Field CoT ablation on identical Alpaca escalated ticks.

Runs 200 escalated ticks x 2 arms (F0, F1) x 2 models (gpt-6.1-sol, deepseek-v4.1-flash) = 800 calls.
Keeps Sections 1-4 of FinancialHandoffBuilder identical to the 4-field run; replaces only Section 5 and
system instructions with Track-1 style single-field JSON decision output.
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

from daemon.clients import JevPrediction, _coerce_decision  # noqa: E402
from daemon.fetcher import StreamTick  # noqa: E402
from daemon.handoff import FinancialHandoffBuilder  # noqa: E402
from daemon.settlement import PortfolioSettlementEngine  # noqa: E402

SHORT_SYSTEM_PROMPT = (
    "You are a senior quantitative risk officer. Analyze the financial data and respond ONLY "
    'with a valid JSON object containing a single field: {"decision": "<one of LONG_ACCUMULATE, '
    'NEUTRAL_HOLD, SHORT_LIQUIDATE>"}. Do not output reasoning fields.'
)

SHORT_SCHEMA_BLOCK = """\n## 5. Decision Output Schema
You must select exactly ONE action from the Allowed Candidate Actions and output strict JSON with a single field (do not include intermediate reasoning or extra fields):
```json
{
  "decision": "<ONE_OF_ALLOWED_CANDIDATE_ACTIONS>"
}
```"""

ACTION_WEIGHTS = {
    "LONG_ACCUMULATE": 1.0,
    "NEUTRAL_HOLD": 0.0,
    "SHORT_LIQUIDATE": -1.0,
}


def load_sandbox_env() -> None:
    env_file = SANDBOX_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def build_short_prompt(tick: StreamTick, arm_id: str, jev_pred: JevPrediction) -> str:
    full_prompt = FinancialHandoffBuilder.build_prompt(tick, arm_id, jev_pred)
    marker = "\n## 5. Decision Output Schema"
    assert marker in full_prompt, f"Missing Section 5 marker in prompt for {tick.tick_id}/{arm_id}"
    prefix = full_prompt.split(marker, 1)[0]
    return prefix + SHORT_SCHEMA_BLOCK


def parse_strict_decision(content: str) -> Tuple[str, bool]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = [ln for ln in cleaned.splitlines() if not ln.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "decision" in parsed:
            dec = _coerce_decision(parsed["decision"])
            return dec, True
    except Exception:
        pass
    dec = _coerce_decision(content)
    return dec, False


def call_model_with_retries(
    model_key: str,
    api_key: str,
    prompt: str,
    arm_id: str,
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
            {"role": "system", "content": SHORT_SYSTEM_PROMPT},
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
                dec, parsed_ok = parse_strict_decision(content)
                return {
                    "tick_id": tick_id,
                    "model_id": model_key,
                    "arm_id": arm_id,
                    "returned_model": data.get("model", api_model),
                    "decision": dec,
                    "parsed_ok": parsed_ok,
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

    raise RuntimeError(f"Failed call {tick_id}/{model_key}/{arm_id}: {last_err}")


def compute_arm_financial_metrics(esc_rows: List[dict], get_dec) -> Dict[str, Any]:
    n = len(esc_rows)
    wrong_jev = [r for r in esc_rows if r["jev_decision"] != r["gold_label"]]
    ok_jev = [r for r in esc_rows if r["jev_decision"] == r["gold_label"]]
    conf_rows = [r for r in esc_rows if r.get("signal_conflict_flag")]

    decs = [get_dec(r) for r in esc_rows]
    acc_cnt = sum(1 for r, d in zip(esc_rows, decs) if d == r["gold_label"])
    ef_cnt = sum(1 for r in wrong_jev if get_dec(r) == r["jev_decision"])
    rescue_cnt = sum(1 for r in wrong_jev if get_dec(r) == r["gold_label"])
    harm_cnt = sum(1 for r in ok_jev if get_dec(r) != r["gold_label"])
    conf_acc_cnt = sum(1 for r in conf_rows if get_dec(r) == r["gold_label"])

    signed_rets = []
    active_wins = 0
    active_total = 0
    gains: List[float] = []
    losses: List[float] = []
    for r, d in zip(esc_rows, decs):
        w = ACTION_WEIGHTS[d]
        pnl = w * float(r["actual_return_4h"])
        signed_rets.append(pnl)
        if w != 0.0:
            active_total += 1
            if pnl > 0.0:
                active_wins += 1
                gains.append(pnl)
            elif pnl < 0.0:
                losses.append(abs(pnl))

    mean_bps = float(np.mean(signed_rets) * 10000.0) if signed_rets else 0.0
    sum_pct = float(np.sum(signed_rets) * 100.0) if signed_rets else 0.0
    dir_win_pct = (100.0 * active_wins / active_total) if active_total > 0 else 0.0
    avg_gain_bps = float(np.mean(gains) * 10000.0) if gains else 0.0
    avg_loss_bps = float(np.mean(losses) * 10000.0) if losses else 0.0
    payoff = (avg_gain_bps / avg_loss_bps) if avg_loss_bps > 0 else 0.0

    return {
        "n_escalated": n,
        "three_class_correct": acc_cnt,
        "three_class_acc_pct": round(100.0 * acc_cnt / max(n, 1), 2),
        "n_wrong_jev": len(wrong_jev),
        "error_follow_count": ef_cnt,
        "error_follow_pct": round(100.0 * ef_cnt / max(len(wrong_jev), 1), 2),
        "rescue_count": rescue_cnt,
        "rescue_pct": round(100.0 * rescue_cnt / max(len(wrong_jev), 1), 2),
        "n_ok_jev": len(ok_jev),
        "harm_count": harm_cnt,
        "harm_pct": round(100.0 * harm_cnt / max(len(ok_jev), 1), 2),
        "conflict_escalated_acc_pct": round(100.0 * conf_acc_cnt / max(len(conf_rows), 1), 2),
        "conflict_escalated_count": len(conf_rows),
        "active_trades": active_total,
        "active_trade_rate_pct": round(100.0 * active_total / max(n, 1), 2),
        "active_wins": active_wins,
        "directional_win_rate_pct": round(dir_win_pct, 2),
        "avg_win_bps": round(avg_gain_bps, 2),
        "avg_loss_bps": round(avg_loss_bps, 2),
        "payoff_ratio": round(payoff, 2),
        "mean_signed_return_bps": round(mean_bps, 2),
        "cumulative_signed_return_pct": round(sum_pct, 2),
    }


def compute_portfolio_for_short(all_rows: List[dict], decs_by_tick: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    config = json.loads((SANDBOX_ROOT / "config.json").read_text(encoding="utf-8"))
    port_cfg = config["portfolio"]
    scale = float(port_cfg.get("position_scale", 1.0 / 32.0))
    init_cap = float(port_cfg.get("initial_capital_usd", 100000.0))
    fee_bps = float(port_cfg.get("transaction_cost_bps", 5.0))

    eng = PortfolioSettlementEngine(
        initial_capital=init_cap,
        transaction_cost_bps=fee_bps,
        position_scale=scale,
    )
    for r in all_rows:
        tid = r["tick_id"]
        s2_map = {}
        if r["was_escalated"]:
            d_short = decs_by_tick[tid]
            # Evaluate F0 and F1 using Short decisions, C0/C1 copied only to satisfy 5-branch signature
            s2_map = {"F0": d_short["F0"], "F1": d_short["F1"], "C0": d_short["F0"], "C1": d_short["F1"]}
        eng.settle_tick(
            symbol=r["symbol"],
            timestamp=r["timestamp"],
            actual_return_4h=r["actual_return_4h"],
            jev_decision=r["jev_decision"],
            was_escalated=r["was_escalated"],
            s2_decisions=s2_map,
        )
    summary = eng.get_summary()
    out = {}
    for k in ("Jev_Standalone", "Cascade_F0", "Cascade_F1"):
        v = dict(summary[k])
        v.pop("nav_history", None)
        out[k] = v
    return out


def main() -> None:
    load_sandbox_env()
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not openai_key or not deepseek_key:
        raise RuntimeError("Missing OPENAI_API_KEY or DEEPSEEK_API_KEY in dev/sandbox/.env")

    real_dir = PROJECT_ROOT / "archives" / "track2_alpaca_real"
    records_path = real_dir / "records.jsonl"
    stream_path = PROJECT_ROOT / "archives" / "track2_alpaca_real" / "alpaca_real_30m_stream.jsonl"

    all_rows = [json.loads(ln) for ln in records_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    esc_rows = [r for r in all_rows if r.get("was_escalated")]
    assert len(esc_rows) == 200, f"Expected 200 escalated rows, got {len(esc_rows)}"

    stream_by_id = {}
    for ln in stream_path.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            obj = json.loads(ln)
            stream_by_id[obj["tick_id"]] = StreamTick(**obj)

    ckpt_path = PROJECT_ROOT / "archives" / "track2_alpaca_real" / "alpaca_short_ablation_200.jsonl"
    done_by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    if ckpt_path.exists():
        for ln in ckpt_path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                rec = json.loads(ln)
                if rec.get("status") == "valid":
                    done_by_key[(rec["tick_id"], rec["model_id"], rec["arm_id"])] = rec
        print(f"Loaded {len(done_by_key)} completed valid rows from checkpoint.", flush=True)

    tasks: List[Tuple[str, str, str, str]] = []
    for r in esc_rows:
        tid = r["tick_id"]
        tick = stream_by_id[tid]
        jev_pred = JevPrediction(
            decision=r["jev_decision"],
            confidence=round(1.0 - float(r["jev_uncertainty"]), 4),
            uncertainty=float(r["jev_uncertainty"]),
            probabilities=r["jev_probabilities"],
            top2_candidates=r["jev_top2"],
            input_tokens=0,
            cost_usd=0.0,
        )
        for arm_id in ("F0", "F1"):
            prompt = build_short_prompt(tick, arm_id, jev_pred)
            for model_key in ("gpt-6.1-sol", "deepseek-v4.1-flash"):
                if (tid, model_key, arm_id) not in done_by_key:
                    tasks.append((tid, model_key, arm_id, prompt))

    print(f"Pending Short-Direct ablation calls: {len(tasks)} / 800", flush=True)
    file_lock = threading.Lock()
    t_start = time.time()

    if tasks:
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
            futs = {}
            for tid, mkey, arm_id, prompt in tasks:
                key = openai_key if mkey == "gpt-6.1-sol" else deepseek_key
                fut = pool.submit(call_model_with_retries, mkey, key, prompt, arm_id, tid)
                futs[fut] = (tid, mkey, arm_id)

            completed = 0
            for fut in concurrent.futures.as_completed(futs):
                rec = fut.result()
                done_by_key[(rec["tick_id"], rec["model_id"], rec["arm_id"])] = rec
                with file_lock:
                    with ckpt_path.open("a", encoding="utf-8") as fw:
                        fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                completed += 1
                if completed % 50 == 0 or completed == len(tasks):
                    print(
                        f"[{time.time()-t_start:6.1f}s] {completed}/{len(tasks)} completed "
                        f"(total valid={len(done_by_key)}/800)",
                        flush=True,
                    )

    assert len(done_by_key) == 800, f"Expected 800 results, got {len(done_by_key)}"
    parse_failures = sum(1 for rec in done_by_key.values() if not rec.get("parsed_ok"))
    print(f"All 800 calls finished. strict JSON parse failures={parse_failures}", flush=True)

    # Aggregate token usage and cost
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
    total_ablation_cost = round(sum(v["cost_usd"] for v in cost_by_model.values()), 6)

    # Compute comparative metrics
    short_decs = {}
    for mkey in ("gpt-6.1-sol", "deepseek-v4.1-flash"):
        per_tick: Dict[str, Dict[str, str]] = {}
        for r in esc_rows:
            tid = r["tick_id"]
            per_tick[tid] = {
                arm: done_by_key[(tid, mkey, arm)]["decision"]
                for arm in ("F0", "F1")
            }
        short_decs[mkey] = per_tick

    summary = {
        "metadata": {
            "description": "In-track controlled prompt contract ablation on identical 200 upgraded real Alpaca ticks",
            "total_escalated_ticks": len(esc_rows),
            "total_settled_ticks": len(all_rows),
            "ablation_requests": len(done_by_key),
            "strict_json_parse_failures": parse_failures,
            "ablation_cost_usd": total_ablation_cost,
            "cost_by_model": cost_by_model,
        },
        "jev_standalone_escalated": compute_arm_financial_metrics(esc_rows, lambda r: r["jev_decision"]),
        "cot_4field": {
            "gpt-6.1-sol": {
                arm: compute_arm_financial_metrics(esc_rows, lambda r, a=arm: r["s2_decisions_gpt"][a])
                for arm in ("F0", "F1", "C0", "C1")
            },
            "deepseek-v4.1-flash": {
                arm: compute_arm_financial_metrics(esc_rows, lambda r, a=arm: r["s2_decisions_deepseek"][a])
                for arm in ("F0", "F1", "C0", "C1")
            },
        },
        "short_direct": {
            mkey: {
                arm: compute_arm_financial_metrics(
                    esc_rows, lambda r, mk=mkey, a=arm: short_decs[mk][r["tick_id"]][a]
                )
                for arm in ("F0", "F1")
            }
            for mkey in ("gpt-6.1-sol", "deepseek-v4.1-flash")
        },
        "short_direct_portfolios": {
            mkey: compute_portfolio_for_short(all_rows, short_decs[mkey])
            for mkey in ("gpt-6.1-sol", "deepseek-v4.1-flash")
        },
    }

    # Add excess anchoring (F1 - F0 on N_wrong=135) and CoT delta metrics cleanly
    for regime in ("cot_4field", "short_direct"):
        summary[f"{regime}_excess_anchoring_pp"] = {
            mkey: round(
                summary[regime][mkey]["F1"]["error_follow_pct"]
                - summary[regime][mkey]["F0"]["error_follow_pct"],
                2,
            )
            for mkey in ("gpt-6.1-sol", "deepseek-v4.1-flash")
        }

    out_path = real_dir / "cot_ablation_200.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    # Sync to sandbox_final_data for convenience without disturbing SHA256SUMS of the frozen stream files
    (PROJECT_ROOT / "archives" / "track2_alpaca_real" / "cot_ablation_200.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved summary to {out_path} (ablation cost=${total_ablation_cost:.6f})", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False)[:3500])


if __name__ == "__main__":
    main()
