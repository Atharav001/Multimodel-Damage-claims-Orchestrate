import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

import data_loader
import judge_engine
import vision_engine

CACHE_PATH = ".cache.json"
MAX_WORKERS = 3
RETRY_WAITS = [2, 4, 8]

OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]


def load_cache() -> dict[str, dict[str, Any]]:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict[str, dict[str, Any]]) -> None:
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def call_with_retry(fn, *args, **kwargs):
    last_exc = None
    for wait in RETRY_WAITS:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            time.sleep(wait)
    raise last_exc


def process_one_claim(
    row: pd.Series,
    user_history: pd.DataFrame,
    evidence_reqs: pd.DataFrame,
    client: OpenAI,
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    claim_id = str(row.name)
    if claim_id in cache:
        return cache[claim_id]

    ctx_list = data_loader.get_claim_context(row, user_history, evidence_reqs)
    ctx = ctx_list[0]

    def run_vision():
        return vision_engine.run_blind_perception(
            ctx["claim_object"], ctx["image_paths"], client
        )

    blind_facts = call_with_retry(run_vision)

    def run_judge():
        return judge_engine.run_adjudication(ctx, blind_facts, client)

    judgement = call_with_retry(run_judge)

    result = {
        "user_id": ctx["user_id"],
        "image_paths": ";".join(ctx["image_paths"]),
        "user_claim": ctx["user_claim"],
        "claim_object": ctx["claim_object"],
        "evidence_standard_met": judgement.get("evidence_standard_met", "false"),
        "evidence_standard_met_reason": judgement.get(
            "evidence_standard_met_reason", ""
        ),
        "risk_flags": judgement.get("risk_flags", "manual_review_required"),
        "issue_type": judgement.get("issue_type", "unknown"),
        "object_part": judgement.get("object_part", "unknown"),
        "claim_status": judgement.get("claim_status", "not_enough_information"),
        "claim_status_justification": judgement.get(
            "claim_status_justification", ""
        ),
        "supporting_image_ids": judgement.get("supporting_image_ids", "none"),
        "valid_image": judgement.get("valid_image", "false"),
        "severity": judgement.get("severity", "unknown"),
    }

    cache[claim_id] = result
    save_cache(cache)
    return result


def main() -> None:
    api_key = os.environ["OPENAI_API_KEY"]
    base_url = os.environ.get("OPENAI_BASE_URL")
    model_name = os.environ.get("MODEL_NAME", "gemini-1.5-flash")

    client = OpenAI(api_key=api_key, base_url=base_url)
    cache = load_cache()

    claims, user_history, evidence_reqs = data_loader.load_data()
    results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for idx, row in claims.iterrows():
            future = executor.submit(
                process_one_claim,
                row,
                user_history,
                evidence_reqs,
                client,
                cache,
            )
            futures[future] = idx

        with tqdm(total=len(futures), desc="Processing claims") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    idx = futures[future]
                    results.append(
                        {
                            "user_id": claims.at[idx, "user_id"],
                            "image_paths": claims.at[idx, "image_paths"],
                            "user_claim": claims.at[idx, "user_claim"],
                            "claim_object": claims.at[idx, "claim_object"],
                            "evidence_standard_met": "false",
                            "evidence_standard_met_reason": str(e),
                            "risk_flags": "manual_review_required",
                            "issue_type": "unknown",
                            "object_part": "unknown",
                            "claim_status": "not_enough_information",
                            "claim_status_justification": (
                                "Processing failed after all retries"
                            ),
                            "supporting_image_ids": "none",
                            "valid_image": "false",
                            "severity": "unknown",
                        }
                    )
                finally:
                    pbar.update(1)

    df = pd.DataFrame(results, columns=OUTPUT_COLUMNS)
    df.to_csv("output.csv", index=False, quoting=1)
    print(f"\nDone. {len(results)} claims written to output.csv")


if __name__ == "__main__":
    main()
