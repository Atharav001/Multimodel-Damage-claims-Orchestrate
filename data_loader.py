import pandas as pd
from typing import List, Dict

CLAIMS_PATH = "claims/claims.csv"
USER_HISTORY_PATH = "claims/user_history.csv"
EVIDENCE_REQUIREMENTS_PATH = "claims/evidence_requirements.csv"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    claims = pd.read_csv(CLAIMS_PATH)
    user_history = pd.read_csv(USER_HISTORY_PATH)
    evidence_reqs = pd.read_csv(EVIDENCE_REQUIREMENTS_PATH)
    return claims, user_history, evidence_reqs


def get_claim_context(
    row: pd.Series,
    user_history: pd.DataFrame,
    evidence_reqs: pd.DataFrame,
) -> List[Dict]:
    user_id = row["user_id"]
    user_row = user_history.loc[user_history["user_id"] == user_id]

    history_flags = user_row["history_flags"].iloc[0] if not user_row.empty else ""
    history_summary = user_row["history_summary"].iloc[0] if not user_row.empty else ""

    claim_object = row["claim_object"]
    matching_reqs = evidence_reqs[
        (evidence_reqs["claim_object"] == claim_object)
        | (evidence_reqs["claim_object"] == "all")
    ]
    evidence_rules = "\n".join(
        f"- [{row["requirement_id"]}] ({row["applies_to"]}) {row["minimum_image_evidence"]}"
        for _, row in matching_reqs.iterrows()
    )

    image_paths = str(row["image_paths"]).split(";")

    return [
        {
            "user_id": user_id,
            "image_paths": image_paths,
            "user_claim": row["user_claim"],
            "claim_object": claim_object,
            "history_flags": history_flags,
            "history_summary": history_summary,
            "evidence_rules": evidence_rules,
        }
    ]


def build_context(
    claims: pd.DataFrame,
    user_history: pd.DataFrame,
    evidence_reqs: pd.DataFrame,
) -> List[Dict]:
    context = []
    for _, row in claims.iterrows():
        context.extend(get_claim_context(row, user_history, evidence_reqs))
    return context


if __name__ == "__main__":
    claims, user_history, evidence_reqs = load_data()
    context = build_context(claims, user_history, evidence_reqs)
    print(f"Built context for {len(context)} claims")
    print(context[0])
