import base64
import json
from typing import List
from openai import OpenAI


SYSTEM_PROMPT = (
    "You are an expert damage inspector. "
    "Look at the provided images. Do NOT read any user text. "
    "Output ONLY JSON with these keys: "
    "image_quality (clear, blurry, low_light, cropped), "
    "visible_object (car, laptop, package, unknown), "
    "visible_parts (list of parts seen), "
    "visible_issues (list of issues seen like dent, crack, none), "
    "estimated_severity (none, low, medium, high, unknown)."
)


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def run_blind_perception(
    claim_object: str,
    image_paths: List[str],
    client: OpenAI,
) -> dict:
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                f"The claim object category is '{claim_object}'. "
                "Evaluate the images below for this category."
            ),
        }
    ]
    for i, path in enumerate(image_paths, start=1):
        content.append({"type": "text", "text": f"Image ID: img_{i}"})
        b64 = encode_image(path)
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "auto",
                },
            }
        )

    try:
        response = client.chat.completions.create(
            model="gemini-1.5-flash",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {
            "image_quality": "unknown",
            "visible_object": "unknown",
            "visible_parts": [],
            "visible_issues": [],
            "estimated_severity": "unknown",
        }
