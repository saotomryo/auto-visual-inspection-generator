import json
from typing import Dict, Any
from PIL import Image
from .llm_providers import LLMProvider

def run_vision_eval(provider: LLMProvider, prompt_bundle: Dict[str, Any], img: Image.Image) -> Dict[str, Any]:
    """画像全体を評価対象としてVLMに判定を依頼する"""
    system = prompt_bundle["system"]
    width, height = img.size
    user = {
        "spec_text": prompt_bundle["user"]["spec_text"],
        "instruction": "画像全体が仕様に合致するか検証し、OK/NGと理由をJSONで一貫して回答してください。",
        "image_size": {"width": width, "height": height},
    }
    datauri = provider.pil_to_datauri(img)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": {"text": json.dumps(user, ensure_ascii=False), "image_url": datauri}},
    ]
    resp = provider.chat_vision(messages)
    result = resp.get("json", {})
    verdict = str(result.get("verdict", "")).upper()
    if verdict not in {"OK", "NG"}:
        result["verdict"] = "NG"
        result["details"] = result.get("details") or "モデルから有効な判定が返らなかったためNGとします。"
    result.setdefault("details", "")
    return result
