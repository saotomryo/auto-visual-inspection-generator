import ast
import json
import re
from pathlib import Path

from scripts.generate_runtime_app import generate_runtime_app


def test_generate_runtime_app_embeds_full_image_roi(tmp_path):
    prompt_bundle = {
        "system": "test system",
        "user": {"spec_text": "spec", "instruction": "do it", "roi_map": {"dummy.png": []}},
        "few_shots": [{"roi": [], "model_decision": {"verdict": "OK"}}],
    }

    out_dir = tmp_path / "prod"
    abs_path, rel_path = generate_runtime_app(prompt_bundle, out_dir=str(out_dir))

    runtime_file = Path(abs_path)
    assert runtime_file.exists(), "generated runtime app should exist"
    assert runtime_file.parent == out_dir
    assert runtime_file.name == rel_path

    code = runtime_file.read_text(encoding="utf-8")

    # run_vision_eval should accept only provider, bundle, img
    assert re.search(
        r"def run_vision_eval\(provider: LLMProvider, prompt_bundle: Dict\[str, Any], img: Image\.Image\)",
        code,
    )

    # ensure全画像評価向けの情報が出力される
    assert '"image_size": {"width":' in code
    assert '"instruction": "画像全体が仕様に合致するか検証し' in code

    # ensure response formatting enforcementは含まれる
    assert '"response_format": {"type": "json_object"}' in code
    assert '"responseMimeType": "application/json"' in code
    assert 'if verdict not in {"OK", "NG"}:' in code
    assert '"roi_map"' not in code
