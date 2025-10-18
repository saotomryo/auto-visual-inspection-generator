from typing import Dict, Any, List
import textwrap, json

SYSTEM_PROMPT = """あなたは製造業の外観検査エキスパートです。
ユーザが入力した日本語仕様に基づき、与えられた画像全体に対して厳密で一貫したOK/NG判定を行います。
応答はJSONのみで出力してください（自然文は出力しない）。JSONスキーマ: {"verdict": "OK|NG", "details": "日本語説明", "checks": [{"result": "OK|NG", "reason": "str"}]}。
視点の傾き・遠近がある場合も可能な限り判定のロバスト性を維持し、根拠をdetailsに明記してください。
"""

def build_prompt_bundle(spec_text: str, few_shots: List[Dict[str, Any]]) -> Dict[str, Any]:
    """System / User / Few-shot をまとめたバンドル（各プロバイダで適宜整形して使う）"""
    fewshot_snippets = []
    for fs in few_shots[-6:]:  # 直近最大6件
        fewshot_snippets.append({
            "spec_text": fs.get("spec_text", ""),
            "human_feedback": fs.get("human_feedback", ""),
            "expected_or_correction": fs.get("model_decision", {})
        })

    user_payload = {
        "spec_text": spec_text,
        "instruction": "画像全体が仕様に合致するか検証し、OK/NGと理由をJSONで一貫して回答してください。"
    }

    return {
        "system": SYSTEM_PROMPT,
        "user": user_payload,
        "few_shots": fewshot_snippets
    }
