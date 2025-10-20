import json
import os
import shutil
from pathlib import Path
from textwrap import dedent

LLM_SRC_PATH = Path("src/llm_providers.py")
VISION_SRC_PATH = Path("src/vision_eval.py")


def _sanitize_prompt_bundle(bundle: dict) -> dict:
    # 深いコピーを作ってROI関連フィールドを除去
    clean = json.loads(json.dumps(bundle))
    user = clean.get("user", {})
    if "roi_map" in user:
        user.pop("roi_map", None)
    clean["user"] = user
    clean.pop("few_shots", None)
    return clean


def generate_runtime_app(prompt_bundle: dict, out_dir: str = "prod_app"):
    os.makedirs(out_dir, exist_ok=True)
    app_path = os.path.join(out_dir, "runtime_app.py")
    app_path = _ensure_unique_path(app_path)
    rel_app_path = os.path.relpath(app_path, out_dir)

    prompt_bundle = _sanitize_prompt_bundle(prompt_bundle)

    llm_source = _augment_llm_module_source(_load_llm_module_source())
    vision_source = _rewrite_run_vision_eval(_load_vision_module_source())

    header_code = dedent(
        """\
        import json
        import os
        from pathlib import Path
        from typing import Dict

        import requests
        import streamlit as st
        from PIL import Image


        APP_DIR = Path(__file__).resolve().parent
        ENV_PATH = APP_DIR / ".env"


        def _read_env_file(path: Path = ENV_PATH) -> Dict[str, str]:
            env: Dict[str, str] = {}
            if path.exists():
                for raw_line in path.read_text(encoding="utf-8").splitlines():
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    env[key.strip()] = value.strip()
            return env


        def _apply_env(env: Dict[str, str]) -> None:
            for key, value in env.items():
                if value:
                    os.environ[key] = value


        def _write_env(updates: Dict[str, str]) -> bool:
            current = _read_env_file()
            changed = False
            for key, value in updates.items():
                value = value.strip()
                if value and current.get(key) != value:
                    current[key] = value
                    changed = True
            if changed:
                lines = [f"{key}={value}" for key, value in current.items()]
                ENV_PATH.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
            _apply_env({key: value.strip() for key, value in updates.items() if value.strip()})
            return changed


        _APPLIED_ENV = _read_env_file()
        _apply_env(_APPLIED_ENV)
        """
    ).strip()

    prompt_json = json.dumps(prompt_bundle, ensure_ascii=False, indent=2)

    ui_code = f"""
PROMPT_BUNDLE = {prompt_json}

with st.sidebar:
    st.header("APIキー設定")
    openai_key = st.text_input("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""), type="password")
    gemini_key = st.text_input("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""), type="password")
    if st.button("APIキーを保存", key="save_api_keys"):
        updates = {{"OPENAI_API_KEY": openai_key, "GEMINI_API_KEY": gemini_key}}
        if _write_env(updates):
            st.success(".env にAPIキーを保存しました。")
        else:
            st.info("変更はありませんでした。")


st.title("外観検査 - 最終アプリ")
provider = st.selectbox("プロバイダ", ["OpenAI", "Gemini"])
model = st.text_input("モデル名", os.getenv("OPENAI_MODEL" if provider == "OpenAI" else "GEMINI_MODEL", ""))
temperature = st.slider("temperature", 0.0, 1.5, 0.2, 0.05)
max_tokens = st.number_input("max_output_tokens", 256, 16384, 4096, step=128)

up = st.file_uploader("画像をアップロード", type=["png", "jpg", "jpeg"])
if up:
    img = Image.open(up).convert("RGB")
    st.image(img, caption=up.name)
    if st.button("判定する"):
        client = LLMProvider(provider_name=provider, model=model, temperature=temperature, max_tokens=int(max_tokens))
        decision = run_vision_eval(client, PROMPT_BUNDLE, img)
        st.write(f"判定: {{decision.get('verdict', 'UNKNOWN')}} / 理由: {{decision.get('details', '-')}}")
""".strip()

    app_code_parts = [
        header_code,
        "# === Embedded from src/llm_providers.py ===",
        llm_source,
        "# === Embedded from src/vision_eval.py ===",
        vision_source,
        ui_code,
    ]

    app_code = "\n\n".join(app_code_parts) + "\n"

    with open(app_path, "w", encoding="utf-8") as f:
        f.write(app_code)

    for extra in ["requirements.txt", ".env.example"]:
        if os.path.exists(extra):
            shutil.copy(extra, os.path.join(out_dir, extra))

    return os.path.abspath(app_path), rel_app_path


def _load_llm_module_source() -> str:
    if not LLM_SRC_PATH.exists():
        raise FileNotFoundError(f"LLM provider module not found: {LLM_SRC_PATH}")
    return LLM_SRC_PATH.read_text(encoding="utf-8").strip()


def _load_vision_module_source() -> str:
    if not VISION_SRC_PATH.exists():
        raise FileNotFoundError(f"Vision evaluation module not found: {VISION_SRC_PATH}")
    return VISION_SRC_PATH.read_text(encoding="utf-8").strip()


def _augment_llm_module_source(source: str) -> str:
    augmentation = dedent(
        """\
        # --- Runtime augmentation for LLMProvider (inject real API calls) ---
        import json as _runtime_json
        import re as _runtime_re
        from typing import Any as _runtime_Any, Dict as _runtime_Dict, List as _runtime_List, Optional as _runtime_Optional, Tuple as _runtime_Tuple
        import requests as _runtime_requests


        def _runtime_split_messages(messages: _runtime_List[_runtime_Dict[str, _runtime_Any]]) -> _runtime_Tuple[str, str, _runtime_Optional[str], _runtime_Optional[_runtime_Tuple[str, str]]]:
            system_text = ""
            user_text = ""
            image_uri = None
            inline_data = None
            for message in messages:
                role = message.get("role")
                if role == "system":
                    system_text = str(message.get("content", ""))
                elif role == "user":
                    content = message.get("content", {})
                    if isinstance(content, dict):
                        user_text = str(content.get("text", ""))
                        image_uri = content.get("image_url")
                        if isinstance(image_uri, str) and image_uri.startswith("data:"):
                            header, encoded = image_uri.split(",", 1)
                            mime = "image/png"
                            if ";" in header:
                                meta = header[5:]
                                mime = meta.split(";")[0] or mime
                            inline_data = (mime, encoded)
                    else:
                        user_text = str(content)
            return system_text, user_text, image_uri, inline_data


        def _runtime_parse_json_response(text: str, fallback_reason: _runtime_Optional[str] = None) -> _runtime_Dict[str, _runtime_Any]:
            if not text:
                details = fallback_reason or "empty response"
                return {"verdict": "NG", "details": details, "checks": []}
            try:
                return _runtime_json.loads(text)
            except _runtime_json.JSONDecodeError:
                match = _runtime_re.search(r"\\{.*\\}", text, _runtime_re.DOTALL)
                if match:
                    try:
                        return _runtime_json.loads(match.group(0))
                    except _runtime_json.JSONDecodeError:
                        pass
            details = fallback_reason or text
            return {"verdict": "NG", "details": details, "checks": []}


        def _extract_error_details(response: _runtime_requests.Response) -> str:
            try:
                payload = response.json()
                if isinstance(payload, dict):
                    err = payload.get("error")
                    if isinstance(err, dict):
                        message = err.get("message")
                        if message:
                            return message
                    return str(payload)
            except ValueError:
                pass
            return response.text or f"HTTP {response.status_code}"


        def _runtime_openai_chat(self, messages: _runtime_List[_runtime_Dict[str, _runtime_Any]]) -> _runtime_Dict[str, _runtime_Any]:
            openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
            if not openai_api_key:
                raise RuntimeError("OPENAI_API_KEYが設定されていません。")
            system_text, user_text, image_uri, _ = _runtime_split_messages(messages)
            model = self.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            content: _runtime_List[_runtime_Dict[str, _runtime_Any]] = []
            if user_text:
                content.append({"type": "text", "text": user_text})
            if image_uri:
                content.append({"type": "image_url", "image_url": {"url": image_uri}})
            if not content:
                content.append({"type": "text", "text": ""})

            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": content},
                ],
                "response_format": {"type": "json_object"},
            }
            if self.temperature not in (None, 1):
                payload["temperature"] = self.temperature
            if self.max_tokens:
                payload["max_completion_tokens"] = self.max_tokens
            request_kwargs = {
                "url": "https://api.openai.com/v1/chat/completions",
                "headers": {
                    "Authorization": f"Bearer {openai_api_key}",
                    "Content-Type": "application/json",
                },
                "json": payload,
                "timeout": 120,
            }
            if os.getenv("AVI_DEBUG") in {"1", "true", "True"}:
                printable_messages = []
                for _msg in messages:
                    msg_copy = dict(_msg)
                    content = msg_copy.get("content")
                    if isinstance(content, dict) and "image_url" in content:
                        msg_copy["content"] = dict(content)
                        msg_copy["content"]["image_url"] = "<image omitted>"
                    printable_messages.append(msg_copy)
                print("=== OpenAI メッセージ ===")
                print(json.dumps(printable_messages, ensure_ascii=False, indent=2))
            response = _runtime_requests.post(**request_kwargs)
            try:
                response.raise_for_status()
            except _runtime_requests.HTTPError:
                details = _extract_error_details(response)
                if "maximum context length" in details or "maximum output length" in details or "finish_reason" in details:
                    result = {
                        "verdict": "ERROR",
                        "details": "出力が途中で打ち切られました。max_output_tokens を増やして再実行してください。",
                        "checks": [],
                        "note": details,
                    }
                    return {"output_text": details, "json": result}
                if "temperature" in details and "default (1)" in details:
                    payload.pop("temperature", None)
                    request_kwargs["json"] = payload
                    response = _runtime_requests.post(**request_kwargs)
                    try:
                        response.raise_for_status()
                    except _runtime_requests.HTTPError:
                        details = _extract_error_details(response)
                        return {"output_text": details, "json": {"verdict": "ERROR", "details": details, "checks": []}}
                else:
                    return {"output_text": details, "json": {"verdict": "ERROR", "details": details, "checks": []}}
            data = response.json()
            if os.getenv("AVI_DEBUG") in {"1", "true", "True"}:
                print("=== OpenAI レスポンス ===")
                print(json.dumps(data, ensure_ascii=False, indent=2))
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError("OpenAIレスポンスにchoicesが含まれていません。")
            message_content = choices[0].get("message", {}).get("content", "")
            if isinstance(message_content, list):
                text = "\\n".join(
                    part.get("text", "") for part in message_content if part.get("type") == "text"
                ).strip()
            else:
                text = str(message_content).strip()
            fallback_reason = None
            if data.get("choices"):
                finish_reason = data["choices"][0].get("finish_reason")
                if finish_reason == "length":
                    fallback_reason = "出力が途中で打ち切られました。max_output_tokens を増やして再実行してください。"
            parsed = _runtime_parse_json_response(text, fallback_reason)
            return {"output_text": text, "json": parsed}


        def _runtime_gemini_chat(self, messages: _runtime_List[_runtime_Dict[str, _runtime_Any]]) -> _runtime_Dict[str, _runtime_Any]:
            gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
            if not gemini_api_key:
                raise RuntimeError("GEMINI_API_KEYが設定されていません。")
            system_text, user_text, _, inline_data = _runtime_split_messages(messages)
            model = self.model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
            prompt_text = "\\n\\n".join(filter(None, [system_text, user_text]))
            parts: _runtime_List[_runtime_Dict[str, _runtime_Any]] = []
            if prompt_text:
                parts.append({"text": prompt_text})
            if inline_data:
                mime, data_b64 = inline_data
                parts.append({"inlineData": {"mimeType": mime, "data": data_b64}})
            generation_config = {}
            if self.temperature not in (None, 1):
                generation_config["temperature"] = self.temperature
            if self.max_tokens:
                generation_config["maxOutputTokens"] = self.max_tokens
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": parts or [{"text": prompt_text}],
                    }
                ],
                "generationConfig": generation_config,
                "responseMimeType": "application/json",
            }
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
                f"?key={gemini_api_key}"
            )
            request_kwargs = {
                "url": url,
                "headers": {"Content-Type": "application/json"},
                "json": payload,
                "timeout": 120,
            }
            print("=== Gemini メッセージ ===")
            print(json.dumps(messages, ensure_ascii=False, indent=2))
            if os.getenv("AVI_DEBUG") in {"1", "true", "True"}:
                printable_messages = []
                for _msg in messages:
                    msg_copy = dict(_msg)
                    content = msg_copy.get("content")
                    if isinstance(content, dict) and "image_url" in content:
                        msg_copy["content"] = dict(content)
                        msg_copy["content"]["image_url"] = "<image omitted>"
                    printable_messages.append(msg_copy)
                print("=== Gemini メッセージ ===")
                print(json.dumps(printable_messages, ensure_ascii=False, indent=2))
            response = _runtime_requests.post(**request_kwargs)
            try:
                response.raise_for_status()
            except _runtime_requests.HTTPError:
                details = _extract_error_details(response)
                if "maximum" in details and "tokens" in details:
                    result = {
                        "verdict": "ERROR",
                        "details": "出力が途中で打ち切られました。max_output_tokens を増やして再実行してください。",
                        "checks": [],
                        "note": details,
                    }
                    return {"output_text": details, "json": result}
                if "temperature" in details and "default (1)" in details:
                    payload["generationConfig"].pop("temperature", None)
                    request_kwargs["json"] = payload
                    response = _runtime_requests.post(**request_kwargs)
                    try:
                        response.raise_for_status()
                    except _runtime_requests.HTTPError:
                        details = _extract_error_details(response)
                        return {"output_text": details, "json": {"verdict": "ERROR", "details": details, "checks": []}}
                else:
                    return {"output_text": details, "json": {"verdict": "ERROR", "details": details, "checks": []}}
            data = response.json()
            if os.getenv("AVI_DEBUG") in {"1", "true", "True"}:
                print("=== Gemini レスポンス ===")
                print(json.dumps(data, ensure_ascii=False, indent=2))
            candidates = data.get("candidates") or []
            text = ""
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join(part.get("text", "") for part in parts if "text" in part).strip()
            fallback_reason = None
            if candidates:
                finish_reason = candidates[0].get("finishReason")
                if finish_reason == "MAX_TOKENS":
                    fallback_reason = "出力が途中で打ち切られました。max_output_tokens を増やして再実行してください。"
            parsed = _runtime_parse_json_response(text, fallback_reason)
            return {"output_text": text, "json": parsed}


        LLMProvider._openai_chat = _runtime_openai_chat  # type: ignore[attr-defined]
        LLMProvider._gemini_chat = _runtime_gemini_chat  # type: ignore[attr-defined]
        """
    ).strip()
    return f"{source}\n\n{augmentation}"


def _rewrite_run_vision_eval(source: str) -> str:
    rewritten = dedent(
        """\
        import json
        from typing import Dict, Any
        from PIL import Image

        def run_vision_eval(provider: LLMProvider, prompt_bundle: Dict[str, Any], img: Image.Image) -> Dict[str, Any]:
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
        """
    ).strip()
    return rewritten


def _ensure_unique_path(initial_path: str) -> str:
    if not os.path.exists(initial_path):
        return initial_path
    base = Path(initial_path)
    stem, suffix = base.stem, base.suffix
    counter = 1
    while True:
        candidate = base.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return str(candidate)
        counter += 1
