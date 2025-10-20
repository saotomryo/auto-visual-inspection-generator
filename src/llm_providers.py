import os, base64, io, json, re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import requests
from dotenv import load_dotenv
from PIL import Image


load_dotenv()

@dataclass
class LLMProvider:
    provider_name: str = "OpenAI"  # or "Gemini"
    model: str = ""
    temperature: float = 0.2
    max_tokens: int = 1024

    # 画像を data:uri に変換（OpenAI Vision系のinput向け）
    @staticmethod
    def pil_to_datauri(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{data}"

    def chat_vision(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """プロバイダ別のVisionチャット呼び出し (MVP: 疑似実装/ホンモノ実装の両方に対応)"""
        if self.provider_name.lower() == "openai":
            return self._openai_chat(messages)
        elif self.provider_name.lower() == "gemini":
            return self._gemini_chat(messages)
        else:
            raise ValueError("Unsupported provider")

    @staticmethod
    def _split_messages(messages: List[Dict[str, Any]]) -> tuple[str, str, Optional[str], Optional[tuple[str, str]]]:
        system_text = ""
        user_text = ""
        image_uri: Optional[str] = None
        inline_data: Optional[tuple[str, str]] = None
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

    @staticmethod
    def _extract_error_details(response: requests.Response) -> str:
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

    @staticmethod
    def _parse_json_response(text: str, fallback_reason: Optional[str] = None) -> Dict[str, Any]:
        if not text:
            details = fallback_reason or "empty response"
            return {"verdict": "NG", "details": details, "checks": []}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
        details = fallback_reason or text
        return {"verdict": "NG", "details": details, "checks": []}

    @staticmethod
    def _debug_print(title: str, payload: Any) -> None:
        if os.getenv("AVI_DEBUG") in {"1", "true", "True"}:
            print(title)
            print(json.dumps(payload, ensure_ascii=False, indent=2) if isinstance(payload, (dict, list)) else payload)

    def _openai_chat(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEYが設定されていません。")

        system_text, user_text, image_uri, _ = self._split_messages(messages)
        model = self.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        content: List[Dict[str, Any]] = []
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

        printable_messages = []
        for msg in messages:
            msg_copy = dict(msg)
            content_dict = msg_copy.get("content")
            if isinstance(content_dict, dict) and "image_url" in content_dict:
                msg_copy["content"] = dict(content_dict)
                msg_copy["content"]["image_url"] = "<image omitted>"
            printable_messages.append(msg_copy)
        self._debug_print("=== OpenAI メッセージ ===", printable_messages)

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError:
            details = self._extract_error_details(response)
            if "temperature" in details and "default (1)" in details and "temperature" in payload:
                payload.pop("temperature", None)
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=120,
                )
                try:
                    response.raise_for_status()
                except requests.HTTPError:
                    details = self._extract_error_details(response)
                    return {"output_text": details, "json": {"verdict": "ERROR", "details": details, "checks": []}}
            elif "maximum" in details and "tokens" in details:
                note = "出力が途中で打ち切られました。max_output_tokens を増やして再実行してください。"
                return {"output_text": details, "json": {"verdict": "ERROR", "details": note, "checks": [], "note": details}}
            else:
                return {"output_text": details, "json": {"verdict": "ERROR", "details": details, "checks": []}}

        data = response.json()
        self._debug_print("=== OpenAI レスポンス ===", data)

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("OpenAIレスポンスにchoicesが含まれていません。")

        message_content = choices[0].get("message", {}).get("content", "")
        if isinstance(message_content, list):
            text = "\n".join(part.get("text", "") for part in message_content if part.get("type") == "text").strip()
        else:
            text = str(message_content).strip()

        fallback_reason = None
        if choices[0].get("finish_reason") == "length":
            fallback_reason = "出力が途中で打ち切られました。max_output_tokens を増やして再実行してください。"

        parsed = self._parse_json_response(text, fallback_reason)
        return {"output_text": text, "json": parsed}

    def _gemini_chat(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEYが設定されていません。")

        system_text, user_text, _, inline_data = self._split_messages(messages)
        model = self.model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        prompt_text = "\n\n".join(filter(None, [system_text, user_text]))

        parts: List[Dict[str, Any]] = []
        if prompt_text:
            parts.append({"text": prompt_text})
        if inline_data:
            mime, data_b64 = inline_data
            parts.append({"inlineData": {"mimeType": mime, "data": data_b64}})

        generation_config: Dict[str, Any] = {}
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

        printable_messages = []
        for msg in messages:
            msg_copy = dict(msg)
            content_dict = msg_copy.get("content")
            if isinstance(content_dict, dict) and "image_url" in content_dict:
                msg_copy["content"] = dict(content_dict)
                msg_copy["content"]["image_url"] = "<image omitted>"
            printable_messages.append(msg_copy)
        self._debug_print("=== Gemini メッセージ ===", printable_messages)

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            f"?key={api_key}"
        )
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError:
            details = self._extract_error_details(response)
            if "temperature" in details and "default (1)" in details and "temperature" in payload.get("generationConfig", {}):
                payload["generationConfig"].pop("temperature", None)
                response = requests.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=120,
                )
                try:
                    response.raise_for_status()
                except requests.HTTPError:
                    details = self._extract_error_details(response)
                    return {"output_text": details, "json": {"verdict": "ERROR", "details": details, "checks": []}}
            elif "maximum" in details and "tokens" in details:
                note = "出力が途中で打ち切られました。max_output_tokens を増やして再実行してください。"
                return {"output_text": details, "json": {"verdict": "ERROR", "details": note, "checks": [], "note": details}}
            else:
                return {"output_text": details, "json": {"verdict": "ERROR", "details": details, "checks": []}}

        data = response.json()
        self._debug_print("=== Gemini レスポンス ===", data)

        candidates = data.get("candidates") or []
        text = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(part.get("text", "") for part in parts if "text" in part).strip()

        fallback_reason = None
        if candidates and candidates[0].get("finishReason") == "MAX_TOKENS":
            fallback_reason = "出力が途中で打ち切られました。max_output_tokens を増やして再実行してください。"

        parsed = self._parse_json_response(text, fallback_reason)
        return {"output_text": text, "json": parsed}
