import os, base64, io
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from PIL import Image

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

    def _openai_chat(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 実運用ではOpenAI公式SDK or Azure OpenAIのRESTを組み込んでください。
        # ここではデモ用に擬似応答を返す。
        return {"output_text": "DEMO(OpenAI): OK", "json": {"verdict": "OK", "details": "demo response"}}

    def _gemini_chat(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 実運用では google-generativeai を組み込んでください（ここでは擬似応答）。
        return {"output_text": "DEMO(Gemini): NG", "json": {"verdict": "NG", "details": "demo response"}}
