import os
import json
from typing import List, Dict, Any

import streamlit as st
from PIL import Image
from dotenv import load_dotenv

from src.prompt_factory import build_prompt_bundle
from src.llm_providers import LLMProvider
from src.vision_eval import run_vision_eval
from scripts.generate_runtime_app import generate_runtime_app


load_dotenv()


st.set_page_config(page_title="外観検査アプリ自動生成(MVP)", layout="wide")

st.title("外観検査アプリ **自動生成** (MVP)")

with st.expander("1) 画像サンプルのアップロード", expanded=True):
    uploaded_files = st.file_uploader("検査したい画像を複数選択", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    sample_images: List[tuple[str, Image.Image]] = []
    expected_map: Dict[str, str] = st.session_state.get("expected_verdicts", {})
    if uploaded_files:
        cols = st.columns(min(3, len(uploaded_files)))
        for i, uf in enumerate(uploaded_files):
            img = Image.open(uf).convert("RGB")
            sample_images.append((uf.name, img))
            with cols[i % len(cols)]:
                st.image(img, caption=uf.name, use_column_width=True)
                default_choice = expected_map.get(uf.name, "OK")
                choice = st.selectbox(
                    "想定判定",
                    ["OK", "NG"],
                    index=0 if default_choice == "OK" else 1,
                    key=f"expected-{uf.name}-{i}",
                )
                expected_map[uf.name] = choice
        st.session_state["expected_verdicts"] = expected_map
    else:
        st.session_state.pop("expected_verdicts", None)

with st.expander("3) 検査仕様を日本語で記述", expanded=True):
    spec_text = st.text_area(
        "例) 画像全体でネジが6本すべてシール済みか確認し、どれか外れていればNGと判断する...",
        height=140,
    )

with st.sidebar:
    st.header("LLM設定")
    provider = st.selectbox("プロバイダ", ["OpenAI", "Gemini"])
    model = st.text_input("モデル名", os.getenv("OPENAI_MODEL" if provider=="OpenAI" else "GEMINI_MODEL", ""))
    temperature = st.slider("temperature", 0.0, 1.5, 0.2, 0.05)
max_tokens = st.number_input("max_output_tokens", 256, 16384, 4096, step=128)
st.caption("※ APIキーは環境変数(.env) または ランタイム環境で設定してください。")

st.divider()

col_a, col_b = st.columns([1,1])

with col_a:
    if st.button("A) 外観検査プロンプトを生成", disabled=not(spec_text and sample_images)):
        prompt_bundle = build_prompt_bundle(
            spec_text=spec_text,
        )
        st.session_state["prompt_bundle"] = prompt_bundle
        st.success("プロンプトを生成しました。右側で確認できます。")

def _generate_prompt_suggestion(provider: LLMProvider, spec_text: str, image_name: str, expected: str, decision: Dict[str, Any]) -> str:
    system_prompt = """あなたは製造業の外観検査プロンプトを改善する専門家です。
検査仕様はそのまま別のアプリに貼り付けられる完成形の文章で提示してください。"""
    user_prompt = f"""
現在の検査仕様:
{spec_text}

サンプル名: {image_name}
想定している判定: {expected}
AIの判定結果: {decision.get('verdict', 'UNKNOWN')}
AIが出力した詳細: {decision.get('details', '-')}

対象サンプルが想定した判定になるように検査仕様を改訂してください。
サンプルの画像全体を参照し、必要に応じて注目すべき部位や判定基準を明示してください。
最終的な検査仕様の文章だけを日本語で出力してください。箇条書きや説明文は不要です。
"""
    try:
        suggestion = provider.chat_text(system_prompt, user_prompt)
        return suggestion.strip() or "修正候補を取得できませんでした。"
    except Exception as exc:
        return f"修正候補の取得に失敗しました: {exc}"

with col_b:
    if st.button("B) サンプルで検査", disabled="prompt_bundle" not in st.session_state):
        provider_client = LLMProvider(provider_name=provider, model=model, temperature=temperature, max_tokens=int(max_tokens))
        results = []
        expected_map = st.session_state.get("expected_verdicts", {})
        with st.spinner("VLMで判定中..."):
            for name, img in sample_images:
                decision = run_vision_eval(provider_client, st.session_state["prompt_bundle"], img)
                expected = expected_map.get(name)
                suggestion = ""
                if expected and expected.upper() != decision.get("verdict", "").upper():
                    suggestion = _generate_prompt_suggestion(provider_client, spec_text, name, expected, decision)
                results.append({"image": name, "decision": decision, "expected": expected, "suggestion": suggestion})
        st.session_state["eval_results"] = results
        st.success("判定完了。下の結果と修正候補をご確認ください。")

st.divider()

if "prompt_bundle" in st.session_state:
    st.subheader("生成されたプロンプト（System / User）")
    st.code(json.dumps(st.session_state["prompt_bundle"], ensure_ascii=False, indent=2))

if "eval_results" in st.session_state:
    st.subheader("判定結果")
    for item in st.session_state["eval_results"]:
        st.markdown(f"**サンプル画像**: {item['image']}")
        decision = item["decision"]
        st.write(f"- 判定: {decision.get('verdict', 'UNKNOWN')} / 理由: {decision.get('details', '-')}")
        expected = item.get("expected")
        if expected:
            st.write(f"- 想定判定: {expected}")
        if item["suggestion"]:
            st.markdown("**プロンプト修正候補:**")
            st.code(item["suggestion"])
        st.divider()

st.subheader("C) 生成されたプロンプトから **最終アプリ** を組み立てる")
out_dir = st.text_input("出力先ディレクトリ", "prod_app")
if st.button("ビルド（/prod_app に生成）", disabled="prompt_bundle" not in st.session_state):
    try:
        abs_path, rel_path = generate_runtime_app(st.session_state["prompt_bundle"], out_dir=out_dir)
    except Exception as exc:
        st.error(f"生成に失敗しました: {exc}")
    else:
        st.success(f"生成しました → `{abs_path}` を `streamlit run` で実行できます。")
        st.caption(f"※ 同じフォルダに既存ファイルがある場合は自動でリネームされます（例: {rel_path}）。")
# moved _generate_prompt_suggestion earlier in the file
