import os
import json
import time
from typing import List

import streamlit as st
from PIL import Image

from src.prompt_factory import build_prompt_bundle
from src.fewshot import FewShotStore
from src.llm_providers import LLMProvider
from src.vision_eval import run_vision_eval
from scripts.generate_runtime_app import generate_runtime_app


st.set_page_config(page_title="外観検査アプリ自動生成(MVP)", layout="wide")

st.title("外観検査アプリ **自動生成** (MVP)")

with st.expander("1) 画像サンプルのアップロード", expanded=True):
    uploaded_files = st.file_uploader("検査したい画像を複数選択", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    sample_images: List[tuple[str, Image.Image]] = []
    if uploaded_files:
        cols = st.columns(min(3, len(uploaded_files)))
        for i, uf in enumerate(uploaded_files):
            img = Image.open(uf).convert("RGB")
            sample_images.append((uf.name, img))
            with cols[i % len(cols)]:
                st.image(img, caption=uf.name, use_column_width=True)

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
        few = FewShotStore("data/few_shots.jsonl")
        prompt_bundle = build_prompt_bundle(
            spec_text=spec_text,
            few_shots=few.load_all()
        )
        st.session_state["prompt_bundle"] = prompt_bundle
        st.success("プロンプトを生成しました。右側で確認できます。")

with col_b:
    if st.button("B) サンプルで検査 → フィードバックをFew-shotへ追加", disabled="prompt_bundle" not in st.session_state):
        provider_client = LLMProvider(provider_name=provider, model=model, temperature=temperature, max_tokens=int(max_tokens))
        results = []
        with st.spinner("VLMで判定中..."):
            for name, img in sample_images:
                decision = run_vision_eval(provider_client, st.session_state["prompt_bundle"], img)
                results.append({"image": name, "decision": decision})
        st.session_state["eval_results"] = results
        st.success("判定完了。下で人手フィードバックを追記し、Few-shotに保存できます。")

st.divider()

if "prompt_bundle" in st.session_state:
    st.subheader("生成されたプロンプト（System / User / Few-shot注入後）")
    st.code(json.dumps(st.session_state["prompt_bundle"], ensure_ascii=False, indent=2))

if "eval_results" in st.session_state:
    st.subheader("判定結果 & フィードバック登録")
    few = FewShotStore("data/few_shots.jsonl")
    for item in st.session_state["eval_results"]:
        st.write(f"**{item['image']}** → 判定: `{item['decision'].get('verdict','unknown')}` / 詳細: {item['decision'].get('details','-')}")
        fb = st.text_input(f"日本語フィードバック（期待する正解や指摘） - {item['image']}", key=f"fb-{item['image']}")
        if st.button(f"保存（Few-shot追加） - {item['image']}"):
            rec = {
                "spec_text": spec_text,
                "roi": [],
                "model_decision": item["decision"],
                "human_feedback": fb,
                "timestamp": time.time(),
            }
            few.append(rec)
            st.success("Few-shotを保存しました。")

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
