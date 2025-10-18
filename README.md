# Auto Visual Inspection **App Generator** (MVP)

このプロジェクトは、**画像アップロード → 検査箇所UI → 日本語仕様の入力**を受け、
1) 外観検査プロンプトの自動生成、
2) いくつかのサンプルに対する日本語フィードバックをFew-shotとして蓄積、
3) 生成プロンプトを用いた検査アプリ(ストリームリット)の自動生成・実行、
までの**最小実装(MVP)**です。

> 目標：Colabでもローカルでも動作。まずはStreamlit UI + LLM(VLM) APIに接続できる骨組みを提供します。

---

## フォルダ構成
```
auto-visual-inspection-generator/
├─ app_streamlit.py         # Streamlitアプリ (仕様入力・LLM接続)
├─ src/
│  ├─ llm_providers.py      # OpenAI/GeminiのAPIラッパ
│  ├─ prompt_factory.py     # プロンプト生成（System / User / Few-shot注入）
│  ├─ fewshot.py            # Few-shotの保存・読み込み・反映
│  └─ vision_eval.py        # 画像+プロンプトで評価(VLM呼び出し)の窓口
├─ data/
│  └─ few_shots.jsonl       # 日本語フィードバック(少数例)の蓄積ファイル
├─ scripts/
│  └─ generate_runtime_app.py # 生成された最終アプリを/prod_appに出力
├─ .env.example
└─ requirements.txt
```

## セットアップ（ローカル）
```bash
# Python 3.10+ を推奨
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# APIキーを設定（必要に応じて）
cp .env.example .env
# .env を編集して OPENAI_API_KEY / GEMINI_API_KEY を設定
```

## 起動（ローカル）
```bash
streamlit run app_streamlit.py
```

## Google Colab での実行
1. このZIPをColabにアップロード & 展開
2. 下記を実行
```python
%pip install -r requirements.txt
import os
os.environ['OPENAI_API_KEY'] = 'sk-...'
# or
os.environ['GEMINI_API_KEY'] = '...'
```
3. Colab上でStreamlitをトンネリングして使うか、`app_streamlit.py`を直接Pythonとして実行し、プロンプト生成だけ検証することも可能です。
（`colabcode`や`pyngrok`でポート公開する例は必要に応じて追記してください）

## Few-shot の考え方（MVP）
- `data/few_shots.jsonl` に、入力仕様/期待出力/日本語フィードバック を1行1JSONで保存。
- 判定用の出力トークン数はデフォルトで4096に設定されています。必要に応じて `max_output_tokens` を調整してください。
- `app_streamlit.py` で検査結果に対する「正解/NG理由」をユーザが追記→保存。
- 次回プロンプト生成時にFew-shotとしてプロンプトへ自動注入。

## 最終アプリ自動生成
- `scripts/generate_runtime_app.py` を実行すると、`prod_app/` に実行用Streamlitアプリを出力します。
- 生成物は**固定化したプロンプト**と**Few-shot**を含み、運用者/顧客へ配布しやすい形にします。

## 注意
- 画像処理/幾何ロジックそのものはVLM(例: GPT-4.1V / GPT-4o / Gemini 2.0 Vision)に委譲する設計です。
- 厳密な寸法計測や射影補正が必要な場合は、OpenCV等の**前処理**を`vision_eval.py`に追加してください。
- 本MVPは**自動生成のワークフロー**に焦点を当てています。精度/速度チューニングは業務要件に合わせて調整してください。
