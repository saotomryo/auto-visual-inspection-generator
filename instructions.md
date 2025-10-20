# 開発者向けインストラクション

本ファイルは開発・検証作業時に必ず参照してください。READMEはエンドユーザー向けドキュメントのため、本ファイルに内部仕様やテスト手順をまとめています。

## 最新仕様（2025-10-17）
- **ワークフロー**: 画像アップロード → 検査仕様入力 → プロンプト生成の3ステップ。ROI描画UIは撤廃済み。
- **判定範囲**: すべての判定はアップロード画像全体を対象とする。生成するプロンプトにはROIを含めず、モデルへの問い合わせには画像全体のサイズ情報のみを渡す。
- **最終アプリ生成**: `scripts/generate_runtime_app.py` が `src/llm_providers.py` / `src/vision_eval.py` を取り込み、API呼び出し＋全画面判定ロジックを埋め込んだ単一ファイル（`prod_app/runtime_app*.py`）を出力。ファイルが衝突する場合は自動リネームされる。
- **APIキー管理**: 生成されたアプリはサイドバーから `OPENAI_API_KEY` / `GEMINI_API_KEY` を保存でき、`.env` に書き込み + 環境変数反映を同時に行う。
- **APIエラー処理**: OpenAI / Gemini へのリクエストで `temperature` や `max_tokens` がサポートされない場合、テキストからエラー内容を抽出して UI に表示。必要に応じて `temperature=1` へフォールバックし、それでも不成立ならアプリ側で `verdict=NG` と理由メッセージを自動補完する。
- **トークン設定**: `max_output_tokens` のデフォルトは 4096。finish_reason が `length` などのトークン超過を示した場合は、アプリ側が NG 判定とともに「max_output_tokens を増やして再実行」メッセージを表示する。

## テスト
開発時には `pytest` を利用して変更の影響範囲を確認してください。

```bash
pytest
```

現在のテスト内容:
- `tests/test_generate_runtime_app.py`: 単一ファイル生成がROIを含まない指示でモデルを呼び出し、JSON出力の強制や判定フォールバックを含むことを検証。
- `tests/test_app_streamlit.py`: `app_streamlit.py` に ROI 描画コードが残っていないこと、`run_vision_eval` 呼び出しが新しいシグネチャ（3引数）に従っていることを検証。

## デバッグログの取得
- 生成した単体アプリ (`prod_app/runtime_app_*.py`) は、環境変数 `AVI_DEBUG=1`（または `true`）を設定して起動すると、AI に送信するメッセージ／レスポンスを標準出力にダンプします（画像データは `<image omitted>` に置き換え）。
- 例: `AVI_DEBUG=1 streamlit run prod_app/runtime_app_latest.py`

## 運用上の注意
- このファイルに記載されていない仕様変更を行う場合は、README ではなく本ファイルを更新してから作業を進めてください。
- 新しい最終アプリを生成した際は、不要になった旧 `runtime_app_*.py` を整理しておくと利用者が迷わない。
- APIキーは実行環境の `.env` に保存されるため、配布前に含めないよう注意してください（`prod_app/.env` を適宜クリア）。
