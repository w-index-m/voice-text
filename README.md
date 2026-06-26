# 🎙️ 音声議事録メーカー

スマホ等の音声ファイルをアップロードし、文字起こし＋話者分離を行い、
LLMで議事録に整形するStreamlitアプリ。

## 機能
- 音声アップロード（m4a / mp3 / wav など）
- AssemblyAI による文字起こし＋話者分離
- 検出話者の実名リネーム
- 要約の粒度切替（3行 / 詳細 / 決定事項のみ）
- 議事録(.md) / 話者分離テキスト(.txt) / ToDo(.csv) のダウンロード

## 必要なAPIキー
- OpenAI API キー（議事録整形）
- AssemblyAI API キー（文字起こし＋話者分離）

## ローカル実行
```bash
pip install -r requirements.txt
# .streamlit/secrets.toml.example をコピーして secrets.toml を作成し、キーを記入
streamlit run app.py
```

## Streamlit Community Cloud へのデプロイ
1. このリポジトリをGitHubにpush
2. https://share.streamlit.io でリポジトリを選択
3. Main file path に `app.py` を指定
4. Settings → Secrets に以下を貼り付け
   ```
   OPENAI_API_KEY = "sk-..."
   ASSEMBLYAI_API_KEY = "..."
   ```
