# 🎙️ 音声議事録メーカー (Voice-Text)

**[🇯🇵 日本語](#日本語) ｜ [🇬🇧 English](#english)**

---

<a id="日本語"></a>
## 🇯🇵 日本語

音声・動画をアップロード（またはマイク録音）して、文字起こし・話者分離・議事録/セミナーまとめ作成・翻訳・日英リアルタイム通訳・日本語吹き替え生成までを行う Streamlit アプリです。

### ✨ 主な機能

#### 📝 議事録・まとめ作成
- 音声・動画のアップロード、またはブラウザのマイクで直接録音
- AssemblyAI による文字起こし＋話者分離
- 検出話者の実名リネーム（出席者名から LLM が自動推定）
- **文書の種類**を「会議」「セミナー・講演」から選択
  - 会議: 参加者・議論の要点・決定事項・ToDo を中心に構造化
  - セミナー: テーマ・要点・インサイト・Q&A・学びを中心に構造化
- 要約の粒度切替（3行サマリ / 詳細 / 決定事項のみ・重要ポイントのみ）
- **聞いたままの再現（逐語）モード**: 要約せず全発言を忠実に再現（日本語翻訳も可）
- 生成した議事録/まとめをその場で編集可能
- 議事録を英語⇄日本語に翻訳（Markdown構造を保持）
- ToDo 抽出・CSV 出力
- 会議の温度感レポート（熱量・対立・合意・発言バランスの分析）

#### 📊 可視化・分析
- 録音/アップロード音声の波形表示（音量の強弱を色分け）
- 話者ごとの発言時間の可視化（横棒グラフ＋表）
- 話者ごとの話し方の傾向分析（話速・ポーズ比率・ピッチ変動の音響分析、実験的機能）
- 文字起こしの RTF（Real-Time Factor）表示で処理速度を把握

#### 🌐 翻訳・リアルタイム機能（実験的機能）
- 📝 **ライブ文字起こし**: 録音しながら数秒遅れで文字起こしを表示
- 🌐 **リアルタイム翻訳**: 話しながら日本語⇄英語（自動判定・双方向）に翻訳して表示
- 🔊 **音声通訳 日英**: 話した内容を翻訳し、音声で読み上げ（双方向会話通訳）
- 🎬 **日本語吹き替え生成**: 英語などの音声・動画講演を、文字起こし→全文翻訳→日本語音声化。動画は音声トラックを日本語に差し替えた動画も生成可能（ファイルアップロード / マイク録音どちらにも対応）

#### 🔧 音声前処理・品質向上
- フィラー除去（「えーと」「あのー」等の自動除去）
- ノイズ除去（環境音の軽減、任意）
- アップロード失敗時のリトライ処理

#### 🤖 LLM・文字起こしエンジン
- 議事録整形: Gemini 2.5 Flash（無料） / Groq Llama 3.3（無料・高速） / Claude Sonnet（AssemblyAI クレジット消費）から選択
- 翻訳は失敗時に Groq → AssemblyAI(Claude) へ自動フォールバック
- 文字起こしは AssemblyAI（高精度・単一言語）/ Whisper via Groq（多言語混在対応）から選択可能

#### 📱 その他
- モバイル最適化・PWA対応（ホーム画面に追加してアプリのように利用可能）
- Transcript ID によるセッション復元

### 🔑 必要な API キー

| キー | 用途 | 必須 |
|---|---|---|
| `ASSEMBLYAI_API_KEY` | 文字起こし・話者分離・Claude 利用（LLM Gateway） | ✅ 必須 |
| `GEMINI_API_KEY` | 議事録整形（無料枠あり） | 任意 |
| `GROQ_API_KEY` | 議事録整形・Whisper文字起こし・翻訳フォールバック（無料枠あり・推奨） | 任意（推奨） |

- AssemblyAI: https://www.assemblyai.com/
- Gemini: https://aistudio.google.com/apikey （クレジットカード不要）
- Groq: https://console.groq.com/keys （無料）

### 💻 ローカル実行

```bash
pip install -r requirements.txt
# .streamlit/secrets.toml.example をコピーして secrets.toml を作成し、キーを記入
cp secrets.toml.example .streamlit/secrets.toml
streamlit run app.py
```

### ☁️ Streamlit Community Cloud へのデプロイ

1. このリポジトリを GitHub に push
2. https://share.streamlit.io でリポジトリを選択
3. Main file path に `app.py` を指定
4. Settings → Secrets に以下を貼り付け

   ```toml
   ASSEMBLYAI_API_KEY = "..."
   GEMINI_API_KEY = "..."
   GROQ_API_KEY = "..."
   ```

### 🧪 実験的機能について

リアルタイム翻訳・音声通訳・ライブ文字起こし・日本語吹き替え生成は `streamlit-webrtc` を使った実験的機能です。ブラウザ・ネットワーク環境によっては接続が不安定になる場合があります。また、生成される音声は元の話速・間（ま）に完全同期するものではなく、ナレーション的な読み上げです。

### 📄 ライセンス

このリポジトリのライセンスは特に指定がない場合、リポジトリ管理者の設定に従います。

---

<a id="english"></a>
## 🇬🇧 English

**[⬆ Back to language menu](#-音声議事録メーカー-voice-text)**

A Streamlit app that uploads (or records) audio/video and performs transcription, speaker diarization, meeting minutes / seminar summary generation, translation, real-time Japanese⇄English interpretation, and Japanese dubbing generation.

### ✨ Features

#### 📝 Minutes & Summary Generation
- Upload audio/video files, or record directly from your browser's microphone
- Transcription + speaker diarization via AssemblyAI
- Rename detected speakers with real names (LLM can auto-guess from an attendee list)
- Choose **document type**: "Meeting" or "Seminar/Lecture"
  - Meeting: structured around participants, discussion points, decisions, and action items
  - Seminar: structured around topic, key points, insights, Q&A, and takeaways
- Adjustable summary granularity (3-line summary / detailed / decisions-only or key-points-only)
- **Verbatim mode**: faithfully reproduces every utterance without summarizing (optional translation to Japanese)
- Edit the generated minutes/summary in place
- Translate minutes between English and Japanese (preserves Markdown structure)
- Extract action items and export as CSV
- Meeting "temperature" report (energy, conflict, consensus, speaking-time balance analysis)

#### 📊 Visualization & Analysis
- Waveform visualization of recorded/uploaded audio (color-coded by volume)
- Per-speaker talk-time visualization (bar chart + table)
- Per-speaker speaking-style analysis (speech rate, pause ratio, pitch variation via acoustic analysis — experimental)
- RTF (Real-Time Factor) display for transcription to gauge processing speed

#### 🌐 Translation & Real-Time Features (Experimental)
- 📝 **Live Transcription**: shows transcribed text with a few seconds of delay while you record
- 🌐 **Real-Time Translation**: translates speech between Japanese and English (auto-detect, bidirectional) as you talk
- 🔊 **Voice Interpreter (JA/EN)**: translates what you say and reads it aloud (bidirectional conversational interpretation)
- 🎬 **Japanese Dubbing Generator**: takes an English (or other language) audio/video lecture and runs it through transcription → full translation → Japanese speech synthesis. For video input, it can also produce a video with the audio track replaced by the Japanese dub (works with both file upload and microphone recording)

#### 🔧 Audio Preprocessing & Quality
- Filler-word removal (e.g. Japanese "um"/"uh" equivalents)
- Noise reduction (optional, reduces ambient noise)
- Retry logic for failed uploads

#### 🤖 LLM & Transcription Engines
- Minutes generation: choose from Gemini 2.5 Flash (free) / Groq Llama 3.3 (free, fast) / Claude Sonnet (consumes AssemblyAI credits)
- Translation automatically falls back from Groq → AssemblyAI (Claude) on failure
- Transcription engine: choose between AssemblyAI (high accuracy, single language) or Whisper via Groq (handles mixed languages)

#### 📱 Other
- Mobile-optimized, PWA-ready (can be added to your home screen and used like a native app)
- Session restore via Transcript ID

### 🔑 Required API Keys

| Key | Purpose | Required |
|---|---|---|
| `ASSEMBLYAI_API_KEY` | Transcription, speaker diarization, Claude access (via LLM Gateway) | ✅ Required |
| `GEMINI_API_KEY` | Minutes generation (has a free tier) | Optional |
| `GROQ_API_KEY` | Minutes generation, Whisper transcription, translation fallback (free tier, recommended) | Optional (recommended) |

- AssemblyAI: https://www.assemblyai.com/
- Gemini: https://aistudio.google.com/apikey (no credit card required)
- Groq: https://console.groq.com/keys (free)

### 💻 Run Locally

```bash
pip install -r requirements.txt
# Copy .streamlit/secrets.toml.example to secrets.toml and fill in your keys
cp secrets.toml.example .streamlit/secrets.toml
streamlit run app.py
```

### ☁️ Deploy to Streamlit Community Cloud

1. Push this repository to GitHub
2. Select the repository at https://share.streamlit.io
3. Set the main file path to `app.py`
4. Paste the following into Settings → Secrets:

   ```toml
   ASSEMBLYAI_API_KEY = "..."
   GEMINI_API_KEY = "..."
   GROQ_API_KEY = "..."
   ```

### 🧪 About Experimental Features

Real-time translation, voice interpretation, live transcription, and Japanese dubbing generation are experimental features built on `streamlit-webrtc`. Connection stability may vary depending on your browser and network environment. Generated speech is not perfectly synced to the original speaker's pacing or pauses — it's a narration-style dub rather than lip-synced audio.

### 📄 License

Unless otherwise specified, this repository follows the licensing terms set by the repository owner.
