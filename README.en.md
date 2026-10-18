# 🎙️ Voice-Text (Voice Meeting Minutes Maker)

[日本語版はこちら / Japanese README is here](README.md)

A Streamlit app that uploads (or records) audio/video and performs transcription, speaker diarization, meeting minutes / seminar summary generation, translation, real-time Japanese⇄English interpretation, and Japanese dubbing generation.

## ✨ Features

### 📝 Minutes & Summary Generation
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

### 📊 Visualization & Analysis
- Waveform visualization of recorded/uploaded audio (color-coded by volume)
- Per-speaker talk-time visualization (bar chart + table)
- Per-speaker speaking-style analysis (speech rate, pause ratio, pitch variation via acoustic analysis — experimental)
- RTF (Real-Time Factor) display for transcription to gauge processing speed

### 🌐 Translation & Real-Time Features (Experimental)
- 📝 **Live Transcription**: shows transcribed text with a few seconds of delay while you record
- 🌐 **Real-Time Translation**: translates speech between Japanese and English (auto-detect, bidirectional) as you talk
- 🔊 **Voice Interpreter (JA/EN)**: translates what you say and reads it aloud (bidirectional conversational interpretation)
- 🎬 **Japanese Dubbing Generator**: takes an English (or other language) audio/video lecture and runs it through transcription → full translation → Japanese speech synthesis. For video input, it can also produce a video with the audio track replaced by the Japanese dub (works with both file upload and microphone recording)

### 🔧 Audio Preprocessing & Quality
- Filler-word removal (e.g. Japanese "um"/"uh" equivalents)
- Noise reduction (optional, reduces ambient noise)
- Retry logic for failed uploads

### 🤖 LLM & Transcription Engines
- Minutes generation: choose from Gemini 2.5 Flash (free) / Groq Llama 3.3 (free, fast) / Claude Sonnet (consumes AssemblyAI credits)
- Translation automatically falls back from Groq → AssemblyAI (Claude) on failure
- Transcription engine: choose between AssemblyAI (high accuracy, single language) or Whisper via Groq (handles mixed languages)

### 📱 Other
- Mobile-optimized, PWA-ready (can be added to your home screen and used like a native app)
- Session restore via Transcript ID

## 🔑 Required API Keys

| Key | Purpose | Required |
|---|---|---|
| `ASSEMBLYAI_API_KEY` | Transcription, speaker diarization, Claude access (via LLM Gateway) | ✅ Required |
| `GEMINI_API_KEY` | Minutes generation (has a free tier) | Optional |
| `GROQ_API_KEY` | Minutes generation, Whisper transcription, translation fallback (free tier, recommended) | Optional (recommended) |

- AssemblyAI: https://www.assemblyai.com/
- Gemini: https://aistudio.google.com/apikey (no credit card required)
- Groq: https://console.groq.com/keys (free)

## 💻 Run Locally

```bash
pip install -r requirements.txt
# Copy .streamlit/secrets.toml.example to secrets.toml and fill in your keys
cp secrets.toml.example .streamlit/secrets.toml
streamlit run app.py
```

## ☁️ Deploy to Streamlit Community Cloud

1. Push this repository to GitHub
2. Select the repository at https://share.streamlit.io
3. Set the main file path to `app.py`
4. Paste the following into Settings → Secrets:

   ```toml
   ASSEMBLYAI_API_KEY = "..."
   GEMINI_API_KEY = "..."
   GROQ_API_KEY = "..."
   ```

## 🧪 About Experimental Features

Real-time translation, voice interpretation, live transcription, and Japanese dubbing generation are experimental features built on `streamlit-webrtc`. Connection stability may vary depending on your browser and network environment. Generated speech is not perfectly synced to the original speaker's pacing or pauses — it's a narration-style dub rather than lip-synced audio.

## 📄 License

Unless otherwise specified, this repository follows the licensing terms set by the repository owner.
