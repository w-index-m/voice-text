"""
音声議事録アプリ (Streamlit)
- AssemblyAI で文字起こし＋話者分離
- LLM は Gemini（無料・恒常）/ AssemblyAI LLM Gateway(Claude) を選択可能
- 話者自動リネーム / 粒度切替 / ToDo CSV出力 / 議事録編集 / 波形表示

キー構成:
  Gemini使用時  → ASSEMBLYAI_API_KEY + GEMINI_API_KEY
  Claude使用時  → ASSEMBLYAI_API_KEY のみ（LLM Gatewayで消費）

.streamlit/secrets.toml:
  ASSEMBLYAI_API_KEY = "..."
  GEMINI_API_KEY = "..."  # Gemini使用時のみ必要

起動: streamlit run app.py
"""

import io
import csv
import json
import os
import subprocess
import tempfile
import time

import numpy as np
import matplotlib.pyplot as plt
import requests
import streamlit as st
import streamlit.components.v1 as components
from streamlit_mic_recorder import mic_recorder

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False

try:
    import imageio_ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False

try:
    import queue
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

from openai import OpenAI

VIDEO_EXTS = ["mp4", "mov", "avi", "mkv", "webm", "m4v"]

AAI_BASE = "https://api.assemblyai.com/v2"

LLM_BACKENDS = {
    "Gemini 2.5 Flash（無料・推奨）": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.5-flash",
        "key_secret": "GEMINI_API_KEY",
    },
    "Groq Llama 3.3 70B（高速・無料枠）": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "key_secret": "GROQ_API_KEY",
    },
    "Claude Sonnet（AssemblyAIクレジット消費）": {
        "base_url": "https://llm-gateway.assemblyai.com/v1",
        "model": "claude-sonnet-4-5-20250929",
        "key_secret": "ASSEMBLYAI_API_KEY",
    },
}


def has_backend_key(backend_name: str) -> bool:
    secret = LLM_BACKENDS[backend_name]["key_secret"]
    return bool(st.secrets.get(secret) or os.environ.get(secret))

# ----- カスタムCSS（ダークモード対応） -----
def apply_custom_css():
    st.markdown("""
    <style>
    /* ヘッダー */
    .stApp header { background: transparent; }

    /* タイトル */
    h1 { font-size: 2rem !important; font-weight: 700 !important; }

    /* プライマリボタン */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        border: none;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        transition: opacity 0.2s;
    }
    .stButton > button[kind="primary"]:hover { opacity: 0.85; }

    /* セカンダリボタン */
    .stButton > button[kind="secondary"] {
        border-radius: 8px;
        font-weight: 500;
    }

    /* カード風コンテナ */
    .card {
        background: var(--background-color, #1e1e2e);
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
    }

    /* プログレスバー */
    .stProgress > div > div { border-radius: 99px; }

    /* テキストエリア */
    .stTextArea textarea {
        border-radius: 8px;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* サイドバー */
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(99,102,241,0.2);
    }

    /* ダウンロードボタン */
    .stDownloadButton > button {
        border-radius: 8px;
        width: 100%;
    }

    /* 成功・警告メッセージ */
    .stAlert { border-radius: 8px; }

    /* ===== モバイル最適化 / PWA ===== */
    /* タップターゲットを大きく（44px以上推奨） */
    .stButton > button, .stDownloadButton > button {
        min-height: 44px;
    }
    /* iOSでの意図しないズームを防ぐ（入力は16px以上） */
    input, textarea, select { font-size: 16px !important; }

    @media (max-width: 640px) {
        /* 余白を詰めて画面を有効活用 */
        .block-container { padding: 1rem 0.8rem 3rem 0.8rem !important; }
        h1 { font-size: 1.5rem !important; }
        /* 横並びカラムをスマホでは縦積みに */
        [data-testid="stHorizontalBlock"] { flex-direction: column; }
        [data-testid="column"] { width: 100% !important; flex: 1 1 100% !important; }
        /* タブを折り返し可能に */
        [data-testid="stTabs"] [role="tablist"] { flex-wrap: wrap; }
    }

    /* iOSのセーフエリア（ノッチ/ホームバー）対応 */
    .stApp { padding-bottom: env(safe-area-inset-bottom); }
    </style>

    <!-- PWA / モバイル用メタ情報 -->
    <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
    <meta name="mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="音声議事録">
    <meta name="theme-color" content="#6366f1">
    """, unsafe_allow_html=True)


# ----- キー -----
def get_aai_key() -> str:
    key = st.secrets.get("ASSEMBLYAI_API_KEY") or os.environ.get("ASSEMBLYAI_API_KEY")
    if not key:
        st.error("ASSEMBLYAI_API_KEY が未設定です。secrets.toml を確認してください。")
        st.stop()
    return key


def get_llm_client(backend_name: str) -> OpenAI:
    cfg = LLM_BACKENDS[backend_name]
    key = st.secrets.get(cfg["key_secret"]) or os.environ.get(cfg["key_secret"])
    if not key:
        st.error(f"{cfg['key_secret']} が未設定です。secrets.toml を確認してください。")
        st.stop()
    return OpenAI(api_key=key, base_url=cfg["base_url"])


def get_llm_model(backend_name: str) -> str:
    return LLM_BACKENDS[backend_name]["model"]


# ----- 音声の長さ算出 -----
def get_audio_duration(audio_bytes: bytes) -> float:
    """音声の長さ（秒）を返す。取得できなければ0.0。"""
    # まず標準ライブラリのwaveで試す（WAVなら依存なしで高速）
    try:
        import wave
        with wave.open(io.BytesIO(audio_bytes), "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
            if rate:
                return frames / float(rate)
    except Exception:
        pass
    # WAV以外はlibrosaにフォールバック
    if LIBROSA_AVAILABLE and SOUNDFILE_AVAILABLE:
        try:
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
            return len(y) / sr
        except Exception:
            pass
    return 0.0


def fmt_duration_ja(seconds: float) -> str:
    """秒数を日本語表記に整形。1時間以上は「○時間○分○秒」。"""
    total = int(seconds)
    h, m, s = total // 3600, (total % 3600) // 60, total % 60
    if h > 0:
        return f"{h}時間{m}分{s}秒"
    if m > 0:
        return f"{m}分{s}秒"
    return f"{s}秒"


# ----- 音声波形の可視化 -----
def render_waveform(audio_bytes: bytes, file_name: str = ""):
    if not LIBROSA_AVAILABLE or not SOUNDFILE_AVAILABLE:
        return
    try:
        import matplotlib.colors as mcolors
        from matplotlib.collections import LineCollection

        buf = io.BytesIO(audio_bytes)
        y, sr = librosa.load(buf, sr=None, mono=True, duration=300)
        duration = len(y) / sr

        # RMS（音の強弱）をフレームごとに算出
        hop = 512
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
        # 0-1 に正規化（強弱の色分け用）
        rms_norm = rms / (rms.max() + 1e-9)

        fig, (ax_wave, ax_bar) = plt.subplots(
            2, 1, figsize=(10, 2.6), gridspec_kw={"height_ratios": [3, 1]})
        fig.patch.set_alpha(0)

        # 上段: 波形を強弱で色分け（静か=青 → 大きい=赤）
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "vol", ["#3b82f6", "#22c55e", "#eab308", "#ef4444"])
        ax_wave.set_facecolor("none")
        env = np.interp(np.linspace(0, len(rms_norm) - 1, len(y)),
                        np.arange(len(rms_norm)), rms_norm)
        times = np.linspace(0, duration, len(y))
        # 強弱に応じて縦線の色を変えて塗る（区間ごとに色付け）
        step = max(1, len(y) // 1500)
        for i in range(0, len(y) - step, step):
            ax_wave.fill_between(times[i:i + step + 1],
                                 y[i:i + step + 1], -y[i:i + step + 1],
                                 color=cmap(env[i]), linewidth=0)
        ax_wave.set_xlim(0, duration)
        ax_wave.set_yticks([])
        ax_wave.set_xticks([])
        for spine in ax_wave.spines.values():
            spine.set_visible(False)

        # 下段: 音量バー（強弱のヒートバー）
        ax_bar.imshow(rms_norm[np.newaxis, :], aspect="auto", cmap=cmap,
                      extent=[0, duration, 0, 1], vmin=0, vmax=1)
        ax_bar.set_yticks([])
        ax_bar.set_xlabel("秒", color="#aaa", fontsize=9)
        ax_bar.tick_params(colors="#aaa")
        for spine in ax_bar.spines.values():
            spine.set_visible(False)

        fig.tight_layout(pad=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        peak = rms_times[int(np.argmax(rms))]
        st.caption(
            f"音声長: {fmt_duration_ja(duration)} ／ "
            f"最大音量: 約{fmt_duration_ja(peak)}付近 "
            f"（🔵静か → 🔴大きい で色分け）"
        )
    except Exception:
        pass  # 波形表示に失敗しても続行


# ----- 動画から音声を抽出 -----
def extract_audio_from_video(video_bytes: bytes, suffix: str = ".mp4") -> bytes:
    """ffmpeg(imageio-ffmpeg同梱)で動画から音声をWAV抽出する。"""
    if not FFMPEG_AVAILABLE:
        raise RuntimeError(
            "動画の音声抽出には imageio-ffmpeg が必要です。requirements.txt を確認してください。")
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as vf:
        vf.write(video_bytes)
        video_path = vf.name
    audio_path = video_path + ".wav"
    try:
        cmd = [ffmpeg_exe, "-y", "-i", video_path,
               "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", audio_path]
        subprocess.run(cmd, check=True, capture_output=True)
        with open(audio_path, "rb") as f:
            return f.read()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"音声抽出に失敗しました: {e.stderr.decode(errors='ignore')[:300]}")
    finally:
        for p in (video_path, audio_path):
            try:
                os.remove(p)
            except OSError:
                pass


# ----- 音声前処理: ノイズ除去 -----
def reduce_noise(audio_bytes: bytes) -> bytes:
    """文字起こし前に音声からノイズ（空調音・環境音等）を除去する。
    失敗時・ライブラリ未導入時は元の音声をそのまま返す。"""
    if not (NOISEREDUCE_AVAILABLE and LIBROSA_AVAILABLE and SOUNDFILE_AVAILABLE):
        return audio_bytes
    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
        reduced = nr.reduce_noise(y=y, sr=sr, stationary=False)
        out = io.BytesIO()
        sf.write(out, reduced, sr, format="WAV")
        return out.getvalue()
    except Exception:
        return audio_bytes


# ----- AssemblyAI: 文字起こし -----
def aai_upload(api_key: str, file_bytes: bytes, max_retries: int = 4) -> str:
    headers = {"authorization": api_key}
    for attempt in range(max_retries):
        try:
            resp = requests.post(f"{AAI_BASE}/upload", headers=headers,
                                 data=file_bytes, timeout=120)
            resp.raise_for_status()
            return resp.json()["upload_url"]
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            time.sleep(wait)
    raise RuntimeError("アップロード失敗")


def aai_transcribe(api_key: str, audio_url: str, language: str,
                   audio_duration_hint: float | None = None,
                   remove_disfluencies: bool = True) -> dict:
    headers = {"authorization": api_key}
    payload = {"audio_url": audio_url, "speaker_labels": True}
    if language:
        payload["language_code"] = language
    # disfluencies=False で「えーと」「あのー」等のフィラーを文字起こしから除去
    payload["disfluencies"] = not remove_disfluencies

    # リトライ付きでジョブ投入
    for attempt in range(4):
        try:
            resp = requests.post(f"{AAI_BASE}/transcript", json=payload,
                                 headers=headers, timeout=30)
            resp.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)

    tid = resp.json()["id"]
    start_time = time.time()

    # 残り時間推定: AssemblyAI は概ね音声長の 20〜30% の処理時間が目安
    estimated_total = (audio_duration_hint * 0.25) if audio_duration_hint else 60.0
    estimated_total = max(estimated_total, 15.0)

    progress = st.progress(0.0, text="文字起こし中（話者分離あり）...")
    status_text = st.empty()
    poll = 0
    consecutive_errors = 0

    while True:
        poll += 1
        try:
            r = requests.get(f"{AAI_BASE}/transcript/{tid}", headers=headers, timeout=15)
            r.raise_for_status()
            consecutive_errors = 0
        except requests.RequestException as e:
            consecutive_errors += 1
            if consecutive_errors >= 5:
                progress.empty()
                status_text.empty()
                raise RuntimeError(f"AssemblyAI への接続に繰り返し失敗しました: {e}")
            time.sleep(3)
            continue

        data = r.json()
        status = data["status"]

        elapsed = time.time() - start_time
        ratio = min(elapsed / estimated_total, 0.92)
        remaining = max(0, estimated_total - elapsed)

        if status == "completed":
            progress.progress(1.0, text="文字起こし完了!")
            time.sleep(0.4)
            progress.empty()
            status_text.empty()
            return data

        if status == "error":
            progress.empty()
            status_text.empty()
            raise RuntimeError(f"AssemblyAI エラー: {data.get('error')}")

        progress.progress(ratio, text=f"文字起こし中... ({status})")
        if remaining > 5:
            status_text.caption(f"推定残り時間: 約 {int(remaining)} 秒")
        else:
            status_text.caption("もうすぐ完了します...")

        time.sleep(3)


def build_diarized_text(data: dict, name_map: dict | None = None) -> str:
    utterances = data.get("utterances")
    if not utterances:
        return data.get("text", "")
    name_map = name_map or {}
    lines = []
    for u in utterances:
        label = u["speaker"]
        display = name_map.get(label) or f"話者{label}"
        lines.append(f"{display}: {u['text']}")
    return "\n".join(lines)


# ----- リアルタイム翻訳（実験）用ヘルパー -----
def frames_to_wav_bytes(frames: list, target_rate: int = 16000) -> bytes | None:
    """streamlit-webrtc の音声フレーム列を16kHzモノラル16bit WAVに変換する。
    PyAVのリサンプラーでフォーマット/チャンネルを正しく揃える。"""
    import wave
    if not frames:
        return None
    try:
        import av
        resampler = av.AudioResampler(format="s16", layout="mono", rate=target_rate)
        pcm = bytearray()
        for f in frames:
            out = resampler.resample(f)
            # PyAVのバージョンにより単一フレーム/リストのどちらも返り得る
            if not isinstance(out, list):
                out = [out] if out is not None else []
            for rf in out:
                pcm += bytes(rf.planes[0])
        if not pcm:
            return None
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(target_rate)
            w.writeframes(bytes(pcm))
        return buf.getvalue()
    except Exception:
        # フォールバック: 手動変換（フォーマット依存で崩れる可能性あり）
        try:
            chunks = []
            src_rate = None
            for f in frames:
                arr = f.to_ndarray()
                src_rate = f.sample_rate
                if arr.ndim == 2:
                    arr = arr.mean(axis=0)
                arr = arr.flatten().astype(np.float32)
                if np.max(np.abs(arr)) > 1.5:
                    arr = arr / 32768.0
                chunks.append(arr)
            if not chunks or not src_rate:
                return None
            samples = np.concatenate(chunks)
            if src_rate != target_rate:
                n = int(len(samples) * target_rate / src_rate)
                if n <= 0:
                    return None
                samples = np.interp(
                    np.linspace(0, len(samples) - 1, n),
                    np.arange(len(samples)), samples)
            pcm2 = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(target_rate)
                w.writeframes(pcm2.tobytes())
            return buf.getvalue()
        except Exception:
            return None


def get_groq_client() -> OpenAI | None:
    """GroqのOpenAI互換クライアント。キー未設定ならNone。"""
    key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    if not key:
        return None
    return OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")


def whisper_transcribe_bytes(groq_client: OpenAI, wav_bytes: bytes) -> str:
    """Groq上のWhisper(large-v3)で文字起こし。言語自動判定・多言語混在に比較的強い。"""
    try:
        bio = io.BytesIO(wav_bytes)
        bio.name = "audio.wav"
        resp = groq_client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=bio,
        )
        return (getattr(resp, "text", "") or "").strip()
    except Exception:
        return ""


def aai_transcribe_quick(api_key: str, audio_url: str, language: str = "en") -> str:
    """話者分離なしの軽量な文字起こし（リアルタイム用・完了までポーリング）。"""
    headers = {"authorization": api_key}
    payload = {"audio_url": audio_url}
    if language:
        payload["language_code"] = language
    resp = requests.post(f"{AAI_BASE}/transcript", json=payload,
                         headers=headers, timeout=30)
    resp.raise_for_status()
    tid = resp.json()["id"]
    for _ in range(40):  # 最大約40秒
        r = requests.get(f"{AAI_BASE}/transcript/{tid}", headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        if data["status"] == "completed":
            return data.get("text", "") or ""
        if data["status"] == "error":
            return ""
        time.sleep(1)
    return ""


def translate_to_ja(client: OpenAI, model: str, text: str) -> str:
    """英語などのテキストを自然な日本語に翻訳する（要約しない）。エラー時は例外を送出。"""
    if not text.strip():
        return ""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": "入力テキストを、意味を変えず自然な日本語に翻訳するアシスタント。"
                        "翻訳文のみを返し、注釈や原文は付けない。"},
            {"role": "user", "content": text},
        ],
        max_tokens=1000,
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


def translate_to_en(client: OpenAI, model: str, text: str) -> str:
    """日本語などのテキストを自然な英語に翻訳する（要約しない）。エラー時は例外を送出。"""
    if not text.strip():
        return ""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": "Translate the input into natural English, preserving meaning. "
                        "Return only the translation, no notes or original text."},
            {"role": "user", "content": text},
        ],
        max_tokens=1000,
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


def translate_with_fallback(client: OpenAI, model: str, text: str,
                            direction: str) -> tuple[str, str]:
    """翻訳を実行し、失敗/空ならGroqに自動フォールバックする。
    戻り値: (翻訳結果, 使用エンジンの注記)。両方失敗なら ("", エラー内容)。"""
    fn = translate_to_en if direction == "en" else translate_to_ja
    err = ""
    try:
        out = fn(client, model, text)
        if out.strip():
            return out, ""
        err = "主エンジンが空を返しました"
    except Exception as e:
        err = f"主エンジンエラー: {e}"
    # Groqへフォールバック
    gc = get_groq_client()
    if gc is not None:
        try:
            out = fn(gc, "llama-3.3-70b-versatile", text)
            if out.strip():
                return out, "（Groqにフォールバック）"
        except Exception as e:
            return "", f"{err} / Groqも失敗: {e}"
    return "", f"{err}（GroqキーGROQ_API_KEY未設定でフォールバック不可）"


# ----- LLM: 話者自動推定 -----
def auto_map_speakers(client: OpenAI, model: str,
                      speakers: list[str], attendees: str,
                      sample_text: str) -> dict[str, str]:
    if not attendees.strip() or not speakers:
        return {}
    cands = [a.strip() for a in attendees.replace(",", "\n").splitlines() if a.strip()]
    if not cands:
        return {}

    system = (
        "会議の文字起こしと出席者リストから、各話者の実名を推定するアシスタント。"
        "発言内容・話し方・呼びかけ（「○○さん」等）・議題の主導者かどうかといった手がかりから、"
        "話者の役職や立場も考慮して、最も確からしい出席者名を割り当てます。"
    )
    user = f"""以下の情報を元に、各話者ラベルに最も対応しそうな出席者名を推定してください。
発言内容から推測できる役職・立場（司会/上長/担当者など）や、他者からの呼びかけも手がかりにしてください。
推定が難しい場合は空文字にしてください。
JSON オブジェクトのみ返してください（説明不要）。

話者ラベル: {speakers}
出席者候補: {cands}

文字起こしサンプル（最初の1500文字）:
{sample_text[:1500]}

出力例: {{"A": "田中太郎", "B": "鈴木花子", "C": ""}}
"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=300,
            temperature=0.0,
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        mapping = json.loads(raw)
        return {k: v for k, v in mapping.items() if isinstance(v, str) and v}
    except Exception:
        return {}


# ----- LLM: 議事録 -----
GRANULARITY_SPEC = {
    "3行サマリ": "要点を3行程度で簡潔にまとめる。詳細は省く。",
    "詳細議事録": "発言サマリ・議論の要点・決定事項・ToDoまで網羅した詳細な議事録。",
    "決定事項のみ": "決定事項と次アクションだけを抽出。雑談や経緯は省く。",
}


def summarize(client: OpenAI, model: str, diarized_text: str,
              meeting_title: str, meeting_date: str,
              attendees: str, granularity: str) -> str:
    system = (
        "あなたは優秀な議事録作成者です。話者ラベル付きの文字起こしを読み、"
        "日本語で構造化された議事録を Markdown で作成してください。"
    )
    spec = GRANULARITY_SPEC.get(granularity, GRANULARITY_SPEC["詳細議事録"])
    user = f"""以下は会議の文字起こしです。各行が「話者: 発言」の形式です。

# 粒度の指示
{spec}

# メタ情報
会議名: {meeting_title or "（不明）"}
日時: {meeting_date or "（不明）"}
出席者（事前入力）: {attendees or "（未入力）"}

# 出力フォーマット（粒度に応じて不要な節は省略可）
- ## 会議名 / 日時
- ## 参加者
- ## 要約
- ## 発言サマリ（話者ごと）
- ## 議論の要点
- ## 決定事項
- ## ToDo（担当者・期限が分かれば併記）

---
文字起こし（話者分離済み）:
{diarized_text}
"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        max_tokens=4000,
        temperature=0.3,
    )
    return resp.choices[0].message.content


def reproduce_verbatim(client: OpenAI, model: str, diarized_text: str,
                       meeting_title: str, meeting_date: str,
                       translate_ja: bool = False) -> str:
    """要約せず、聞いたままの発言を忠実に再現する（読みやすさのみ整える）。
    translate_ja=True の場合は、内容を省略せず日本語に翻訳して再現する。"""
    if translate_ja:
        system = (
            "あなたは正確な通訳・書き起こし編集者です。話者ラベル付きの文字起こしを、"
            "内容を要約・省略せず、全ての発言を自然な日本語に翻訳して再現してください。"
            "発言の順序・内容・ニュアンスは保持し、省略や意訳のしすぎは避けます。"
        )
        instruction = ("以下の文字起こしを、話者ごとの発言として日本語に翻訳し忠実に再現してください。"
                       "要約や省略はせず、全ての発言を「話者: 日本語訳」の形式で Markdown 出力してください。")
    else:
        system = (
            "あなたは正確な書き起こし編集者です。話者ラベル付きの文字起こしを、"
            "内容を要約・省略・言い換えせず、聞いたままを忠実に再現してください。"
            "行うのは読みやすさの調整のみ（句読点の補正、明らかな誤変換の修正、"
            "重複した言い淀みの軽い整理）で、発言の順序・内容・ニュアンスは保持します。"
        )
        instruction = ("以下の文字起こしを、話者ごとの発言として忠実に再現してください。"
                       "要約や省略はせず、全ての発言を「話者: 発言」の形式で Markdown 出力してください。")
    header = ""
    if meeting_title or meeting_date:
        header = (f"# {meeting_title or '会議'}"
                  + (f"（{meeting_date}）" if meeting_date else "") + "\n\n")
    user = f"""{instruction}

---
{diarized_text}
"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        max_tokens=8000,
        temperature=0.0,
    )
    return header + (resp.choices[0].message.content or "")


def analyze_sentiment(client: OpenAI, model: str, diarized_text: str) -> str:
    system = (
        "あなたは会議のファシリテーション分析の専門家です。"
        "話者ラベル付きの文字起こしを読み、会議の温度感・対立・合意・熱量を分析し、"
        "日本語の Markdown レポートを作成してください。"
    )
    user = f"""以下の会議の文字起こしを分析し、温度感レポートを Markdown で作成してください。

# 出力フォーマット
- ## 全体の雰囲気（一言で / 例: 建設的・緊張気味・和やか など）
- ## 熱量の推移（議論が盛り上がった/停滞した場面）
- ## 対立・意見の相違（あれば、論点と双方の立場）
- ## 合意・前向きな点（合意に至った点、ポジティブな発言）
- ## 発言バランス（話者ごとの発言量・積極性の偏り）
- ## ファシリテーションの観点での気づき（改善提案があれば）

---
文字起こし（話者分離済み）:
{diarized_text}
"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        max_tokens=2500,
        temperature=0.4,
    )
    return resp.choices[0].message.content


def extract_todos(client: OpenAI, model: str, diarized_text: str) -> list[dict]:
    system = "会議の文字起こしからToDo（アクションアイテム）を抽出するアシスタント。"
    user = f"""以下の文字起こしから ToDo を抽出し、JSON配列のみを返してください。
前後の説明やMarkdownのコードフェンスは付けないでください。
各要素のキー: task(タスク内容), owner(担当者/不明なら空文字), due(期限/不明なら空文字)
ToDoが無ければ [] を返してください。

---
{diarized_text}
"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        max_tokens=2000,
        temperature=0.0,
    )
    text = (resp.choices[0].message.content or "").strip()
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        todos = json.loads(text)
        return todos if isinstance(todos, list) else []
    except Exception:
        return []


def todos_to_csv(todos: list[dict]) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["タスク", "担当者", "期限"])
    for t in todos:
        writer.writerow([t.get("task", ""), t.get("owner", ""), t.get("due", "")])
    return ("﻿" + buf.getvalue()).encode("utf-8")


# ----- セッション状態の永続化ヘルパー -----
def save_transcript_cache(aai_data: dict, audio_name: str):
    """文字起こし結果をキャッシュキーとして保存（同一セッション内での再利用）"""
    ss = st.session_state
    ss["aai_data"] = aai_data
    # utterances は話者が検出されないと None になり得るため or [] で保護
    utterances = aai_data.get("utterances") or []
    ss["speakers"] = sorted({u["speaker"] for u in utterances})
    ss["audio_name"] = audio_name
    # セッション間永続化: transcript_id をキャッシュ
    ss["cached_transcript_id"] = aai_data.get("id")


@st.cache_data(show_spinner=False)
def fetch_cached_transcript(api_key: str, transcript_id: str) -> dict | None:
    """transcript_id から文字起こし結果を取得（st.cache_data でキャッシュ）"""
    try:
        headers = {"authorization": api_key}
        r = requests.get(f"{AAI_BASE}/transcript/{transcript_id}", headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        if data.get("status") == "completed":
            return data
    except Exception:
        pass
    return None


# ----- UI -----
st.set_page_config(page_title="音声議事録", page_icon="🎙️", layout="wide",
                   initial_sidebar_state="auto")
apply_custom_css()

st.title("🎙️ 音声議事録メーカー")
st.caption("話者分離 / 自動リネーム / 粒度切替 / ToDo CSV出力 / 議事録編集")

with st.sidebar:
    st.header("LLM設定")
    # キーが設定済みのエンジンのみ表示（Geminiは案内のため常に表示）
    available_backends = [
        name for name in LLM_BACKENDS
        if has_backend_key(name) or "Gemini" in name
    ]
    backend_name = st.radio(
        "議事録整形エンジン",
        available_backends,
        help="Gemini/Groqは無料枠あり。Claudeはコスト有。"
             " GroqはSecretに GROQ_API_KEY を追加すると選べます。",
    )

    st.divider()
    st.header("会議情報")
    meeting_title = st.text_input("会議名（任意）")
    meeting_date = st.text_input("日時（任意・例: 2026-06-27 14:00）")
    attendees = st.text_area("出席者（任意・改行 or カンマ区切り）",
                             help="入力しておくと話者の実名を自動推定できます。")
    language = st.selectbox("音声の言語",
                            options=[("日本語", "ja"), ("英語", "en"), ("自動判定", "")],
                            format_func=lambda x: x[0])[1]
    output_mode = st.radio(
        "出力タイプ",
        ["議事録（要約・構造化）", "聞いたままの再現（逐語）"],
        help="「議事録」は要点をまとめて構造化。「聞いたまま」は要約せず発言を忠実に再現します。",
    )
    is_verbatim = output_mode.startswith("聞いたまま")
    verbatim_translate_ja = False
    if not is_verbatim:
        granularity = st.radio("要約の粒度", list(GRANULARITY_SPEC.keys()), index=1)
    else:
        granularity = "詳細議事録"  # 逐語モードでは未使用
        verbatim_lang = st.radio(
            "逐語の言語",
            ["原文のまま", "日本語に翻訳"],
            help="英語などの音声を、そのまま再現するか日本語に翻訳して再現するか選べます。",
        )
        verbatim_translate_ja = verbatim_lang == "日本語に翻訳"
    show_transcript = st.checkbox("話者分離テキスト全文も表示", value=True)

    st.divider()
    st.header("音声前処理")
    remove_disfluencies = st.checkbox(
        "フィラー除去（えーと/あのー 等）", value=True,
        help="文字起こしテキストから意味のないつなぎ言葉を除去します。")
    use_noise_reduction = st.checkbox(
        "ノイズ除去（文字起こし前に音声を前処理）", value=False,
        disabled=not NOISEREDUCE_AVAILABLE,
        help="空調音・環境音などを軽減して文字起こし精度を上げます。処理に少し時間がかかります。")
    if not NOISEREDUCE_AVAILABLE:
        st.caption("ノイズ除去には noisereduce が必要です（requirements.txt に追加済み）。")

    st.divider()
    # セッションリストア
    st.header("セッション復元")
    restore_id = st.text_input("Transcript ID（前回の結果を再利用）",
                               placeholder="例: abc123...",
                               help="前回の文字起こしIDを入力するとAPIを再実行せず結果を取得します。")
    if st.button("復元", disabled=not restore_id.strip()):
        aai_key = get_aai_key()
        with st.spinner("文字起こし結果を取得中..."):
            restored = fetch_cached_transcript(aai_key, restore_id.strip())
        if restored:
            save_transcript_cache(restored, st.session_state.get("audio_name", "復元音声"))
            st.success("復元しました。")
        else:
            st.error("取得できませんでした。IDを確認してください。")

    st.divider()
    with st.expander("📱 スマホアプリのように使う"):
        st.markdown(
            "ホーム画面に追加すると、アプリのように起動できます。\n\n"
            "**Android (Chrome)**\n"
            "1. 右上の ⋮ メニューを開く\n"
            "2. 「ホーム画面に追加」をタップ\n\n"
            "**iPhone (Safari)**\n"
            "1. 共有ボタン（□↑）をタップ\n"
            "2. 「ホーム画面に追加」を選択\n\n"
            "追加後はアイコンから全画面で起動できます。"
        )

# Gemini選択時のキー案内
if "Gemini" in backend_name:
    gemini_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        st.warning(
            "Geminiを使うには GEMINI_API_KEY が必要です。"
            " [Google AI Studio](https://aistudio.google.com/apikey) で無料発行できます。"
            " secrets.toml に追加してください。"
        )

ss = st.session_state
ss.setdefault("aai_data", None)
ss.setdefault("speakers", [])
ss.setdefault("audio_name", "")
ss.setdefault("recorded_audio", None)
ss.setdefault("minutes_edited", None)
ss.setdefault("audio_duration", None)
ss.setdefault("sentiment", None)

tab_rec, tab_up, tab_rt, tab_vo = st.tabs(
    ["🎤 マイク録音", "📁 ファイルアップロード",
     "🌐 リアルタイム翻訳(実験)", "🔊 日→英 音声通訳(実験)"])

file_bytes = None
audio_name = ""

with tab_rec:
    st.caption("「録音開始」を押してマイクに話しかけ、終わったら「録音停止」を押す")

    # 録音中のリアルタイム経過時間タイマー（録音ボタンの状態をJSで監視）
    components.html(
        """
        <div id="rec-timer" style="font-family:sans-serif;font-size:1.4rem;
             font-weight:700;color:#888;padding:6px 12px;border-radius:8px;
             display:inline-block;">⏱ 待機中 00:00</div>
        <script>
        (function(){
          const el = document.getElementById('rec-timer');
          const fmt = function(s){
            const h=Math.floor(s/3600), m=Math.floor((s%3600)/60), sec=s%60;
            const mm=String(m).padStart(2,'0'), ss=String(sec).padStart(2,'0');
            return h>0 ? (h+':'+mm+':'+ss) : (mm+':'+ss);
          };
          function recordingState(){
            try{
              const frames = window.parent.document.querySelectorAll('iframe');
              for(const f of frames){
                let d; try{ d = f.contentDocument; }catch(e){ continue; }
                if(!d) continue;
                for(const b of d.querySelectorAll('button')){
                  const t=(b.innerText||b.textContent||'').trim();
                  if(t.indexOf('録音停止')>=0) return true;
                  if(t.indexOf('録音開始')>=0) return false;
                }
              }
            }catch(e){}
            return null;
          }
          let recording=false, start=0;
          setInterval(function(){
            const s = recordingState();
            if(s===true){
              if(!recording){ recording=true; start=Date.now(); }
              const sec=Math.floor((Date.now()-start)/1000);
              el.textContent='🔴 録音中 '+fmt(sec);
              el.style.color='#ef4444';
            } else if(s===false){
              recording=false;
              el.textContent='⏱ 待機中 00:00';
              el.style.color='#888';
            }
          }, 250);
        })();
        </script>
        """,
        height=48,
    )

    recorded = mic_recorder(
        start_prompt="🎤 録音開始",
        stop_prompt="⏹ 録音停止",
        just_once=False,
        use_container_width=True,
        key="mic_recorder",
    )
    if recorded:
        ss.recorded_audio = recorded["bytes"]
    if ss.recorded_audio:
        file_bytes = ss.recorded_audio
        audio_name = "録音音声.wav"
        st.audio(ss.recorded_audio, format="audio/wav")
        render_waveform(ss.recorded_audio, audio_name)
        dur = get_audio_duration(ss.recorded_audio)
        dur_text = f"録音時間 {fmt_duration_ja(dur)} / " if dur else ""
        st.success(f"録音完了 / {dur_text}{len(ss.recorded_audio)/1024:.1f} KB")
        if st.button("録音をクリア", key="clear_rec"):
            ss.recorded_audio = None
            st.rerun()

with tab_up:
    uploaded = st.file_uploader(
        "音声・動画ファイルをアップロード",
        type=["m4a", "mp3", "wav", "mp4", "mpeg", "mpga", "webm", "ogg", "flac"]
        + VIDEO_EXTS,
        help="動画は音声を自動抽出して文字起こしします（画面内容の解析は行いません）。",
    )
    if uploaded is not None:
        raw_bytes = uploaded.getvalue()
        audio_name = uploaded.name
        ext = os.path.splitext(uploaded.name)[1].lower().lstrip(".")
        is_video = ext in VIDEO_EXTS
        st.info(f"ファイル: {uploaded.name} / {len(raw_bytes)/1024/1024:.1f} MB"
                + ("（動画）" if is_video else ""))

        if is_video:
            st.video(raw_bytes)
            with st.spinner("動画から音声を抽出中..."):
                try:
                    file_bytes = extract_audio_from_video(raw_bytes, suffix=f".{ext}")
                    st.success("音声を抽出しました。")
                    st.audio(file_bytes, format="audio/wav")
                    render_waveform(file_bytes, audio_name)
                except Exception as e:
                    st.error(f"{e}")
                    file_bytes = None
        else:
            file_bytes = raw_bytes
            st.audio(file_bytes)
            render_waveform(file_bytes, audio_name)

with tab_rt:
    st.caption("英語などの音声を、話しながら数秒遅れで日本語に翻訳して表示します（実験機能）。")
    if not WEBRTC_AVAILABLE:
        st.warning("この機能には streamlit-webrtc が必要です。requirements.txt に追加済みなので"
                   "デプロイ後に利用できます。")
    else:
        groq_ready = get_groq_client() is not None
        engine_opts = ["AssemblyAI（単一言語・高精度）"]
        if groq_ready:
            engine_opts.insert(0, "Whisper via Groq（多言語混在対応）")
        rt_engine = st.radio(
            "文字起こしエンジン", engine_opts, key="rt_engine",
            help="複数言語が混じる音声はWhisperが比較的得意です（GROQ_API_KEYが必要）。",
        )
        use_whisper = rt_engine.startswith("Whisper")
        if not groq_ready:
            st.caption("Whisper（多言語混在対応）を使うには GROQ_API_KEY をSecretに追加してください。")

        if not use_whisper:
            rt_lang = st.selectbox(
                "話す言語",
                options=[("英語", "en"), ("韓国語", "ko"), ("日本語", "ja"),
                         ("自動判定", "")],
                format_func=lambda x: x[0],
                key="rt_lang")[1]
        else:
            rt_lang = ""  # Whisperは自動判定
            st.caption("Whisperは言語を自動判定します（英語＋韓国語などの混在も可）。")
        st.caption("「START」を押してマイクを許可 → 話す。止めるには「STOP」。")

        ss.setdefault("rt_log", [])
        col_a, col_b = st.columns([1, 1])
        with col_b:
            if st.button("🗑 表示をクリア", key="rt_clear"):
                ss.rt_log = []

        webrtc_ctx = webrtc_streamer(
            key="realtime-translate",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=2048,
            media_stream_constraints={"audio": True, "video": False},
            rtc_configuration={"iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]}]},
        )

        output_box = st.empty()
        if ss.rt_log:
            output_box.markdown("### 📝 日本語訳\n\n" + "\n\n".join(ss.rt_log))

        if webrtc_ctx.state.playing:
            aai_key = get_aai_key()
            client = get_llm_client(backend_name)
            model = get_llm_model(backend_name)
            groq_client = get_groq_client() if use_whisper else None
            status = st.empty()
            chunk_seconds = 6
            frame_buffer: list = []

            while True:
                if webrtc_ctx.audio_receiver:
                    try:
                        frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
                    except queue.Empty:
                        frames = []
                else:
                    break

                frame_buffer.extend(frames)
                # バッファ長（秒）を概算
                total = sum(f.samples for f in frame_buffer) if frame_buffer else 0
                rate = frame_buffer[0].sample_rate if frame_buffer else 48000
                dur = total / rate if rate else 0

                if dur >= chunk_seconds:
                    status.caption("🔊 変換中...")
                    wav = frames_to_wav_bytes(frame_buffer)
                    frame_buffer = []
                    if wav:
                        try:
                            if use_whisper and groq_client is not None:
                                en = whisper_transcribe_bytes(groq_client, wav)
                            else:
                                url = aai_upload(aai_key, wav)
                                en = aai_transcribe_quick(aai_key, url, rt_lang or "en")
                            if en.strip():
                                ja, _ = translate_with_fallback(client, model, en, "ja")
                                if ja:
                                    ss.rt_log.append(ja)
                                    output_box.markdown(
                                        "### 📝 日本語訳\n\n" + "\n\n".join(ss.rt_log))
                        except Exception as e:
                            status.caption(f"変換エラー: {e}")
                    status.caption("🎤 録音中...")
        else:
            st.info("STARTを押すと翻訳が始まります。ページを開いたままにしてください。")

with tab_vo:
    st.caption("日本語で話すと、数秒遅れで英語に翻訳し、英語音声で読み上げます（実験機能）。")
    if not WEBRTC_AVAILABLE:
        st.warning("この機能には streamlit-webrtc が必要です（デプロイ後に利用可能）。")
    else:
        vo_groq = get_groq_client()
        vo_use_whisper = vo_groq is not None
        st.caption(
            ("文字起こしはWhisper(Groq)を使用します。" if vo_use_whisper
             else "文字起こしはAssemblyAIを使用します。")
            + " 「START」→ マイク許可 → 日本語で話す。")

        ss.setdefault("vo_log", [])
        if st.button("🗑 表示をクリア", key="vo_clear"):
            ss.vo_log = []

        vo_ctx = webrtc_streamer(
            key="voice-interpret",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=2048,
            media_stream_constraints={"audio": True, "video": False},
            rtc_configuration={"iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]}]},
        )

        vo_box = st.empty()
        vo_speak = st.empty()
        if ss.vo_log:
            vo_box.markdown("### 🔊 English\n\n" + "\n\n".join(ss.vo_log))

        if vo_ctx.state.playing:
            aai_key = get_aai_key()
            client = get_llm_client(backend_name)
            model = get_llm_model(backend_name)
            vo_status = st.empty()
            vo_debug = st.empty()
            chunk_seconds = 5
            frame_buffer2: list = []

            while True:
                if vo_ctx.audio_receiver:
                    try:
                        frames = vo_ctx.audio_receiver.get_frames(timeout=1)
                    except queue.Empty:
                        frames = []
                else:
                    break

                frame_buffer2.extend(frames)
                total = sum(f.samples for f in frame_buffer2) if frame_buffer2 else 0
                rate = frame_buffer2[0].sample_rate if frame_buffer2 else 48000
                dur = total / rate if rate else 0

                if dur >= chunk_seconds:
                    vo_status.caption("🔊 翻訳中...")
                    # 取得音声の音量レベルを算出（0に近い=拾えていない）
                    try:
                        levels = []
                        for f in frame_buffer2:
                            a = f.to_ndarray().astype(np.float32).flatten()
                            if a.size:
                                if np.max(np.abs(a)) > 1.5:
                                    a = a / 32768.0
                                levels.append(float(np.sqrt(np.mean(a ** 2))))
                        level = max(levels) if levels else 0.0
                    except Exception:
                        level = -1.0
                    wav = frames_to_wav_bytes(frame_buffer2)
                    frame_buffer2 = []
                    if not wav:
                        vo_debug.caption("⚠️ 音声データを取得できませんでした（マイク未接続の可能性）")
                        continue
                    try:
                        if vo_use_whisper:
                            ja = whisper_transcribe_bytes(vo_groq, wav)
                        else:
                            url = aai_upload(aai_key, wav)
                            ja = aai_transcribe_quick(aai_key, url, "ja")
                        if not ja.strip():
                            vo_debug.caption(
                                f"🔇 音声が認識されませんでした（音量レベル={level:.4f}）"
                                "／ レベルがほぼ0ならマイクが拾えていません")
                            vo_status.caption("🎤 録音中...")
                            continue
                        en, note = translate_with_fallback(client, model, ja, "en")
                        if not en:
                            vo_debug.caption(f"認識: {ja} ／ ⚠️ 英訳失敗: {note}")
                            vo_status.caption("🎤 録音中...")
                            continue
                        vo_debug.caption(f"認識(日本語): {ja}{note}")
                        ss.vo_log.append(en)
                        vo_box.markdown(
                            "### 🔊 English\n\n" + "\n\n".join(ss.vo_log))
                        # ブラウザTTSで英語読み上げ
                        safe = json.dumps(en)
                        with vo_speak:
                            components.html(
                                f"""<script>
                                try{{
                                  const u=new SpeechSynthesisUtterance({safe});
                                  u.lang='en-US'; u.rate=1.0;
                                  window.speechSynthesis.cancel();
                                  window.speechSynthesis.speak(u);
                                }}catch(e){{}}
                                </script>""", height=0)
                    except Exception as e:
                        vo_debug.caption(f"エラー: {e}")
                    vo_status.caption("🎤 録音中...")
        else:
            st.info("STARTを押すと通訳が始まります。端末の音量を上げておいてください。")

# ステップ1: 文字起こし
if file_bytes is not None:
    # 音声長を推定（波形取得済みなら利用）
    audio_duration_hint = None
    if LIBROSA_AVAILABLE and SOUNDFILE_AVAILABLE:
        try:
            y, sr = librosa.load(io.BytesIO(file_bytes), sr=None, mono=True, duration=300)
            audio_duration_hint = len(y) / sr
            ss.audio_duration = audio_duration_hint
        except Exception:
            pass

    if st.button("① 文字起こし（話者分離）", type="primary"):
        aai_key = get_aai_key()
        upload_bytes = file_bytes
        if use_noise_reduction and NOISEREDUCE_AVAILABLE:
            with st.spinner("ノイズ除去中..."):
                upload_bytes = reduce_noise(file_bytes)
        with st.spinner("音声をアップロード中..."):
            try:
                audio_url = aai_upload(aai_key, upload_bytes)
            except Exception as e:
                st.error(f"アップロード失敗: {e}")
                st.stop()
        try:
            data = aai_transcribe(aai_key, audio_url, language,
                                  audio_duration_hint=ss.get("audio_duration"),
                                  remove_disfluencies=remove_disfluencies)
        except Exception as e:
            st.error(f"文字起こし失敗: {e}")
            st.stop()
        save_transcript_cache(data, audio_name)
        ss.minutes_edited = None
        st.success(f"完了。検出話者: {len(ss.speakers)}名 / Transcript ID: `{data.get('id')}`")
        st.caption("※ このIDをサイドバーに入力すると次回リロード後も結果を復元できます。")

# ステップ2: リネーム & 議事録
if ss.aai_data is not None:
    st.divider()
    st.subheader("② 話者の名前を割り当て")

    raw_text = build_diarized_text(ss.aai_data)
    base_name = os.path.splitext(ss.audio_name)[0]
    st.download_button(
        "📄 文字起こし全文を .txt で保存",
        raw_text.encode("utf-8"),
        file_name=f"{base_name}_文字起こし.txt",
        mime="text/plain",
    )

    # 自動推定ボタン
    col_auto, col_hint = st.columns([1, 3])
    with col_auto:
        auto_map_clicked = st.button("🤖 話者を自動推定", disabled=not attendees.strip(),
                                     help="出席者欄の名前をもとにLLMが話者を推定します。")
    with col_hint:
        if not attendees.strip():
            st.caption("サイドバーの「出席者」を入力すると自動推定が使えます。")

    auto_mapping: dict[str, str] = ss.get("auto_mapping", {})
    if auto_map_clicked and attendees.strip():
        with st.spinner("話者を推定中..."):
            client = get_llm_client(backend_name)
            model = get_llm_model(backend_name)
            auto_mapping = auto_map_speakers(client, model, ss.speakers, attendees, raw_text)
            ss["auto_mapping"] = auto_mapping
        if auto_mapping:
            st.success("自動推定完了。必要に応じて以下で修正してください。")
        else:
            st.warning("推定できませんでした。手動で入力してください。")

    if attendees.strip():
        cand = [a.strip() for a in attendees.replace(",", "\n").splitlines() if a.strip()]
        if cand:
            st.caption("出席者候補: " + " / ".join(cand))

    name_map = {}
    cols = st.columns(min(4, max(1, len(ss.speakers))))
    for i, sp in enumerate(ss.speakers):
        with cols[i % len(cols)]:
            default_name = auto_mapping.get(sp, "")
            name_map[sp] = st.text_input(f"話者{sp} の名前", key=f"name_{sp}",
                                         value=default_name,
                                         placeholder=f"話者{sp}")

    diarized = build_diarized_text(ss.aai_data, name_map)

    if show_transcript:
        with st.expander("話者分離テキスト全文", expanded=False):
            st.text_area("transcript", diarized, height=300)

    gen_label = "③ 逐語再現を作成" if is_verbatim else "③ 議事録を作成"
    if st.button(gen_label, type="primary"):
        client = get_llm_client(backend_name)
        model = get_llm_model(backend_name)

        spin_label = "聞いたまま再現中" if is_verbatim else "議事録を整形中"
        with st.spinner(f"{spin_label}...（{backend_name}）"):
            try:
                if is_verbatim:
                    minutes = reproduce_verbatim(client, model, diarized,
                                                 meeting_title, meeting_date,
                                                 translate_ja=verbatim_translate_ja)
                else:
                    minutes = summarize(client, model, diarized, meeting_title,
                                        meeting_date, attendees, granularity)
            except Exception as e:
                st.error(f"生成に失敗しました: {e}")
                st.stop()

        # ToDo抽出は議事録モードのみ
        if is_verbatim:
            todos = []
        else:
            with st.spinner("ToDoを抽出中..."):
                todos = extract_todos(client, model, diarized)

        ss.minutes_edited = minutes
        ss["todos"] = todos
        ss["sentiment"] = None  # 温度感は別ボタンで生成

    # 議事録表示＆編集
    if ss.get("minutes_edited") is not None:
        out_heading = "📝 逐語再現（聞いたまま）" if is_verbatim else "📝 議事録"
        st.subheader(out_heading)

        tab_preview, tab_edit = st.tabs(["プレビュー", "✏️ 編集"])
        with tab_preview:
            st.markdown(ss.minutes_edited)
        with tab_edit:
            edited = st.text_area(
                "内容を直接編集できます",
                value=ss.minutes_edited,
                height=500,
                key="minutes_editor",
            )
            if edited != ss.minutes_edited:
                ss.minutes_edited = edited
                st.rerun()

        todos = ss.get("todos", [])
        if todos:
            st.subheader("✅ ToDo")
            st.dataframe(todos, use_container_width=True)
        elif not is_verbatim:
            st.caption("抽出されたToDoはありませんでした。")

        # 感情・温度感レポート
        st.divider()
        st.subheader("🌡️ 会議の温度感レポート")
        st.caption("議論の熱量・対立・合意・発言バランスを分析します。")
        if st.button("温度感レポートを生成", key="gen_sentiment"):
            client = get_llm_client(backend_name)
            model = get_llm_model(backend_name)
            with st.spinner(f"会議の温度感を分析中...（{backend_name}）"):
                try:
                    ss["sentiment"] = analyze_sentiment(client, model, diarized)
                except Exception as e:
                    st.error(f"分析失敗: {e}")

        if ss.get("sentiment"):
            st.markdown(ss["sentiment"])

        base = os.path.splitext(ss.audio_name)[0]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            dl_label = "逐語 .md" if is_verbatim else "議事録 .md"
            dl_name = f"{base}_逐語.md" if is_verbatim else f"{base}_議事録.md"
            st.download_button(dl_label, ss.minutes_edited.encode("utf-8"),
                               file_name=dl_name, mime="text/markdown")
        with c2:
            st.download_button("話者分離 .txt", diarized.encode("utf-8"),
                               file_name=f"{base}_話者分離.txt", mime="text/plain")
        with c3:
            st.download_button("ToDo .csv", todos_to_csv(todos),
                               file_name=f"{base}_ToDo.csv", mime="text/csv",
                               disabled=not todos)
        with c4:
            st.download_button("温度感 .md",
                               (ss.get("sentiment") or "").encode("utf-8"),
                               file_name=f"{base}_温度感レポート.md",
                               mime="text/markdown",
                               disabled=not ss.get("sentiment"))
