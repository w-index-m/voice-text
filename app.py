"""
音声議事録アプリ (Streamlit) - 話者分離 + 出席者リネーム + 粒度切替 + ToDo CSV
- 音声アップロード -> AssemblyAI で文字起こし＋話者分離
- 検出話者を実名にリネーム
- 要約の粒度（3行 / 詳細 / 決定事項のみ）を選択
- GPT で議事録生成 + ToDo を JSON で抽出して CSV ダウンロード

セットアップ:
  pip install streamlit openai requests
  .streamlit/secrets.toml:
    OPENAI_API_KEY = "sk-..."
    ASSEMBLYAI_API_KEY = "..."
起動:
  streamlit run app.py
"""

import io
import csv
import json
import os
import time

import requests
import streamlit as st
from openai import OpenAI

AAI_BASE = "https://api.assemblyai.com/v2"


# ----- キー / クライアント -----
def get_openai() -> OpenAI:
    key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        st.error("OPENAI_API_KEY が未設定です。secrets.toml を確認してください。")
        st.stop()
    return OpenAI(api_key=key)


def get_aai_key() -> str:
    key = st.secrets.get("ASSEMBLYAI_API_KEY") or os.environ.get("ASSEMBLYAI_API_KEY")
    if not key:
        st.error("ASSEMBLYAI_API_KEY が未設定です。secrets.toml を確認してください。")
        st.stop()
    return key


# ----- AssemblyAI -----
def aai_upload(api_key: str, file_bytes: bytes) -> str:
    headers = {"authorization": api_key}
    resp = requests.post(f"{AAI_BASE}/upload", headers=headers, data=file_bytes)
    resp.raise_for_status()
    return resp.json()["upload_url"]


def aai_transcribe(api_key: str, audio_url: str, language: str) -> dict:
    headers = {"authorization": api_key}
    payload = {"audio_url": audio_url, "speaker_labels": True}
    if language:
        payload["language_code"] = language
    resp = requests.post(f"{AAI_BASE}/transcript", json=payload, headers=headers)
    resp.raise_for_status()
    tid = resp.json()["id"]

    progress = st.progress(0.0, text="文字起こし中（話者分離あり）...")
    poll = 0
    while True:
        poll += 1
        r = requests.get(f"{AAI_BASE}/transcript/{tid}", headers=headers)
        r.raise_for_status()
        data = r.json()
        status = data["status"]
        if status == "completed":
            progress.empty()
            return data
        if status == "error":
            progress.empty()
            raise RuntimeError(f"AssemblyAI エラー: {data.get('error')}")
        progress.progress(min(0.9, poll * 0.03), text=f"文字起こし中... ({status})")
        time.sleep(3)


def build_diarized_text(data: dict, name_map: dict | None = None) -> str:
    """utterances を「話者名: 発言」形式に。name_map で実名へ置換可能。"""
    utterances = data.get("utterances")
    if not utterances:
        return data.get("text", "")
    name_map = name_map or {}
    lines = []
    for u in utterances:
        label = u["speaker"]  # 'A' / 'B' ...
        display = name_map.get(label) or f"話者{label}"
        lines.append(f"{display}: {u['text']}")
    return "\n".join(lines)


# ----- GPT: 議事録 -----
GRANULARITY_SPEC = {
    "3行サマリ": "要点を3行程度で簡潔にまとめる。詳細は省く。",
    "詳細議事録": "発言サマリ・議論の要点・決定事項・ToDoまで網羅した詳細な議事録。",
    "決定事項のみ": "決定事項と次アクションだけを抽出。雑談や経緯は省く。",
}


def summarize(client: OpenAI, diarized_text: str, meeting_title: str,
              meeting_date: str, attendees: str, granularity: str) -> str:
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
        model="gpt-4o",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.3,
    )
    return resp.choices[0].message.content


def extract_todos(client: OpenAI, diarized_text: str) -> list[dict]:
    """ToDoを構造化JSONで抽出。"""
    system = "会議の文字起こしからToDo（アクションアイテム）を抽出するアシスタント。"
    user = f"""以下の文字起こしから ToDo を抽出し、JSON配列のみを返してください。
前後の説明やMarkdownのコードフェンスは付けないでください。
各要素は以下のキー: task(タスク内容), owner(担当者/不明なら空文字), due(期限/不明なら空文字)

ToDoが無ければ [] を返してください。

---
{diarized_text}
"""
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.0,
    )
    text = resp.choices[0].message.content.strip()
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
    # Excel(Windows)での文字化け回避に BOM 付き UTF-8
    return ("\ufeff" + buf.getvalue()).encode("utf-8")


# ----- UI -----
st.set_page_config(page_title="音声議事録", page_icon="🎙️", layout="wide")
st.title("🎙️ 音声議事録メーカー")
st.caption("話者分離 / 出席者リネーム / 粒度切替 / ToDo CSV出力")

with st.sidebar:
    st.header("会議情報")
    meeting_title = st.text_input("会議名（任意）")
    meeting_date = st.text_input("日時（任意・例: 2026-06-26 14:00）")
    attendees = st.text_area("出席者（任意・改行 or カンマ区切り）",
                             help="事前に名前を入れておくと話者の実名推定に使われます。")
    language = st.selectbox("音声の言語",
                            options=[("日本語", "ja"), ("英語", "en"), ("自動判定", "")],
                            format_func=lambda x: x[0])[1]
    granularity = st.radio("要約の粒度", list(GRANULARITY_SPEC.keys()), index=1)
    show_transcript = st.checkbox("話者分離テキスト全文も表示", value=True)

# セッション状態の初期化
ss = st.session_state
ss.setdefault("aai_data", None)       # AssemblyAI の結果
ss.setdefault("speakers", [])         # 検出話者ラベル ['A','B',...]
ss.setdefault("audio_name", "")

uploaded = st.file_uploader(
    "音声ファイルをアップロード",
    type=["m4a", "mp3", "wav", "mp4", "mpeg", "mpga", "webm", "ogg", "flac"],
    help="アップロード・分割・話者分離は AssemblyAI 側で処理します。",
)

# ステップ1: 文字起こし
if uploaded is not None:
    file_bytes = uploaded.getvalue()
    st.info(f"ファイル: {uploaded.name} / {len(file_bytes)/1024/1024:.1f} MB")
    st.audio(file_bytes)

    if st.button("① 文字起こし（話者分離）", type="primary"):
        aai_key = get_aai_key()
        with st.spinner("音声をアップロード中..."):
            try:
                audio_url = aai_upload(aai_key, file_bytes)
            except Exception as e:
                st.error(f"アップロード失敗: {e}")
                st.stop()
        try:
            data = aai_transcribe(aai_key, audio_url, language)
        except Exception as e:
            st.error(f"文字起こし失敗: {e}")
            st.stop()
        ss.aai_data = data
        ss.speakers = sorted({u["speaker"] for u in data.get("utterances", [])})
        ss.audio_name = uploaded.name
        st.success(f"完了。検出話者: {len(ss.speakers)}名")

# ステップ2: 話者リネーム & 議事録
if ss.aai_data is not None:
    st.divider()
    st.subheader("② 話者の名前を割り当て（任意）")

    # 出席者の候補をサジェストとして表示
    if attendees.strip():
        cand = [a.strip() for a in attendees.replace(",", "\n").splitlines() if a.strip()]
        if cand:
            st.caption("出席者候補: " + " / ".join(cand))

    name_map = {}
    cols = st.columns(min(4, max(1, len(ss.speakers))))
    for i, sp in enumerate(ss.speakers):
        with cols[i % len(cols)]:
            name_map[sp] = st.text_input(f"話者{sp} の名前", key=f"name_{sp}",
                                         placeholder=f"話者{sp}")

    diarized = build_diarized_text(ss.aai_data, name_map)

    if show_transcript:
        with st.expander("話者分離テキスト全文", expanded=False):
            st.text_area("transcript", diarized, height=300)

    if st.button("③ 議事録を作成", type="primary"):
        client = get_openai()
        with st.spinner("議事録を整形中..."):
            try:
                minutes = summarize(client, diarized, meeting_title,
                                    meeting_date, attendees, granularity)
            except Exception as e:
                st.error(f"議事録整形失敗: {e}")
                st.stop()
        with st.spinner("ToDoを抽出中..."):
            todos = extract_todos(client, diarized)

        st.subheader("📝 議事録")
        st.markdown(minutes)

        if todos:
            st.subheader("✅ ToDo")
            st.dataframe(todos, use_container_width=True)
        else:
            st.caption("抽出されたToDoはありませんでした。")

        base = os.path.splitext(ss.audio_name)[0]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("議事録 .md", minutes.encode("utf-8"),
                               file_name=f"{base}_議事録.md", mime="text/markdown")
        with c2:
            st.download_button("話者分離 .txt", diarized.encode("utf-8"),
                               file_name=f"{base}_話者分離.txt", mime="text/plain")
        with c3:
            st.download_button("ToDo .csv", todos_to_csv(todos),
                               file_name=f"{base}_ToDo.csv", mime="text/csv",
                               disabled=not todos)
