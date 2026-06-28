import streamlit as st
import requests
import os
import json
import tempfile
from pyvis.network import Network
import streamlit.components.v1 as components
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ── .env 로드 ────────────────────────────────────────────────────
load_dotenv(Path(".env"))

# ── API 설정 ────────────────────────────────────────────────────
_OAI_URL   = "https://api.openai.com/v1/chat/completions"
_OAI_MODEL = "gpt-4o-mini"

_ANT_URL   = "https://api.anthropic.com/v1/messages"
_ANT_MODEL = "claude-sonnet-4-6"


def _detect_api():
    """사용 가능한 API 자동 감지. 우선순위: Anthropic → OpenAI"""
    k = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if k:
        return "anthropic", k
    k = os.environ.get("OPENAI_API_KEY", "").strip()
    if k:
        return "openai", k
    return None, ""


_api_mode, _api_key = _detect_api()

if not _api_key:
    st.error(
        "API 키가 설정되지 않았습니다.\n\n"
        "**Anthropic Claude:** .env 파일에 `ANTHROPIC_API_KEY=<키>` 추가\n\n"
        "**OpenAI:** .env 파일에 `OPENAI_API_KEY=<키>` 추가\n\n"
        "`.env.example` 파일을 참고해 `.env` 파일을 만들어주세요."
    )
    st.stop()

# ── 지식 저장 경로 ───────────────────────────────────────────────
KB_PATH = Path("./knowledge_db.json")


def _load_kb() -> list:
    if KB_PATH.exists():
        return json.loads(KB_PATH.read_text(encoding="utf-8"))
    return []


def _save_kb(kb: list) -> None:
    KB_PATH.write_text(json.dumps(kb, ensure_ascii=False, indent=2), encoding="utf-8")


# ── LLM 호출 (Anthropic / OpenAI 자동 전환) ─────────────────────
def chat(system: str, user: str) -> str:
    if _api_mode == "anthropic":
        headers = {
            "Content-Type": "application/json",
            "x-api-key": _api_key,
            "anthropic-version": "2023-06-01",
        }
        body = {
            "model": _ANT_MODEL,
            "max_tokens": 4096,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        r = requests.post(_ANT_URL, headers=headers, json=body, timeout=180)
        r.raise_for_status()
        return r.json()["content"][0]["text"]

    else:  # openai
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_api_key}",
        }
        body = {
            "model": _OAI_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "max_tokens": 4096,
        }
        r = requests.post(_OAI_URL, headers=headers, json=body, timeout=180)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


# ── TF-IDF 유사도 검색 (로컬, 외부 연결 없음) ───────────────────
def search_kb(kb: list, query: str, top_k: int = 3) -> list:
    if not kb:
        return []
    docs = [item["document"] for item in kb]
    corpus = docs + [query]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    tfidf_matrix = vectorizer.fit_transform(corpus)
    scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [{**kb[idx], "score": float(scores[idx])}
            for idx in top_indices if scores[idx] > 0.05]


# ── UI ────────────────────────────────────────────────────────────
st.set_page_config(page_title="나의 성장 파트너", page_icon="🧠", layout="wide")
st.title("🧠 나의 성장 파트너")

_mode_label = {"anthropic": "Anthropic Claude", "openai": "OpenAI (GPT-4o mini)"}.get(_api_mode, "Unknown")
st.caption(f"업무 지식을 저장하고 · AI로 검색하고 · 성과 언어로 회고하세요 &nbsp;|&nbsp; API: {_mode_label}")

tab1, tab2, tab3, tab4 = st.tabs(["📝 지식 저장", "🔍 AI 검색", "📊 업무 회고", "🕸 지식 그래프"])


# ── Tab 1: 지식 저장 ──────────────────────────────────────────────
with tab1:
    st.subheader("지식 / 메모 저장")
    st.caption("업무 중 배운 것, 해결한 문제, 아이디어를 기록하세요.")

    category = st.selectbox(
        "카테고리",
        ["업무 노하우", "문제 해결", "배운 것", "아이디어", "기타"]
    )
    title   = st.text_input("제목", placeholder="예) 엑셀 피벗 #REF 오류 해결 방법")
    content = st.text_area("내용", height=200,
                           placeholder="해결 방법, 과정, 핵심 포인트 등을 자유롭게 적어주세요.")

    if st.button("저장", type="primary", key="save_btn"):
        if not title or not content:
            st.warning("제목과 내용을 모두 입력해주세요.")
        else:
            kb = _load_kb()
            kb.append({
                "id":       datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
                "category": category,
                "title":    title,
                "date":     datetime.now().strftime("%Y-%m-%d"),
                "document": f"{title}\n{content}",
            })
            _save_kb(kb)
            st.success(f"✅ '{title}' 저장 완료! (총 {len(kb)}개 저장됨)")

    kb_now = _load_kb()
    if kb_now:
        st.divider()
        st.markdown(f"**저장된 지식: {len(kb_now)}개**")
        if st.button("목록 보기", key="list_btn"):
            for item in kb_now:
                st.markdown(f"- `{item['date']}` **{item['title']}** ({item['category']})")


# ── Tab 2: AI 검색 ────────────────────────────────────────────────
with tab2:
    st.subheader("AI로 내 지식 검색")
    st.caption("저장한 지식을 자연어로 검색하세요. 단어가 달라도 관련 내용을 찾아냅니다.")

    query = st.text_input("무엇이 궁금하신가요?",
                          placeholder="예) 저번에 해결했던 엑셀 오류 어떻게 했더라?")

    if st.button("검색", type="primary", key="search_btn"):
        if not query:
            st.warning("검색어를 입력해주세요.")
        else:
            kb = _load_kb()
            if not kb:
                st.warning("저장된 지식이 없습니다. 먼저 '지식 저장' 탭에서 내용을 저장해주세요.")
            else:
                with st.spinner("검색 중..."):
                    hits = search_kb(kb, query, top_k=3)

                if not hits:
                    st.info("관련된 지식을 찾지 못했습니다. 다른 표현으로 검색해보세요.")
                else:
                    context = "\n\n---\n\n".join([h["document"] for h in hits])
                    try:
                        answer = chat(
                            system=(
                                "당신은 사용자의 개인 지식 베이스 AI입니다. "
                                "아래 [참고 자료]를 바탕으로 질문에 답변하세요. "
                                "참고 자료에 없는 내용은 '저장된 지식에서 찾을 수 없습니다'라고 솔직하게 말하세요. "
                                "답변은 한국어로 작성하세요."
                            ),
                            user=f"[참고 자료]\n{context}\n\n[질문]\n{query}"
                        )
                    except Exception as e:
                        st.error(f"AI 응답 오류: {e}")
                        st.stop()

                    st.markdown("### 💡 AI 답변")
                    st.markdown(answer)

                    with st.expander("📄 참고된 지식 원문 보기"):
                        for h in hits:
                            similarity = round(h["score"] * 100, 1)
                            st.markdown(
                                f"**{h['title']}** | {h['category']} | {h['date']} | 관련도 {similarity}%"
                            )
                            doc_preview = h["document"]
                            st.text(doc_preview[:300] + ("..." if len(doc_preview) > 300 else ""))
                            st.divider()


# ── Tab 3: 업무 기록 & 성과 보고서 ────────────────────────────────
with tab3:
    st.subheader("📋 업무 기록 & 성과 보고서")

    rec_tab1, rec_tab2, rec_tab3 = st.tabs(["✏️ 오늘 기록", "📅 기록 목록", "🏆 성과 보고서 생성"])

    # ── 오늘 기록 ──────────────────────────────────────────────────
    with rec_tab1:
        st.caption("매일 퇴근 전 3분, 오늘 한 일을 편하게 적어두세요. 누적된 기록이 나중에 성과 보고서가 됩니다.")

        log_date = st.date_input("날짜", value=datetime.now().date())
        log_text = st.text_area(
            "오늘 한 일",
            height=200,
            placeholder=(
                "예) CMP 파티클 불량 원인 분석. KLA 데이터 맵핑해보니 엣지 집중 패턴.\n"
                "슬러리 필터 교체 이후 발생해서 필터 재질 문제로 추정, 구매팀에 벤더 변경 검토 요청.\n"
                "오후엔 수율 데이터 ANOVA 분석, 장비 3대 편차 유의미 (p=0.03) 확인.\n"
                "PM 우선순위 조정 건의. 신입 결함 분류 기준 교육."
            )
        )

        if st.button("기록 저장", type="primary", key="log_save_btn"):
            if not log_text.strip():
                st.warning("내용을 입력해주세요.")
            else:
                kb = _load_kb()
                date_str = log_date.strftime("%Y-%m-%d")
                kb = [item for item in kb if not (
                    item["category"] == "일일기록" and item["date"] == date_str
                )]
                kb.append({
                    "id":       f"log_{log_date.strftime('%Y%m%d')}",
                    "category": "일일기록",
                    "title":    f"일일기록_{date_str}",
                    "date":     date_str,
                    "document": f"[{date_str}] 업무 기록\n{log_text.strip()}",
                })
                _save_kb(kb)
                st.success(f"✅ {date_str} 기록 저장 완료!")

    # ── 기록 목록 ──────────────────────────────────────────────────
    with rec_tab2:
        kb_logs = [item for item in _load_kb() if item["category"] == "일일기록"]
        kb_logs.sort(key=lambda x: x["date"], reverse=True)

        if not kb_logs:
            st.info("아직 기록이 없습니다. '오늘 기록' 탭에서 입력해주세요.")
        else:
            st.markdown(f"**총 {len(kb_logs)}일 기록됨**")
            for item in kb_logs:
                with st.expander(f"📅 {item['date']}"):
                    lines = item["document"].split("\n")[1:]
                    st.markdown("\n".join(lines))
                    if st.button("삭제", key=f"del_{item['id']}"):
                        kb_all = _load_kb()
                        kb_all = [i for i in kb_all if i["id"] != item["id"]]
                        _save_kb(kb_all)
                        st.rerun()

    # ── 성과 보고서 생성 ───────────────────────────────────────────
    with rec_tab3:
        st.caption("누적된 일일 기록을 AI가 읽고 기간별 성과 보고서를 자동으로 만들어줍니다.")

        kb_logs_all = [item for item in _load_kb() if item["category"] == "일일기록"]
        kb_logs_all.sort(key=lambda x: x["date"])

        if not kb_logs_all:
            st.info("기록이 없습니다. 먼저 일일 업무를 기록해주세요.")
        else:
            date_min = datetime.strptime(kb_logs_all[0]["date"],  "%Y-%m-%d").date()
            date_max = datetime.strptime(kb_logs_all[-1]["date"], "%Y-%m-%d").date()

            col_a, col_b = st.columns(2)
            with col_a:
                start_date = st.date_input("시작일", value=date_min, min_value=date_min, max_value=date_max)
            with col_b:
                end_date = st.date_input("종료일", value=date_max, min_value=date_min, max_value=date_max)

            report_type = st.selectbox(
                "보고서 유형",
                ["월간 성과 요약", "분기 성과 요약", "연간 성과 요약 / 자기평가", "자유 형식"]
            )

            in_range = [
                item for item in kb_logs_all
                if start_date <= datetime.strptime(item["date"], "%Y-%m-%d").date() <= end_date
            ]
            st.caption(f"선택 기간 내 기록: {len(in_range)}일")

            if st.button("📊 성과 보고서 생성", type="primary", key="report_btn"):
                if not in_range:
                    st.warning("선택한 기간에 기록이 없습니다.")
                else:
                    all_logs = "\n\n".join([item["document"] for item in in_range])
                    type_prompt = {
                        "월간 성과 요약":           "월간 성과 보고서",
                        "분기 성과 요약":           "분기 성과 보고서",
                        "연간 성과 요약 / 자기평가": "연간 자기평가 보고서 (인사고과용)",
                        "자유 형식":               "성과 요약 보고서",
                    }[report_type]

                    with st.spinner(f"AI가 {len(in_range)}일치 기록을 분석 중..."):
                        try:
                            report = chat(
                                system=(
                                    f"당신은 직장인의 업무 기록을 분석해 {type_prompt}를 작성하는 커리어 코치 AI입니다.\n"
                                    "아래 일별 업무 기록들을 읽고 다음 형식으로 보고서를 작성하세요.\n\n"
                                    "## 📌 핵심 성과 요약 (3줄)\n"
                                    "이 기간의 가장 중요한 성과 3가지를 임팩트 있게\n\n"
                                    "## 🏆 주요 성과 목록\n"
                                    "- 해결한 문제, 완료한 프로젝트, 기여한 사항을 성과 언어로 (수치 포함)\n\n"
                                    "## 🛠 활용 및 성장한 역량\n"
                                    "이 기간에 반복적으로 사용하거나 새롭게 개발한 스킬\n\n"
                                    "## 📈 정량적 성과 (추정 포함)\n"
                                    "수치로 나타낼 수 있는 성과 (불량률 개선, 시간 절감 등)\n\n"
                                    "## 💡 인사이트 & 성장 포인트\n"
                                    "이 기간 동안 배우고 성장한 점\n\n"
                                    "전문적이고 설득력 있는 한국어로 작성하세요."
                                ),
                                user=(
                                    f"[분석 기간: {start_date} ~ {end_date}, 총 {len(in_range)}일]\n\n"
                                    f"{all_logs}"
                                )
                            )
                        except Exception as e:
                            st.error(f"AI 응답 오류: {e}")
                            st.stop()

                    st.markdown(f"### 📊 {report_type} ({start_date} ~ {end_date})")
                    st.markdown(report)

                    st.download_button(
                        label="📥 보고서 다운로드 (.txt)",
                        data=(
                            f"[{report_type}] {start_date} ~ {end_date}\n"
                            f"기록 일수: {len(in_range)}일\n\n"
                            f"{report}"
                        ),
                        file_name=f"성과보고서_{start_date}_{end_date}.txt",
                        mime="text/plain"
                    )


# ── Tab 4: 지식 그래프 ──────────────────────────────────────────────
with tab4:
    st.subheader("🕸 지식 그래프")
    st.caption("비슷한 내용의 지식끼리 자동으로 연결됩니다. 그래프로 관계를 탐색하고, 아래 목록에서 내용을 확인하세요.")

    kb_graph = _load_kb()

    if not kb_graph:
        st.warning("저장된 지식이 없습니다. 먼저 지식을 저장해주세요.")
    else:
        COLOR_MAP = {
            "업무 노하우": "#4A90D9",
            "문제 해결":   "#E74C3C",
            "배운 것":     "#2ECC71",
            "아이디어":    "#F39C12",
            "업무 회고":   "#9B59B6",
            "기타":        "#95A5A6",
        }

        docs = [item["document"] for item in kb_graph]
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        tfidf_matrix = vectorizer.fit_transform(docs)

        ctrl_cols = st.columns([2, 2, 2, 2])
        with ctrl_cols[0]:
            threshold = st.slider("연결 강도", 0.05, 0.5, 0.12, 0.01,
                                  help="낮을수록 더 많은 연결 표시")
        with ctrl_cols[1]:
            cat_filter = st.multiselect(
                "카테고리 필터",
                options=list(COLOR_MAP.keys()),
                default=list(COLOR_MAP.keys()),
            )

        filtered_kb = [item for item in kb_graph if item["category"] in cat_filter]
        if not filtered_kb:
            st.info("선택한 카테고리에 해당하는 지식이 없습니다.")
            st.stop()

        f_docs = [item["document"] for item in filtered_kb]
        f_vec  = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(f_docs)
        f_sim  = cosine_similarity(f_vec)

        net = Network(height="560px", width="100%",
                      bgcolor="#0E1117", font_color="#FFFFFF", directed=False)
        net.set_options("""{
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -80,
              "centralGravity": 0.01,
              "springLength": 130,
              "springConstant": 0.08
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 150}
          },
          "nodes": {"borderWidth": 2, "shadow": true},
          "edges": {"smooth": {"type": "continuous"}, "shadow": true},
          "interaction": {"hover": true, "tooltipDelay": 80}
        }""")

        for i, item in enumerate(filtered_kb):
            color   = COLOR_MAP.get(item["category"], "#95A5A6")
            label   = item["title"][:16] + ("…" if len(item["title"]) > 16 else "")
            preview = item["document"][:180].replace("\n", " ")
            tooltip = (
                f"<div style='max-width:280px;font-family:sans-serif'>"
                f"<b style='font-size:14px'>{item['title']}</b><br>"
                f"<span style='color:#aaa;font-size:12px'>"
                f"[{item['category']}] {item['date']}</span><br><br>"
                f"<span style='font-size:12px'>{preview}…</span></div>"
            )
            net.add_node(i, label=label, title=tooltip,
                         color=color, size=22,
                         font={"size": 13, "color": "#FFFFFF"})

        for i in range(len(filtered_kb)):
            for j in range(i + 1, len(filtered_kb)):
                score = f_sim[i][j]
                if score >= threshold:
                    net.add_edge(i, j,
                                 width=min(1 + score * 8, 6),
                                 color={"color": "#445566", "highlight": "#88AAFF"},
                                 title=f"유사도 {score:.2f}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".html",
                                         mode="w", encoding="utf-8") as f:
            net.save_graph(f.name)
            html_path = f.name
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        components.html(html_content, height=570, scrolling=False)

        legend_cols = st.columns(len(COLOR_MAP))
        for col, (cat, color) in zip(legend_cols, COLOR_MAP.items()):
            col.markdown(
                f'<span style="background:{color};padding:3px 9px;border-radius:10px;'
                f'color:white;font-size:12px">{cat}</span>',
                unsafe_allow_html=True)
        st.caption(f"총 {len(filtered_kb)}개 노드 표시 중 · 연결 임계값 {threshold:.2f} "
                   f"· 마우스 올리면 미리보기, 드래그/스크롤로 탐색")

        st.divider()
        st.markdown("### 📖 지식 내용 보기")
        st.caption("항목을 선택하면 전체 내용과 연결된 지식을 확인할 수 있습니다.")

        titles   = [f"[{item['category']}] {item['title']}" for item in filtered_kb]
        selected = st.selectbox("항목 선택", titles, label_visibility="collapsed")
        sel_idx  = titles.index(selected)
        sel_item = filtered_kb[sel_idx]

        color = COLOR_MAP.get(sel_item["category"], "#95A5A6")
        st.markdown(
            f'<div style="border-left:4px solid {color};padding:12px 16px;'
            f'background:#1a1a2e;border-radius:6px;margin-bottom:8px">'
            f'<span style="background:{color};padding:2px 8px;border-radius:8px;'
            f'color:white;font-size:12px">{sel_item["category"]}</span>'
            f'&nbsp;&nbsp;<span style="color:#aaa;font-size:12px">{sel_item["date"]}</span><br><br>'
            f'<b style="font-size:16px">{sel_item["title"]}</b>'
            f'</div>',
            unsafe_allow_html=True
        )
        content_lines = sel_item["document"].split("\n")
        for line in content_lines:
            if line.strip():
                st.markdown(line)

        scores = [(j, f_sim[sel_idx][j]) for j in range(len(filtered_kb)) if j != sel_idx]
        scores.sort(key=lambda x: x[1], reverse=True)
        top_related = [(filtered_kb[j], s) for j, s in scores[:3] if s >= threshold]

        if top_related:
            st.markdown("**🔗 연결된 지식**")
            for rel_item, rel_score in top_related:
                rel_color = COLOR_MAP.get(rel_item["category"], "#95A5A6")
                with st.expander(f"{rel_item['title']} — 유사도 {rel_score:.0%}"):
                    st.markdown(
                        f'<span style="background:{rel_color};padding:2px 8px;'
                        f'border-radius:8px;color:white;font-size:11px">'
                        f'{rel_item["category"]}</span>',
                        unsafe_allow_html=True
                    )
                    for line in rel_item["document"].split("\n"):
                        if line.strip():
                            st.markdown(line)
