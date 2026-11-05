import streamlit as st
import requests
import re
import html
import altair as alt
import pandas as pd
from datetime import date

st.set_page_config(page_title="우리학교 다이어트 지키미", page_icon="🍱")

# API 키 (.streamlit/secrets.toml)
NEIS_API_KEY = st.secrets["NEIS_API_KEY"]
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

# 함수 정의
@st.cache_data
def find_school(office_code, school_name):
    url = "https://open.neis.go.kr/hub/schoolInfo"
    params = {
        "KEY": NEIS_API_KEY, "Type": "json",
        "pIndex": 1, "pSize": 5,
        "ATPT_OFCDC_SC_CODE": office_code,
        "SCHUL_NM": school_name
    }
    res = requests.get(url, params=params, timeout=10).json()
    if "schoolInfo" in res:
        return res["schoolInfo"][1]["row"][0]["SD_SCHUL_CODE"]
    return None

@st.cache_data(ttl=3600)
def get_meals_range(office_code, school_code, start, end):
    url = "https://open.neis.go.kr/hub/mealServiceDietInfo"
    rows = []
    page = 1
    while True:
        params = {
            "KEY": NEIS_API_KEY, "Type": "json",
            "pIndex": page, "pSize": 1000,
            "ATPT_OFCDC_SC_CODE": office_code,
            "SD_SCHUL_CODE": school_code,
            "MMEAL_SC_CODE": "2",  # 중식만
            "MLSV_FROM_YMD": start,
            "MLSV_TO_YMD": end
        }
        res = requests.get(url, params=params, timeout=10).json()
        if "mealServiceDietInfo" not in res:
            break
        total = res["mealServiceDietInfo"][0]["head"][0]["list_total_count"]
        rows += res["mealServiceDietInfo"][1]["row"]
        if len(rows) >= total:
            break
        page += 1
    return rows

def parse_cal(cal_info):
    m = re.search(r"[\d.]+", cal_info or "")
    return float(m.group()) if m else None

DAILY_KCAL = 2500  # 고등학생 하루 권장 열량(가정값)

def parse_nutrients(ntr):
    """'탄수화물(g) : 105.8<br/>...' 문자열을 {이름: 숫자} 딕셔너리로 변환"""
    result = {}
    if not isinstance(ntr, str):
        return result
    for line in ntr.split("<br/>"):
        m = re.match(r"\s*(.+?)\(.*?\)\s*:\s*([\d.]+)", line)
        if m:
            result[m.group(1)] = float(m.group(2))
    return result

@st.dialog("🧾 영양정보")
def show_nutrition(nutrients):
    st.dataframe(pd.DataFrame({"함량": nutrients}), use_container_width=True)

# 인기쟁이 메뉴에서 제외할 음식 판별
KIMCHI_WORDS = ["김치", "깍두기", "겉절이", "석박지"]
KIMCHI_MAIN = ["찌개", "볶음", "전", "국", "죽", "만두", "찜", "밥"]  # 반찬이 아닌 요리
RICE_MAIN = ["볶음", "비빔", "덮", "김", "초", "주먹", "마요", "카레", "오므", "유부", "쌈", "죽"]  # 요리가 된 밥

def is_kimchi_or_rice(dish):
    if any(w in dish for w in KIMCHI_WORDS):
        return not any(m in dish for m in KIMCHI_MAIN)
    if dish.endswith("밥"):
        return not any(m in dish for m in RICE_MAIN)
    return False

PASTEL = ["#FFB5C2", "#FFD6A5", "#FDEBA0", "#B9E8C4", "#A8DDF0", "#B8C6F5", "#D4BDF5"]

def pastel_bar(data, unit=""):
    """{이름: 값} 딕셔너리를 파스텔 색 가로 막대그래프로 표시"""
    d = pd.DataFrame({"이름": list(data.keys()), "값": list(data.values())})
    d["색"] = [PASTEL[i % len(PASTEL)] for i in range(len(d))]
    bars = alt.Chart(d, height=alt.Step(34)).mark_bar(
        cornerRadiusTopRight=8, cornerRadiusBottomRight=8
    ).encode(
        x=alt.X("값:Q", title=unit, axis=alt.Axis(grid=False)),
        y=alt.Y("이름:N", sort=None, title=None, axis=alt.Axis(labelFontSize=13)),
        color=alt.Color("색:N", scale=None, legend=None),
        tooltip=["이름", "값"],
    )
    labels = bars.mark_text(align="left", dx=5, fontSize=13).encode(
        text=alt.Text("값:Q", format=".0f"), color=alt.value("gray"),
    )
    st.altair_chart((bars + labels).configure_view(strokeWidth=0), use_container_width=True)

# ---------- 개인트레이너 챗봇 (Gemini) ----------
# (모델, 생각 수준) - 앞에서부터 시도. 생각 수준을 낮추면 응답이 훨씬 빨라진다.
GEMINI_MODELS = [
    ("gemini-3.1-flash-lite", "minimal"),
    ("gemini-3.6-flash", "minimal"),
    ("gemini-3.5-flash", "minimal"),
]

def ask_trainer(system_prompt, history):
    """Gemini REST API 호출. 모델이 혼잡하면 다음 모델로 넘어간다."""
    if not GEMINI_API_KEY:
        return "GEMINI_API_KEY가 설정되지 않았어요. .streamlit/secrets.toml을 확인해 주세요."
    body = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [
            {"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
            for m in history
        ],
    }
    last_error = ""
    for model, level in GEMINI_MODELS:
        body["generationConfig"] = {"thinkingConfig": {"thinkingLevel": level}}
        try:
            res = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
                json=body,
                timeout=20,
            )
            data = res.json()
            if "candidates" in data:
                parts = data["candidates"][0]["content"]["parts"]
                return "".join(p.get("text", "") for p in parts)
            last_error = data.get("error", {}).get("message", str(data))
        except Exception as e:
            last_error = str(e)
    return f"트레이너가 지금 자리를 비웠어요 😅 잠시 후 다시 말을 걸어 주세요. ({last_error})"

def format_reply(text):
    """마크다운에서 한 줄 바꿈이 무시되지 않게, 문장마다 줄을 나눠 보여준다."""
    text = text.strip()
    if "\n" not in text:  # 줄바꿈이 전혀 없으면 문장 끝(. ! ?)에서 나눈다
        text = re.sub(r"(?<=[.!?])\s+", "\n", text)
    return re.sub(r"\n+", "\n\n", text)

def stat_card(container, title, value, unit="", note=None):
    """제목·수치·단위·보조설명의 글씨 크기를 통일한 카드"""
    with container.container(border=True):
        st.markdown(
            f'<div style="font-size:0.9rem;opacity:0.7;">{title}</div>'
            f'<div style="font-size:1.8rem;font-weight:700;line-height:1.3;">{value}'
            f'<span style="font-size:0.9rem;font-weight:400;opacity:0.7;"> {unit}</span></div>',
            unsafe_allow_html=True,
        )
        if note:
            st.caption(note)

def clean_dish_name(dish):
    dish = re.sub(r"[\(\[].*?[\)\]]", "", dish)
    return dish.strip("*! ")

def clean_menu(raw):
    return [clean_dish_name(d) for d in raw.split("<br/>") if d.strip()]

###############################

# 기간 설정 (3월 2일 ~ 오늘)
today = date.today()
year = today.year if today >= date(today.year, 3, 2) else today.year - 1
start = f"{year}0302"
end = today.strftime("%Y%m%d")
WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]

# 사이드바 메뉴 (처음 접속할 때부터 항상 표시)
pages = ["📅 오늘의 급식", "🏆 칼로리 TOP", "🍚 우리학교 인기쟁이 메뉴", "💪 개인트레이너 잔소리"]
if "page" not in st.session_state:
    st.session_state["page"] = pages[0]
st.sidebar.markdown(
    '<div style="font-size:1.4rem;font-weight:800;line-height:1.3;margin-bottom:0.8rem;'
    'white-space:nowrap;">🍱 다이어트 지키미</div>',
    unsafe_allow_html=True,
)
for p in pages:
    if st.sidebar.button(
        p,
        key=f"nav_{p}",
        type="primary" if st.session_state["page"] == p else "secondary",
        use_container_width=True,
    ):
        st.session_state["page"] = p
        st.rerun()
page = st.session_state["page"]

# 다른 페이지로 이동해도 입력값이 유지되도록 복원
for _k in ["office_name", "school_input", "height", "weight"]:
    if _k in st.session_state:
        st.session_state[_k] = st.session_state[_k]

on_top = "rows" in st.session_state and page != pages[0]

if not on_top:
    st.title("🍱 우리학교 다이어트 지키미")
    st.write("오늘 급식 칼로리를 확인하고, 먹은 만큼 움직여 보세요! 우리 학교 칼로리 TOP도 함께 볼 수 있어요.")

    offices = {
        "서울특별시교육청": "B10", "부산광역시교육청": "C10",
        "대구광역시교육청": "D10", "인천광역시교육청": "E10",
        "광주광역시교육청": "F10", "대전광역시교육청": "G10",
        "울산광역시교육청": "H10", "세종특별자치시교육청": "I10",
        "경기도교육청": "J10", "강원특별자치도교육청": "K10",
        "충청북도교육청": "M10", "충청남도교육청": "N10",
        "전북특별자치도교육청": "P10", "전라남도교육청": "Q10",
        "경상북도교육청": "R10", "경상남도교육청": "S10",
        "제주특별자치도교육청": "T10"
    }

    office_name = st.selectbox("교육청을 선택하세요", list(offices.keys()), index=8, key="office_name")
    office_code = offices[office_name]
    school_name = st.text_input("학교 이름을 입력하세요", value="광명고등학교", key="school_input")
    col_h, col_w, col_b = st.columns(3)
    height = col_h.slider("키 (cm)", min_value=130, max_value=210, value=170, step=1, key="height")
    weight = col_w.slider("몸무게 (kg)", min_value=30, max_value=150, value=60, step=1, key="weight")
    bmi = weight / (height / 100) ** 2
    if bmi < 18.5:
        bmi_label = "저체중"
    elif bmi < 23:
        bmi_label = "정상"
    elif bmi < 25:
        bmi_label = "과체중"
    else:
        bmi_label = "비만"
    stat_card(col_b, "📏 내 BMI", f"{bmi:.1f}", f"({bmi_label})")

    st.caption(f"조회 기간: {start} ~ {end}")

    _, col_btn, _ = st.columns([1, 1, 1])
    scan_clicked = col_btn.button("급식 스캔 시작! 🔍", type="primary", use_container_width=True)
    if "rows" not in st.session_state and page != pages[0]:
        st.info("먼저 '급식 스캔 시작!'을 눌러 급식 데이터를 불러와 주세요.")

    if scan_clicked:
        if not school_name.strip():
            st.warning("학교 이름을 입력해주세요.")
        else:
            try:
                with st.spinner("급식 정보를 불러오는 중..."):
                    school_code = find_school(office_code, school_name)
                    if not school_code:
                        st.error("학교를 찾을 수 없습니다. 교육청과 학교 이름을 확인해 주세요.")
                    else:
                        rows = get_meals_range(office_code, school_code, start, end)
                        if not rows:
                            st.info("해당 기간에 급식 정보가 없습니다.")
                        else:
                            st.session_state["rows"] = rows
                            st.session_state["school_name"] = school_name
            except Exception as e:
                st.error(f"API 호출 중 오류가 발생했습니다: {e}")

# 결과 표시 (session_state에 데이터가 있을 때)
if "rows" in st.session_state:
    df = pd.DataFrame(st.session_state["rows"])
    if not on_top:
        st.success(f"{st.session_state['school_name']} 급식 {len(df)}건을 불러왔습니다.")

    # 전처리 (중식만 사용)
    df = df[df["MMEAL_SC_CODE"] == "2"].copy()
    df["칼로리"] = df["CAL_INFO"].apply(parse_cal)
    df = df.dropna(subset=["칼로리"])
    df["메뉴"] = df["DDISH_NM"].apply(lambda x: ", ".join(clean_menu(x)))
    df["날짜"] = pd.to_datetime(df["MLSV_YMD"]).dt.strftime("%Y-%m-%d")

    if df.empty:
        st.info("중식 데이터가 없습니다.")
    else:

        if page == "📅 오늘의 급식":
            st.header(f"📅 오늘의 급식 · {today.month}월 {today.day}일 ({WEEKDAYS[today.weekday()]})")
            today_df = df[df["MLSV_YMD"] == end]
            if today_df.empty:
                st.info("오늘은 중식 정보가 없습니다. (주말·휴일일 수 있어요)")
            else:
                t = today_df.iloc[0]
                kcal = float(t["칼로리"])
                dishes = clean_menu(t["DDISH_NM"])
                nutrients = parse_nutrients(t.get("NTR_INFO"))

                left, right = st.columns([3, 2])

                # 왼쪽: 메뉴 카드 + 영양소 그래프
                with left.container(border=True):
                    st.markdown("##### 🍽️ 오늘의 메뉴")
                    chips = "".join(
                        f'<span style="display:inline-block;margin:3px 4px 3px 0;padding:4px 12px;'
                        f'border-radius:14px;background:rgba(255,140,60,0.15);font-size:0.9rem;">'
                        f'{html.escape(d)}</span>'
                        for d in dishes
                    )
                    st.markdown(chips, unsafe_allow_html=True)

                    macro = {k: nutrients[k] for k in ["탄수화물", "단백질", "지방"] if k in nutrients}
                    if macro:
                        st.markdown("##### 🥗 영양소 (g)")
                        pastel_bar(macro, "g")

                # 오른쪽: 총 칼로리 카드
                stat_card(
                    right, "🔥 총 칼로리", f"{kcal:.0f}", "Kcal",
                    f"하루 권장 열량({DAILY_KCAL:,} Kcal 가정)의 {kcal / DAILY_KCAL * 100:.0f}%",
                )

                # 영양정보 카드 (누르면 팝업)
                if nutrients:
                    with right.container(border=True):
                        st.markdown("##### 🧾 영양정보")
                        if st.button("전체 영양정보 보기", use_container_width=True):
                            show_nutrition(nutrients)

                # 운동 카드
                st.markdown("##### 🏃 먹은 만큼 움직이려면?")
                exercises = {"🚶 걷기": 3.5, "🏃 조깅": 7.0, "🚴 자전거": 6.8, "🪢 줄넘기": 11.0}
                cols = st.columns(len(exercises))
                minutes = {}
                for col, (name, met) in zip(cols, exercises.items()):
                    kcal_per_min = met * 3.5 * weight / 200
                    minutes[name] = round(kcal / kcal_per_min)
                    stat_card(col, name, f"{minutes[name]}", "분")
                pastel_bar(minutes, "필요 시간 (분)")

        elif page == pages[1]:
            st.header("🏆 칼로리 TOP (중식 기준)")
            n = st.radio("표시 개수", [5, 10], horizontal=True)
            top = (
                df[["날짜", "칼로리", "메뉴"]]
                .sort_values("칼로리", ascending=False)
                .head(n)
                .reset_index(drop=True)
            )
            top.index = top.index + 1

            # 막대그래프: TOP3는 강조색, 나머지는 회색
            chart_df = top.reset_index().rename(columns={"index": "순위"})
            chart_df["라벨"] = chart_df["순위"].astype(str) + "위 · " + chart_df["날짜"]
            rank_colors = ["#FF4B4B", "#FF8A3D", "#FFC53D"]
            chart_df["색"] = [
                rank_colors[i] if i < 3 else "#BDBDBD" for i in range(len(chart_df))
            ]
            # x축을 0이 아닌 데이터 범위 근처에서 시작 (차이를 크게 보이게)
            lo = int((chart_df["칼로리"].min() - 30) // 50 * 50)
            hi = chart_df["칼로리"].max()
            hi = hi + (hi - lo) * 0.15  # 수치 라벨 공간
            bars = alt.Chart(chart_df, height=alt.Step(36)).mark_bar(clip=True).encode(
                x=alt.X(
                    "칼로리:Q",
                    title="칼로리 (Kcal)",
                    scale=alt.Scale(domain=[lo, hi], zero=False, nice=False),
                ),
                y=alt.Y("라벨:N", sort=None, title=None),
                color=alt.Color("색:N", scale=None, legend=None),
                tooltip=["날짜", "칼로리", "메뉴"],
            )
            labels = bars.mark_text(align="left", dx=4).encode(
                text=alt.Text("칼로리:Q", format=".0f"),
                color=alt.value("gray"),
            )
            st.altair_chart(bars + labels, use_container_width=True)
            st.caption(f"※ 차이를 잘 보이게 하려고 가로축을 {lo} Kcal부터 시작했어요.")

            st.dataframe(top, use_container_width=True)
            st.write(f"집계 일수 {len(df)}일, 평균 {df['칼로리'].mean():.1f} Kcal")

        elif page == pages[2]:
            st.header("🍚 우리학교 인기쟁이 메뉴")
            st.caption("3월 2일부터 중식에 많이 나온 음식이에요. 글씨가 클수록 자주 나왔어요! (인기투표가 아니라 등장 횟수예요)")

            skip_basic = st.checkbox("김치류·흰밥 종류 제외", value=False, key="skip_basic")

            counts = {}
            for raw in df["DDISH_NM"]:
                for dish in set(clean_menu(raw)):  # 하루에 같은 음식이 두 번 세어지지 않도록
                    if skip_basic and is_kimchi_or_rice(dish):
                        continue
                    counts[dish] = counts.get(dish, 0) + 1
            if not counts:
                st.info("집계할 메뉴가 없어요.")
                st.stop()

            menu_df = (
                pd.DataFrame({"메뉴": list(counts.keys()), "횟수": list(counts.values())})
                .sort_values(["횟수", "메뉴"], ascending=[False, True])
                .head(40)
                .reset_index(drop=True)
            )

            # 워드클라우드 (글씨 크기 = 나온 횟수)
            lo_c, hi_c = menu_df["횟수"].min(), menu_df["횟수"].max()
            palette = ["#FF4B4B", "#FF8A3D", "#F2B01E", "#2E9E6B", "#3B82F6", "#8B5CF6", "#EC4899"]
            shuffled = menu_df.sample(frac=1, random_state=7).reset_index(drop=True)
            words = []
            for i, r in shuffled.iterrows():
                ratio = (r["횟수"] - lo_c) / (hi_c - lo_c) if hi_c > lo_c else 1
                size = 0.95 + ratio * 2.4
                words.append(
                    f'<span title="{r["횟수"]}회" style="font-size:{size:.2f}rem;font-weight:{700 if ratio > 0.3 else 500};'
                    f'color:{palette[i % len(palette)]};margin:4px 10px;display:inline-block;line-height:1.3;">'
                    f'{html.escape(r["메뉴"])}</span>'
                )
            with st.container(border=True):
                st.markdown(
                    '<div style="text-align:center;padding:12px 4px;">' + "".join(words) + "</div>",
                    unsafe_allow_html=True,
                )
            st.caption("글씨에 마우스를 올리면 나온 횟수를 볼 수 있어요.")

            st.markdown("##### 🥇 많이 나온 음식 TOP 3")
            cols3 = st.columns(3)
            for i, col in enumerate(cols3[: len(menu_df)]):
                stat_card(col, f"{i + 1}위", html.escape(menu_df.loc[i, "메뉴"]), f"{menu_df.loc[i, '횟수']}회")

        else:
            st.header("💪 개인트레이너 잔소리")
            today_df = df[df["MLSV_YMD"] == end]
            if today_df.empty:
                st.info("오늘은 중식 정보가 없어서 잔소리할 거리가 없어요. (주말·휴일일 수 있어요)")
                st.stop()

            t = today_df.iloc[0]
            kcal = float(t["칼로리"])
            menu = ", ".join(clean_menu(t["DDISH_NM"]))
            nutrients = parse_nutrients(t.get("NTR_INFO"))
            h = st.session_state.get("height", 170)
            w = st.session_state.get("weight", 60)
            bmi_v = w / (h / 100) ** 2
            nutri_text = ", ".join(f"{k} {v:g}" for k, v in nutrients.items() if k in ["탄수화물", "단백질", "지방", "나트륨"])

            system_prompt = f"""너는 고등학생 전담 '개인 트레이너'야. 말투는 츤데레처럼 잔소리가 많지만 속은 따뜻해.
반말과 존댓말을 섞지 말고 친근한 반말로, 이모지를 조금 섞어서 짧게(5~8문장) 말해.
읽기 쉽게 한 문장이나 한 항목마다 줄을 바꾸고, 운동·식습관 제안은 "- "로 시작하는 목록으로 써.
아래는 학생의 오늘 학교 급식과 신체 정보야. 이 데이터를 근거로 구체적으로 잔소리하고, 살 빼는 방법(운동, 식습관)을 제안해.
- 오늘 중식 메뉴: {menu}
- 오늘 중식 칼로리: {kcal:.0f} Kcal
- 영양소(g, mg): {nutri_text}
- 키 {h}cm, 몸무게 {w}kg, BMI {bmi_v:.1f}
안전 규칙: 굶기, 구토, 약물, 극단적 절식은 절대 권하지 마. 외모나 몸을 비하하지 마.
BMI가 18.5 미만이면 살을 빼라고 하지 말고 균형 잡힌 식사를 권해.
학생이 식이장애가 의심되는 말을 하면 걱정해 주고 보건교사나 부모님과 이야기해 보라고 권해.
의학적 진단은 하지 말고, 심각한 건강 고민은 전문가와 상의하라고 안내해."""

            st.caption(f"오늘 급식 {kcal:.0f} Kcal · 키 {h}cm · 몸무게 {w}kg 기준으로 잔소리해요. (AI가 만든 답변이라 틀릴 수 있어요)")

            if "chat" not in st.session_state:
                st.session_state["chat"] = []

            pending = None  # 이번 실행에서 새로 보낸 말
            col_a, col_b = st.columns([3, 1])
            if col_a.button("📣 오늘 급식 잔소리 듣기", use_container_width=True, type="primary"):
                pending = "오늘 급식 먹은 거 보고 잔소리해줘!"
            if col_b.button("🗑️ 대화 지우기", use_container_width=True):
                st.session_state["chat"] = []
                st.rerun()
            typed = st.chat_input("트레이너에게 말해 보세요 (예: 오늘 운동 뭐 하면 좋아?)")
            if typed:
                pending = typed

            # 새로 보낸 말도 대기 중에 바로 보이도록, 먼저 기록에 넣고 화면을 그린다
            if pending:
                st.session_state["chat"].append({"role": "user", "content": pending})

            def show_message(m):
                is_user = m["role"] == "user"
                with st.chat_message("user" if is_user else "assistant", avatar="🧑‍🎓" if is_user else "💪"):
                    st.markdown(format_reply(m["content"]))

            for m in st.session_state["chat"]:
                show_message(m)

            if pending:
                with st.chat_message("assistant", avatar="💪"):
                    with st.spinner("트레이너가 생각 중..."):
                        reply = ask_trainer(system_prompt, st.session_state["chat"])
                    st.markdown(format_reply(reply))
                st.session_state["chat"].append({"role": "assistant", "content": reply})
