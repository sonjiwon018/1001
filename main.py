import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)


# =========================================================
# 화면 디자인
# =========================================================
st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            180deg,
            #fff8fb 0%,
            #f8fbff 100%
        );
    }

    .main-title {
        text-align: center;
        font-size: 48px;
        font-weight: 800;
        color: #333333;
        margin-top: 10px;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        color: #777777;
        margin-bottom: 35px;
    }

    .info-card {
        background-color: white;
        border-radius: 18px;
        padding: 22px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.06);
        border: 1px solid #f0e5ea;
    }

    .card-title {
        font-size: 15px;
        color: #888888;
        margin-bottom: 8px;
    }

    .card-value {
        font-size: 30px;
        font-weight: 800;
        color: #e86a92;
    }

    .slope-card {
        background: linear-gradient(
            135deg,
            #ffe8f0,
            #eaf5ff
        );
        border-radius: 22px;
        padding: 30px 20px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(220, 150, 180, 0.12);
        border: 1px solid #f3dce5;
        min-height: 170px;
    }

    .slope-title {
        font-size: 18px;
        color: #666666;
        margin-bottom: 12px;
        font-weight: 600;
    }

    .slope-value {
        font-size: 40px;
        font-weight: 900;
        color: #e85d88;
    }

    .slope-unit {
        font-size: 16px;
        color: #777777;
        margin-top: 5px;
    }

    .temperature-card {
        background: linear-gradient(
            135deg,
            #ffe8f0,
            #eaf5ff
        );
        border-radius: 25px;
        padding: 35px;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 25px;
        box-shadow: 0 8px 25px rgba(220, 150, 180, 0.15);
        border: 1px solid #f3dce5;
    }

    .temperature-year {
        font-size: 22px;
        color: #666666;
        margin-bottom: 5px;
    }

    .temperature-value {
        font-size: 64px;
        font-weight: 900;
        color: #e85d88;
    }

    .temperature-description {
        font-size: 15px;
        color: #777777;
        margin-top: 5px;
    }

    .notice {
        background-color: white;
        border-left: 5px solid #f28bad;
        padding: 15px 20px;
        border-radius: 10px;
        color: #666666;
        margin: 15px 0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 제목
# =========================================================
st.markdown(
    '<div class="main-title">🌡️ 기온 예측기</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    '서울의 과거 기온 데이터를 이용해 연도에 따른 기온 변화를 살펴봅니다.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# 데이터 주소
# =========================================================
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


# =========================================================
# 데이터 불러오기
# =========================================================
@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8"
    )

    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["날짜", "평균기온"]
    )

    df["연도"] = df["날짜"].dt.year

    return df


# =========================================================
# 데이터 불러오기
# =========================================================
try:

    df = load_data()

except Exception as e:

    st.error("기상 데이터를 불러오는 중 문제가 발생했습니다.")
    st.error(str(e))
    st.stop()


# =========================================================
# 2025년까지의 데이터만 사용
# =========================================================
df = df[
    df["연도"] <= 2025
].copy()


# =========================================================
# 연도별 관측일 수
# =========================================================
year_count = (
    df.groupby("연도")
    .size()
    .reset_index(name="관측일수")
)


# =========================================================
# 연도별 평균기온
# =========================================================
year_temp = (
    df.groupby("연도")["평균기온"]
    .mean()
    .reset_index(name="연평균기온")
)


# =========================================================
# 연도별 데이터 합치기
# =========================================================
annual = pd.merge(
    year_temp,
    year_count,
    on="연도",
    how="inner"
)


# =========================================================
# 관측일수가 300일 미만인 연도 제외
# =========================================================
annual = annual[
    annual["관측일수"] >= 300
].copy()


# =========================================================
# 1908년부터 지난 연수 계산
# =========================================================
annual["지난연수"] = (
    annual["연도"] - 1908
)


# 1908년 이전 데이터 제외
annual = annual[
    annual["지난연수"] >= 0
].copy()


# =========================================================
# 데이터가 충분한지 확인
# =========================================================
if len(annual) < 2:

    st.error(
        "회귀분석을 할 수 있는 데이터가 충분하지 않습니다."
    )

    st.stop()


# =========================================================
# 전체 기간 회귀분석
# 독립변수 = 1908년부터 지난 연수
# 종속변수 = 연평균기온
# =========================================================
x = annual["지난연수"].to_numpy()
y = annual["연평균기온"].to_numpy()


slope, intercept = np.polyfit(
    x,
    y,
    1
)


# =========================================================
# 전체 기간의 100년당 기온 변화
# =========================================================
slope_per_100_years = slope * 100


# =========================================================
# 전체 기간 회귀 예측값
# =========================================================
annual["회귀예측기온"] = (
    slope * annual["지난연수"]
    + intercept
)


# =========================================================
# 상관계수
# =========================================================
correlation = np.corrcoef(
    x,
    y
)[0, 1]


# =========================================================
# 회귀에 사용된 정보
# =========================================================
used_years = len(annual)

start_year = int(
    annual["연도"].min()
)

end_year = int(
    annual["연도"].max()
)


# =========================================================
# =========================================================
# 최근 20년 회귀분석
# =========================================================
# 끝 연도를 기준으로 최근 20년
recent_start_year = end_year - 19

recent_annual = annual[
    annual["연도"] >= recent_start_year
].copy()


# 최근 20년 데이터가 충분한지 확인
if len(recent_annual) >= 2:

    recent_x = (
        recent_annual["연도"] - 1908
    ).to_numpy()

    recent_y = (
        recent_annual["연평균기온"]
        .to_numpy()
    )

    recent_slope, recent_intercept = np.polyfit(
        recent_x,
        recent_y,
        1
    )

    # 100년당 변화량
    recent_slope_per_100_years = (
        recent_slope * 100
    )

else:

    recent_slope = np.nan
    recent_intercept = np.nan
    recent_slope_per_100_years = np.nan


# =========================================================
# 회귀식
# =========================================================
regression_text = (
    f"연평균기온 = "
    f"{slope:.4f} × (연도 - 1908) "
    f"+ {intercept:.4f}"
)


# =========================================================
# 회귀 분석 정보
# =========================================================
st.markdown(
    "## 📊 회귀 분석 정보"
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown(
        f"""
        <div class="info-card">
            <div class="card-title">
                직선을 만든 해의 개수
            </div>
            <div class="card-value">
                {used_years}개
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        f"""
        <div class="info-card">
            <div class="card-title">
                시작 연도
            </div>
            <div class="card-value">
                {start_year}년
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        f"""
        <div class="info-card">
            <div class="card-title">
                끝 연도
            </div>
            <div class="card-value">
                {end_year}년
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col4:

    st.markdown(
        f"""
        <div class="info-card">
            <div class="card-title">
                상관계수
            </div>
            <div class="card-value">
                {correlation:.3f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# 회귀식 표시
# =========================================================
st.markdown(
    f"""
    <div class="notice">
        <b>전체 기간 회귀식</b><br>
        {regression_text}
    </div>
    """,
    unsafe_allow_html=True
)


st.caption(
    "2025년까지의 자료 중 관측일수가 300일 이상인 연도만 "
    "회귀분석에 사용했습니다."
)


# =========================================================
# 100년에 몇 도 오르는가?
# =========================================================
st.markdown("---")

st.markdown(
    "## 🌡️ 100년에 몇 ℃ 변하는가?"
)

st.write(
    "회귀선의 기울기를 100년 기준으로 환산한 값입니다."
)


slope_col1, slope_col2 = st.columns(2)


# =========================================================
# 전체 기간
# =========================================================
with slope_col1:

    st.markdown(
        f"""
        <div class="slope-card">

            <div class="slope-title">
                📚 전체 기간
            </div>

            <div class="slope-value">
                {slope_per_100_years:+.2f}℃
            </div>

            <div class="slope-unit">
                100년에 변화하는 연평균기온
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# 최근 20년
# =========================================================
with slope_col2:

    if not np.isnan(recent_slope_per_100_years):

        st.markdown(
            f"""
            <div class="slope-card">

                <div class="slope-title">
                    🕒 최근 20년
                    ({recent_start_year}~{end_year})
                </div>

                <div class="slope-value">
                    {recent_slope_per_100_years:+.2f}℃
                </div>

                <div class="slope-unit">
                    100년에 변화하는 연평균기온
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.warning(
            "최근 20년 회귀분석에 사용할 데이터가 부족합니다."
        )


# =========================================================
# 산점도 + 전체 기간 회귀선
# =========================================================
st.markdown("---")

st.markdown(
    "## 📈 연도별 평균기온과 회귀선"
)


# =========================================================
# 전체 기간 회귀선용 연도
# =========================================================
line_years = np.linspace(
    annual["연도"].min(),
    annual["연도"].max(),
    300
)


line_x = (
    line_years - 1908
)


line_y = (
    slope * line_x
    + intercept
)


# =========================================================
# Plotly 그래프
# =========================================================
fig = go.Figure()


# =========================================================
# 실제 연평균기온 산점도
# =========================================================
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(
            size=8,
            opacity=0.75
        ),
        customdata=annual[
            ["관측일수"]
        ],
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "연평균기온: %{y:.2f}℃<br>"
            "관측일수: %{customdata[0]}일"
            "<extra></extra>"
        )
    )
)


# =========================================================
# 전체 기간 회귀선
# =========================================================
fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_y,
        mode="lines",
        name="전체 기간 회귀선",
        line=dict(
            width=4
        ),
        hovertemplate=(
            "<b>%{x:.0f}년</b><br>"
            "회귀선 예상기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# =========================================================
# 그래프 디자인
# =========================================================
fig.update_layout(

    height=600,

    plot_bgcolor="white",

    paper_bgcolor="rgba(0,0,0,0)",

    xaxis=dict(
        title="연도",
        tickmode="auto",
        dtick=10,
        gridcolor="#eeeeee"
    ),

    yaxis=dict(
        title="연평균기온 (℃)",
        gridcolor="#eeeeee"
    ),

    hovermode="x unified",

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),

    margin=dict(
        l=50,
        r=30,
        t=40,
        b=50
    )
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# 연도별 예상 평균기온
# =========================================================
st.markdown("---")

st.markdown(
    "## 🔮 연도별 예상 평균기온"
)


st.write(
    "슬라이더에서 연도를 선택하면 "
    "전체 기간의 회귀선을 이용해 해당 연도의 "
    "예상 연평균기온을 계산합니다."
)


# =========================================================
# 연도 슬라이더
# =========================================================
selected_year = st.slider(
    "예측할 연도",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)


# =========================================================
# 선택한 연도의 지난 연수
# =========================================================
selected_x = (
    selected_year - 1908
)


# =========================================================
# 선택한 연도의 예상 기온
# =========================================================
predicted_temp = (
    slope * selected_x
    + intercept
)


# =========================================================
# 예상 기온 크게 표시
# =========================================================
st.markdown(
    f"""
    <div class="temperature-card">

        <div class="temperature-year">
            {selected_year}년 예상 연평균기온
        </div>

        <div class="temperature-value">
            {predicted_temp:.2f}℃
        </div>

        <div class="temperature-description">
            전체 기간의 선형 회귀식을 이용한 예상값
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 실제 데이터와 비교
# =========================================================
selected_actual = annual[
    annual["연도"] == selected_year
]


if not selected_actual.empty:

    actual_temp = (
        selected_actual.iloc[0]["연평균기온"]
    )

    difference = (
        predicted_temp - actual_temp
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "실제 연평균기온",
            f"{actual_temp:.2f}℃"
        )

    with col2:

        st.metric(
            "회귀선과 실제 기온의 차이",
            f"{difference:+.2f}℃"
        )

else:

    st.caption(
        "선택한 연도에는 조건을 만족하는 실제 관측자료가 없어 "
        "회귀식을 이용한 예상값만 표시됩니다."
    )


# =========================================================
# 데이터 처리 기준
# =========================================================
st.markdown("---")

st.markdown(
    "## ℹ️ 데이터 처리 기준"
)


st.markdown(
    f"""
    - **자료:** 서울 기상 관측 데이터
    - **사용 기간:** 2025년까지
    - **연평균기온:** 해당 연도의 일별 평균기온의 평균
    - **제외 기준:** 관측일수가 300일 미만인 연도
    - **독립 변수:** 연도 - 1908
    - **종속 변수:** 연평균기온
    - **전체 기간:** {start_year}년 ~ {end_year}년
    - **최근 20년:** {recent_start_year}년 ~ {end_year}년
    - **회귀 방법:** 1차 선형회귀
    - **예측 연도 범위:** 1900년 ~ 2100년
    """
)


st.caption(
    "※ 100년에 몇 ℃ 변하는지는 회귀선의 기울기를 100배하여 계산한 값입니다."
)

st.caption(
    "※ 회귀선에 의한 예상값은 과거 자료의 선형 관계를 "
    "이용한 계산값이며 실제 미래 기온을 보장하는 예보값은 아닙니다."
)
