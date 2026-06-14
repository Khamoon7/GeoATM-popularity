import requests
import streamlit as st
import folium
import plotly.graph_objects as go
from streamlit_folium import st_folium

# ВАЖНО: set_page_config должен быть самым первым вызовом Streamlit
st.set_page_config(
    page_title="GeoATM • Популярность банкоматов",
    page_icon="🏧",
    layout="wide",
)

SESSION = requests.Session()
SESSION.trust_env = False

# =========================================================================
# Константы
# =========================================================================

# Интуитивная цветовая семантика сегментов: высокий = хорошо = зелёный
SEG_HEX = {"high": "#16a34a", "medium": "#d97706", "low": "#dc2626"}
SEG_BG = {"high": "#dcfce7", "medium": "#fef3c7", "low": "#fee2e2"}
SEG_FOLIUM = {"high": "green", "medium": "orange", "low": "red"}

# Функции банкомата, сгруппированные по категориям: key (флаг API) -> подпись
FLAG_DEFS = {
    "Сервисы": {
        "is_24_7": "Круглосуточно 24/7",
        "contactless_tech": "Бесконтакт (NFC)",
        "qr_codes": "QR-коды",
        "cash_in": "Приём наличных",
        "cash_out": "Выдача наличных",
        "cashless_pay": "Безналичная оплата",
        "account_statement": "Выписка по счёту",
        "access_for_disabled": "Доступ для МГН",
    },
    "Валюта": {
        "usd_available": "Доллары (USD)",
        "eur_available": "Евро (EUR)",
    },
    "Переводы и платежи": {
        "transfer_p2p": "Переводы P2P",
        "transfer_a2a": "Переводы A2A",
        "loan_payments": "Платежи по кредитам",
    },
}
ALL_FLAG_KEYS = [k for cat in FLAG_DEFS.values() for k in cat]

PRESETS = {
    "Базовый": {"cash_out", "cash_in", "contactless_tech"},
    "Премиум": set(ALL_FLAG_KEYS),
    "Сброс": set(),
}

BANK_OPTIONS = [
    "ВТБ", "Альфа-Банк", "Росбанк", "Россельхозбанк",
    "Газпромбанк", "Ак Барс Банк", "Уралсиб", "Другое",
]

GAUGE_LO, GAUGE_HI = -0.15, 0.22


# =========================================================================
# Стили
# =========================================================================

def inject_css() -> None:
    st.markdown(
        """
        <style>
        footer {visibility: hidden;}
        .block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1280px;}

        /* Карточки (st.container(border=True)) — мягкая тень + скругление */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            box-shadow: 0 6px 20px rgba(15, 23, 42, 0.06);
            border-radius: 16px !important;
            border-color: #e8ebf3 !important;
        }

        .geo-hero {
            background: linear-gradient(110deg, #4f46e5 0%, #6366f1 45%, #14b8a6 110%);
            border-radius: 20px;
            padding: 26px 32px;
            color: #ffffff;
            box-shadow: 0 12px 30px rgba(79, 70, 229, 0.28);
            margin-bottom: 18px;
        }
        .geo-hero h1 {font-size: 28px; font-weight: 800; margin: 0 0 6px 0; line-height: 1.2;}
        .geo-hero p {font-size: 15px; opacity: .92; margin: 0;}
        .geo-hero .pill {
            display: inline-block; margin-top: 14px; padding: 4px 12px;
            background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.35);
            border-radius: 999px; font-size: 12.5px; font-weight: 600; letter-spacing: .3px;
        }
        .geo-badge {
            display: inline-block; padding: 5px 16px; border-radius: 999px;
            font-weight: 800; font-size: 14px; letter-spacing: .4px;
        }
        .geo-fact {
            background: #f8fafc; border: 1px solid #eef1f7; border-radius: 12px;
            padding: 10px 14px; margin-bottom: 8px;
        }
        .geo-fact .k {font-size: 12px; color: #64748b; margin-bottom: 2px;}
        .geo-fact .v {font-size: 15px; color: #0f172a; font-weight: 600; word-break: break-word;}
        .geo-empty {
            text-align: center; color: #64748b; padding: 48px 16px;
            border: 1.5px dashed #d7dce8; border-radius: 16px; background: #fbfcff;
        }
        .geo-empty .ico {font-size: 46px;}
        .status-dot {font-weight: 700;}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================================
# Хелперы
# =========================================================================

def fmt_float(x, ndigits: int = 5) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return str(x)


def segment_ru(seg) -> str:
    if not seg:
        return "—"
    return {"high": "Высокий", "medium": "Средний", "low": "Низкий"}.get(seg.lower(), seg)


def segment_badge_html(seg) -> str:
    seg_l = (seg or "").lower()
    color = SEG_HEX.get(seg_l, "#64748b")
    bg = SEG_BG.get(seg_l, "#f1f5f9")
    return (
        f'<span class="geo-badge" style="background:{bg};color:{color};'
        f'border:1px solid {color}40;">{segment_ru(seg).upper()}</span>'
    )


def gauge_figure(pop: float, seg) -> go.Figure:
    needle = SEG_HEX.get((seg or "").lower(), "#4f46e5")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(pop),
            number={"valueformat": ".4f", "font": {"size": 30, "color": "#0f172a"}},
            gauge={
                "axis": {"range": [GAUGE_LO, GAUGE_HI], "tickwidth": 1, "tickcolor": "#94a3b8"},
                "bar": {"color": "rgba(0,0,0,0)"},
                "borderwidth": 0,
                "steps": [
                    {"range": [GAUGE_LO, -0.02], "color": "#fecaca"},
                    {"range": [-0.02, 0.02], "color": "#fde68a"},
                    {"range": [0.02, GAUGE_HI], "color": "#bbf7d0"},
                ],
                "threshold": {
                    "line": {"color": needle, "width": 5},
                    "thickness": 0.9,
                    "value": float(pop),
                },
            },
        )
    )
    fig.update_layout(
        height=240,
        margin=dict(l=24, r=24, t=12, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#1e293b"},
    )
    return fig


def fact_html(key: str, value: str) -> str:
    return f'<div class="geo-fact"><div class="k">{key}</div><div class="v">{value}</div></div>'


def build_display_address(payload: dict, coords: dict) -> str:
    addr = payload.get("address")
    if addr:
        return addr
    lat, lon = coords.get("lat"), coords.get("lon")
    if lat is not None and lon is not None:
        return f"{fmt_float(lat, 6)}, {fmt_float(lon, 6)}"
    return "—"


def build_payload(*, use_address, address, lat, lon, bank_name, flags) -> dict:
    payload: dict = {}
    if use_address:
        if address and address.strip():
            payload["address"] = address.strip()
    else:
        if lat is not None and lon is not None:
            payload["lat"] = float(lat)
            payload["lon"] = float(lon)
    if bank_name and bank_name.strip():
        payload["bank_name"] = bank_name.strip()
    payload.update(flags)
    return payload


def call_forward(base_url: str, payload: dict, timeout_s: float):
    url = f"{base_url}/forward"
    try:
        r = SESSION.post(url, json=payload, timeout=float(timeout_s))
    except requests.RequestException as e:
        return 0, None, f"Сеть недоступна: {e}"
    try:
        data = r.json() if r.text else None
    except Exception:
        data = None
    if r.status_code >= 400:
        detail = data.get("detail") if isinstance(data, dict) else (r.text[:600] or None)
        return r.status_code, data, str(detail) if detail else "Неизвестная ошибка"
    return r.status_code, data, None


@st.cache_data(ttl=15, show_spinner=False)
def api_health(base_url: str) -> bool:
    try:
        return SESSION.get(f"{base_url}/health", timeout=3).status_code == 200
    except Exception:
        return False


@st.cache_data(ttl=60, show_spinner=False)
def api_model_info(base_url: str):
    try:
        r = SESSION.get(f"{base_url}/model_info", timeout=3)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def apply_preset(preset_keys: set) -> None:
    for cat, items in FLAG_DEFS.items():
        st.session_state[f"pills_{cat}"] = [lbl for k, lbl in items.items() if k in preset_keys]


# =========================================================================
# Инициализация состояния
# =========================================================================

inject_css()

if "last_result" not in st.session_state:
    st.session_state.last_result = None
for _cat in FLAG_DEFS:
    st.session_state.setdefault(f"pills_{_cat}", [])


# =========================================================================
# Hero
# =========================================================================

st.markdown(
    """
    <div class="geo-hero">
      <h1>🏧 GeoATM — прогноз популярности локаций для банкоматов</h1>
      <p>Оцените потенциал точки размещения: индекс популярности и сегмент по геоданным,
      инфраструктуре, плотности населения и экономике региона.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================================
# Sidebar
# =========================================================================

with st.sidebar:
    st.markdown("### 🏧 GeoATM")
    st.caption("Сервис прогнозирования популярности банкоматов")
    st.divider()

    st.markdown("#### 🔌 Подключение")
    api_base_url = st.text_input("API Base URL", value="http://127.0.0.1:8000").rstrip("/")

    online = api_health(api_base_url)
    if online:
        st.markdown(
            '<span class="status-dot" style="color:#16a34a;">🟢 API online</span>',
            unsafe_allow_html=True,
        )
        info = api_model_info(api_base_url)
        if info:
            model_name = (info.get("model_name") or "—").split("(")[0].strip()
            st.caption(f"**Модель:** {model_name}")
    else:
        st.markdown(
            '<span class="status-dot" style="color:#dc2626;">🔴 API offline</span>',
            unsafe_allow_html=True,
        )
        st.caption("Проверь, что запущен `uvicorn main:app` и URL верный.")

    with st.expander("⚙️ Настройки"):
        timeout_s = st.number_input(
            "Timeout (сек)", min_value=10, max_value=600, value=180, step=10,
            help="Сбор 67 признаков через OSM занимает ~10–30с.",
        )

    st.divider()
    st.caption("Эндпоинт: `POST /forward`")
    st.caption("© GeoATM · Happy Data Year")


# =========================================================================
# Основной layout
# =========================================================================

col_left, col_right = st.columns([1, 1.05], gap="large")

# ----- Левая колонка: ввод -----
with col_left:
    with st.container(border=True):
        st.subheader("📍 Локация")
        mode = st.segmented_control(
            "Способ ввода", ["Адрес", "Координаты"], default="Адрес",
            selection_mode="single", label_visibility="collapsed",
        ) or "Адрес"
        use_address = (mode == "Адрес")

        address = lat = lon = None
        if use_address:
            address = st.text_input("Адрес", value="Москва, Тверская 1",
                                    placeholder="Город, улица, дом")
        else:
            c1, c2 = st.columns(2)
            lat = c1.number_input("Широта (lat)", value=55.7558, format="%.6f")
            lon = c2.number_input("Долгота (lon)", value=37.6173, format="%.6f")

    with st.container(border=True):
        st.subheader("🏦 Банк")
        bank_choice = st.selectbox("Банк", BANK_OPTIONS, index=0, label_visibility="collapsed")
        bank_name = bank_choice
        if bank_choice == "Другое":
            bank_name = st.text_input("Название банка", value="", placeholder="Введите название").strip()

    with st.container(border=True):
        st.subheader("⚙️ Функции банкомата")
        pc1, pc2, pc3 = st.columns(3)
        if pc1.button("Базовый", use_container_width=True):
            apply_preset(PRESETS["Базовый"]); st.rerun()
        if pc2.button("Премиум", use_container_width=True):
            apply_preset(PRESETS["Премиум"]); st.rerun()
        if pc3.button("Сброс", use_container_width=True):
            apply_preset(PRESETS["Сброс"]); st.rerun()

        for cat, items in FLAG_DEFS.items():
            st.pills(cat, list(items.values()), selection_mode="multi", key=f"pills_{cat}")

    # Собираем флаги из состояния pills
    flags = {k: False for k in ALL_FLAG_KEYS}
    for cat, items in FLAG_DEFS.items():
        label_to_key = {lbl: k for k, lbl in items.items()}
        for lbl in st.session_state.get(f"pills_{cat}", []):
            if lbl in label_to_key:
                flags[label_to_key[lbl]] = True

    payload = build_payload(
        use_address=use_address, address=address, lat=lat, lon=lon,
        bank_name=bank_name, flags=flags,
    )

    if st.button("🚀 Рассчитать популярность", type="primary", use_container_width=True):
        with st.spinner("Считаем… сбор признаков через OSM (~10–30с)"):
            status, data, err = call_forward(api_base_url, payload, timeout_s)
        st.session_state.last_result = {
            "status": status, "data": data, "err": err, "payload": payload,
        }


# ----- Правая колонка: результат -----
with col_right:
    with st.container(border=True):
        st.subheader("📊 Результат")
        res = st.session_state.last_result

        if res is None:
            st.markdown(
                '<div class="geo-empty"><div class="ico">🗺️</div>'
                "<p>Заполните параметры слева и нажмите<br><b>«Рассчитать популярность»</b></p></div>",
                unsafe_allow_html=True,
            )
        elif res["status"] != 200:
            status, err = res["status"], res["err"]
            msg = {
                0: "Сеть недоступна — проверь, что API запущен.",
                400: "Некорректный запрос (bad request).",
                403: "Модель не смогла обработать данные (геокодер / OSM / признаки).",
            }.get(status, f"Ошибка {status}.")
            st.error(f"**{msg}**\n\n{err or ''}")
        else:
            data = res["data"] or {}
            payload_used = res.get("payload") or {}
            pop = data.get("popularity_index")
            seg = data.get("segment")
            coords = data.get("coords") or {}
            plat, plon = coords.get("lat"), coords.get("lon")
            addr = build_display_address(payload_used, coords)
            bank_out = payload_used.get("bank_name") or "—"

            tab_res, tab_map = st.tabs(["Итог", "🗺️ Карта"])

            with tab_res:
                g1, g2 = st.columns([1.1, 1])
                with g1:
                    if pop is not None:
                        st.plotly_chart(gauge_figure(pop, seg), use_container_width=True,
                                        config={"displayModeBar": False})
                with g2:
                    st.metric("Индекс популярности", fmt_float(pop, 4))
                    st.markdown("**Сегмент**", help="Высокий = перспективная локация")
                    st.markdown(segment_badge_html(seg), unsafe_allow_html=True)

                st.markdown(fact_html("🏦 Банк", bank_out), unsafe_allow_html=True)
                st.markdown(fact_html("📍 Адрес", addr), unsafe_allow_html=True)
                st.markdown(
                    fact_html("🧭 Координаты", f"{fmt_float(plat, 6)}, {fmt_float(plon, 6)}"),
                    unsafe_allow_html=True,
                )

            with tab_map:
                if plat is None or plon is None:
                    st.warning("В ответе нет координат — карту не построить.")
                else:
                    m = folium.Map(
                        location=[plat, plon], zoom_start=16, control_scale=True,
                        tiles="CartoDB positron",
                    )
                    color = SEG_FOLIUM.get((seg or "").lower(), "blue")
                    folium.Circle(
                        location=[plat, plon], radius=300, color=color,
                        weight=1.5, fill=True, fill_opacity=0.08,
                        tooltip="Зона охвата POI — 300 м",
                    ).add_to(m)
                    popup_html = (
                        f'<div style="font-size:13px;">'
                        f"<b>Банкомат · {bank_out}</b><br>"
                        f"<b>Индекс:</b> {fmt_float(pop, 4)}<br>"
                        f"<b>Сегмент:</b> {segment_ru(seg)}<br>"
                        f"<b>Адрес:</b> {addr}</div>"
                    )
                    folium.Marker(
                        location=[plat, plon],
                        tooltip=f"Индекс {fmt_float(pop, 4)} · {segment_ru(seg)}",
                        popup=folium.Popup(popup_html, max_width=320),
                        icon=folium.Icon(color=color, icon="credit-card", prefix="fa"),
                    ).add_to(m)
                    st_folium(m, use_container_width=True, height=460)
