import json
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium

SESSION = requests.Session()
SESSION.trust_env = False

col1, col2, col3 = st.columns([2, 3, 1])
with col2:
    st.image("https://img.ridus.ru/images/2020/9/16/1149775/in_article_563225e9af.webp", width=650)


st.set_page_config(page_title="GeoATM Popularity", layout="wide")

if "last_result" not in st.session_state:
    st.session_state.last_result = None

st.title("🏧 Сервис прогнозирования популярности локаций для размещения банкоматов")
st.markdown(
    """
Это приложение помогает оценить **популярность банкомата в выбранной точке**.

Сервис анализирует расположение и доступные функции банкомата и рассчитывает
**индекс популярности**, а также относит точку к одному из сегментов
(низкий, средний или высокий).

**Как пользоваться:**
1) Укажи **адрес** или **координаты** предполагаемого размещения банкомата  
2) Выбери **банк** и отметь доступные **функции банкомата**  
3) Нажми **«Рассчитать популярность»** и посмотри результат на карте
"""
)

st.sidebar.header("Настройки")
api_base_url = st.sidebar.text_input("Base URL API", value="http://127.0.0.1:8000").rstrip("/")
timeout_s = st.sidebar.number_input("Timeout (сек)", min_value=1, max_value=120, value=60, step=1)

st.sidebar.markdown("---")
st.sidebar.caption("Эндпоинт: POST /forward")


def build_payload(
    *,
    use_address: bool,
    address: str | None,
    lat: float | None,
    lon: float | None,
    bank_name: str | None,
    flags: dict,
) -> dict:
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



def call_forward(payload: dict) -> tuple[int, dict | None, str | None, str | None]:
    url = f"{api_base_url}/forward"
    try:
        r = SESSION.post(url, json=payload, timeout=float(timeout_s))
    except requests.RequestException as e:
        return 0, None, f"Network error: {e}", None

    raw_text = r.text

    try:
        data = r.json() if r.text else None
    except Exception:
        data = None

    if r.status_code >= 400:
        detail = None
        if isinstance(data, dict) and "detail" in data:
            detail = data["detail"]
        else:
            detail = raw_text[:800] if raw_text else None
        return r.status_code, data, (str(detail) if detail is not None else "Unknown error"), raw_text

    return r.status_code, data, None, raw_text


def fmt_float(x: float | None, ndigits: int = 5) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return str(x)

def segment_ru(seg: str | None) -> str:
    if not seg:
        return "—"
    m = {
        "high": "высокий",
        "medium": "средний",
        "low": "низкий",
    }
    return m.get(seg.lower(), seg)

def segment_color(seg: str | None) -> str:
    if not seg:
        return "blue"
    seg = seg.lower()
    return {"high": "red", "medium": "orange", "low": "green"}.get(seg, "blue")

def build_display_address(payload: dict, coords: dict) -> str:
    addr = payload.get("address")
    if addr:
        return addr
    lat = coords.get("lat")
    lon = coords.get("lon")
    if lat is not None and lon is not None:
        return f"{fmt_float(lat, 6)}, {fmt_float(lon, 6)}"
    return "—"


col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("1) Локация")

    mode = st.radio("Как задаём локацию?", ["Адрес", "Координаты"], horizontal=True)
    use_address = (mode == "Адрес")

    address = None
    lat = None
    lon = None

    if use_address:
        address = st.text_input("Адрес", value="Москва, Тверская 1")
        st.caption("Адрес и lat/lon взаимоисключающие.")
    else:
        lat = st.number_input("lat", value=55.7558, format="%.6f")
        lon = st.number_input("lon", value=37.6173, format="%.6f")
        st.caption("Координаты должны быть заданы парой.")

    st.subheader("2) Данные банкомата")

    bank_options = [
        "ВТБ",
        "Альфа-Банк",
        "Росбанк",
        "Россельхозбанк",
        "Газпромбанк",
        "Ак Барс Банк",
        "Уралсиб",
        "Другое",
    ]
    bank_choice = st.selectbox("Банк", bank_options, index=0)

    bank_name_custom = ""
    if bank_choice == "Другое":
        bank_name_custom = st.text_input("Название банка (если 'Другое')", value="")

    bank_name = bank_name_custom.strip() if bank_choice == "Другое" else bank_choice

    st.subheader("3) Функции и особенности банкомата")

    flag_labels = {
        "is_24_7": "Круглосуточный (24/7)",
        "contactless_tech": "Бесконтактное обслуживание (NFC)",
        "qr_codes": "Поддержка QR-кодов",
        "usd_available": "Выдаёт доллары (USD)",
        "eur_available": "Выдаёт евро (EUR)",
        "cash_in": "Приём наличных (cash-in)",
        "cash_out": "Выдача наличных (cash-out)",
        "cashless_pay": "Безналичная оплата услуг",
        "account_statement": "Печать выписки по счёту",
        "access_for_disabled": "Доступен для маломобильных",
        "transfer_p2p": "Переводы P2P",
        "transfer_a2a": "Переводы счёт-счёт (A2A)",
        "loan_payments": "Платежи по кредитам",
    }

    flags = {}
    for key, label in flag_labels.items():
        flags[key] = st.checkbox(label, value=False)

    payload = build_payload(
        use_address=use_address,
        address=address,
        lat=lat,
        lon=lon,
        bank_name=bank_name,
        flags=flags,
    )

    if st.button("🚀 Рассчитать популярность", type="primary", use_container_width=True):
        status, data, err, raw_text = call_forward(payload)
        st.session_state.last_result = {
            "url": f"{api_base_url}/forward",
            "status": status,
            "data": data,
            "err": err,
            "raw_text": raw_text,
            "payload": payload,
        }


with col_right:
    st.subheader("Результат")
    res = st.session_state.last_result
    if res is None:
        st.info("Заполни поля слева и нажми **Рассчитать популярность**.")
    else:
        status = res["status"]
        data = res["data"]
        err = res["err"]
        raw_text = res["raw_text"]

        if status == 0:
            st.error(err or "Network error")
        elif status == 400:
            st.error(f"400 — bad request\n\n{err}")
        elif status == 403:
            st.error(f"403 — модель не смогла обработать данные\n\n{err}")
        elif status != 200:
            st.error(f"{status} — ошибка\n\n{err}")
        else:
            payload_used = res.get("payload") or {}
            pop = (data or {}).get("popularity_index")
            seg = (data or {}).get("segment")
            seg_ru = segment_ru(seg)

            coords = (data or {}).get("coords") or {}
            plat = coords.get("lat")
            plon = coords.get("lon")
            addr = build_display_address(payload_used, coords)
            bank_out = payload_used.get("bank_name") or "—"

            st.success("✅ Успешно")

            st.markdown("### Итог")
            st.markdown(
                f"""
            **Индекс популярности:** {fmt_float(pop, 5)}  
            **Сегмент:** {seg_ru}  
            **Банк:** {bank_out}  
            **Адрес:** {addr}  
            **Координаты:** {fmt_float(plat, 6)}, {fmt_float(plon, 6)}
            """
            )

            if plat is None or plon is None:
                st.warning("В ответе нет coords.lat/lon — карту не построить.")
            else:
                st.markdown("### Карта")

                m = folium.Map(location=[plat, plon], zoom_start=16, control_scale=True)

                popup_html = f"""
                    <div style="font-size: 14px;">
                      <b>Банкомат</b><br>
                      <b>Индекс:</b> {fmt_float(pop, 5)}<br>
                      <b>Сегмент:</b> {seg_ru}<br>
                      <b>Адрес:</b> {addr}<br>
                      <b>Коорд:</b> {fmt_float(plat, 6)}, {fmt_float(plon, 6)}
                    </div>
                    """

                folium.Marker(
                    location=[plat, plon],
                    tooltip=f"Индекс: {fmt_float(pop, 5)} | Сегмент: {seg_ru}",
                    popup=folium.Popup(popup_html, max_width=350),
                    icon=folium.Icon(
                        color=segment_color(seg),
                        icon="credit-card",
                        prefix="fa",
                    ),
                ).add_to(m)

                st_folium(m, width=900, height=520)
