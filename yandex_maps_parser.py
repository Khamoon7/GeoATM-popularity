#Для написания данного парсера использовался ChatGPT 5

from __future__ import annotations
import os
import pandas as pd
from typing import Dict, Iterable
import re
from typing import Optional, List
import time
import random
from dotenv import load_dotenv
import sys
from urllib.parse import quote_plus
from contextlib import suppress
from throttling import human_delay

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

load_dotenv()
USE_UC = os.getenv("USE_UC", "false").lower() == "true"

TRAIN_URL = "https://raw.githubusercontent.com/userusr/rosbank_happy_data_year/master/data/train.csv"
TRAIN_URL = "/final_for_parse_maps.csv"
OUTPUT_PATH = "output/parsed.csv"

NEEDED_COLS = ["id", "geo_lat", "geo_lon", "atm_group", "geo_address"]

ATM_GROUP_TO_BANK = {
    5478.0: "ВТБ",
    1942.0: "Альфа-Банк",
    8083.0: "РОСБАНК",
    496.5:  "Россельхозбанк",
    3185.5: "Газпромбанк",
    1022.0: "АК Барс",
    32.0:   "УРАЛСИБ БАНК",
}


def _float_env(name: str, default: float) -> float:
    """Преобразует переменную окружения в float.

Параметры:
- name: имя переменной.
- default: значение по умолчанию.

Возвращает:
float."""
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

BASE_DELAY = _float_env("BASE_DELAY", 1.5)
MIN_JITTER = _float_env("MIN_JITTER", 0.6)
MAX_JITTER = _float_env("MAX_JITTER", 1.8)

def human_delay(a: float | None = None, b: float | None = None):
    """Пауза с базовой задержкой и джиттером."""
    low = MIN_JITTER if a is None else a
    high = MAX_JITTER if b is None else b
    time.sleep(BASE_DELAY + random.uniform(low, high))

def resolve_bank_name(atm_group) -> str | None:
    """Возвращает название банка по коду группы банкомата."""
    try:
        return ATM_GROUP_TO_BANK.get(float(atm_group))
    except Exception:
        return None

def ensure_output_dir():
    """Создает директорию для выходных файлов, если её нет."""
    os.makedirs("output", exist_ok=True)

def load_train(src: str = IN_URL) -> pd.DataFrame:
    """Загружает входной CSV из пути или URL и проверяет обязательные колонки.

Параметры:
- src: путь или URL к CSV с колонками NEEDED_COLS.

Возвращает:
DataFrame только с нужными колонками.
    """
    df = pd.read_csv(src)
    missing = [c for c in NEEDED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"В train.csv нет колонок: {missing}")
    df = df[NEEDED_COLS].copy()
    return df

def append_result_row(row_dict: Dict, out_path: str = OUTPUT_PATH):
    """Добавляет одну запись в CSV результатов (append)."""
    ensure_output_dir()
    header = not os.path.exists(out_path)
    pd.DataFrame([row_dict]).to_csv(out_path, index=False, mode="a", header=header)

def append_result_batch(rows: Iterable[Dict], out_path: str = OUTPUT_PATH):
    """Добавляет батч записей в CSV результатов."""
    rows = list(rows)
    if not rows:
        return
    ensure_output_dir()
    header = not os.path.exists(out_path)
    pd.DataFrame(rows).to_csv(out_path, index=False, mode="a", header=header)

def already_processed_ids(out_path: str = OUTPUT_PATH) -> set:
    """Читает CSV результатов и возвращает множество уже обработанных id."""
    if not os.path.exists(out_path):
        return set()
    try:
        df = pd.read_csv(out_path, usecols=["id"])
        return set(df["id"].tolist())
    except Exception:
        return set()

def _clean_lines(lines: List[str]) -> List[str]:
    """Очищает многострочный текст: обрезает пробелы, удаляет пустые строки."""
    out = []
    for t in lines:
        t = (t or "").strip()
        if not t:
            continue
        # убираем служебные символы
        t = re.sub(r"\s+", " ", t)
        out.append(t)
    # уберём повторы, сохранив порядок
    seen = set(); uniq = []
    for t in out:
        if t not in seen:
            seen.add(t); uniq.append(t)
    return uniq


def parse_operations(driver, root=None) -> str:
    """
    Ищем в активной карточке блоки с перечислением операций.
    Возвращаем одной строкой через '; '.
    """
    scope = root
    if scope is None:
        # корень карточки
        for css in [
            '[data-zone-name="businessCard"]',
            '[class*="BusinessCard"]',
            '[class*="business-card-view"]',
        ]:
            els = driver.find_elements(By.CSS_SELECTOR, css)
            if els:
                scope = els[0]; break
    if scope is None:
        return ""

    # собираем тексты вокруг заголовков «Операции», «Виды операций», «Особенности»
    texts: List[str] = []
    xps = [
        './/*[contains(translate(normalize-space(.),"ОЕРАЦИВД","оерацивд"),"операци")]/following::*[self::ul or self::ol or self::div][1]//*[self::li or self::div or self::span or self::p]',
        './/*[contains(normalize-space(.),"Виды операций") or contains(normalize-space(.),"Доступные операции")]/following::*[self::ul or self::ol or self::div][1]//*[self::li or self::div or self::span or self::p]',
        './/*[contains(normalize-space(.),"Особенности")]/following::*[self::ul or self::ol or self::div][1]//*[self::li or self::div or self::span or self::p]',
    ]
    for xp in xps:
        try:
            for el in scope.find_elements(By.XPATH, xp):
                txt = (el.text or "").strip()
                if txt:
                    texts.append(txt)
        except Exception:
            continue

    items = _clean_lines(texts)
    items = [t for t in items if not re.search(r"(подробнее|ещё|развернуть|показать|скрыть)", t, flags=re.I)]
    return "; ".join(items[:30])

from typing import Dict, List

def _card_root(driver):
    """Ищет корень карточки места на странице Яндекс.Карт."""
    for css in [
        '[data-zone-name="businessCard"]',
        '[class*="BusinessCard"]',
        '[class*="business-card-view"]',
    ]:
        els = driver.find_elements(By.CSS_SELECTOR, css)
        if els:
            return els[0]
    return None

def _clean_text(s: str) -> str:
    """Возвращает очищенный видимый текст из WebElement."""
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _collect_block_text(el) -> str:
    """Собираем текст из ближайшего блока/списка."""
    txts: List[str] = []
    for xp in ['.//*', '.']:
        try:
            for node in el.find_elements(By.XPATH, xp):
                t = _clean_text(node.text)
                if t:
                    txts.append(t)
            if not txts:
                t = _clean_text(el.text)
                if t:
                    txts.append(t)
        except Exception:
            continue
        if txts:
            break
    out = _clean_text(" ".join(txts))
    # чуть укорачиваем шум: «Подробнее / Показать все» и т.п.
    out = re.sub(r"(Подробнее|Показать все|Ещё|Свернуть)\b.*", "", out, flags=re.I)
    return out

def _find_label_scope(scope, label_variants: List[str]):
    """Находим узел-лейбл по любому из вариантов (регистр игнорируем)."""
    for lab in label_variants:
        xp = f'.//*[contains(translate(normalize-space(.),"АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ","абвгдеёжзийклмнопрстуфхцчшщъыьэюя"),"{lab.lower()}")]'
        try:
            els = scope.find_elements(By.XPATH, xp)
            if els:
                return els[0]
        except Exception:
            continue
    return None

def _grab_after_label(scope, label_variants: List[str]) -> str:
    """Берём текст рядом/после заголовка секции."""
    label_el = _find_label_scope(scope, label_variants)
    if not label_el:
        return ""

    candidates = []
    try:
        # следующий логический блок
        candidates += label_el.find_elements(By.XPATH, 'following::*[self::div or self::ul or self::ol][1]')
    except Exception:
        pass
    try:
        # родительский контейнер
        p = label_el.find_element(By.XPATH, 'ancestor::*[self::section or self::div][1]')
        candidates.append(p)
    except Exception:
        pass
    # внутри того же контейнера, ближайший список/блок после лейбла
    if candidates:
        try:
            candidates += candidates[0].find_elements(By.XPATH, './/following-sibling::*[self::div or self::ul or self::ol][1]')
        except Exception:
            pass

    for c in candidates:
        txt = _collect_block_text(c)
        if txt:
            # вырезаем сам заголовок, чтобы не «дублировался»
            for lab in label_variants:
                txt = re.sub(lab, "", txt, flags=re.I)
            return _clean_text(txt)
    return ""

def parse_features(driver) -> Dict[str, object]:
    """
    Возвращает словарь колонок по вкладке «Особенности».
    Поля:
      - Валюта банкомата
      - Валюта приема
      - Виды операций
      - Бесконтактные технологии (bool)
      - QR-коды (bool)
      - Операции со счетами и картами
    """
    scope = _card_root(driver)
    if scope is None:
        return {}

    # Основные текстовые секции
    atm_currency = _grab_after_label(scope, ["Валюта банкомата"])
    accept_currency = _grab_after_label(scope, ["Валюта приема", "Валюта приёма"])
    operations_types = _grab_after_label(scope, ["Виды операций"])
    accounts_ops = _grab_after_label(scope, ["Операции со счетами и картами"])

    # Бинарные признаки
    text_all = _clean_text(scope.text).lower()

    has_contactless = bool(re.search(r"бесконтактн", text_all))
    has_qr = bool(re.search(r"\bqr\b|\bqr-код|\bqr код", text_all))

    return {
        "Валюта банкомата": atm_currency or "нет информации",
        "Валюта приема": accept_currency or "нет информации",
        "Виды операций": operations_types or "нет информации",
        "Бесконтактные технологии": has_contactless,
        "QR-коды": has_qr,
        "Операции со счетами и картами": accounts_ops or "нет информации",
    }


def build_driver(headless: bool = False):
    """Создает и настраивает Chrome WebDriver."""
    headless_env = os.getenv("HEADLESS", "false").lower() == "true"
    headless = headless or headless_env

    ua = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    )

    if USE_UC:
        import undetected_chromedriver as uc
        opts = uc.ChromeOptions()
        if headless:
            opts.add_argument("--headless=new")
        for arg in [
            "--disable-gpu", "--no-sandbox", "--start-maximized",
            "--disable-blink-features=AutomationControlled",
            "--window-size=1920,1080", f"--user-agent={ua}"
        ]:
            opts.add_argument(arg)
        driver = uc.Chrome(options=opts)
        driver.set_page_load_timeout(45)
        return driver

    opts = ChromeOptions()
    if headless:
        opts.add_argument("--headless=new")
    for arg in [
        "--disable-gpu", "--no-sandbox", "--start-maximized",
        "--disable-blink-features=AutomationControlled",
        "--window-size=1920,1080", f"--user-agent={ua}"
    ]:
        opts.add_argument(arg)
    opts.page_load_strategy = "eager"

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=opts)
    driver.set_page_load_timeout(45)
    return driver

def open_search(driver, lat: float, lon: float, bank_name: str, zoom: int = 17, query_suffix: str = " банкомат"):
    """Открывает поиск на Яндекс.Картах с заданным запросом."""
    base = "https://yandex.ru/maps/?"
    text_q = (bank_name or "") + (query_suffix or "")
    query = f"ll={lon}%2C{lat}&z={int(zoom)}&text={quote_plus(text_q)}"
    driver.get(base + query)

    _handle_popups(driver)
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, '[class*="search"]'))
    )
    human_delay(0.2, 0.4)


def _handle_popups(driver):
    """Пытается закрыть всплывающие окна (cookies, антибот, уведомления)."""
    human_delay(0.2, 0.4)
    for xp in [
        '//button[contains(text(),"Понятно")]',
        '//button[contains(text(),"Принять")]',
        '//button[contains(text(),"Согласен") or contains(text(),"Согласна")]',
    ]:
        for el in driver.find_elements(By.XPATH, xp):
            try:
                driver.execute_script("arguments[0].click();", el)
                human_delay(0.1, 0.2)
                break
            except Exception:
                pass


def get_first_snippet(driver):
    """Возвращает первый сниппет из результатов поиска."""
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((
        By.XPATH, '//*[contains(@class,"search-business-snippet-view") or contains(@class,"business-list-item") or contains(@class,"mini-card")]'
    )))
    snippets = driver.find_elements(By.CSS_SELECTOR, '[class*="search-business-snippet-view"], [class*="business-list-item"], [class*="mini-card"]')
    return snippets[0] if snippets else None


def _text(el) -> str:
    """Безопасно читает .text у WebElement."""
    try:
        return (el.text or "").strip()
    except Exception:
        return ""


def parse_first_snippet_meta(snippet) -> dict:
    """Из первого сниппета: title, address, labels(круглосуточно/больше не работает), rating."""
    # заголовок
    title = ""
    for css in ['.search-business-snippet-view__title', 'h3', 'h2', 'a[role="link"]', 'a[aria-label]']:
        try:
            t = _text(snippet.find_element(By.CSS_SELECTOR, css))
            if t:
                title = t; break
        except Exception:
            continue

    # адрес
    address = ""
    for css in ['a.search-business-snippet-view__address', '.search-business-snippet-view__address', '[class*="address"]']:
        try:
            a = _text(snippet.find_element(By.CSS_SELECTOR, css))
            if a:
                address = a; break
        except Exception:
            continue

    # 24/7 и «Больше не работает»
    big_text = _text(snippet).lower()
    is_24_7 = "круглосуточно" in big_text
    is_closed = "больше не работает" in big_text

    # рейтинг «рядом со звёздами» (строго в сниппете)
    rating = ""
    # 1) aria-label "... из 5"
    try:
        el = snippet.find_element(By.XPATH, './/*[@aria-label and contains(@aria-label,"из 5")]')
        rating = (el.get_attribute("aria-label") or "").split("из")[0].strip().replace(",", ".")
    except Exception:
        pass
    # 2) явные блоки рейтинга внутри сниппета
    if not rating:
        for css in ['[class*="rating"]', '[class*="Rating"]', '[class*="stars"]']:
            try:
                el = snippet.find_element(By.CSS_SELECTOR, css)
                txt = (el.get_attribute("aria-label") or el.text or "").strip()
                m = __import__("re").search(r'([0-5](?:[.,]\d)?)', txt)
                if m:
                    rating = m.group(1).replace(",", ".")
                    break
            except Exception:
                continue

    return {
        "Bank_Title": title or "—",
        "Snippet_Address": address or "—",
        "Is_24_7": bool(is_24_7),
        "Is_Closed": bool(is_closed),
        "Snippet_Rating": rating or "нет оценки",
    }


def click_first_snippet(driver, snippet):
    """
    Упрощённый и быстрый переход: если есть href на /org/ — идём по нему.
    Иначе — один клик по заголовку/контейнеру.
    """
    # пробуем извлечь прямой href
    for css in [
        '.search-business-snippet-view__title a[href*="/org/"]',
        '.search-business-snippet-view__head a[href*="/org/"]',
        'a[role="link"][href*="/org/"]',
        'a[href*="/org/"]',
        'a.search-business-snippet-view__address[href*="/org/"]',
        '.search-business-snippet-view__address a[href*="/org/"]',
    ]:
        try:
            a = snippet.find_element(By.CSS_SELECTOR, css)
            href = a.get_attribute("href")
            if href:
                driver.get(href)
                WebDriverWait(driver, 15).until(EC.presence_of_element_located((
                    By.CSS_SELECTOR, '[data-zone-name="businessCard"], [class*="BusinessCard"], [class*="business-card-view"]'
                )))
                human_delay(0.15, 0.3)
                return
        except Exception:
            continue

    # fallback — клик по заголовку/контейнеру
    for css in ['.search-business-snippet-view__head', '.search-business-snippet-view__title']:
        try:
            el = snippet.find_element(By.CSS_SELECTOR, css)
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
            el.click()
            WebDriverWait(driver, 15).until(EC.presence_of_element_located((
                By.CSS_SELECTOR, '[data-zone-name="businessCard"], [class*="BusinessCard"], [class*="business-card-view"]'
            )))
            human_delay(0.15, 0.3)
            return
        except Exception:
            continue

    # совсем простой клик по сниппету
    try:
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", snippet)
        ActionChains(driver).move_to_element(snippet).click().perform()
        WebDriverWait(driver, 15).until(EC.presence_of_element_located((
            By.CSS_SELECTOR, '[data-zone-name="businessCard"], [class*="BusinessCard"], [class*="business-card-view"]'
        )))
        human_delay(0.15, 0.3)
    except Exception:
        pass



def open_features_section(driver):
    """Открывает вкладку 'Особенности' (features-tab) максимально надёжно."""
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    human_delay(0.15, 0.3)

    selectors = [
        'div.tabs-select-view__title._name_features',
        'a.tabs-select-view__label[href*="tab=features"]',
        '//*[contains(@class,"_name_features")]',
        '//*[contains(@href,"tab=features")]',
        '//*[contains(normalize-space(.),"Особенности")]',
    ]

    btn = None
    for sel in selectors:
        try:
            els = driver.find_elements(By.CSS_SELECTOR, sel) if not sel.startswith("//") else driver.find_elements(By.XPATH, sel)
            if els:
                btn = els[0]
                break
        except Exception:
            continue

    if not btn:
        return

    try:
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
    except Exception:
        pass

    # предпочитаем клик по <a href="...tab=features">
    try:
        link = btn.find_element(By.CSS_SELECTOR, 'a[href]')
        driver.execute_script("arguments[0].click();", link)
    except Exception:
        try:
            driver.execute_script("arguments[0].click();", btn)
        except Exception:
            try:
                btn.click()
            except Exception:
                pass

    # короткое ожидание появления контента «Особенности»
    try:
        WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((
                By.XPATH,
                '//*[contains(translate(.,"ОЕРАЦИВД","оерацивд"),"операци") or contains(.,"QR") or contains(.,"Бесконтакт")]'
            ))
        )
    except Exception:
        pass

    human_delay(0.15, 0.3)


def process_row(driver, row: Dict) -> Dict:
    """
    1) открываем поиск
    2) если сразу открылась карточка — сразу открываем 'Особенности'
       иначе: берём ПЕРВЫЙ сниппет слева и парсим метаданные
    3) открываем вкладку «Особенности», парсим Operations
    4) возвращаем результат в унифицированном формате
    """
    rid, lat, lon, ag = row["id"], row["geo_lat"], row["geo_lon"], row["atm_group"]
    bank_name = resolve_bank_name(ag) or "UNKNOWN"

    print(f"\n-> Обработка ID={rid} ({bank_name})")

    # Шаг 1: поиск
    open_search(driver, lat=lat, lon=lon, bank_name=bank_name)

    # Проверим: открылась ли сразу карточка банка
    try:
        WebDriverWait(driver, 3).until(
            EC.presence_of_element_located((By.XPATH, '//*[contains(@class, "orgpage-header-view__header") or contains(@class, "business-card-view")]'))
        )
        print("Сразу открылась карточка банка, переходим на вкладку 'Особенности'")
        with suppress(Exception):
            open_features_section(driver)
        with suppress(Exception):
            features = parse_features(driver) or {}
        operations = parse_operations(driver) or "нет информации"

        return {
            "id": rid,
            "lat": lat,
            "long": lon,
            "atm_group": ag,
            "bank_name": bank_name,
            "Bank_Title": bank_name,
            "Snippet_Address": "—",
            "Is_24_7": False,
            "Is_Closed": False,
            "Snippet_Rating": "нет оценки",
            "Working_hours": "нет информации",
            "YaMaps_Rating": "нет оценки",
            "Operations": operations,
            "Валюта банкомата": features.get("Валюта банкомата", "нет информации"),
            "Валюта приема": features.get("Валюта приема", "нет информации"),
            "Виды операций": features.get("Виды операций", "нет информации"),
            "Бесконтактные технологии": features.get("Бесконтактные технологии", False),
            "QR-коды": features.get("QR-коды", False),
            "Операции со счетами и картами": features.get("Операции со счетами и картами", "нет информации"),
        }

    except TimeoutException:
        print("INFO После поиска открыт список объектов (не карточка). Парсим сниппет...")

    # Шаг 2: первый сниппет и его мета
    snippet = get_first_snippet(driver)
    if snippet is None:
        print("WARNING Не найден сниппет — пропуск.")
        return {
            "id": rid, "lat": lat, "long": lon, "atm_group": ag,
            "bank_name": bank_name,
            "Bank_Title": "—",
            "Snippet_Address": "—",
            "Is_24_7": False,
            "Is_Closed": False,
            "Snippet_Rating": "нет оценки",
            "Working_hours": "нет информации",
            "YaMaps_Rating": "нет оценки",
            "Operations": "нет информации",
            "Валюта банкомата": "нет информации",
            "Валюта приема": "нет информации",
            "Виды операций": "нет информации",
            "Бесконтактные технологии": False,
            "QR-коды": False,
            "Операции со счетами и картами": "нет информации",
        }

    meta = parse_first_snippet_meta(snippet)

    # Шаг 3: кликаем сниппет и открываем «Особенности»
    with suppress(Exception):
        click_first_snippet(driver, snippet)
    with suppress(Exception):
        open_features_section(driver)

    operations = parse_operations(driver) or "нет информации"

    # Шаг 4: структурные поля из вкладки «Особенности»
    features = {}
    with suppress(Exception):
        features = parse_features(driver) or {}

    working_hours = "Круглосуточно" if meta.get("Is_24_7") else "нет информации"
    ya_rating = meta.get("Snippet_Rating") or "нет оценки"

    # Итог
    return {
        "id": rid,
        "lat": lat,
        "long": lon,
        "atm_group": ag,
        "bank_name": bank_name,
        "Bank_Title": meta.get("Bank_Title", "—"),
        "Snippet_Address": meta.get("Snippet_Address", "—"),
        "Is_24_7": bool(meta.get("Is_24_7")),
        "Is_Closed": bool(meta.get("Is_Closed")),
        "Snippet_Rating": ya_rating,
        "Working_hours": working_hours,
        "YaMaps_Rating": ya_rating,
        "Operations": operations,
        "Валюта банкомата": features.get("Валюта банкомата", "нет информации"),
        "Валюта приема": features.get("Валюта приема", "нет информации"),
        "Виды операций": features.get("Виды операций", "нет информации"),
        "Бесконтактные технологии": features.get("Бесконтактные технологии", False),
        "QR-коды": features.get("QR-коды", False),
        "Операции со счетами и картами": features.get("Операции со счетами и картами", "нет информации"),
    }



def main(start: int | None = None, limit: int | None = None, skip_processed: bool = True):
    """
    start/limit — батчи;
    skip_processed=True — пропускаем id, уже записанные в output/parsed.csv.
    """
    df = load_train()
    total_len = len(df)

    if start is not None:
        if limit:
            df = df.iloc[start:start + limit]
        else:
            df = df.iloc[start:]
    elif limit:
        df = df.head(limit)

    base_index = df.index.start if hasattr(df.index, "start") else (start or 0)
    done_ids = already_processed_ids() if skip_processed else set()

    driver = build_driver(headless=False)
    try:
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            rid = row["id"]
            if skip_processed and rid in done_ids:
                print(f"[skip] #{i} id={rid} уже в parsed.csv")
                continue

            print(f"[{i}/{len(df)} | global≈{base_index + i}] обрабатываю id={rid}, atm_group={row['atm_group']}")
            try:
                result = process_row(driver, row)
            except Exception as e:
                result = {
                    "id": row["id"], "lat": row["geo_lat"], "long": row["geo_lon"],
                    "atm_group": row["atm_group"], "bank_name": resolve_bank_name(row["atm_group"]) or "UNKNOWN",
                    "Bank_Title": "—",
                    "Snippet_Address": "—",
                    "Is_24_7": False,
                    "Is_Closed": False,
                    "Snippet_Rating": "нет оценки",
                    "Working_hours": "нет информации",
                    "YaMaps_Rating": "нет оценки",
                    "Operations": f"ошибка: {type(e).__name__}",
                }
                print(f"ошибка: {type(e).__name__}: {e}")

            append_result_row(result)  # сохраняем CSV после каждого объекта
            human_delay(0.2, 0.5)
    finally:
        with suppress(Exception):
            driver.quit()



if __name__ == "__main__":
    arg_start = int(sys.argv[1]) if len(sys.argv) > 1 else None
    arg_limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(start=arg_start, limit=arg_limit)