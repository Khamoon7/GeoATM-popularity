# Реестр регионов и городов России 2025

## Описание файлов

### 1. regions_registry_2025.csv
Полный реестр всех 84 субъектов Российской Федерации с актуальными статистическими данными за 2024-2025 годы.

**Структура данных:**

| Колонка | Описание | Тип данных |
|---------|----------|------------|
| `region_name` | Название субъекта РФ | string |
| `population` | Численность населения | integer |
| `area_km2` | Площадь территории (км²) | float |
| `population_density` | Плотность населения (чел/км²) | float |
| `federal_district` | Федеральный округ (ЦФО, СЗФО, ЮФО, СКФО, ПФО, УФО, СФО, ДФО) | string |
| `is_federal_district_capital` | Является ли столицей федерального округа | boolean |
| `is_federal_city` | Город федерального значения | boolean |
| `macro_region` | Макрорегион (central, north, south, ural, siberia, far_east) | string |
| `grp_per_capita_2023_rub` | ВРП на душу населения 2023 (руб.) | float |
| `avg_salary_oct_2025_rub` | Средняя зарплата октябрь 2025 (руб./мес) | float |
| `avg_income_q3_2025_rub` | Среднедушевые доходы Q3 2025 (руб./мес) | float |

**Источники данных:**
- Население, площадь, плотность: исходный CSV-файл
- ВРП на душу населения 2023: Росстат, statbase.ru
- Средняя зарплата октябрь 2025: Росстат, finance.mail.ru
- Среднедушевые доходы Q3 2025: Росстат, РИА Рейтинг

---

### 2. cities_statistics_2025.csv
Статистика по 85 крупнейшим городам России с привязкой к региональным показателям.

**Структура данных:**

| Колонка | Описание | Тип данных |
|---------|----------|------------|
| `city_name` | Название города | string |
| `population` | Численность населения города | integer |
| `region_name` | Субъект РФ, к которому относится город | string |
| `city_type` | Тип города | string |
| `city_size_category` | Категория размера города | string |
| `federal_district` | Федеральный округ | string |
| `grp_per_capita_2023_rub` | ВРП на душу населения региона 2023 (руб.) | float |
| `avg_salary_oct_2025_rub` | Средняя зарплата региона октябрь 2025 (руб./мес) | float |
| `avg_income_q3_2025_rub` | Среднедушевые доходы региона Q3 2025 (руб./мес) | float |
| `macro_region` | Макрорегион | string |

**Типы городов (`city_type`):**
- `federal_city` — Город федерального значения (Москва, Санкт-Петербург, Севастополь)
- `regional_capital` — Столица субъекта РФ (областной центр, столица республики/края)
- `city` — Обычный город (не административный центр)

**Категории размера (`city_size_category`):**
- `million+` — Города-миллионники (≥ 1 000 000 чел)
- `large` — Крупные города (500 000 – 999 999 чел)
- `medium` — Средние города (100 000 – 499 999 чел)
- `small` — Малые города (< 100 000 чел)

**Источники данных:**
- Население городов: Росстат, ria.ru, mojgorod.ru
- Региональные показатели: джойн с regions_registry_2025.csv

---

## Фичи для ML-модели банкоматов

### Региональные фичи (C):

1. **region_gdp_per_capita** → `grp_per_capita_2023_rub` — ВРП на душу населения по субъектам РФ (руб.)
2. **region_income_per_capita** → `avg_income_q3_2025_rub` — Среднедушевые доходы по регионам (руб./мес)
3. **avg_salary_region** → `avg_salary_oct_2025_rub` — Средняя зарплата по регионам (руб./мес)

### Фичи городов:

4. **city_population** → `population` — Население города, где стоит банкомат
5. **city_type** → `city_type` — Тип города (federal_city, regional_capital, city)

### Производные фичи:

10. **city_size_category** → `city_size_category` — Размер города как категория:
    - `small` (< 100k)
    - `medium` (100k–500k)
    - `large` (500k–1M)
    - `million+` (≥ 1M)

11. **region_type** → Комбинация флагов:
    - `is_federal_city` — Город федерального значения (0/1)
    - `is_federal_district_capital` — Столица федерального округа (0/1)
    - `federal_district` — Федеральный округ (ЦФО, СЗФО, ЮФО, СКФО, ПФО, УФО, СФО, ДФО)
    - `macro_region` — Макрорегион (north, south, central, ural, siberia, far_east)

---

## Пример использования (pandas)

```python
import pandas as pd

# Загрузка данных
regions = pd.read_csv('regions_registry_2025.csv')
cities = pd.read_csv('cities_statistics_2025.csv')

# Пример джойна для банкоматов
# Предположим, у вас есть таблица банкоматов с колонкой 'city_name'
atms = pd.read_csv('your_atms.csv')

# Джойн банкоматов с городами
atms_with_city = atms.merge(
    cities[['city_name', 'population', 'city_type', 'city_size_category', 
            'region_name', 'federal_district', 'macro_region',
            'grp_per_capita_2023_rub', 'avg_salary_oct_2025_rub', 
            'avg_income_q3_2025_rub']],
    on='city_name',
    how='left'
)

# Если у вас только регион (без города)
atms_with_region = atms.merge(
    regions[['region_name', 'federal_district', 'macro_region',
             'grp_per_capita_2023_rub', 'avg_salary_oct_2025_rub',
             'avg_income_q3_2025_rub', 'is_federal_city', 
             'is_federal_district_capital']],
    on='region_name',
    how='left'
)

# Кодирование категориальных признаков
from sklearn.preprocessing import LabelEncoder

le_city_type = LabelEncoder()
atms_with_city['city_type_encoded'] = le_city_type.fit_transform(
    atms_with_city['city_type']
)

le_city_size = LabelEncoder()
atms_with_city['city_size_encoded'] = le_city_size.fit_transform(
    atms_with_city['city_size_category']
)
```

---

## Статистика по данным

### Регионы:
- Всего регионов: **84**
- Федеральных округов: **8**
- Городов федерального значения: **3**
- Столиц федеральных округов: **8**

### Города:
- Всего городов: **85**
- Городов-миллионников: **16**
- Крупных городов (500k–1M): **20**
- Средних городов (100k–500k): **43**
- Малых городов (< 100k): **6**

### Федеральные округа (кол-во регионов):
- ЦФО: 18
- СЗФО: 11
- ЮФО: 7
- СКФО: 7
- ПФО: 14
- УФО: 6
- СФО: 11
- ДФО: 10

---

## Примечания

1. **Данные за 2025 год** — актуальны на март 2026 года
2. **ВРП на душу населения** — последние доступные данные за 2023 год (публикуются с задержкой)
3. **Зарплаты** — данные за октябрь 2025 года (последний доступный месяц)
4. **Доходы** — данные за Q3 2025 года
5. Для регионов без точных данных использованы средние значения по федеральному округу

---

## Контакты и источники

Собрано на основе официальных данных:
- Росстат (rosstat.gov.ru)
- РИА Рейтинг (riarating.ru)
- Statbase.ru
- Finance.mail.ru
- Wikipedia (административное деление)

Дата создания: 08.03.2026
