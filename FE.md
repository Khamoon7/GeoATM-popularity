## 1. Признаки, которые можно допарсить из открытых источников (OSM, Росстат, open-data городов)

### 1.1. Городская среда и застройка (радиус 100–300 м)
- buildings_count_100m — количество зданий в радиусе 100 м  
- buildings_count_300m — количество зданий в радиусе 300 м  
- avg_building_levels_100m — средняя этажность зданий в 100 м  
- avg_building_levels_300m — средняя этажность зданий в 300 м  
- residential_buildings_share_300m — доля жилых зданий  
- commercial_buildings_share_300m — доля коммерческих зданий  
- industrial_buildings_share_300m — доля промышленных зданий  
- landuse_residential_share_300m — доля жилой landuse  
- landuse_commercial_share_300m — доля коммерческой landuse  
- landuse_industrial_share_300m — доля индустриальной landuse  
- landuse_mixed_flag — смешанная зона  
- area_type_osm ∈ {residential, commercial, industrial, mixed}

### 1.2. Дорожная сеть и транспорт (OSM)
- primary_roads_count_300m — количество магистралей  
- secondary_roads_count_300m — второстепенные дороги  
- road_intersections_count_100m — число перекрёстков  
- distance_to_primary_road_m — расстояние до ближайшей магистрали  
- crosswalks_count_100m — пешеходные переходы  
- footways_count_100m — пешие дорожки  

### 1.3. Дополнительные типы POI (OSM)
- count_offices_300m / 500m — офисы  
- count_banks_branches_300m — отделения банков  
- count_sport_300m — спорт  
- count_entertainment_300m — кино/ТРК  
- count_hotels_300m — гостиницы  
- count_attractions_500m — достопримечательности  
- office_area_flag — офисный район  
- tourist_area_flag — туристический район

### 1.4. Соц-дем и экономика (Росстат)
- median_income_region — доход  
- unemployment_rate_region — безработица  
- share_pensioners_region — пенсионеры  
- share_youth_region — молодёжь  
- urbanization_level_region — урбанизация  
- income_region_bucket — бины  
- pensioner_share_bucket  
- youth_share_bucket

### 1.5. Макро-показатели города
- city_population — население  
- city_area_km2 — площадь  
- city_population_density — плотность  
- city_is_touristic_flag  
- city_size_bucket ∈ {small, medium, large, mega}  
- city_tourism_score  

---

## 2. Признаки, которые можно построить из текущих колонок

### 2.1. Услуги банкомата
- services_cnt — количество услуг  
- payments_services_cnt — платёжные функции  
- has_foreign_currency — есть валюта  
- is_full_functional_atm — полнофункциональный банкомат

### 2.2. Нормализация на плотность населения
- banks_atms_per_density_300m  
- supermarkets_per_density_300m  
- malls_per_density_500m  
- poi_total_300m — общий индекс POI  
- poi_per_density_300m — нормировано на плотность

### 2.3. Внутренний vs внешний круг
- supermarkets_inner_share  
- cafes_inner_share  
- restaurants_inner_share  
- banks_inner_share  

### 2.4. Агрегаты по типам активности
- food_100m / 300m / 500m  
- everyday_300m  
- education_100m / 300m  
- transport_300m / 500m  
- services_300m  
- food_share_300m  
- transport_share_300m  
- education_share_300m  

### 2.5. Признаки по расстояниям
- has_mall_nearby, has_supermarket_nearby, has_subway_nearby  
- nearest_malls_dist_log и подобные  
- subway_dist_bucket ∈ {'<100','100-300','300-700','>700','no_metro'}
