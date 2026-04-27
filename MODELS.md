# GeoATM Popularity - Описание моделей

### LinearRegression (`models/linear_model_best.pkl`)

CPU, препроцессинг через `ColumnTransformer` (StandardScaler + OrdinalEncoder; дистанции - log1p).

**Базовая модель** (без регуляризации, sklearn `LinearRegression`)

- Fit time: ~0.04 s

**Ridge** - `RidgeCV(alphas=[0.001…1000], cv=5)` → лучший alpha = **0.01** (почти OLS, L2 не даёт прироста)

**Lasso** - `GridSearchCV(alphas=logspace(−5, 2), cv=5)` → лучший alpha = **0.00001** (1 из 72 признаков обнулён, по качеству ≈ LR)

- Все три модели дают CV RMSE ≈ 0.0573 ± 0.0011 - регуляризация при текущих признаках не улучшает baseline.
- Топ-признаки: `is_federal_city`, `cash_out`, `transfer_a2a` - бинарные ATM-фичи доминируют над геопризнаками.
- **Сохранена базовая LinearRegression** как лучшая среди трёх (RMSE 0.0591, R² 0.5406).

---

### Decision Tree (`models/dt_best.joblib`)

CPU, sklearn `DecisionTreeRegressor`, препроцессинг через `ColumnTransformer` (медианный imputer + OrdinalEncoder - без OHE, деревьям хватает порядкового кодирования).

**Базовая модель** (`max_depth=8, min_samples_leaf=20, min_samples_split=40`)

**Optuna** (80 триалов, 5-fold CV на train; early stopping для деревьев отсутствует)


| Гиперпараметр       | Диапазон поиска                       | Лучшее значение |
| ------------------- | ------------------------------------- | --------------- |
| `max_depth`         | целое [3, 20]                         | -               |
| `min_samples_leaf`  | целое [1, 50]                         | -               |
| `min_samples_split` | целое [2, 300]                        | -               |
| `max_features`      | ['sqrt', 'log2', 0.3, 0.5, 0.8, None] | -               |
| `criterion`         | ['squared_error', 'friedman_mse']     | -               |


- Финал переобучен на train+val и сохранён в Pipeline вместе с препроцессором.

---

### Random Forest (`models/rf_best.joblib`)

CPU, sklearn `RandomForestRegressor` (n_jobs=-1), те же препроцессинг и сплит что у DT.

**Базовая модель** (`n_estimators=300, max_depth=None, min_samples_leaf=5`)

**Optuna** (100 триалов, 5-fold CV на train)


| Гиперпараметр           | Диапазон поиска                              | Лучшее значение |
| ----------------------- | -------------------------------------------- | --------------- |
| `n_estimators`          | целое [100, 800]                             | -               |
| `max_depth`             | целое [5, 40]                                | -               |
| `min_samples_leaf`      | целое [1, 20]                                | -               |
| `min_samples_split`     | целое [2, 20]                                | -               |
| `max_features`          | равномерное [0.1, 1.0]                       | -               |
| `bootstrap`             | [True, False]                                | -               |
| `max_samples`           | равномерное [0.5, 1.0] (если bootstrap=True) | -               |
| `min_impurity_decrease` | log-равномерное [1e-7, 1e-3]                 | -               |


- CV на train+val: 0.0449 ± 0.0011 - стабильно.

---

### CatBoost (`models/catboost_best.cbm`)

GPU, симметричные деревья, нативная поддержка категорий (кодирование не требуется).

**Базовая модель** (`iterations=1000, lr=0.05, depth=6, l2_leaf_reg=3.0`, ранняя остановка 50)

- Лучшая итерация: 489

**Optuna** (100 триалов, оптимизация по RMSE на валидации)


| Гиперпараметр         | Диапазон поиска              | Лучшее значение |
| --------------------- | ---------------------------- | --------------- |
| `learning_rate`       | log-равномерное [0.01, 0.15] | 0.0713          |
| `depth`               | целое [4, 10]                | 10              |
| `l2_leaf_reg`         | равномерное [1.0, 10.0]      | 4.41            |
| `bagging_temperature` | равномерное [0.0, 1.0]       | 0.679           |
| `random_strength`     | равномерное [0.0, 2.0]       | 1.12            |
| `border_count`        | целое [32, 128]              | 83              |


- Лучшая итерация: 232 (финал переобучен на train+val за 233 итерации)

---

### XGBoost (`models/xgboost_best.json`)

GPU (`tree_method=hist, device=cuda`), препроцессинг через `ColumnTransformer` (StandardScaler + OrdinalEncoder).

**Базовая модель** (`num_boost_round=1000, lr=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8`, ранняя остановка 50)

- Лучшая итерация: 249

**Optuna** (100 триалов) - не улучшил базовую модель на тесте; **сохранена базовая модель**.


| Гиперпараметр       | Диапазон поиска              | Лучшее значение |
| ------------------- | ---------------------------- | --------------- |
| `learning_rate`     | log-равномерное [0.01, 0.08] | 0.0462          |
| `max_depth`         | целое [3, 10]                | 10              |
| `min_child_weight`  | целое [1, 20]                | 8               |
| `subsample`         | равномерное [0.5, 1.0]       | 0.566           |
| `colsample_bytree`  | равномерное [0.5, 1.0]       | 0.774           |
| `colsample_bylevel` | равномерное [0.5, 1.0]       | 0.801           |
| `reg_alpha`         | log-равномерное [1e-8, 10]   | 0.0032          |
| `reg_lambda`        | log-равномерное [1e-8, 10]   | 0.0173          |
| `gamma`             | равномерное [0.0, 5.0]       | 0.007           |


---

### LightGBM (`models/lightgbm_best.txt`)

CPU, нативная поддержка категорий через тип `category`.

**Базовая модель** (`n_estimators=2000, lr=0.05, max_depth=6, num_leaves=63`, ранняя остановка 50)

- Лучшая итерация: 259

**Optuna** (100 триалов, оптимизация по RMSE на валидации)


| Гиперпараметр       | Диапазон поиска              | Лучшее значение |
| ------------------- | ---------------------------- | --------------- |
| `learning_rate`     | log-равномерное [0.01, 0.15] | 0.0359          |
| `num_leaves`        | целое [20, 200]              | 200             |
| `max_depth`         | целое [3, 12]                | 10              |
| `min_child_samples` | целое [5, 50]                | 49              |
| `subsample`         | равномерное [0.6, 1.0]       | 0.986           |
| `colsample_bytree`  | равномерное [0.6, 1.0]       | 0.970           |
| `reg_alpha`         | log-равномерное [1e-8, 10]   | 7.25e-05        |
| `reg_lambda`        | log-равномерное [1e-8, 10]   | 4.43e-08        |


- Лучшая итерация: 428 (финал переобучен на train+val за 428 итераций)

---

## Результаты на тестовой выборке


| Модель                    | RMSE       | R²         | Файл                           |
| ------------------------- | ---------- | ---------- | ------------------------------ |
| LinearRegression          | 0.0591     | 0.5406     | `models/linear_model_best.pkl` |
| TabNet (базовая)          | 0.0500     | 0.6647     |                                |
| Decision Tree (базовая)   | 0.0493     | 0.6742     |                                |
| TabNet (Optuna)           | 0.0491     | 0.6766     | `models/tabnet_best.pt`        |
| Decision Tree (Optuna)    | 0.0484     | 0.6858     | `models/dt_best.joblib`        |
| CatBoost (базовая)        | 0.0455     | 0.7223     |                                |
| CatBoost (Optuna)         | 0.0453     | 0.7249     | `models/catboost_best.cbm`     |
| LightGBM (базовая)        | 0.0452     | 0.7259     |                                |
| XGBoost (базовая)         | 0.0451     | 0.7276     | `models/xgboost_best.json`     |
| **LightGBM (Optuna)**     | **0.0448** | **0.7307** | `models/lightgbm_best.txt`     |
| Random Forest (базовая)   | 0.0446     | 0.7327     |                                |
| Random Forest (Optuna) ⚠️ | 0.0444     | 0.7359     | `models/rf_best.joblib`        |


> **Лучшая по надёжности: LightGBM (Optuna)** - RMSE 0.0448, R² 0.7307.  
> ⚠️ RF Optuna формально ниже по RMSE (0.0444), но train R² ≈ 0.94 vs test R² ≈ 0.74 - сильный gap; LightGBM стабильнее и предпочтительнее для продакшна.  

