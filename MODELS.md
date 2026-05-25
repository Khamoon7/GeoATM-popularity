# GeoATM Popularity - Описание моделей

### LinearRegression (`models/linear_model_best.pkl`)

CPU, препроцессинг через `ColumnTransformer` (StandardScaler + OrdinalEncoder; дистанции - log1p).

**Базовая модель** (без регуляризации, sklearn `LinearRegression`)

- Fit time: ~0.04 s

**Ridge** - `RidgeCV(alphas=[0.001…1000], cv=5)` → лучший alpha = **0.1** (умеренная L2, качество ≈ LR)

**Lasso** - `GridSearchCV(alphas=logspace(−5, 2), cv=5)` → лучший alpha = **0.00001** (1 из 67 признаков обнулён, по качеству ≈ LR)

- Все три модели дают CV RMSE ≈ 0.0573 ± 0.0008 - регуляризация при текущих признаках не улучшает baseline.
- Топ-признаки: `is_federal_city`, `cash_out`, `transfer_a2a` - бинарные ATM-фичи доминируют над геопризнаками.
- **Сохранена базовая LinearRegression** как лучшая среди трёх (LR/Ridge/Lasso без poly) (RMSE 0.0596, R² 0.5235).

---

### Decision Tree (`models/dt_best.joblib`)

CPU, sklearn `DecisionTreeRegressor`, препроцессинг через `ColumnTransformer` (медианный imputer + OrdinalEncoder - без OHE, деревьям хватает порядкового кодирования).

**Базовая модель** (`max_depth=8, min_samples_leaf=20, min_samples_split=40`)

**Optuna** (80 триалов, 5-fold CV на train; early stopping для деревьев отсутствует)


| Гиперпараметр       | Диапазон поиска                        | Лучшее значение  |
| ------------------- | -------------------------------------- | ---------------- |
| `max_depth`         | целое [3, 20]                          | 12               |
| `min_samples_leaf`  | целое [1, 50]                          | 24               |
| `min_samples_split` | целое [2, 300]                         | 246              |
| `max_features`      | ['sqrt', 'log2', 0.3, 0.5, 0.8, None] | None (все)       |
| `criterion`         | ['squared_error', 'friedman_mse']      | friedman_mse     |


- Финал переобучен на train+val и сохранён в Pipeline вместе с препроцессором.

---

### Random Forest (`models/rf_best.joblib`)

CPU, sklearn `RandomForestRegressor` (n_jobs=-1), те же препроцессинг и сплит что у DT.

**Базовая модель** (`n_estimators=300, max_depth=None, min_samples_leaf=5`)

**Optuna** (100 триалов, 5-fold CV на train)


| Гиперпараметр           | Диапазон поиска                              | Лучшее значение |
| ----------------------- | -------------------------------------------- | --------------- |
| `n_estimators`          | целое [100, 800]                             | 371             |
| `max_depth`             | целое [5, 40]                                | 38              |
| `min_samples_leaf`      | целое [1, 20]                                | 2               |
| `min_samples_split`     | целое [2, 20]                                | 17              |
| `max_features`          | равномерное [0.1, 1.0]                       | 0.347           |
| `bootstrap`             | [True, False]                                | False           |
| `max_samples`           | равномерное [0.5, 1.0] (если bootstrap=True) | — (bootstrap=False) |
| `min_impurity_decrease` | log-равномерное [1e-7, 1e-3]                 | 1.31e-7         |


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

### MLP (`models/mlp_baseline.pt`, `models/mlp_optuna.pt`)

GPU, PyTorch, препроцессинг вручную (медианный imputer + log1p для расстояний + StandardScaler + OrdinalEncoder).

Архитектура: `Input → BatchNorm → [Linear → BatchNorm → ReLU → Dropout] × N → Linear → Output`

Входной BatchNorm нормализует масштаб признаков. Dropout в каждом блоке. Выходной слой без активации. Оптимизатор: Adam + ReduceLROnPlateau (×0.5 при стагнации 10 эпох). Early stopping по val RMSE (patience=20).

**Базовая модель** (`hidden_dims=[256, 128, 64], dropout=0.3, lr=1e-3, weight_decay=1e-4`)

- Обучался все 200 эпох (early stopping не сработал), лучший val RMSE = 0.0484

**Optuna** (50 триалов, оптимизация по val RMSE; внутри каждого триала 100 эпох + early stopping patience=10)


| Гиперпараметр  | Диапазон поиска              | Лучшее значение |
| -------------- | ---------------------------- | --------------- |
| `n_layers`     | целое [1, 4]                 | 1               |
| `hidden_i`     | [64, 128, 256, 512]          | 128             |
| `dropout`      | равномерное [0.1, 0.5]       | 0.209           |
| `lr`           | log-равномерное [1e-4, 1e-2] | 4.09e-3         |
| `weight_decay` | log-равномерное [1e-5, 1e-3] | 3.87e-4         |


- Лучшая архитектура: один скрытый слой [128] - задача не потребовала глубокой сети
- Early stopping сработал на эпохе 166, лучший val RMSE = 0.0477

---

### MLP ResNet-like (`models/mlp_resnet_baseline.pt`, `models/mlp_resnet_optuna.pt`)

GPU, PyTorch, тот же препроцессинг что у MLP.

Архитектура: `Input → Linear(in→hidden) → BN → ReLU → [ResidualBlock] × N → Linear → Output`

Каждый ResidualBlock: `Linear → BN → ReLU → Dropout → Linear → BN + skip (identity)`. Skip-connection позволяет блоку вырождаться в тождественный при F(x)≈0, что устраняет проблему затухания градиентов при N>3. Оптимизатор: Adam + ReduceLROnPlateau (×0.5 при стагнации 10 эпох). Early stopping по val RMSE (patience=20).

**Базовая модель** (`hidden_dim=128, n_blocks=3, dropout=0.3, lr=1e-3, weight_decay=1e-4`)

- Обучался все 200 эпох (early stopping не сработал), лучший val RMSE = 0.0542

**Optuna v1** (50 триалов, 100 эпох на триал + early stopping patience=10)


| Гиперпараметр  | Диапазон поиска              | Лучшее значение |
| -------------- | ---------------------------- | --------------- |
| `hidden_dim`   | [64, 128, 256, 512]          | 64              |
| `n_blocks`     | целое [1, 5]                 | 4               |
| `dropout`      | равномерное [0.1, 0.5]       | 0.408           |
| `lr`           | log-равномерное [1e-4, 1e-2] | 7.77e-3         |
| `weight_decay` | log-равномерное [1e-5, 1e-3] | 2.9e-4          |


- Early stopping на эпохе 115, лучший val RMSE = 0.0477
- **Сохранена как финальная модель**: `models/mlp_resnet_optuna.pt` (обучена только на train, без дообучения на train+val)

**Optuna v2** (80 триалов, 150 эпох на триал + early stopping patience=15; HyperbandPruner)

Расширено пространство поиска: добавлены `batch_size`, выбор оптимизатора (Adam/AdamW), Huber loss, `grad_clip`; scheduler заменён на CosineAnnealingLR.


| Гиперпараметр  | Диапазон поиска               | Лучшее значение |
| -------------- | ----------------------------- | --------------- |
| `hidden_dim`   | [64, 128, 192, 256, 384, 512] | 128             |
| `n_blocks`     | целое [2, 8]                  | 4               |
| `dropout`      | равномерное [0.05, 0.6]       | 0.394           |
| `lr`           | log-равномерное [5e-5, 5e-2]  | 3.38e-3         |
| `weight_decay` | log-равномерное [1e-6, 1e-2]  | 1.66e-4         |
| `batch_size`   | [128, 256, 512]               | 128             |
| `optimizer`    | [Adam, AdamW]                 | Adam            |
| `use_huber`    | [False, True]                 | True            |
| `grad_clip`    | равномерное [0.5, 5.0]        | 1.69            |


- Лучший val RMSE = 0.0470; лучший trial 16 из 80 (27 завершено, 53 pruned)
- **v2 не улучшил v1 на тесте**: v2 train 0.0502, v2 train+val 0.0523 vs v1 0.0501 - финальной остаётся v1

---

### TabNet (`models/tabnet_best.zip`)

GPU, pytorch-tabnet, препроцессинг вручную (медианный imputer + OrdinalEncoder; категориальные признаки передаются как целочисленные индексы через `cat_idxs`/`cat_dims`).

Архитектура: последовательное внимание по N_steps шагам (Attentive Transformer + Feature Transformer на каждом шаге). Маска внимания типа `sparsemax`, `n_independent=2, n_shared=2, cat_emb_dim=3`. Scheduler: StepLR (step=30, γ=0.9). Early stopping patience=50, max_epochs=500, batch_size=1024.

**Базовая модель** (`n_d=n_a=32, n_steps=5, gamma=1.3, lambda_sparse=1e-3, lr=2e-2`)

**Optuna** (30 триалов, оптимизация по val RMSE)


| Гиперпараметр    | Диапазон поиска              | Лучшее значение |
| ---------------- | ---------------------------- | --------------- |
| `n_d` (=`n_a`)   | целое [8, 64] с шагом 8      | 16              |
| `n_steps`        | целое [3, 8]                 | 7               |
| `gamma`          | равномерное [1.0, 2.0]       | 1.436           |
| `lambda_sparse`  | log-равномерное [1e-4, 1e-2] | 3.0e-4          |
| `lr`             | log-равномерное [5e-3, 5e-2] | 0.0174          |
| `momentum`       | равномерное [0.01, 0.4]      | 0.013           |


- Лучший val RMSE = 0.0492; финальная модель переобучена на train+val за 196 эпох
- Optuna выявила устойчивый кластер: `n_d=16, n_steps=7` - умеренная ширина при большом числе шагов внимания

---

### FT-Transformer (`models/fttransformer_base.pt`, `models/fttransformer_best.pt`)

GPU, PyTorch + rtdl, тот же препроцессинг что у TabNet (числовые → float32, категориальные → int64; передаются раздельно как `(x_num, x_cat)`).

Архитектура: каждый признак токенизируется в вектор размерности `d_token` (числовые - отдельная линейная проекция, категориальные - таблица эмбеддингов). Добавляется обучаемый `[CLS]`-токен. Далее - `n_blocks` Pre-LN Transformer-блоков (LayerNorm → Multi-Head Self-Attention → Residual → LayerNorm → FFN с ReGLU → Residual). Выход `[CLS]` подаётся на линейный выход. Оптимизатор: AdamW + CosineAnnealingLR. Batch size: 256, max_epochs: 300, early stopping patience: 30.

**Базовая модель** (`d_token=192, n_blocks=3, attention_dropout=0.2, ffn_d_hidden=256, ffn_dropout=0.1, residual_dropout=0.0, lr=1e-4, weight_decay=1e-5`)

**Optuna** (30 триалов, оптимизация по val RMSE)


| Гиперпараметр         | Диапазон поиска              | Лучшее значение |
| --------------------- | ---------------------------- | --------------- |
| `d_token`             | [96, 128, 192, 256]          | 256             |
| `n_blocks`            | целое [1, 6]                 | 3               |
| `attention_dropout`   | равномерное [0.0, 0.4]       | 0.317           |
| `ffn_d_hidden_factor` | равномерное [1.0, 4.0]       | 3.607           |
| `ffn_dropout`         | равномерное [0.0, 0.3]       | 0.013           |
| `residual_dropout`    | равномерное [0.0, 0.2]       | 0.027           |
| `lr`                  | log-равномерное [5e-5, 5e-3] | 2.34e-4         |
| `weight_decay`        | log-равномерное [1e-6, 1e-3] | 1.05e-4         |


- Лучший val RMSE = 0.0455; финальная модель переобучена на train+val
- Baseline и Optuna вышли на одинаковый RMSE (0.0475) - параметры по умолчанию из статьи уже близки к оптимуму для этого датасета

---


## Результаты на тестовой выборке

| Модель                    | RMSE       | R²         | Файл                             |
|---------------------------|-----------:|-----------:|----------------------------------|
| LinearRegression          | 0.0596     | 0.5235     | `models/linear_model_best.pkl`  |
| MLP ResNet-like (базовая) | 0.0571     | 0.5623     | `models/mlp_resnet_baseline.pt` |
| MLP (базовая)             | 0.0513     | 0.6469     | `models/mlp_baseline.pt`        |
| MLP ResNet-like (Optuna)  | 0.0501     | 0.6637     | `models/mlp_resnet_optuna.pt`   |
| TabNet (базовая)          | 0.0500     | 0.6647     |                                  |
| MLP (Optuna)              | 0.0495     | 0.6710     | `models/mlp_optuna.pt`          |
| Decision Tree (базовая)   | 0.0493     | 0.6742     |                                  |
| TabNet (Optuna)           | 0.0491     | 0.6766     | `models/tabnet_best.zip`        |
| Decision Tree (Optuna)    | 0.0484     | 0.6858     | `models/dt_best.joblib`         |
| FT-Transformer (базовая)  | 0.0475     | 0.6973     | `models/fttransformer_base.pt`  |
| FT-Transformer (Optuna)   | 0.0475     | 0.6975     | `models/fttransformer_best.pt`  |
| CatBoost (базовая)        | 0.0455     | 0.7223     |                                  |
| CatBoost (Optuna)         | 0.0453     | 0.7249     | `models/catboost_best.cbm`      |
| LightGBM (базовая)        | 0.0452     | 0.7259     |                                  |
| XGBoost (базовая)         | 0.0451     | 0.7276     | `models/xgboost_best.json`      |
| **LightGBM (Optuna)**     | **0.0448** | **0.7307** | `models/lightgbm_best.txt`      |
| Random Forest (базовая)   | 0.0446     | 0.7327     |                                  |
| Random Forest (Optuna) ⚠️ | 0.0444     | 0.7359     | `models/rf_best.joblib`         |

> **Лучшая по надёжности: LightGBM (Optuna)** - RMSE 0.0448, R² 0.7307.  
> ⚠️ RF Optuna формально ниже по RMSE (0.0444), но train R² ≈ 0.94 vs test R² ≈ 0.74 - сильный gap; LightGBM стабильнее и предпочтительнее для продакшна.  

