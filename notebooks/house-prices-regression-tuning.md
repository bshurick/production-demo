# Production model example

This uses a [Kaggle dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
to create a simple house price predictive model.

To goal of this notebook is to come up with optimal LightGBM parameters to use for production model training.


```python
# data
import hashlib
import pandas as pd
import numpy as np

# parameter tuning
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

# model
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from joblib import dump

# sampling
from scipy.stats import uniform, randint

# custom objects 
from production_demo import CategoriesTransformer, OUTPUT

CATEGORIES = [
    "BldgType",
    "CentralAir",
    "Electrical",
    #"ExterCond",
    #"ExterQual",
    "Fence",
    "FireplaceQu",
    "Foundation",
    "Functional",
    #"GarageCond",
   # "GarageQual",
    "GarageType",
    "Heating",
    "HeatingQC",
    "HouseStyle",
    #"KitchenQual",
    "LotConfig",
    "MasVnrType",
    "MSSubClass",
    "PavedDrive",
    "RoofStyle",
]

NUMERICS = [
    "1stFlrSF",
    "2ndFlrSF",
    "BedroomAbvGr",
    "EnclosedPorch",
    "Fireplaces",
    "FullBath",
    "GarageArea",
    "GarageCars",
    "GrLivArea",
    "HalfBath",
    "KitchenAbvGr",
    "LotArea",
    "OpenPorchSF",
    #"OverallCond",
    #"OverallQual",
    "PoolArea",
    "TotRmsAbvGrd",
    "TotalBsmtSF",
    "WoodDeckSF",
    "YearBuilt",
    "YearRemodAdd",
]
```

### Dataprep


```python
train = pd.read_csv('../data/train.csv')
print(train.shape)
print(train[CATEGORIES[:5]].head().to_markdown())
```

    (1460, 81)
    |    | BldgType   | CentralAir   | Electrical   |   Fence | FireplaceQu   |
    |---:|:-----------|:-------------|:-------------|--------:|:--------------|
    |  0 | 1Fam       | Y            | SBrkr        |     nan | nan           |
    |  1 | 1Fam       | Y            | SBrkr        |     nan | TA            |
    |  2 | 1Fam       | Y            | SBrkr        |     nan | TA            |
    |  3 | 1Fam       | Y            | SBrkr        |     nan | Gd            |
    |  4 | 1Fam       | Y            | SBrkr        |     nan | TA            |



```python
# hasher 
hct = CategoriesTransformer(CATEGORIES)


# prepare train/test splitting
train.sort_values(by=['YrSold', 'MoSold'], 
                  inplace=True)
tss = TimeSeriesSplit(n_splits=5)


# parameter space
param_distributions = dict(
    LGBM__num_leaves=randint(2, 5000),
    LGBM__max_depth=randint(2, 20),
    LGBM__learning_rate=uniform(0.01, 0.9),
    LGBM__n_estimators=randint(5, 1000),
    LGBM__min_split_gain=uniform(0.0, 0.1),
    LGBM__min_child_weight=uniform(0.0, 0.1),
    LGBM__subsample=uniform(0.1, 0.9),
    LGBM__colsample_bytree=uniform(0.1, 0.9),
    LGBM__reg_alpha=uniform(0.0, 5000.0),
    LGBM__reg_lambda=uniform(0.0, 5000.0),
)
```


```python
print(train[CATEGORIES[:5]].head().to_markdown())
```

    |    | BldgType   | CentralAir   | Electrical   |   Fence | FireplaceQu   |
    |---:|:-----------|:-------------|:-------------|--------:|:--------------|
    |  0 | 1Fam       | Y            | SBrkr        |     nan | nan           |
    |  1 | 1Fam       | Y            | SBrkr        |     nan | TA            |
    |  2 | 1Fam       | Y            | SBrkr        |     nan | TA            |
    |  3 | 1Fam       | Y            | SBrkr        |     nan | Gd            |
    |  4 | 1Fam       | Y            | SBrkr        |     nan | TA            |


### Features subset

We're subsetting features here based on what we will have **at time of prediction**; in other words, not all 80+ features from training are going to be available to us at prediction time, or we want to make it easier to fill out a form to on our web page to make a prediction. We are saying that we will only *require* the below features in order to make a prediction. 


```python
print(f'Categories used:\n{CATEGORIES}')
print(f'\nNumerics used:\n{NUMERICS}')
print(f'\n Total features used: {len(CATEGORIES) + len(NUMERICS)}')
```

    Categories used:
    ['BldgType', 'CentralAir', 'Electrical', 'Fence', 'FireplaceQu', 'Foundation', 'Functional', 'GarageType', 'Heating', 'HeatingQC', 'HouseStyle', 'LotConfig', 'MasVnrType', 'MSSubClass', 'PavedDrive', 'RoofStyle']
    
    Numerics used:
    ['1stFlrSF', '2ndFlrSF', 'BedroomAbvGr', 'EnclosedPorch', 'Fireplaces', 'FullBath', 'GarageArea', 'GarageCars', 'GrLivArea', 'HalfBath', 'KitchenAbvGr', 'LotArea', 'OpenPorchSF', 'PoolArea', 'TotRmsAbvGrd', 'TotalBsmtSF', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd']
    
     Total features used: 35


### Parameter tuning


```python
model = Pipeline([
    ('hash', hct),
    ('LGBM', LGBMRegressor(random_state=22)),
])
rsv = RandomizedSearchCV(estimator=model,
                         param_distributions=param_distributions,
                         n_iter=10000,
                         cv=tss,
                         n_jobs=-1,
                         scoring='neg_mean_squared_log_error')
_ = rsv.fit(train[NUMERICS + CATEGORIES], train[OUTPUT])
```

    /Users/brandonshurick/ProdDemo/production-demo/lib/python3.9/site-packages/joblib/externals/loky/process_executor.py:702: UserWarning: A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.
      warnings.warn(



```python
best_params_dict = rsv.best_params_

print(f'Best params:\n {best_params_dict}')
print(f'\nBest score:\n {rsv.best_score_:.4f}')

# save 
model = LGBMRegressor(**best_params_dict)
```

    Best params:
     {'LGBM__colsample_bytree': 0.4364670911371453, 'LGBM__learning_rate': 0.769420294059774, 'LGBM__max_depth': 7, 'LGBM__min_child_weight': 0.01822587243541679, 'LGBM__min_split_gain': 0.03308747951729114, 'LGBM__n_estimators': 600, 'LGBM__num_leaves': 921, 'LGBM__reg_alpha': 286.5027269365633, 'LGBM__reg_lambda': 2429.7823135073622, 'LGBM__subsample': 0.7760855497537651}
    
    Best score:
     -0.0278


## Train


```python
model.fit(train[NUMERICS], train[OUTPUT])
```




    LGBMRegressor(LGBM__colsample_bytree=0.4364670911371453,
                  LGBM__learning_rate=0.769420294059774, LGBM__max_depth=7,
                  LGBM__min_child_weight=0.01822587243541679,
                  LGBM__min_split_gain=0.03308747951729114, LGBM__n_estimators=600,
                  LGBM__num_leaves=921, LGBM__reg_alpha=286.5027269365633,
                  LGBM__reg_lambda=2429.7823135073622,
                  LGBM__subsample=0.7760855497537651)




```python
#save model artifacts
dump(model, '../data/trained_model')
```




    ['../data/trained_model']


