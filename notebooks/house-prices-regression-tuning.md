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
train.head().to_markdown()
```

    (1460, 81)





    '|    |   Id |   MSSubClass | MSZoning   |   LotFrontage |   LotArea | Street   |   Alley | LotShape   | LandContour   | Utilities   | LotConfig   | LandSlope   | Neighborhood   | Condition1   | Condition2   | BldgType   | HouseStyle   |   OverallQual |   OverallCond |   YearBuilt |   YearRemodAdd | RoofStyle   | RoofMatl   | Exterior1st   | Exterior2nd   | MasVnrType   |   MasVnrArea | ExterQual   | ExterCond   | Foundation   | BsmtQual   | BsmtCond   | BsmtExposure   | BsmtFinType1   |   BsmtFinSF1 | BsmtFinType2   |   BsmtFinSF2 |   BsmtUnfSF |   TotalBsmtSF | Heating   | HeatingQC   | CentralAir   | Electrical   |   1stFlrSF |   2ndFlrSF |   LowQualFinSF |   GrLivArea |   BsmtFullBath |   BsmtHalfBath |   FullBath |   HalfBath |   BedroomAbvGr |   KitchenAbvGr | KitchenQual   |   TotRmsAbvGrd | Functional   |   Fireplaces | FireplaceQu   | GarageType   |   GarageYrBlt | GarageFinish   |   GarageCars |   GarageArea | GarageQual   | GarageCond   | PavedDrive   |   WoodDeckSF |   OpenPorchSF |   EnclosedPorch |   3SsnPorch |   ScreenPorch |   PoolArea |   PoolQC |   Fence |   MiscFeature |   MiscVal |   MoSold |   YrSold | SaleType   | SaleCondition   |   SalePrice |\n|---:|-----:|-------------:|:-----------|--------------:|----------:|:---------|--------:|:-----------|:--------------|:------------|:------------|:------------|:---------------|:-------------|:-------------|:-----------|:-------------|--------------:|--------------:|------------:|---------------:|:------------|:-----------|:--------------|:--------------|:-------------|-------------:|:------------|:------------|:-------------|:-----------|:-----------|:---------------|:---------------|-------------:|:---------------|-------------:|------------:|--------------:|:----------|:------------|:-------------|:-------------|-----------:|-----------:|---------------:|------------:|---------------:|---------------:|-----------:|-----------:|---------------:|---------------:|:--------------|---------------:|:-------------|-------------:|:--------------|:-------------|--------------:|:---------------|-------------:|-------------:|:-------------|:-------------|:-------------|-------------:|--------------:|----------------:|------------:|--------------:|-----------:|---------:|--------:|--------------:|----------:|---------:|---------:|:-----------|:----------------|------------:|\n|  0 |    1 |           60 | RL         |            65 |      8450 | Pave     |     nan | Reg        | Lvl           | AllPub      | Inside      | Gtl         | CollgCr        | Norm         | Norm         | 1Fam       | 2Story       |             7 |             5 |        2003 |           2003 | Gable       | CompShg    | VinylSd       | VinylSd       | BrkFace      |          196 | Gd          | TA          | PConc        | Gd         | TA         | No             | GLQ            |          706 | Unf            |            0 |         150 |           856 | GasA      | Ex          | Y            | SBrkr        |        856 |        854 |              0 |        1710 |              1 |              0 |          2 |          1 |              3 |              1 | Gd            |              8 | Typ          |            0 | nan           | Attchd       |          2003 | RFn            |            2 |          548 | TA           | TA           | Y            |            0 |            61 |               0 |           0 |             0 |          0 |      nan |     nan |           nan |         0 |        2 |     2008 | WD         | Normal          |      208500 |\n|  1 |    2 |           20 | RL         |            80 |      9600 | Pave     |     nan | Reg        | Lvl           | AllPub      | FR2         | Gtl         | Veenker        | Feedr        | Norm         | 1Fam       | 1Story       |             6 |             8 |        1976 |           1976 | Gable       | CompShg    | MetalSd       | MetalSd       | None         |            0 | TA          | TA          | CBlock       | Gd         | TA         | Gd             | ALQ            |          978 | Unf            |            0 |         284 |          1262 | GasA      | Ex          | Y            | SBrkr        |       1262 |          0 |              0 |        1262 |              0 |              1 |          2 |          0 |              3 |              1 | TA            |              6 | Typ          |            1 | TA            | Attchd       |          1976 | RFn            |            2 |          460 | TA           | TA           | Y            |          298 |             0 |               0 |           0 |             0 |          0 |      nan |     nan |           nan |         0 |        5 |     2007 | WD         | Normal          |      181500 |\n|  2 |    3 |           60 | RL         |            68 |     11250 | Pave     |     nan | IR1        | Lvl           | AllPub      | Inside      | Gtl         | CollgCr        | Norm         | Norm         | 1Fam       | 2Story       |             7 |             5 |        2001 |           2002 | Gable       | CompShg    | VinylSd       | VinylSd       | BrkFace      |          162 | Gd          | TA          | PConc        | Gd         | TA         | Mn             | GLQ            |          486 | Unf            |            0 |         434 |           920 | GasA      | Ex          | Y            | SBrkr        |        920 |        866 |              0 |        1786 |              1 |              0 |          2 |          1 |              3 |              1 | Gd            |              6 | Typ          |            1 | TA            | Attchd       |          2001 | RFn            |            2 |          608 | TA           | TA           | Y            |            0 |            42 |               0 |           0 |             0 |          0 |      nan |     nan |           nan |         0 |        9 |     2008 | WD         | Normal          |      223500 |\n|  3 |    4 |           70 | RL         |            60 |      9550 | Pave     |     nan | IR1        | Lvl           | AllPub      | Corner      | Gtl         | Crawfor        | Norm         | Norm         | 1Fam       | 2Story       |             7 |             5 |        1915 |           1970 | Gable       | CompShg    | Wd Sdng       | Wd Shng       | None         |            0 | TA          | TA          | BrkTil       | TA         | Gd         | No             | ALQ            |          216 | Unf            |            0 |         540 |           756 | GasA      | Gd          | Y            | SBrkr        |        961 |        756 |              0 |        1717 |              1 |              0 |          1 |          0 |              3 |              1 | Gd            |              7 | Typ          |            1 | Gd            | Detchd       |          1998 | Unf            |            3 |          642 | TA           | TA           | Y            |            0 |            35 |             272 |           0 |             0 |          0 |      nan |     nan |           nan |         0 |        2 |     2006 | WD         | Abnorml         |      140000 |\n|  4 |    5 |           60 | RL         |            84 |     14260 | Pave     |     nan | IR1        | Lvl           | AllPub      | FR2         | Gtl         | NoRidge        | Norm         | Norm         | 1Fam       | 2Story       |             8 |             5 |        2000 |           2000 | Gable       | CompShg    | VinylSd       | VinylSd       | BrkFace      |          350 | Gd          | TA          | PConc        | Gd         | TA         | Av             | GLQ            |          655 | Unf            |            0 |         490 |          1145 | GasA      | Ex          | Y            | SBrkr        |       1145 |       1053 |              0 |        2198 |              1 |              0 |          2 |          1 |              4 |              1 | Gd            |              9 | Typ          |            1 | TA            | Attchd       |          2000 | RFn            |            3 |          836 | TA           | TA           | Y            |          192 |            84 |               0 |           0 |             0 |          0 |      nan |     nan |           nan |         0 |       12 |     2008 | WD         | Normal          |      250000 |'




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
train.head().to_markdown()
```




    '|     |   Id |   MSSubClass | MSZoning   |   LotFrontage |   LotArea | Street   |   Alley | LotShape   | LandContour   | Utilities   | LotConfig   | LandSlope   | Neighborhood   | Condition1   | Condition2   | BldgType   | HouseStyle   |   OverallQual |   OverallCond |   YearBuilt |   YearRemodAdd | RoofStyle   | RoofMatl   | Exterior1st   | Exterior2nd   | MasVnrType   |   MasVnrArea | ExterQual   | ExterCond   | Foundation   | BsmtQual   | BsmtCond   | BsmtExposure   | BsmtFinType1   |   BsmtFinSF1 | BsmtFinType2   |   BsmtFinSF2 |   BsmtUnfSF |   TotalBsmtSF | Heating   | HeatingQC   | CentralAir   | Electrical   |   1stFlrSF |   2ndFlrSF |   LowQualFinSF |   GrLivArea |   BsmtFullBath |   BsmtHalfBath |   FullBath |   HalfBath |   BedroomAbvGr |   KitchenAbvGr | KitchenQual   |   TotRmsAbvGrd | Functional   |   Fireplaces | FireplaceQu   | GarageType   |   GarageYrBlt | GarageFinish   |   GarageCars |   GarageArea | GarageQual   | GarageCond   | PavedDrive   |   WoodDeckSF |   OpenPorchSF |   EnclosedPorch |   3SsnPorch |   ScreenPorch |   PoolArea |   PoolQC |   Fence |   MiscFeature |   MiscVal |   MoSold |   YrSold | SaleType   | SaleCondition   |   SalePrice |\n|----:|-----:|-------------:|:-----------|--------------:|----------:|:---------|--------:|:-----------|:--------------|:------------|:------------|:------------|:---------------|:-------------|:-------------|:-----------|:-------------|--------------:|--------------:|------------:|---------------:|:------------|:-----------|:--------------|:--------------|:-------------|-------------:|:------------|:------------|:-------------|:-----------|:-----------|:---------------|:---------------|-------------:|:---------------|-------------:|------------:|--------------:|:----------|:------------|:-------------|:-------------|-----------:|-----------:|---------------:|------------:|---------------:|---------------:|-----------:|-----------:|---------------:|---------------:|:--------------|---------------:|:-------------|-------------:|:--------------|:-------------|--------------:|:---------------|-------------:|-------------:|:-------------|:-------------|:-------------|-------------:|--------------:|----------------:|------------:|--------------:|-----------:|---------:|--------:|--------------:|----------:|---------:|---------:|:-----------|:----------------|------------:|\n| 141 |  142 |           20 | RL         |            78 |     11645 | Pave     |     nan | Reg        | Lvl           | AllPub      | Inside      | Gtl         | CollgCr        | Norm         | Norm         | 1Fam       | 1Story       |             7 |             5 |        2005 |           2005 | Gable       | CompShg    | VinylSd       | VinylSd       | None         |            0 | Gd          | TA          | PConc        | Gd         | TA         | Av             | GLQ            |         1300 | Unf            |            0 |         434 |          1734 | GasA      | Ex          | Y            | SBrkr        |       1734 |          0 |              0 |        1734 |              1 |              0 |          2 |          0 |              3 |              1 | Gd            |              7 | Typ          |            0 | nan           | Attchd       |          2005 | Fin            |            2 |          660 | TA           | TA           | Y            |          160 |            24 |               0 |           0 |             0 |          0 |      nan |     nan |           nan |         0 |        1 |     2006 | WD         | Normal          |      260000 |\n| 169 |  170 |           20 | RL         |           nan |     16669 | Pave     |     nan | IR1        | Lvl           | AllPub      | Corner      | Gtl         | Timber         | Norm         | Norm         | 1Fam       | 1Story       |             8 |             6 |        1981 |           1981 | Hip         | WdShake    | Plywood       | Plywood       | BrkFace      |          653 | Gd          | TA          | CBlock       | Gd         | TA         | No             | Unf            |            0 | Unf            |            0 |        1686 |          1686 | GasA      | TA          | Y            | SBrkr        |       1707 |          0 |              0 |        1707 |              0 |              0 |          2 |          1 |              2 |              1 | TA            |              6 | Typ          |            1 | TA            | Attchd       |          1981 | RFn            |            2 |          511 | TA           | TA           | Y            |          574 |            64 |               0 |           0 |             0 |          0 |      nan |     nan |           nan |         0 |        1 |     2006 | WD         | Normal          |      228000 |\n| 302 |  303 |           20 | RL         |           118 |     13704 | Pave     |     nan | IR1        | Lvl           | AllPub      | Corner      | Gtl         | CollgCr        | Norm         | Norm         | 1Fam       | 1Story       |             7 |             5 |        2001 |           2002 | Gable       | CompShg    | VinylSd       | VinylSd       | BrkFace      |          150 | Gd          | TA          | PConc        | Gd         | TA         | No             | Unf            |            0 | Unf            |            0 |        1541 |          1541 | GasA      | Ex          | Y            | SBrkr        |       1541 |          0 |              0 |        1541 |              0 |              0 |          2 |          0 |              3 |              1 | Gd            |              6 | Typ          |            1 | TA            | Attchd       |          2001 | RFn            |            3 |          843 | TA           | TA           | Y            |          468 |            81 |               0 |           0 |             0 |          0 |      nan |     nan |           nan |         0 |        1 |     2006 | WD         | Normal          |      205000 |\n| 370 |  371 |           60 | RL         |           nan |      8121 | Pave     |     nan | IR1        | Lvl           | AllPub      | Inside      | Gtl         | Gilbert        | Norm         | Norm         | 1Fam       | 2Story       |             6 |             5 |        2000 |           2000 | Gable       | CompShg    | VinylSd       | VinylSd       | None         |            0 | TA          | TA          | PConc        | Gd         | TA         | No             | Unf            |            0 | Unf            |            0 |         953 |           953 | GasA      | Ex          | Y            | SBrkr        |        953 |        711 |              0 |        1664 |              0 |              0 |          2 |          1 |              3 |              1 | TA            |              7 | Typ          |            1 | TA            | Attchd       |          2000 | RFn            |            2 |          460 | TA           | TA           | Y            |          100 |            40 |               0 |           0 |             0 |          0 |      nan |     nan |           nan |         0 |        1 |     2006 | WD         | Normal          |      172400 |\n| 411 |  412 |          190 | RL         |           100 |     34650 | Pave     |     nan | Reg        | Bnk           | AllPub      | Inside      | Gtl         | Gilbert        | Norm         | Norm         | 2fmCon     | 1Story       |             5 |             5 |        1955 |           1955 | Hip         | CompShg    | Wd Sdng       | Wd Sdng       | None         |            0 | TA          | TA          | CBlock       | TA         | TA         | Mn             | Rec            |         1056 | Unf            |            0 |           0 |          1056 | GasA      | TA          | N            | SBrkr        |       1056 |          0 |              0 |        1056 |              1 |              0 |          1 |          0 |              3 |              1 | TA            |              5 | Typ          |            0 | nan           | Attchd       |          1955 | Fin            |            2 |          572 | TA           | TA           | Y            |          264 |             0 |               0 |           0 |             0 |          0 |      nan |     nan |           nan |         0 |        1 |     2006 | WD         | Normal          |      145000 |'



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


