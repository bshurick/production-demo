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
from production_demo import (CategoriesTransformer, 
                             CATEGORIES, 
                             NUMERICS, 
                             OUTPUT)
```

### Dataprep


```python
train = pd.read_csv('../data/train.csv')
print(train.shape)
train.head()
```

    (1460, 81)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>




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
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>141</th>
      <td>142</td>
      <td>20</td>
      <td>RL</td>
      <td>78.0</td>
      <td>11645</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>260000</td>
    </tr>
    <tr>
      <th>169</th>
      <td>170</td>
      <td>20</td>
      <td>RL</td>
      <td>NaN</td>
      <td>16669</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>228000</td>
    </tr>
    <tr>
      <th>302</th>
      <td>303</td>
      <td>20</td>
      <td>RL</td>
      <td>118.0</td>
      <td>13704</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>205000</td>
    </tr>
    <tr>
      <th>370</th>
      <td>371</td>
      <td>60</td>
      <td>RL</td>
      <td>NaN</td>
      <td>8121</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>172400</td>
    </tr>
    <tr>
      <th>411</th>
      <td>412</td>
      <td>190</td>
      <td>RL</td>
      <td>100.0</td>
      <td>34650</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Bnk</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>145000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>



### Features subset

We're subsetting features here based on what we will have **at time of prediction**; in other words, not all 80+ features from training are going to be available to us at prediction time, or we want to make it easier to fill out a form to on our web page to make a prediction. We are saying that we will only *require* the below features in order to make a prediction. 


```python
print(f'Categories used:\n{CATEGORIES}')
print(f'\nNumerics used:\n{NUMERICS}')
print(f'\n Total features used: {len(CATEGORIES) + len(NUMERICS)}')
```

    Categories used:
    ['BldgType', 'CentralAir', 'Electrical', 'ExterCond', 'ExterQual', 'Fence', 'FireplaceQu', 'Foundation', 'Functional', 'GarageCond', 'GarageQual', 'GarageType', 'Heating', 'HeatingQC', 'HouseStyle', 'KitchenQual', 'LotConfig', 'MasVnrType', 'MSSubClass', 'PavedDrive', 'RoofStyle']
    
    Numerics used:
    ['1stFlrSF', '2ndFlrSF', 'BedroomAbvGr', 'EnclosedPorch', 'Fireplaces', 'FullBath', 'GarageArea', 'GarageCars', 'GrLivArea', 'HalfBath', 'KitchenAbvGr', 'LotArea', 'OpenPorchSF', 'OverallCond', 'OverallQual', 'PoolArea', 'TotRmsAbvGrd', 'TotalBsmtSF', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd']
    
     Total features used: 42


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
     {'LGBM__colsample_bytree': 0.21551578658017545, 'LGBM__learning_rate': 0.1826774879834436, 'LGBM__max_depth': 15, 'LGBM__min_child_weight': 0.010980670712808305, 'LGBM__min_split_gain': 0.088343583973319, 'LGBM__n_estimators': 929, 'LGBM__num_leaves': 80, 'LGBM__reg_alpha': 4695.787133251388, 'LGBM__reg_lambda': 795.5697908590797, 'LGBM__subsample': 0.9850671378256838}
    
    Best score:
     -0.0236


## Train


```python
model.fit(train[NUMERICS], train[OUTPUT])
```




    LGBMRegressor(LGBM__colsample_bytree=0.21551578658017545,
                  LGBM__learning_rate=0.1826774879834436, LGBM__max_depth=15,
                  LGBM__min_child_weight=0.010980670712808305,
                  LGBM__min_split_gain=0.088343583973319, LGBM__n_estimators=929,
                  LGBM__num_leaves=80, LGBM__reg_alpha=4695.787133251388,
                  LGBM__reg_lambda=795.5697908590797,
                  LGBM__subsample=0.9850671378256838)




```python
#save model artifacts
dump(model, '../data/trained_model')
```




    ['../data/trained_model']


