CATEGORIES = [
    'BldgType',
    'CentralAir',
    'Electrical',
    'ExterCond',
    'ExterQual',
    'Fence',
    'FireplaceQu',
    'Foundation',
    'Functional',
    'GarageCond',
    'GarageQual',
    'GarageType',
    'Heating',
    'HeatingQC',
    'HouseStyle',
    'KitchenQual',
    'LotConfig',
    'MasVnrType',
    'MSSubClass',
    'PavedDrive',
    'RoofStyle',
]

NUMERICS = [
    '1stFlrSF',
    '2ndFlrSF',
    'BedroomAbvGr',
    'EnclosedPorch',
    'Fireplaces',
    'FullBath',
    'GarageArea',
    'GarageCars',
    'GrLivArea',
    'HalfBath',
    'KitchenAbvGr',
    'LotArea',
    'OpenPorchSF',
    'OverallCond',
    'OverallQual',
    'PoolArea',
    'TotRmsAbvGrd',
    'TotalBsmtSF',
    'WoodDeckSF',
    'YearBuilt',
    'YearRemodAdd',
]

OUTPUT = 'SalePrice'

MODEL_PARAMS = {'LGBM__colsample_bytree': 0.4569053964451767, 
                'LGBM__learning_rate': 0.060215823699935396, 
                'LGBM__max_depth': 10, 
                'LGBM__min_child_weight': 0.06494958890542044, 
                'LGBM__min_split_gain': 0.05643039388402624, 
                'LGBM__n_estimators': 354, 
                'LGBM__num_leaves': 3192, 
                'LGBM__reg_alpha': 260.8146685873852, 
                'LGBM__reg_lambda': 101.43819465504578, 
                'LGBM__subsample': 0.4382540932521588}
