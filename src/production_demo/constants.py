"""Model constants

This file contains all feature and configuration constants
used during model training and invocation.
"""

CATEGORIES = [
    "BldgType",
    "CentralAir",
    "Electrical",
    "ExterCond",
    "ExterQual",
    "Fence",
    "FireplaceQu",
    "Foundation",
    "Functional",
    "GarageCond",
    "GarageQual",
    "GarageType",
    "Heating",
    "HeatingQC",
    "HouseStyle",
    "KitchenQual",
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
    "OverallCond",
    "OverallQual",
    "PoolArea",
    "TotRmsAbvGrd",
    "TotalBsmtSF",
    "WoodDeckSF",
    "YearBuilt",
    "YearRemodAdd",
]

OUTPUT = "SalePrice"

MODEL_PARAMS = {
    "LGBM__colsample_bytree": 0.21551578658017545,
    "LGBM__learning_rate": 0.1826774879834436,
    "LGBM__max_depth": 15,
    "LGBM__min_child_weight": 0.010980670712808305,
    "LGBM__min_split_gain": 0.088343583973319,
    "LGBM__n_estimators": 929,
    "LGBM__num_leaves": 80,
    "LGBM__reg_alpha": 4695.787133251388,
    "LGBM__reg_lambda": 795.5697908590797,
    "LGBM__subsample": 0.9850671378256838,
}

EVAL_SPLITS = 5
EVAL_METRIC = "neg_mean_squared_log_error"
