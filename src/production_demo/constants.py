"""Model constants

This file contains all feature and configuration constants
used during model training and invocation.
"""

CATEGORIES = [
    "BldgType",
    "CentralAir",
    "Electrical",
    # "ExterCond",
    # "ExterQual",
    "Fence",
    "FireplaceQu",
    "Foundation",
    "Functional",
    # "GarageCond",
    # "GarageQual",
    "GarageType",
    "Heating",
    "HeatingQC",
    "HouseStyle",
    # "KitchenQual",
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
    # "OverallCond",
    # "OverallQual",
    "PoolArea",
    "TotRmsAbvGrd",
    "TotalBsmtSF",
    "WoodDeckSF",
    "YearBuilt",
    "YearRemodAdd",
]

OUTPUT = "SalePrice"

MODEL_PARAMS = {
    "LGBM__colsample_bytree": 0.4364670911371453,
    "LGBM__learning_rate": 0.769420294059774,
    "LGBM__max_depth": 7,
    "LGBM__min_child_weight": 0.01822587243541679,
    "LGBM__min_split_gain": 0.03308747951729114,
    "LGBM__n_estimators": 600,
    "LGBM__num_leaves": 921,
    "LGBM__reg_alpha": 286.5027269365633,
    "LGBM__reg_lambda": 2429.7823135073622,
    "LGBM__subsample": 0.7760855497537651,
}

EVAL_SPLITS = 5
EVAL_METRIC = "neg_mean_squared_log_error"

TRAINED_MODEL_NAME = "trained_model"
