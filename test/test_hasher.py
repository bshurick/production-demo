import pytest
import pandas as pd

from numpy import nan

SAMPLE_RECORD = {
    'Id': 142,
    'MSSubClass': 20,
    'MSZoning': 'RL',
    'LotFrontage': 78.0,
    'LotArea': 11645,
    'Street': 'Pave',
    'Alley': nan,
    'LotShape': 'Reg',
    'LandContour': 'Lvl',
    'Utilities': 'AllPub',
    'LotConfig': 'Inside',
    'LandSlope': 'Gtl',
    'Neighborhood': 'CollgCr',
    'Condition1': 'Norm',
    'Condition2': 'Norm',
    'BldgType': '1Fam',
    'HouseStyle': '1Story',
    'OverallQual': 7,
    'OverallCond': 5,
    'YearBuilt': 2005,
    'YearRemodAdd': 2005,
    'RoofStyle': 'Gable',
    'RoofMatl': 'CompShg',
    'Exterior1st': 'VinylSd',
    'Exterior2nd': 'VinylSd',
    'MasVnrType': 'None',
    'MasVnrArea': 0.0,
    'ExterQual': 'Gd',
    'ExterCond': 'TA',
    'Foundation': 'PConc',
    'BsmtQual': 'Gd',
    'BsmtCond': 'TA',
    'BsmtExposure': 'Av',
    'BsmtFinType1': 'GLQ',
    'BsmtFinSF1': 1300,
    'BsmtFinType2': 'Unf',
    'BsmtFinSF2': 0,
    'BsmtUnfSF': 434,
    'TotalBsmtSF': 1734,
    'Heating': 'GasA',
    'HeatingQC': 'Ex',
    'CentralAir': 'Y',
    'Electrical': 'SBrkr',
    '1stFlrSF': 1734,
    '2ndFlrSF': 0,
    'LowQualFinSF': 0,
    'GrLivArea': 1734,
    'BsmtFullBath': 1,
    'BsmtHalfBath': 0,
    'FullBath': 2,
    'HalfBath': 0,
    'BedroomAbvGr': 3,
    'KitchenAbvGr': 1,
    'KitchenQual': 'Gd',
    'TotRmsAbvGrd': 7,
    'Functional': 'Typ',
    'Fireplaces': 0,
    'FireplaceQu': nan,
    'GarageType': 'Attchd',
    'GarageYrBlt': 2005.0,
    'GarageFinish': 'Fin',
    'GarageCars': 2,
    'GarageArea': 660,
    'GarageQual': 'TA',
    'GarageCond': 'TA',
    'PavedDrive': 'Y',
    'WoodDeckSF': 160,
    'OpenPorchSF': 24,
    'EnclosedPorch': 0,
    '3SsnPorch': 0,
    'ScreenPorch': 0,
    'PoolArea': 0,
    'PoolQC': nan,
    'Fence': nan,
    'MiscFeature': nan,
    'MiscVal': 0,
    'MoSold': 1,
    'YrSold': 2006,
    'SaleType': 'WD',
    'SaleCondition': 'Normal',
    'SalePrice': 260000
}



def test_hasher():
    return True
