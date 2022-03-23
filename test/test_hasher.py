import pytest
import pandas as pd

from numpy import nan
from production_demo.train import CategoriesTransformer


SAMPLE_RECORD = {
    "Id": 142,
    "MSSubClass": 20,
    "MSZoning": "RL",
    "Street": "Pave",
    "Alley": nan,
    "LotShape": "Reg",
    "LandContour": "Lvl",
    "BsmtFinSF2": 0,
    "BsmtUnfSF": 434,
    "TotalBsmtSF": 1734,
    "Heating": "GasA",
    "CentralAir": "Y",
    "SaleType": "WD",
    "SaleCondition": "Normal",
    "SalePrice": 260000,
}


def test_hasher():
    # GIVEN
    xdf = pd.DataFrame([SAMPLE_RECORD], index=[0])
    cxt = CategoriesTransformer(
        category_cols=[
            "MSZoning",
            "Street",
            "Alley",
            "LotShape",
            "LandContour",
            "Heating",
            "CentralAir",
            "SaleCondition",
        ]
    )

    # WHEN
    xdf_trans = cxt.fit_transform(xdf)

    # THEN
    # Validate that the response exactly matches our static example
    assert xdf_trans.to_dict() == {
        "Id": {0: 142},
        "MSSubClass": {0: 20},
        "MSZoning": {0: 65099},
        "Street": {0: 81231},
        "Alley": {0: 78366},
        "LotShape": {0: 70125},
        "LandContour": {0: 10624},
        "BsmtFinSF2": {0: 0},
        "BsmtUnfSF": {0: 434},
        "TotalBsmtSF": {0: 1734},
        "Heating": {0: 51662},
        "CentralAir": {0: 82272},
        "SaleType": {0: "WD"},
        "SaleCondition": {0: 80304},
        "SalePrice": {0: 260000},
    }
