import requests


def test_endpoint():
    """Local endpoint integration test
    """
    # GIVEN 
    TEST_REQUEST = (
        '{"1stFlrSF":896,"2ndFlrSF":0,"BedroomAbvGr":2,'
        '"EnclosedPorch":0,"Fireplaces":0,"FullBath":1,'
        '"GarageArea":730.0,"GarageCars":1.0,"GrLivArea"'
        ':896,"HalfBath":0,"KitchenAbvGr":1,"LotArea":11622'
        ',"OpenPorchSF":0,"OverallCond":6,"OverallQual":5,'
        '"PoolArea":0,"TotRmsAbvGrd":5,"TotalBsmtSF":882.0,'
        '"WoodDeckSF":140,"YearBuilt":1961,"YearRemodAdd":'
        '1961,"BldgType":"1Fam","CentralAir":"Y","Electrical"'
        ':"SBrkr","ExterCond":"TA","ExterQual":"TA","Fence":'
        '"MnPrv","FireplaceQu":null,"Foundation":"CBlock",'
        '"Functional":"Typ","GarageCond":"TA","GarageQual":'
        '"TA","GarageType":"Attchd","Heating":"GasA","HeatingQC"'
        ':"TA","HouseStyle":"1Story","KitchenQual":"TA",'
        '"LotConfig":"Inside","MasVnrType":"None","MSSubClass":20,'
        '"PavedDrive":"Y","RoofStyle":"Gable"}\n'
    ) 

    # WHEN 
    result = requests.post(
        "http://127.0.0.1:8000/invocations",
        data=TEST_REQUEST,
        headers={"Content-Type": "application/json"},
    )

    #THEN
    assert result.content == b"125627.7320"
