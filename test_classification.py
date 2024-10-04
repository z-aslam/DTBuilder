import pytest
import pandas as pd
from io import StringIO

import pages.Classification_Model as CM


@pytest.fixture(scope='function', autouse=True) 
def test_load_data():
    csv_data = "col1,col2\n1,2\n3,4"
    file = StringIO(csv_data)
    df = CM.load_data(file)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
