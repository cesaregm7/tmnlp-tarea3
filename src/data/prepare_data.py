from typing import List
import pandas as pd
import os

def read_sample() -> List[str]:
    basePath = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_json(basePath+'/../../data/raw/newsgroups.json')
    return df.content.values.tolist()