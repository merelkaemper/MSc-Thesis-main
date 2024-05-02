import re
import sys
import unicodedata

import pandas as pd
import networkx as nx
import numpy as np

from datetime import datetime
from tqdm import tqdm
from datetime import date
from glob import glob
from collections import defaultdict

sys.path.append('/Users/merelkamper/Documents/MSc Data Science/Thesis/MSc-Thesis-main/adapted/src')
from feature_extractor import features_extractor


if __name__ == "__main__":
    from_file = '/Users/merelkamper/Documents/MSc Data Science/Thesis/MSc-Thesis-main/adapted/data/data_per_day.csv'
    feature_file = '/Users/merelkamper/Documents/MSc Data Science/Thesis/MSc-Thesis-main/adapted/data/features/features.csv'

    dfs = []
    for file_path in tqdm(glob(from_file)):
        df = pd.read_csv(file_path, engine="python")
        df = df[['YEAR', 'MONTH', 'DAY', 'ORIGIN', 'DESTINATION', 'RIDES PERFORMED']]
        df = df.rename(index=str, columns={"ORIGIN": "source", 
                                            "DESTINATION": "target", 
                                            'RIDES PERFORMED': 'weight'})
        dfs.append(df[df.weight != 0])

    # Concatenate all DataFrames in dfs list into one DataFrame
    data = pd.concat(dfs, ignore_index=True)
    data = data.reset_index().drop(columns='index')
    data.set_index(['YEAR', 'MONTH', 'DAY'], inplace=True)
    data.sort_index(inplace=True)
            
    data = data[data['source'] != data['target']]  # Remove rows where origin and destination are the same
    data = data[data['weight'] != 0]               # Remove rows where weight is zero

    graphs = []
    dates = []
    
    year = list(data.index.get_level_values(0).unique())
    month = list(data.index.get_level_values(1).unique())
    day = list(data.index.get_level_values(2).unique())
    
    for y in year:
        for m in month:
            for d in day:
                if (y,m,d) in data.index:
                    df = data.loc[y,m,d]
                    dates.append(date(y,m,d))
                    #print(dates)
                    G = nx.from_pandas_edgelist(df, edge_attr=True)
                    graphs.append(G)
                    #print(graphs)
    features = features_extractor(graphs, dates)
    features.to_csv(feature_file)