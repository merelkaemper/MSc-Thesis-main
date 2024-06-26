import re
import sys
import os
import unicodedata

import pandas as pd
import networkx as nx
import numpy as np

from tqdm import tqdm
from datetime import date
from glob import glob
from collections import defaultdict

sys.path.append('src')
# ADDED: changed to feature_extractor from features_extractor
from feature_extractor import features_extractor

def strip_accents(text):
    """
    Strip accents from input String.

    :param text: The input string.
    :type text: String.

    :returns: The processed String.
    :rtype: String.
    """
    try:
        text = unicode(text, 'utf-8')
    except (TypeError, NameError): # unicode is a default on python 3
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)

def text_to_id(text):
    """
    Convert input text to id.

    :param text: The input string.
    :type text: String.

    :returns: The processed String.
    :rtype: String.
    """
    text = strip_accents(text.lower())
    text = re.sub(r"\d", "", text)
    text=re.sub(r"^\s+", "", text)
    text=re.sub(r"\s+$", "", text)
    text = re.sub(r"\s+","_", text, flags = re.I)
    #text = re.sub('[ ]+', '_', text)
    text = re.sub('[^a-zA-Z_-]', '', text)
    return text

if __name__ == "__main__":
    path = '/Users/merelkamper/Documents/MSc Data Science/Thesis/transportation_network_evolution-master/data/raw_usair_data'
    files = os.path.join(path, '*.csv')

    feature_file = '/Users/merelkamper/Documents/MSc Data Science/Thesis/transportation_network_evolution-master/data/features/usair_2004_2021.csv'
    to_file = '/Users/merelkamper/Documents/MSc Data Science/Thesis/transportation_network_evolution-master/data/to_file.csv'

    try:
        data = pd.read_csv(to_file, sep=';')
        data.set_index(['YEAR', 'MONTH'], inplace=True)
    except FileNotFoundError:
        print(f'{to_file} not found! Generating graphs from raw data.')
        dfs = []
        
        all_files = glob(files)
        print(f"Files to be read: {all_files}")  # Debugging: Print list of files to be read
        
        for file in tqdm(sorted(all_files)):
            print(f"Processing file: {file}")  # Debugging: Print each file being processed
            if file.endswith('.csv'):
                try:
                    df = pd.read_csv(file, engine="python")
                except pd.errors.ParserError:
                    print(f"Error parsing file: {file}")
                    continue  # Skip to the next file
                #df=pd.read_csv(f,engine="python",error_bad_lines=False)
                # Get these column from the from_file 
                df=df[['YEAR','MONTH','ORIGIN_CITY_NAME','DEST_CITY_NAME','PASSENGERS','DEPARTURES_PERFORMED']]
                # Rename those to these names
                df=df.rename(index=str, columns={"ORIGIN_CITY_NAME": "source",
                                                "DEST_CITY_NAME": "target",
                                                'PASSENGERS':'passengers',
                                                'DEPARTURES_PERFORMED':'weight'})
                #print(df)
                
                # Change the source and target column to id's
                df['source']=df.apply(lambda row: text_to_id(str(row.source)), axis=1)
                df['target']=df.apply(lambda row: text_to_id(str(row.target)), axis=1)
                #print(df)
                # Group on year and month (going from 335.250 rows to 116.288 rows)
                df=df.groupby(['YEAR','MONTH','source','target']).sum()
                #print(df)
                # Reset index so that every source-target combination is linked to a year and month (number of rows remains the same)
                df=df.reset_index()
                #print(df)
                # Add all info from df to empty dfs and skip the ones where the weight (=DEPARTURES_PERFORMED) is 0 (from 116.288 to 116.111 rows)
                # These are probably cancelled flights
                dfs.append(df[df.weight !=0 ])
                #print(dfs)
                #print(df[df.weight ==0 ])
        
        # dfs is put into "data"    
        data=pd.concat(dfs, ignore_index=True)
        data=data.reset_index().drop(columns='index')
        data.set_index(['YEAR', 'MONTH'],inplace=True)
        data.sort_index(inplace=True)
        #print(data)
        data.to_csv(to_file,sep=';')

    # Rows where the source and the target are the same are deleted
    #print(data[data.source == data.target])
    data = data[data.source != data.target]
    #print(data)

    # rows where the weigh is 0 are deleted (none left here)    
    data = data[data.weight!=0]    
    print(data)
    
    year = list(data.index.get_level_values(0).unique())
    month = list(data.index.get_level_values(1).unique())
    graphs_air = []
    date_air = []
    
    # For every year, look at every month
    for y in year:
        for m in month:
            if y==2021 and m==9:
                break
            df = data.loc[y,m]
            date_air.append(date(y,m,1))
            #print(date_air)
            G = nx.from_pandas_edgelist(df, edge_attr=True)
            graphs_air.append(G)
            #graphs_air
    features = features_extractor(graphs_air, date_air)
    #print(features)
    features.to_csv(feature_file)