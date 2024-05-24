import pandas as pd
import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

def calculate_features(data):
    distances = pd.read_csv('/Users/merelkamper/Documents/MSc Data Science/Thesis/MSc-Thesis-main/adapted/data/station_distances.csv')
    populations = pd.read_csv('/Users/merelkamper/Documents/MSc Data Science/Thesis/MSc-Thesis-main/adapted/data/station_population.csv')
    
    features_list = []

    # Group by YearMonth and calculate features for each group
    for year_month, group in data.groupby('YearMonth'):
        print(f"Adding edges for {year_month}:")
        
        # Create a graph for the current YearMonth
        G = nx.Graph()

        # Add edges to the graph with weights and distances
        for index, row in tqdm(group.iterrows(), total=group.shape[0], desc="Adding edges"):
            source, target = row['source'], row['target']
            
            if source == target:
                continue
            
            weight = row['Rides planned']
            
            # Find the distance between source and target
            distance_row = distances[(distances['origin_name'] == source) & (distances['destination_name'] == target)]
            if distance_row.empty:
                distance = np.nan
            else:
                distance = distance_row['distance'].values[0]

            # Find populations
            pop_source_row = populations[populations['Station'] == source]
            pop_target_row = populations[populations['Station'] == target]
            if pop_source_row.empty:
                population_source = np.nan
            else:
                population_source = pop_source_row['Population'].values[0]
                
            if pop_target_row.empty:
                population_target = np.nan
            else:
                population_target = pop_target_row['Population'].values[0]

            G.add_edge(source, target, weight=weight, distance=distance, pop_source=population_source, pop_target=population_target)

        # Precompute degree for all nodes to avoid recalculating in loop
        degree = dict(G.degree())
        weighted_degree = dict(G.degree(weight='weight'))

        def process_edge(row):
            source, target = row['source'], row['target']
            
            if source == target or not G.has_edge(source, target):
                return None
            
            common_neighbors = list(nx.common_neighbors(G, source, target))
            num_common_neighbors = len(common_neighbors)

            if num_common_neighbors == 0:
                return None
            
            # Precompute neighbor sets
            source_neighbors = set(G.neighbors(source))
            target_neighbors = set(G.neighbors(target))
            
            # Unweighted topological feature calculations
            CN = num_common_neighbors
            SA = CN / np.sqrt(len(source_neighbors) * len(target_neighbors))
            JA = CN / len(source_neighbors.union(target_neighbors))
            SO = 2 * CN / (len(source_neighbors) + len(target_neighbors))
            HPI = CN / min(len(source_neighbors), len(target_neighbors))
            HDI = CN / max(len(source_neighbors), len(target_neighbors))
            LHNI = CN / (len(source_neighbors) * len(target_neighbors))
            PA = len(source_neighbors) * len(target_neighbors)
            AA = sum(1 / np.log(len(list(G.neighbors(w)))) for w in common_neighbors if len(list(G.neighbors(w))) > 1)
            RA = sum(1 / len(list(G.neighbors(w))) for w in common_neighbors)
            LPI = sum(1 / (degree[w] ** 0.5) for w in common_neighbors)

            # Weighted topological feature calculations
            weighted_CN = CN
            weighted_SA = CN / np.sqrt(weighted_degree[source] * weighted_degree[target])
            weighted_JA = CN / len(source_neighbors.union(target_neighbors))
            weighted_SO = 2 * CN / (weighted_degree[source] + weighted_degree[target])
            weighted_HPI = CN / min(weighted_degree[source], weighted_degree[target])
            weighted_HDI = CN / max(weighted_degree[source], weighted_degree[target])
            weighted_LHNI = CN / (weighted_degree[source] * weighted_degree[target])
            weighted_PA = weighted_degree[source] * weighted_degree[target]
            weighted_AA = sum(1 / np.log(weighted_degree[w]) for w in common_neighbors if weighted_degree[w] > 1)
            weighted_RA = sum(1 / weighted_degree[w] for w in common_neighbors)
            weighted_LPI = sum(1 / (weighted_degree[w] ** 0.5) for w in common_neighbors)

            distance = G[source][target]['distance']
            population_source = G[source][target]['pop_source']
            population_target = G[source][target]['pop_target']
            gravitational_index = (population_source * population_target) / (distance ** 2) if distance > 0 else np.nan

            return {
                'YearMonth': year_month,
                'source': source,
                'target': target,
                'CN': CN,
                'SA': SA,
                'JA': JA,
                'SO': SO,
                'HPI': HPI,
                'HDI': HDI,
                'LHNI': LHNI,
                'PA': PA,
                'AA': AA,
                'RA': RA,
                'LPI': LPI,
                'weighted_CN': weighted_CN,
                'weighted_SA': weighted_SA,
                'weighted_JA': weighted_JA,
                'weighted_SO': weighted_SO,
                'weighted_HPI': weighted_HPI,
                'weighted_HDI': weighted_HDI,
                'weighted_LHNI': weighted_LHNI,
                'weighted_PA': weighted_PA,
                'weighted_AA': weighted_AA,
                'weighted_RA': weighted_RA,
                'weighted_LPI': weighted_LPI,
                'distance': distance,
                'population_source': population_source,
                'population_target': population_target,
                'gravitational_index': gravitational_index
            }

        # Use parallel processing to speed up the feature calculation
        features = Parallel(n_jobs=-1)(delayed(process_edge)(row) for _, row in tqdm(group.iterrows(), total=group.shape[0], desc="Processing edges"))

        # Filter out None results
        features = [feature for feature in features if feature is not None]

        features_list.extend(features)

    return pd.DataFrame(features_list)
