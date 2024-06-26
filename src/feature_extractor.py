import pandas as pd
import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

def calculate_features(data):
    # Load auxiliary data
    distances = pd.read_csv('/Users/merelkamper/Documents/MSc Data Science/Thesis/MSc_Thesis_code/data/station_distances.csv')
    populations = pd.read_csv('/Users/merelkamper/Documents/MSc Data Science/Thesis/MSc_Thesis_code/data/station_populations.csv')

    # Merge population data with the main data
    data = data.merge(populations, how='left', left_on='source', right_on='Station').rename(columns={'Population': 'source_population'}).drop(columns=['Station'])
    data = data.merge(populations, how='left', left_on='target', right_on='Station').rename(columns={'Population': 'target_population'}).drop(columns=['Station'])

    # Fill NaN population values with 0
    data['source_population'] = data['source_population'].fillna(0)
    data['target_population'] = data['target_population'].fillna(0)

    # Calculate the gravitational index directly
    def calculate_gravitational_index(row):
        if row['distance'] > 0:
            return (row['source_population'] * row['target_population']) / (row['distance'] ** 2)
        else:
            return 0

    # Find distances
    data = data.merge(distances, how='left', left_on=['source', 'target'], right_on=['origin_name', 'destination_name'])
    data['distance'] = data['distance'].fillna(0)

    data['gravitational_index'] = data.apply(calculate_gravitational_index, axis=1)

    features_list = []

    # Group by YearMonth and calculate features for each group
    for year_month, group in data.groupby('YearMonth'):
        print(f"Processing YearMonth: {year_month}")
        
        # Create a graph for the current YearMonth
        G = nx.Graph()

        # Add edges to the graph with weights and distances
        for index, row in tqdm(group.iterrows(), total=group.shape[0], desc="Adding edges"):
            source, target = row['source'], row['target']
            
            if source == target:
                continue
            
            weight = row['Rides planned']
            distance = row['distance']

            # Directly add edges without population data
            G.add_edge(source, target, weight=weight, distance=distance)

        # Precompute degree and strength for all nodes to avoid recalculating in loop
        degree = dict(G.degree())
        weighted_degree = dict(G.degree(weight='weight'))
        strength = dict(G.degree(weight='weight'))

        # Calculate centrality measures
        closeness_centrality = nx.closeness_centrality(G, distance='distance')
        degree_centrality = nx.degree_centrality(G)

        def process_edge(row):
            source, target = row['source'], row['target']
            
            if source == target or not G.has_edge(source, target):
                return None
            
            common_neighbors = list(nx.common_neighbors(G, source, target))
            num_common_neighbors = len(common_neighbors)

            if num_common_neighbors == 0:
                # Set all features to 0 when there are no common neighbors
                CN = SA = JA = SO = HPI = HDI = LHNI = PA = AA = RA = LPI = 0
                weighted_CN = weighted_SA = weighted_JA = weighted_SO = weighted_HPI = weighted_HDI = weighted_LHNI = weighted_PA = weighted_AA = weighted_RA = weighted_LPI = 0
            else:
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
                total_weight = sum(G[source][w]['weight'] + G[w][target]['weight'] for w in common_neighbors)
                weighted_CN = total_weight
                weighted_SA = total_weight / np.sqrt(weighted_degree[source] * weighted_degree[target])
                weighted_JA = total_weight / len(source_neighbors.union(target_neighbors))
                weighted_SO = 2 * total_weight / (weighted_degree[source] + weighted_degree[target])
                weighted_HPI = total_weight / min(weighted_degree[source], weighted_degree[target])
                weighted_HDI = total_weight / max(weighted_degree[source], weighted_degree[target])
                weighted_LHNI = total_weight / (weighted_degree[source] * weighted_degree[target])
                weighted_PA = weighted_degree[source] * weighted_degree[target]
                weighted_AA = sum(1 / np.log(weighted_degree[w]) for w in common_neighbors if weighted_degree[w] > 1)
                weighted_RA = sum(1 / weighted_degree[w] for w in common_neighbors)
                weighted_LPI = sum(1 / (weighted_degree[w] ** 0.5) for w in common_neighbors)

            distance = row['distance']
            population_source = row['source_population']
            population_target = row['target_population']
            gravitational_index = row['gravitational_index']

            # Extract centrality measures for the source and target nodes
            source_closeness = closeness_centrality.get(source, 0)
            target_closeness = closeness_centrality.get(target, 0)
            source_degree = degree_centrality.get(source, 0)
            target_degree = degree_centrality.get(target, 0)
            source_strength = strength.get(source, 0)
            target_strength = strength.get(target, 0)

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
                'gravitational_index': gravitational_index,
                'source_closeness': source_closeness,
                'target_closeness': target_closeness,
                'source_degree': source_degree,
                'target_degree': target_degree,
                'source_strength': source_strength,
                'target_strength': target_strength
            }

        # Use parallel processing to speed up the feature calculation
        features = Parallel(n_jobs=-1)(delayed(process_edge)(row) for _, row in tqdm(group.iterrows(), total=group.shape[0], desc="Processing edges"))

        # Filter out None results
        features = [feature for feature in features if feature is not None]
        features_list.extend(features)

    return pd.DataFrame(features_list)
