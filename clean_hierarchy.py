# python clean_hierarchy.py output/gpt/cached_memories/memory_cache_sample_0.pkl --levels scene semantic
# rm output/gpt/cached_memories/retriever_cache_sample_0_retriever.pkl
# rm output/gpt/cached_memories/retriever_cache_sample_0_embeddings.npy


import pickle
import networkx as nx
import argparse
import os
import sys


def clean_hierarchy(file_path, levels_to_remove=None):
    """
    Removes nodes of specified levels from the memory graph cache.
    
    Args:
        file_path: Path to the .pkl file.
        levels_to_remove: List of level strings to remove (e.g., ['scene', 'semantic']).
                          If None, defaults to ['scene', 'semantic'].
    """
    if levels_to_remove is None:
        levels_to_remove = ['scene', 'semantic']

    print(f"--- Cleaning Hierarchy in: {file_path} ---")
    print(f"Target levels to remove: {levels_to_remove}\n")


    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    # 1. Load the cache
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        if isinstance(data, dict) and 'graph' in data:
            G = data['graph']
            turns = data.get('turns_processed', 0)
            print(f"Successfully loaded graph. Current turns processed: {turns}")
        else:
            print("Error: Unrecognized cache format. Expected dict with 'graph' key.")
            return

    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return

    # 2. Identify nodes to remove
    nodes_to_remove = []
    kept_nodes_count = 0
    
    print("Scanning nodes...")
    for node_id, attrs in G.nodes(data=True):
        level = attrs.get('level')
        
        if level in levels_to_remove:
            nodes_to_remove.append(node_id)
        else:
            kept_nodes_count += 1


    print(f"Found {len(nodes_to_remove)} nodes to remove (Levels: {levels_to_remove}).")
    print(f"Found {kept_nodes_count} nodes to keep.")

    if not nodes_to_remove:
        print("No matching nodes found. The graph is already clean.")
        return

    # 3. Remove nodes
    G.remove_nodes_from(nodes_to_remove)
    print("Nodes removed successfully.")

    # 4. Save back to file
    try:
        with open(file_path, 'wb') as f:
            # We keep turns_processed as is, so it doesn't rebuild Level 0
            pickle.dump({'turns_processed': turns, 'graph': G}, f)
        print(f"--- Cleaned graph saved to {file_path} ---")
        print("Ready for rebuilding.")
        
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean specific levels from A-Mem cache")
    parser.add_argument("file_path", type=str, help="Path to the .pkl memory cache file")
    

    parser.add_argument("--levels", nargs='+', default=['scene', 'semantic'],
                        help="List of levels to remove (e.g. 'scene' 'semantic' 'event'). Default: scene semantic")

    args = parser.parse_args()
    
    clean_hierarchy(args.file_path, args.levels)