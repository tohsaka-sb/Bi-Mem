

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
        levels_to_remove: List of level strings to remove (e.g., ['scene', 'persona']).
                          If None, defaults to ['scene', 'persona'].
    """
    if levels_to_remove is None:
        levels_to_remove = ['scene', 'persona']

    print(f"--- Cleaning Hierarchy in: {file_path} ---")
    print(f"Target levels to remove: {levels_to_remove}\n")

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

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

    G.remove_nodes_from(nodes_to_remove)
    print("Nodes removed successfully.")

    try:
        with open(file_path, 'wb') as f:

            pickle.dump({'turns_processed': turns, 'graph': G}, f)
        print(f"--- Cleaned graph saved to {file_path} ---")
        print("Ready for rebuilding.")
        
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean specific levels from A-Mem cache")
    parser.add_argument("file_path", type=str, help="Path to the .pkl memory cache file")
    

    parser.add_argument("--levels", nargs='+', default=['scene', 'persona'],
                        help="List of levels to remove (e.g. 'scene' 'persona' 'fact'). Default: scene persona")

    args = parser.parse_args()
    
    clean_hierarchy(args.file_path, args.levels)
