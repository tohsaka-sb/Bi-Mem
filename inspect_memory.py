import pickle
import networkx as nx
import argparse
import os
import sys

def inspect_memory_cache(file_path, output_file=None):
    """
    Reads and displays the content of the A-Mem graph cache, including edges.
    Supports writing to a file.
    """
    # Determine where to print: stdout or a file
    if output_file:
        f_out = open(output_file, 'w', encoding='utf-8')
    else:
        f_out = sys.stdout

    def log(message=""):
        print(message, file=f_out)

    log(f"--- Inspecting Memory Cache: {file_path} ---\n")

    if not os.path.exists(file_path):
        log(f"Error: File not found at {file_path}")
        if output_file: f_out.close()
        return

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        if isinstance(data, dict) and 'graph' in data:
            G = data['graph']
            turns = data.get('turns_processed', 'Unknown')
            log(f"Status: {turns} turns processed.")
        else:
            log("Error: Unrecognized cache format. Expected dict with 'graph' key.")
            if output_file: f_out.close()
            return

    except Exception as e:
        log(f"Error loading pickle file: {e}")
        if output_file: f_out.close()
        return

    # --- Analyze Graph Nodes ---
    nodes = G.nodes(data=True)
    
    persona_nodes = []
    scene_nodes = []
    fact_nodes = []
    other_nodes = []

    for node_id, attrs in nodes:
        level = attrs.get('level')
        if level == 'persona':
            persona_nodes.append((node_id, attrs))
        elif level == 'scene':
            scene_nodes.append((node_id, attrs))
        elif level == 'fact':
            fact_nodes.append((node_id, attrs))
        else:
            other_nodes.append((node_id, attrs))

    # --- Print Statistics ---
    log(f"\nTotal Nodes: {len(G.nodes)}")
    log(f"  - Level 2 (Persona): {len(persona_nodes)}")
    log(f"  - Level 1 (Scene)   : {len(scene_nodes)}")
    log(f"  - Level 0 (Fact)   : {len(fact_nodes)}")
    log(f"  - Unknown/Other     : {len(other_nodes)}")
    log(f"Total Edges: {len(G.edges)}")
    log("-" * 50)

    # Helper to format edges (Preserved from your code)
    def get_edges_str(node_id):
        try:
            preds = list(G.predecessors(node_id))
            succs = list(G.successors(node_id))
            return f"  <- In (From {len(preds)}): {preds}\n  -> Out (To {len(succs)}): {succs}"
        except:
            return "  (Edges info unavailable)"

    # --- Print Level 2: Persona Memories ---
    log(f"\n=== Level 2: Persona Memories ({len(persona_nodes)}) ===")
    for i, (nid, attrs) in enumerate(persona_nodes, 1):
        log(f"\n[Node {i}] ID: {nid}")
        log(f"  Attribute: {attrs.get('attribute_type', 'General')}")
        log(f"  Time: {attrs.get('timestamp')}")
        log(f"  Content (Profile): \n{attrs.get('content')}")
        log(get_edges_str(nid))

    # --- Print Level 1: Scene Memories ---
    log(f"\n\n=== Level 1: Scene Memories ({len(scene_nodes)}) ===")
    
    # Check for --all flag in sys.argv manually
    display_limit_scene = len(scene_nodes) if '--all' in sys.argv else 5
    
    for i, (nid, attrs) in enumerate(scene_nodes[:display_limit_scene], 1):
        log(f"\n[Scene {i}] ID: {nid}")
        log(f"  Summary: {attrs.get('content')}")
        log(get_edges_str(nid))

    if len(scene_nodes) > display_limit_scene:
        log(f"\n... and {len(scene_nodes) - display_limit_scene} more scenes.")

    # --- Print Level 0: Fact Memories ---
    log(f"\n\n=== Level 0: Fact Memories ({len(fact_nodes)}) ===")
    try:
        fact_nodes.sort(key=lambda x: x[1].get('timestamp', ''))
    except:
        pass

    display_limit_fact = len(fact_nodes) if '--all' in sys.argv else 5

    for i, (nid, attrs) in enumerate(fact_nodes[:display_limit_fact], 1):
        log(f"\n[Fact {i}] ID: {nid}")
        log(f"  Time: {attrs.get('timestamp')}")
        log(f"  Content: {attrs.get('content')[:]}...") # Truncate long content for readability
        log(get_edges_str(nid))
        

        # Find persona neighbors (successors that are also facts)
        lateral_links = []
        try:
            for succ_id in G.successors(nid):
                # Check if the target node is an fact
                if G.nodes[succ_id].get('level') == 'fact':
                    neighbor_content = G.nodes[succ_id].get('content', '')
                    lateral_links.append((succ_id, neighbor_content))
        except Exception:
            pass

        if lateral_links:
            log(f"  >> Lateral Persona Links ({len(lateral_links)}):")
            for target_id, content in lateral_links:
                preview = (content[:60] + '...') if len(content) > 60 else content
                log(f"     - -> {target_id}: {preview}")


    if len(fact_nodes) > display_limit_fact:
        log(f"\n... and {len(fact_nodes) - display_limit_fact} more facts.")

    log("\n\n=== Inspection Complete ===")
    
    if output_file:
        print(f"Inspection results saved to {output_file}")
        f_out.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect A-Mem Graph Cache")
    parser.add_argument("file_path", type=str, help="Path to the .pkl file")
    parser.add_argument("--output", type=str, default=None, help="Path to save the output text file")
    parser.add_argument("--all", action="store_true", help="Show all nodes instead of first 5")
    
    args = parser.parse_args()
    inspect_memory_cache(args.file_path, args.output)