import networkx as nx
import uuid
from datetime import datetime
from tqdm import tqdm
import json
import re
import torch
from sentence_transformers import util

class BaseManager:
    def __init__(self, llm_controller, retriever):
        self.llm = llm_controller
        self.retriever = retriever

    def _get_llm_json_response(self, prompt: str) -> dict:
        """Helper to handle LLM call and JSON parsing robustly (Shared logic)."""
        response = ""
        try:
            response = self.llm.get_completion(prompt, response_format=None)
            match = re.search(r"```json\s*(\{.*\})\s*```", response, re.DOTALL)
            if match: return json.loads(match.group(1))
            match = re.search(r"```\s*(\{.*\})\s*```", response, re.DOTALL)
            if match: return json.loads(match.group(1))
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match: return json.loads(match.group(0))
            return json.loads(response)
        except Exception as e:
            if response and response != "{}":
                print(f"\n[ERROR] JSON Parsing Failed in Manager! Error: {str(e)}")
            return {}

class SceneManager(BaseManager):
    """Manages Level 1 (Scene) memory construction and refinement."""
    
    def __init__(self, llm_controller, retriever):
        super().__init__(llm_controller, retriever)
        self.prompt_template = """
        You are an AI Scene Analyst specialized in narrative comprehension.
        
        Task: Summarize a cluster of related conversation facts into a coherent 'Scene Memory'.
        
        Input Facts:
        {facts_content}
        
        Instructions:
        1. Identify the core theme connecting these facts.
        2. Create a descriptive summary that captures the progression of the conversation.
        3. Extract key entities and topics.
        
        Format the response as a JSON object:
        {{
            "scene_summary": "A comprehensive summary...",
            "keywords": ["keyword1", "keyword2", ...],
            "tags": ["tag1", "tag2", ...]
        }}
        """

        self.refinement_prompt_template = """
        You are an AI Memory Enhancer. Your task is to Enrich a Scene Memory based on the high-level User Profile, WITHOUT losing existing details.
        
        High-Level User Profile (Level 2):
        {user_profile}
        
        Current Scene Summary (Level 1):
        {current_summary}
        
        Instructions:
        1. Read the User Profile to understand the user's key interests, values, and traits.
        2. Check the Current Scene Summary. Does it fail to mention any specific connection to the User Profile that is likely present in the events?
        3. If yes, create an ADDITION string to append to the summary. This addition should explicitly link the scene to the profile (e.g., "This aligns with her interest in X...").
        4. CRITICAL: DO NOT REWRITE the existing summary. ONLY generate text to ADD.
        5. If the current summary is already perfect, return an empty string for "addition".
        
        Format the response as a JSON object:
        {{
            "needs_enhancement": true/false,
            "addition": "Text to append (or empty string)",
            "reason": "Why you decided to add this"
        }}
        """


    def build_scene(self, graph, community_nodes):
        facts_content = ""
        for nid in community_nodes:
            node_data = graph.nodes[nid]
            facts_content += f"- [{node_data.get('timestamp', '')}] {node_data.get('content', '')}\n"
        
        prompt = self.prompt_template.format(facts_content=facts_content)
        analysis = self._get_llm_json_response(prompt)
        
        return {
            "id": str(uuid.uuid4()),
            "content": analysis.get('scene_summary', 'Scene Summary'),
            "level": 'scene',
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "keywords": analysis.get('keywords', []),
            "tags": analysis.get('tags', [])
        }

    def refine_single_scene(self, current_summary, user_profile_text):
        """Refines a single scene's summary by appending new insights."""
        prompt = self.refinement_prompt_template.format(
            user_profile=user_profile_text,
            current_summary=current_summary
        )
        
        analysis = self._get_llm_json_response(prompt)
        
        # Relaxed logic from Ver 6: as long as 'addition' is not empty, we update
        addition = analysis.get("addition", "").strip()
        if addition and len(addition) > 5:
            # Additive update
            return f"{current_summary}\n[Insight]: {addition}"
            
        return None
    ####

class PersonaManager(BaseManager):
    """Manages Level 2 (Persona) memory construction."""
    
    def __init__(self, llm_controller, retriever):
        super().__init__(llm_controller, retriever)

        self.prompt_template = """
        You are an AI Profiler specialized in psychological and behavioral analysis.
        
        Task: Create a COMPREHENSIVE User Profile based on the provided scene memories.
        
        Input Scenes:
        {scenes_content}
        
        Instructions:
        1. Analyze the scenes deeply. Look for patterns in behavior, emotion, and choices.
        2. For each dimension below, write a DETAILED paragraph (5-10 sentences). Do not be brief.
        3. Use specific examples from the scenes to support your analysis.
        
        Format the response as a JSON object:
        {{
            "basic_info": "Detailed background...",
            "interests": "Comprehensive list of hobbies and how they engage with them...",
            "personality": "In-depth personality analysis...",
            "values": "Core beliefs and motivations...",
            "relationships": "Detailed social dynamics..."
        }}
        """
        self.dimensions = ["basic_info", "interests", "personality", "values", "relationships"]

    def build_profile(self, graph, scene_nodes):
        """Builds Level 2 nodes from a list of scene node IDs."""
        scenes_content = ""
        for nid in scene_nodes:
            node_data = graph.nodes[nid]
            scenes_content += f"- {node_data.get('content', '')}\n"
            
        prompt = self.prompt_template.format(scenes_content=scenes_content)
        analysis = self._get_llm_json_response(prompt)
        
        new_nodes_data = []
        
        # Pre-calculate embeddings for selective linking
        # (This logic is moved from AgenticMemorySystem to here)
        scene_texts = [graph.nodes[nid].get('content', '') for nid in scene_nodes]
        scene_embeddings = self.retriever.model.encode(scene_texts, convert_to_tensor=True)
        
        for dim in self.dimensions:
            content = analysis.get(dim, "")
            if not content or content == "N/A": continue
            
            # Selective Linking Logic
            dim_embedding = self.retriever.model.encode([content], convert_to_tensor=True) 
            cos_scores = util.cos_sim(dim_embedding, scene_embeddings)[0]
            top_k = max(5, int(len(scene_nodes) * 0.2))
            top_results = torch.topk(cos_scores, k=min(top_k, len(scene_nodes)))
            
            links = []
            for score, idx in zip(top_results.values, top_results.indices):
                if score > 0.2:
                    links.append((scene_nodes[idx], score.item()))

            new_nodes_data.append({
                "id": str(uuid.uuid4()),
                "content": content,
                "level": "persona",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tags": [dim.replace("_", " ").title()],
                "attribute_type": dim,
                "links": links # List of (target_id, weight)
            })
            
        return new_nodes_data