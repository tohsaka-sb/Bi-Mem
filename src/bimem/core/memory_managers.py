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
        summary = analysis.get('scene_summary')
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError('Scene generation failed: no valid scene_summary')
        
        return {
            "id": str(uuid.uuid4()),
            "content": summary.strip(),
            "level": 'scene',
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "keywords": analysis.get('keywords', []),
            "tags": analysis.get('tags', [])
        }

    def update_scene(self, scene, fact):
        """Update from the previous summary plus one new fact, not all facts."""
        prompt = (
            'Incrementally update a scene memory using the existing summary and one new fact. '
            'Preserve relevant existing details, people, dates and temporal distinctions. '
            'Treat any existing [Insight] as interpretation, not factual evidence. '
            'Do not invent events or replace newer facts with older evidence. '
            'Return JSON with scene_summary (nonempty string), keywords (string list), '
            'and tags (string list).\nExisting scene:\n'
            + json.dumps(scene, ensure_ascii=False, default=str)
            + '\nNew fact:\n' + json.dumps(fact, ensure_ascii=False, default=str)
        )
        analysis = self._get_llm_json_response(prompt)
        content = analysis.get('scene_summary')
        if not isinstance(content, str) or not content.strip():
            raise ValueError('Incremental scene update failed: no valid scene_summary')
        result = {'content': content.strip()}
        for key in ('keywords', 'tags'):
            value = analysis.get(key, scene.get(key, []))
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError(f'Invalid scene {key}')
            result[key] = value
        return result

    def compress_facts(self, graph, fact_ids):
        """Lossy compression must succeed before its source facts can be deleted."""
        facts = [dict(graph.nodes[n]) for n in fact_ids]
        prompt = (
            'Compress the following old factual memories into ONE concise factual memory. '
            'Retain the core events, entities and chronological changes; omit minor detail. '
            'Preserve distinctions between different people and conflicting dates. '
            'Do not infer personality, invent facts, or use persona insights as evidence. '
            'Return JSON with content (nonempty string), context (string), '
            'keywords (string list), tags (string list).\nFacts:\n'
            + json.dumps(facts, ensure_ascii=False, default=str)
        )
        analysis = self._get_llm_json_response(prompt)
        content = analysis.get('content')
        if not isinstance(content, str) or not content.strip():
            raise ValueError('Fact compression failed: original memories were retained')
        result = {'content': content.strip(), 'context': analysis.get('context', '')}
        if not isinstance(result['context'], str):
            raise ValueError('Invalid compressed context')
        for key in ('keywords', 'tags'):
            value = analysis.get(key, [])
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError(f'Invalid compressed {key}')
            result[key] = value
        return result

    def refine_single_scene(self, current_summary, user_profile_text):
        """Refines a single scene's summary by appending new insights."""
        prompt = self.refinement_prompt_template.format(
            user_profile=user_profile_text,
            current_summary=current_summary
        )
        
        analysis = self._get_llm_json_response(prompt)

        addition = analysis.get("addition", "").strip()
        if addition and len(addition) > 5:

            return f"{current_summary}\n[Insight]: {addition}"
            
        return None

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

    def analyze_scene(self, scene, profiles):
        """Classify relevance/conflicts; never let the LLM decide time precedence."""
        prompt = (
            'Analyze ONE newly created scene against the existing persona claims. '
            'Use scene memories as evidence; no raw fact input is provided. '
            'For each relevant dimension, return at most one proposal. Allowed dimensions: '
            'basic_info, interests, personality, values, relationships. '
            'Classify first: adds = a new compatible standalone conclusion, '
            'supports = confirms existing claims without changing them, '
            'conflicts = contradicts specific existing claims, unrelated = no update. '
            'Do not disguise contradictions as additions. Treat one-off events and '
            'persona-derived [Insight] as insufficient evidence of enduring traits. '
            'For conflicts, identify only the contradictory claim IDs; preserve all '
            'unrelated claims. Do NOT resolve conflicts or choose the newer view: '
            'the program compares source event times after classification. '
            'Content must be supported solely by the incoming scene, not a rewritten '
            'whole profile. For supports/conflicts, target_claim_ids must identify '
            'existing claims in that dimension. Adds/unrelated must have no targets. '
            'Return JSON {"decisions": [{"dimension": "interests", "relation": '
            '"adds|supports|conflicts|unrelated", "content": "standalone claim", '
            '"target_claim_ids": []}]}. Empty decisions means no relevant evidence. '
            '\nExisting persona claims and source evidence:\n'
            + json.dumps(profiles, ensure_ascii=False, default=str)
            + '\nNew scene:\n' + json.dumps(scene, ensure_ascii=False, default=str)
        )
        analysis = self._get_llm_json_response(prompt)
        decisions = analysis.get('decisions') if isinstance(analysis, dict) else None
        if not isinstance(decisions, list):
            raise ValueError('Invalid scene/persona classification')
        validated = []
        seen = set()
        for decision in decisions:
            if not isinstance(decision, dict):
                raise ValueError('Persona decision must be an object')
            dimension = decision.get('dimension')
            relation = decision.get('relation')
            targets = decision.get('target_claim_ids')
            if not isinstance(dimension, str) or dimension not in self.dimensions or dimension in seen:
                raise ValueError('Invalid or repeated persona dimension')
            seen.add(dimension)
            if relation not in ('adds', 'supports', 'conflicts', 'unrelated'):
                raise ValueError('Invalid persona relation')
            if not isinstance(targets, list) or not all(isinstance(t, str) for t in targets):
                raise ValueError('Invalid target_claim_ids')
            known_ids = {claim['id'] for claim in profiles.get(dimension, [])}
            if len(set(targets)) != len(targets) or not set(targets) <= known_ids:
                raise ValueError('Persona decision refers to an unknown or duplicate claim')
            if bool(targets) != (relation in ('supports', 'conflicts')):
                raise ValueError('Persona target claims do not match the relation')
            if relation == 'unrelated':
                continue
            content = decision.get('content')
            if not isinstance(content, str) or not content.strip():
                raise ValueError('Persona proposal must contain nonempty content')
            validated.append({'dimension': dimension, 'relation': relation,
                              'content': content.strip(), 'target_claim_ids': targets})
        return validated

    def build_profile(self, graph, scene_nodes):
        """Builds Level 2 nodes from a list of scene node IDs."""
        scenes_content = ""
        for nid in scene_nodes:
            node_data = graph.nodes[nid]
            scenes_content += f"- {node_data.get('content', '')}\n"
            
        prompt = self.prompt_template.format(scenes_content=scenes_content)
        analysis = self._get_llm_json_response(prompt)
        
        new_nodes_data = []

        scene_texts = [graph.nodes[nid].get('content', '') for nid in scene_nodes]
        scene_embeddings = self.retriever.model.encode(scene_texts, convert_to_tensor=True)
        
        for dim in self.dimensions:
            content = analysis.get(dim, "")
            if not content or content == "N/A": continue

            dim_embedding = self.retriever.model.encode([content], convert_to_tensor=True) 
            cos_scores = util.cos_sim(dim_embedding, scene_embeddings)[0]
            top_k = max(5, int(len(scene_nodes) * 0.2))
            top_results = torch.topk(cos_scores, k=min(top_k, len(scene_nodes)))
            
            links = []
            for score, idx in zip(top_results.values, top_results.indices):
                if score > 0.6:
                    links.append((scene_nodes[idx], score.item()))

            new_nodes_data.append({
                "id": str(uuid.uuid4()),
                "content": content,
                "level": "persona",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tags": [dim.replace("_", " ").title()],
                "attribute_type": dim,
                "links": links
            })
            
        return new_nodes_data
