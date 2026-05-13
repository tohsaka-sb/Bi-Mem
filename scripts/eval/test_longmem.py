from src.bimem.core.memory_layer import LLMController, AgenticMemorySystem
import os
import json
import argparse
import logging
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from src.bimem.core.load_dataset import load_locomo_dataset, QA, Turn, Session, Conversation

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
import nltk

def _parse_judge_score(raw: str) -> float:
    """Parse 0-100 score from LLM output; return 50.0 (neutral) on failure."""
    s = str(raw or "").strip()
    m = re.search(r"\b(\d{1,3})\b", s)
    if m:
        val = float(m.group(1))
        return min(100.0, max(0.0, val))
    return 50.0

def calculate_llm_judge_score(
    question: str,
    prediction: str,
    reference: str,
    client: "OpenAI",
    model: str = "gpt-4o",
) -> float:
    """
    LLM-as-Judge: score how well the prediction answers the question (0-100).
    100=fully correct, 0=completely wrong.
    """
    if not client or not reference:
        return 50.0
    prompt = f"""You are a strict evaluator. Score how well the model's prediction answers the question, given the reference answer.

Question: {question}
Reference answer: {reference}
Model's prediction: {prediction}

Scoring (0-100):
- 100: Fully correct, semantically equivalent or highly accurate
- 75-99: Mostly correct, minor omissions or paraphrasing
- 50-74: Partially correct
- 25-49: Mostly wrong, significant errors
- 0-24: Completely wrong or irrelevant

Reply with ONLY an integer score from 0 to 100 (no other text)."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Reply with only an integer from 0 to 100."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        raw = (response.choices[0].message.content or "").strip()
        return _parse_judge_score(raw)
    except Exception as e:
        print(f"[WARN] LLM Judge API error: {e}")
        return 50.0

@dataclass
class LongMemSample:
    """One QA item from longmemeval_s_cleaned.json with its haystack conversation."""
    sample_id: str
    qa: List[QA]
    conversation: Conversation

def load_longmemeval_dataset(file_path: Union[str, Path], ratio: float = 1.0, max_samples: Optional[int] = None) -> List[LongMemSample]:
    """
    Load longmemeval_s_cleaned.json. Each list item has:
      question_id, question_type, question, question_date, answer,
      haystack_dates (list of date strings), haystack_sessions (list of sessions; each session = list of {role, content}).
    Returns list of LongMemSample compatible with the same eval loop (sample_id, conversation, qa).
    Note: For very large JSON files, the whole file is loaded into memory; use ratio < 1.0 or max_samples to limit.
    """
    file_path = Path(file_path) if isinstance(file_path, str) else file_path
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("longmemeval JSON should be a list of QA items")
    n_total = len(data)
    n_use = max(1, int(n_total * ratio)) if ratio < 1.0 else n_total
    if max_samples is not None:
        n_use = min(n_use, max_samples)
    data = data[:n_use]
    samples = []
    for idx, item in enumerate(data):
        haystack_sessions = item.get("haystack_sessions") or []
        haystack_dates = item.get("haystack_dates") or []
        sessions_dict = {}
        for i, sess in enumerate(haystack_sessions):
            date_time = haystack_dates[i] if i < len(haystack_dates) else ""
            turns = [
                Turn(speaker=msg.get("role", "user"), dia_id="", text=msg.get("content", ""))
                for msg in sess
                if msg.get("content")
            ]
            if turns:
                sessions_dict[i] = Session(session_id=i, date_time=date_time, turns=turns)
        conversation = Conversation(speaker_a="user", speaker_b="assistant", sessions=sessions_dict)
        qa_list = [
            QA(
                question=item.get("question", ""),
                answer=item.get("answer"),
                evidence=item.get("answer_session_ids", []) or [],
                category=4,
                adversarial_answer=None,
            )
        ]
        sample_id = item.get("question_id", str(idx))
        samples.append(LongMemSample(sample_id=sample_id, qa=qa_list, conversation=conversation))
    return samples
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim
import statistics
from collections import defaultdict
import pickle
import random
from tqdm import tqdm
from src.bimem.core.utils import calculate_metrics, aggregate_metrics
from datetime import datetime
import re

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')

try:
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Warning: Could not load SentenceTransformer model: {e}")
    sentence_model = None

INCLUDE_SCENE_INSIGHTS_IN_ANSWER_CONTEXT = False

class advancedMemAgent:
    def __init__(self, model, backend, retrieve_k, api_key=None, retrieval_strategy="graph_traversal", top_k_levels=None, include_scene_insights=None):
        self.include_scene_insights = include_scene_insights if include_scene_insights is not None else INCLUDE_SCENE_INSIGHTS_IN_ANSWER_CONTEXT
        self.memory_system = AgenticMemorySystem(
            model_name='all-MiniLM-L6-v2',
            llm_backend=backend,
            llm_model=model,
            api_key=api_key
        )
        self.retriever_llm = LLMController(
            backend=backend, 
            model=model, 
            api_key=api_key
        )
        self.retrieve_k = retrieve_k
        self.retrieval_strategy = retrieval_strategy
        self.top_k_levels = top_k_levels or [2, 3, 5]

    def add_memory(self, content, time=None):
        self.memory_system.add_note(content, time=time)

    def retrieve_memory(self, content, k=10):
        return self.memory_system.find_related_memories(
            content, 
            k=k, 
            strategy=self.retrieval_strategy, 
            top_k_levels=self.top_k_levels,
            include_scene_insights=self.include_scene_insights
        )
    def retrieve_memory_llm(self, memories_text, query):
        prompt = f"""Given the following conversation memories and a question, select the most relevant parts of the conversation that would help answer the question. Include the date/time if available.

                Conversation memories:
                {memories_text}

                Question: {query}

                Return only the relevant parts of the conversation that would help answer this specific question. Format your response as a JSON object with a "relevant_parts" field containing the selected text. 
                If no parts are relevant, do not do any things just return the input.

                Example response format:
                {{"relevant_parts": "2024-01-01: Speaker A said something relevant..."}}"""
            
        response = self.retriever_llm.llm.get_completion(prompt,response_format={"type": "json_schema", "json_schema": {
                            "name": "response",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "relevant_parts": {
                                        "type": "string",
                                    }
                                },
                                "required": ["relevant_parts"],
                                "additionalProperties": False
                            },
                            "strict": True
                        }})
        return response
    
    def generate_query_llm(self, question):
        prompt = f"""Given the following question, generate several keywords, using 'cosmos' as the separator.

                Question: {question}

                Format your response as a JSON object with a "keywords" field containing the selected text. 

                Example response format:
                {{"keywords": "keyword1, keyword2, keyword3"}}"""
            
        response = self.retriever_llm.llm.get_completion(prompt,response_format={"type": "json_schema", "json_schema": {
                            "name": "response",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "keywords": {
                                        "type": "string",
                                    }
                                },
                                "required": ["keywords"],
                                "additionalProperties": False
                            },
                            "strict": True
                        }})

        try:

            import re
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                response_json = json.loads(match.group(0))
                response = response_json.get("keywords", response)
            else:
                response = response.strip()
        except:
            response = response.strip()
        return response

    def answer_question(self, question: str, category: int, answer: str) -> str:
        """Generate answer for a question given the conversation context."""
        keywords = self.generate_query_llm(question)
        if category == 2:
            current_k = 25
        elif category == 4:
            current_k = 25
        elif category == 3:
            current_k = 30
        else:
            current_k = 35

        raw_context, retrieved_ids = self.retrieve_memory(keywords, k=current_k)
        context = raw_context
        
        assert category in [1,2,3,4,5]

        format_instruction = """
        Write an answer in the form of **a short phrase**, not a complete sentence. 
        Answer with exact words from the context whenever possible.
        """

            
        if category == 2:
            user_prompt = f"""
                            Based on the context provided below:
                            {context}

                            Question: {question}

                            Instructions:
                            1. Identify the specific [EVENT] related to the question.
                            2. Look at the [Time: ...] timestamp of that fact. Treat this as the "Reference Date".
                            
                            3. **Date Format**: 
                               - You MUST use the format "Day Month, Year" (e.g., "15 July, 2023"). 
                               - Note the COMMA after the month.
                            
                            4. **Relative Time Logic (CRITICAL)**:
                               - If the context says "last week" or "a week ago", answer in the format: "The week before [Reference Date]".
                               - If the context says "last month", answer: "The month before [Reference Date]".
                               - If the context says "next week", answer: "The week after [Reference Date]".
                               - If the context says "yesterday" or specific days, calculate the exact date: "14 July, 2023".
                               
                            5. **Examples**:
                               - Context Time: "20 May, 2023". Text: "I went last week." -> Answer: "The week before 20 May, 2023"
                               - Context Time: "April, 2023". Text: "Going next month." -> Answer: "May, 2023"
                            
                            6. If the question asks for duration, answer in "X years/months/days".
                            
                            Short answer:
                            """
            temperature = 0.0 

        elif category == 3:
            inference_format_instruction = """
        Write an answer in the form of **a short phrase**, not a complete sentence.
        The question may need you to **analyze and infer** the answer from the summary, rather than finding exact words.
        """
            temperature = 0.0
            q_lower = question.lower()

            is_choice = " or " in q_lower and "?" in q_lower
            is_yes_no = q_lower.startswith(("is ", "are ", "do ", "does ", "did ", "can ", "could ", "would ", "will ", "should "))
            
            if is_choice:
                cat3_instruction = """
                1. The question asks you to choose between options (e.g., "A or B?").
                2. Answer with **ONLY** the chosen option. Do not add "I think" or reasons.
                3. Example: "beach" (NOT "He prefers the beach").
                """
            elif is_yes_no:
                cat3_instruction = """
                1. Answer "Yes", "No", "Likely", or "Unlikely".
                2. You MAY add a very short reason (3-5 words) starting with ";".
                3. Example: "Yes; she loves art."
                """
            else:
                cat3_instruction = """
                1. Analyze the user's personality and history to infer the answer.
                2. If the answer is not explicitly stated, make an **educated guess** based on context clues. Avoid saying "Not specified".
                3. Write a **very short phrase** (1-5 words). No full sentences.
                """

            user_prompt = f"""
                            Based on the context provided below:
                            {context}

                            Question: {question}

                            Instructions:
                            {cat3_instruction}
                            
                            Short answer:
                            """

        elif category == 1:
            inference_format_instruction = """
        Write an answer in the form of **a short phrase**, not a complete sentence.
        The question may need you to **analyze and infer** the answer from the summary, rather than finding exact words.
        """
            
            temperature = 0.0 
            q_lower = question.lower()
            is_yes_no = q_lower.startswith(("is ", "are ", "do ", "does ", "did ", "can ", "could ", "would ", "will ", "should "))
            is_count = q_lower.startswith(("how many", "how much"))
            
            if is_yes_no:

                cat1_specific_instruction = """
                - Answer with **EXACTLY ONE word** from this list: [Yes, No, Likely, Unlikely]. 
                - Do NOT add any explanation or punctuation.
                """
            elif is_count:

                cat1_specific_instruction = """
                - Answer with **only the number** (e.g., "3" or "three").
                - Do NOT add words like "times", "items", "people".
                - Do NOT use "once" or "twice"; use numeric form instead.
                """
            else:

                cat1_specific_instruction = """
                - **List Exhaustion**: If the question asks for multiple items (e.g., "Which books"), list **ALL** specific entities mentioned in [EVENT] blocks. Do NOT summarize.
                - **Reference Resolution**: If the text mentions a pronoun (e.g., "my home country"), you MUST search the context to find the **specific proper noun** (e.g., "Sweden").
                - **Temporal Filtering**: If the question asks about "recently" or "last", compare [Time: ...] stamps to find the latest fact.
                - **Exact Name**: Provide the full, exact name of the entity if asked.
                """

            user_prompt = f"""
                            Based on the context provided below:
                            {context}

                            Question: {question}

                            Instructions:
                            1. This is a Multi-Hop question. You need to connect pieces of information.
                            2. {cat1_specific_instruction}
                            3. Write an answer in the form of **a short phrase**, not a complete sentence.
                            
                            Short answer:
                            """
            temperature = 0.0
            
        else:

            exact_format_instruction = """
        Write an answer in the form of **a short phrase**, not a complete sentence. 
        Answer with **exact words** from the context whenever possible.
        """
            user_prompt = f"""
                            Based on the context provided below:
                            {context}

                            Question: {question}

                            Instructions:
                            1. {format_instruction}
                            
                            Short answer:
                            """
            temperature = 0.0

        response = self.memory_system.llm_controller.llm.get_completion(
            user_prompt,response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "answer": {
                                    "type": "string",
                                }
                            },
                            "required": ["answer"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }},temperature=temperature
        )
        prediction = ""
        
        try:
            clean_response = response.strip()

            match = re.search(r"```json\s*(\{.*\})\s*```", clean_response, re.DOTALL)
            if match:
                clean_response = match.group(1)
            elif re.search(r"```\s*(\{.*\})\s*```", clean_response, re.DOTALL):
                clean_response = re.search(r"```\s*(\{.*\})\s*```", clean_response, re.DOTALL).group(1)
            
            response_json = json.loads(clean_response)

            local_prediction = None 

            if isinstance(response_json, dict):
                for key in ["answer", "short_answer", "prediction", "result", "content"]:
                    if key in response_json:
                        local_prediction = str(response_json[key])
                        break

                if not local_prediction and response_json:
                    local_prediction = str(list(response_json.values())[0])

                if not local_prediction:
                     local_prediction = str(response_json)

            else:
                local_prediction = str(response_json)

            if local_prediction:
                prediction = local_prediction
                
        except (json.JSONDecodeError, AttributeError, ValueError):
            prediction = response.strip()

        if prediction:
            prefixes_to_remove = ["Answer:", "Short answer:", "The answer is", "Prediction:"]
            for prefix in prefixes_to_remove:
                if prediction.lower().startswith(prefix.lower()):
                    prediction = prediction[len(prefix):].strip()
                    break 
            
            if prediction.startswith('"') and prediction.endswith('"'):
                prediction = prediction[1:-1]
        
        if not prediction and response:
            prediction = response.strip()

                
        return prediction, user_prompt, raw_context, retrieved_ids

def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger('locomo_eval')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def save_checkpoint_internal(agent, mem_file, ret_file, ret_emb_file, turns):
    """Standalone helper to save checkpoints."""
    try:
        with open(mem_file, 'wb') as f:
            pickle.dump({'turns_processed': turns, 'graph': agent.memory_system.graph}, f)
        agent.memory_system.retriever.save(ret_file, ret_emb_file)
    except Exception as e:
        print(f"Error saving checkpoint: {e}")

def evaluate_longmem_dataset(
    dataset_path: str,
    model: str,
    output_path: Optional[str] = None,
    ratio: float = 1.0,
    max_samples: Optional[int] = None,
    backend: str = "openai",
    retrieve_k: int = 10,
    api_key: Optional[str] = None,
    retrieval_strategy: str = "graph_traversal",
    top_k_levels: List[int] = None,
    enable_refinement: bool = False,
    no_llm_judge: bool = False,
    judge_model: str = "gpt-4o",
):
    """
    Evaluate the agent on the LongMemEval (longmemeval_s_cleaned.json) dataset.
    Same pipeline as LoCoMo: build memory from haystack_sessions -> hierarchy -> answer each question.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    model_name_sanitized = model.replace("/", "_").replace(":", "_")
    log_filename = f"eval_longmem_{model_name_sanitized}_{backend}_ratio{ratio}_{timestamp}.log"
    log_path = os.path.join(os.path.dirname(__file__), "logs", log_filename)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = setup_logger(log_path)
    logger.info(f"Loading LongMemEval dataset from {dataset_path}")
    samples = load_longmemeval_dataset(dataset_path, ratio=ratio, max_samples=max_samples)
    logger.info(f"Loaded {len(samples)} LongMemEval samples")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if not os.path.exists(output_path):
            with open(output_path, 'w', encoding='utf-8') as f:
                logger.info(f"Initializing new results file at {output_path}")
                meta_info = {
                    "type": "metadata",
                    "dataset_type": "longmemeval",
                    "model": model,
                    "backend": backend,
                    "dataset": str(dataset_path),
                    "ratio": ratio,
                    "max_samples": max_samples,
                    "retrieval_strategy": retrieval_strategy,
                    "evaluation_start_time": datetime.now().isoformat()
                }
                f.write(json.dumps(meta_info) + "\n")
        else:
            logger.info(f"Found existing results file. Will resume and append to {output_path}")

    all_metrics = []
    all_categories = []
    total_questions = 0
    category_counts = defaultdict(int)
    error_num = 0

    if output_path:
        base_cache_dir = os.path.dirname(output_path)
    else:
        base_cache_dir = os.path.dirname(__file__)

    memories_dir = os.path.join(base_cache_dir, "cached_memories_longmem")
    os.makedirs(memories_dir, exist_ok=True)
    logger.info(f"Using memory cache directory: {memories_dir}")

    judge_client = None
    if not no_llm_judge and OpenAI:
        judge_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if judge_api_key:
            judge_client = OpenAI(api_key=judge_api_key, base_url=os.getenv("OPENAI_API_BASE"))
            logger.info(f"LLM-as-Judge enabled (model={judge_model})")
        else:
            logger.warning("OPENAI_API_KEY not set; skipping LLM-as-Judge")
    elif no_llm_judge:
        logger.info("LLM-as-Judge disabled (--no-llm-judge)")

    allow_categories = [4]

    for sample_idx, sample in enumerate(samples):
        agent = advancedMemAgent(model, backend, retrieve_k, api_key=api_key, retrieval_strategy=retrieval_strategy, top_k_levels=top_k_levels)
        
        memory_cache_file = os.path.join(
            memories_dir,
            f"memory_cache_sample_{sample.sample_id}.pkl"
        )
        retriever_cache_file = os.path.join(
            memories_dir,
            f"retriever_cache_sample_{sample.sample_id}_retriever.pkl"
        )
        retriever_cache_embeddings_file = os.path.join(
            memories_dir,
            f"retriever_cache_sample_{sample.sample_id}_embeddings.npy"
        )
        
        CHECKPOINT_INTERVAL = 10
        turns_processed = 0
        import networkx as nx
        cached_graph = nx.DiGraph() 

        if os.path.exists(memory_cache_file):
            logger.info(f"Loading cached memories for sample {sample.sample_id} from {memory_cache_file}")
            try:
                with open(memory_cache_file, 'rb') as f:
                    saved_data = pickle.load(f)
                    turns_processed = saved_data.get('turns_processed', 0)
                    cached_graph = saved_data.get('graph', nx.DiGraph())
            except (pickle.UnpicklingError, EOFError, AttributeError, ImportError) as e:
                logger.warning(f"Could not load memory checkpoint file (error: {e}). Starting from scratch for this sample.")
                turns_processed = 0
                cached_graph = nx.DiGraph()
            
            agent.memory_system.graph = cached_graph
            
            if turns_processed > 0 and os.path.exists(retriever_cache_file) and os.path.exists(retriever_cache_embeddings_file):
                logger.info(f"Loading retriever from cache for sample {sample.sample_id}")
                agent.memory_system.retriever.load(retriever_cache_file, retriever_cache_embeddings_file)
            else:
                if agent.memory_system.graph.number_of_nodes() > 0:
                    logger.warning(f"Retriever cache not found or is outdated. Rebuilding retriever from loaded graph.")
                    all_node_ids = list(agent.memory_system.graph.nodes())
                    from src.bimem.core.memory_layer import SimpleEmbeddingRetriever
                    try:
                        agent.memory_system.retriever = SimpleEmbeddingRetriever('all-MiniLM-L6-v2')
                    except:
                        pass
                    agent.memory_system._update_retriever(all_node_ids)

            logger.info(f"Successfully loaded {agent.memory_system.graph.number_of_nodes()} memories, resuming from turn {turns_processed}.")
        else:
            logger.info(f"No cached memories found for sample {sample.sample_id}. Starting new memory creation.")

        all_turns = []
        for _, session in sorted(sample.conversation.sessions.items()):
            for turn in session.turns:
                all_turns.append((session.date_time, turn))

        turns_to_process = all_turns[turns_processed:]
        
        if turns_to_process:
            logger.info(f"Processing {len(turns_to_process)} remaining turns for sample {sample.sample_id}.")
            
            try:
                with tqdm(total=len(all_turns), initial=turns_processed, desc=f"Building memory for sample {sample.sample_id}") as pbar:
                    for i, (turn_datetime, turn) in enumerate(turns_to_process):
                        conversation_tmp = f"Speaker {turn.speaker} says: {turn.text}"
                        agent.add_memory(conversation_tmp, time=turn_datetime)
                        
                        current_turn_index = turns_processed + i + 1

                        if current_turn_index % CHECKPOINT_INTERVAL == 0:
                            logger.info(f"\n--- Saving checkpoint at turn {current_turn_index} ---")
                            save_checkpoint_internal(agent, memory_cache_file, retriever_cache_file, retriever_cache_embeddings_file, current_turn_index)
                            logger.info(f"Checkpoint saved successfully.")
                        
                        pbar.update(1)

            except Exception as e:
                current_loop_index = i if 'i' in locals() else 0
                logger.error(f"An error occurred during memory building at global turn index {turns_processed + 1 + current_loop_index}: {e}")
                logger.error("Attempting to save progress before exiting...")
                final_turns_processed = turns_processed + current_loop_index
                save_checkpoint_internal(agent, memory_cache_file, retriever_cache_file, retriever_cache_embeddings_file, final_turns_processed)
                logger.error("Progress saved. Now re-raising the exception.")
                raise e
        else:
            logger.info(f"All turns for sample {sample.sample_id} already processed. Skipping memory building.")

        if turns_to_process:
            logger.info(f"\nFinished building memory for sample {sample.sample_id}. Saving final state.")
            final_turns_processed = len(all_turns)
            save_checkpoint_internal(agent, memory_cache_file, retriever_cache_file, retriever_cache_embeddings_file, final_turns_processed)
            logger.info(f"Final memory state saved for {agent.memory_system.graph.number_of_nodes()} memories.")

        logger.info(f"\nStarting Hierarchy Construction for sample {sample.sample_id}")
        try:

            agent.memory_system.build_hierarchy_batch(
                save_callback=lambda: save_checkpoint_internal(agent, memory_cache_file, retriever_cache_file, retriever_cache_embeddings_file, len(all_turns)),
                enable_refinement=enable_refinement
            )
            logger.info(f"Hierarchy construction completed. Saving final graph state.")
            save_checkpoint_internal(agent, memory_cache_file, retriever_cache_file, retriever_cache_embeddings_file, len(all_turns))
        except Exception as e:
            logger.error(f"An error occurred during hierarchy construction: {e}")
            logger.error("Attempting to save progress before exiting...")
            save_checkpoint_internal(agent, memory_cache_file, retriever_cache_file, retriever_cache_embeddings_file, len(all_turns))
            raise e

        logger.info(f"\nAnswering questions for sample {sample.sample_id}")

        processed_questions_set = set()
        if output_path and os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get("type") == "qa_result":

                             if str(entry.get("sample_id")) == str(sample.sample_id):
                                 q_text = entry.get("question", "").strip()
                                 processed_questions_set.add(q_text)
                    except json.JSONDecodeError:
                        continue
        
        logger.info(f"Found {len(processed_questions_set)} already processed unique questions for sample {sample.sample_id}.")

        qas_in_sample = [qa for qa in sample.qa if int(qa.category) in allow_categories]
        qas_to_process = []

        qas_to_process_with_index = []
        
        for idx, qa in enumerate(qas_in_sample):
            if qa.question.strip() not in processed_questions_set:
                qas_to_process_with_index.append((idx, qa))
        
        logger.info(f"Queueing {len(qas_to_process_with_index)} questions to answer.")

        if qas_to_process_with_index:

            initial_progress = len(qas_in_sample) - len(qas_to_process_with_index)
            
            with tqdm(total=len(qas_in_sample), initial=initial_progress, desc=f"Answering questions for sample {sample.sample_id}") as pbar:
                for original_idx, qa in qas_to_process_with_index:

                    global_question_id = original_idx + 1

                    total_questions += 1
                    category_counts[qa.category] += 1
                    
                    prediction, user_prompt, raw_context, retrieved_node_ids = agent.answer_question(qa.question, qa.category, qa.final_answer)
                    
                    try:

                        if isinstance(prediction, str) and prediction.strip().startswith('{'):
                             try:
                                 pred_json = json.loads(prediction)
                                 prediction = pred_json.get("answer", prediction)
                             except:
                                 pass
                    except:
                        pass

                    prediction = str(prediction)

                    metrics = calculate_metrics(prediction, qa.final_answer) if qa.final_answer else {
                        "exact_match": 0, "f1": 0.0, "rouge1_f": 0.0, "rouge2_f": 0.0, 
                        "rougeL_f": 0.0, "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, 
                        "bleu4": 0.0, "bert_f1": 0.0, "meteor": 0.0, "sbert_similarity": 0.0,
                        "llm_as_judge": 50.0
                    }
                    if judge_client and qa.final_answer:
                        llm_score = calculate_llm_judge_score(
                            qa.question, prediction, qa.final_answer,
                            judge_client, model=judge_model
                        )
                        metrics["llm_as_judge"] = llm_score
                    else:
                        metrics["llm_as_judge"] = 50.0

                    core_metrics = {
                        "f1": metrics.get("f1", 0.0),
                        "bleu1": metrics.get("bleu1", 0.0),
                        "llm_as_judge": metrics.get("llm_as_judge", 50.0),
                    }

                    logger.info(f"\nQuestion {global_question_id} (Sample {sample.sample_id}): {qa.question}")
                    logger.info(f"Prediction: {prediction}")
                    logger.info(f"Reference: {qa.final_answer}")
                    logger.info(f"Metrics: F1={core_metrics['f1']:.4f}, BLEU-1={core_metrics['bleu1']:.4f}, LLM-as-Judge={core_metrics['llm_as_judge']:.1f}")
                    
                    all_metrics.append(metrics)
                    all_categories.append(qa.category)
                    
                    result_to_save = {
                        "type": "qa_result",
                        "sample_id": sample.sample_id,
                        "question_id_in_sample": global_question_id,
                        "question": qa.question,
                        "prediction": prediction,
                        "reference": qa.final_answer,
                        "category": qa.category,
                        "metrics": metrics,
                        "f1": core_metrics["f1"],
                        "bleu1": core_metrics["bleu1"],
                        "llm_as_judge": core_metrics["llm_as_judge"],
                        "retrieved_nodes": retrieved_node_ids
                    }
                    
                    if output_path:
                        with open(output_path, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(result_to_save, ensure_ascii=False) + "\n")
                    
                    pbar.update(1)
        else:
            logger.info("All questions for this sample are already processed.")

    
    if total_questions > 0:
        aggregate_results = aggregate_metrics(all_metrics, all_categories)
        
        run_summary = {
            "type": "run_summary",
            "model": model,
            "questions_processed_in_this_run": total_questions,
            "error_num_in_this_run": error_num,
            "category_distribution_in_this_run": {
                str(cat): count for cat, count in category_counts.items()
            },
            "aggregate_metrics_for_this_run": aggregate_results,
            "run_end_time": datetime.now().isoformat()
        }

        if output_path:
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(run_summary) + "\n")
            logger.info(f"Run finished. Summary for this run appended to {output_path}")

        logger.info("\nSummary for This Run:")
        logger.info(f"Total questions evaluated in this run: {total_questions}")
        logger.info(f"JSON parsing errors on predictions: {error_num}")
        logger.info("\nCategory Distribution in this run:")
        for category, count in sorted(category_counts.items()):
            logger.info(f"Category {category}: {count} questions ({count/total_questions*100:.1f}%)")
        
        logger.info("\nAggregate Metrics for this run:")
        for split_name, metrics in aggregate_results.items():
            logger.info(f"\n{split_name.replace('_', ' ').title()}:")
            for metric_name, stats in metrics.items():
                logger.info(f"  {metric_name}:")
                for stat_name, value in stats.items():
                    logger.info(f"    {stat_name}: {value:.4f}")
    else:
        logger.info("\nNo new questions were processed in this run.")
    
    return {}

def main():
    parser = argparse.ArgumentParser(description="Evaluate Bi-Mem agent on LongMemEval (longmemeval_s_cleaned.json) dataset")
    parser.add_argument("--dataset", type=str, default="data/longmemeval_s_cleaned.json",
                        help="Path to LongMemEval JSON file")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="LLM model to use")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save evaluation results (JSONL)")
    parser.add_argument("--ratio", type=float, default=0.01,
                        help="Fraction of dataset to use (0.0 to 1.0). Default 0.01 for large file.")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max number of samples to evaluate (overrides ratio if set). Useful for huge JSON.")
    parser.add_argument("--backend", type=str, default="openai",
                        help="Backend: openai, ollama, probex")
    parser.add_argument("--retrieve_k", type=int, default=10,
                        help="Base retrieve k for retrieval")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key for the selected backend")
    parser.add_argument("--enable-refinement", action="store_true",
                        help="Enable top-down refinement of scene memories")
    parser.add_argument("--no-llm-judge", action="store_true",
                        help="Disable LLM-as-Judge scoring (saves API calls)")
    parser.add_argument("--judge-model", type=str, default="gpt-4o",
                        help="Model for LLM-as-Judge (default: gpt-4o)")
    parser.add_argument("--retrieval-strategy", type=str, default="graph_traversal",
                        choices=["flat", "graph_traversal", "top_down", "bottom_up"],
                        help="Memory retrieval strategy")
    parser.add_argument("--top-k-levels", type=int, nargs=3, default=[2, 3, 5],
                        help="Top-k for L2, L1, L0 (for top_down/bottom_up). E.g. --top-k-levels 2 3 5")

    args = parser.parse_args()
    if args.ratio <= 0.0 or args.ratio > 1.0:
        raise ValueError("Ratio must be in (0.0, 1.0]")
    dataset_path = os.path.join(os.path.dirname(__file__), args.dataset)
    output_path = args.output if args.output else None
    evaluate_longmem_dataset(
        dataset_path, args.model, output_path,
        ratio=args.ratio, max_samples=args.max_samples,
        backend=args.backend, retrieve_k=args.retrieve_k,
        api_key=args.api_key, retrieval_strategy=args.retrieval_strategy,
        top_k_levels=args.top_k_levels, enable_refinement=args.enable_refinement,
        no_llm_judge=args.no_llm_judge, judge_model=args.judge_model,
    )

if __name__ == "__main__":
    main()
