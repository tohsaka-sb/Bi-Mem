import os
import json
import argparse
import logging
from typing import List, Optional
from collections import defaultdict
from datetime import datetime
import re

from tqdm import tqdm

from src.bimem.core.memory_layer import LLMController
from src.bimem.core.load_dataset import load_locomo_dataset
from src.bimem.core.utils import calculate_metrics, aggregate_metrics

LONG_CONTEXT_MODELS = {"gpt-5o", "gpt-5o-mini"}
DEFAULT_CONTEXT_TOKENS = 12000
LONG_CONTEXT_TOKENS = 32000

def resolve_context_window(model: str, max_context_tokens: Optional[int]) -> int:
    if max_context_tokens is not None:
        return max_context_tokens
    if model.lower() in LONG_CONTEXT_MODELS:
        return LONG_CONTEXT_TOKENS
    return DEFAULT_CONTEXT_TOKENS

def append_jsonl(path: Optional[str], payload: dict):
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

def estimate_tokens(text: str) -> int:
    """Approximate token count using a conservative chars-per-token ratio."""
    if not text:
        return 0
    return max(1, len(text) // 4)

def build_full_context(sample) -> str:
    parts = []
    for _, session in sorted(sample.conversation.sessions.items()):
        for turn in session.turns:
            parts.append(f"[Time: {session.date_time}] Speaker {turn.speaker} says: {turn.text}")
    return "\n".join(parts)

def truncate_keep_latest(text: str, max_tokens: int) -> str:
    """Keep newest lines when context exceeds token budget."""
    if estimate_tokens(text) <= max_tokens:
        return text

    lines = text.splitlines()
    kept = []
    kept_tokens = 0
    for line in reversed(lines):
        line_tokens = estimate_tokens(line + "\n")
        if kept and kept_tokens + line_tokens > max_tokens:
            break
        kept.append(line)
        kept_tokens += line_tokens

    if not kept:
        approx_chars = max_tokens * 4
        return text[-approx_chars:]
    kept.reverse()
    return "\n".join(kept)

class DirectContextAgent:
    def __init__(
        self,
        model: str,
        backend: str,
        api_key: Optional[str],
        max_context_tokens: int,
        empty_answer_retries: int = 2,
        raw_response_dump_path: Optional[str] = None,
    ):
        self.model = model
        self.backend = backend
        self.llm_controller = LLMController(backend=backend, model=model, api_key=api_key)
        self.max_context_tokens = max_context_tokens
        self.empty_answer_retries = max(0, empty_answer_retries)
        self.raw_response_dump_path = raw_response_dump_path

    @staticmethod
    def _extract_prediction(raw_response: str) -> str:
        prediction = ""
        try:
            clean_response = raw_response.strip()
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
            prediction = raw_response.strip()

        if prediction:
            for prefix in ["Answer:", "Short answer:", "The answer is", "Prediction:"]:
                if prediction.lower().startswith(prefix.lower()):
                    prediction = prediction[len(prefix):].strip()
                    break
            if prediction.startswith('"') and prediction.endswith('"'):
                prediction = prediction[1:-1]

        if not prediction and raw_response:
            prediction = raw_response.strip()
        return prediction

    def answer_question(
        self,
        question: str,
        category: int,
        sample_id: Optional[str],
        context: str,
        question_id: Optional[int] = None,
    ):
        assert category in [1, 2, 3, 4, 5]

        if category == 2:
            user_prompt = f"""
                            Based on the context provided below:
                            {context}

                            Question: {question}

                            Instructions:
                            1. Identify the specific [EVENT] related to the question.
                            2. Look at the [Time: ...] timestamp of that fact. Treat this as the "Reference Date".
                            3. Date Format: Day Month, Year (e.g., "15 July, 2023").
                            4. Relative Time Logic:
                               - "last week"/"a week ago" => "The week before [Reference Date]".
                               - "last month" => "The month before [Reference Date]".
                               - "next week" => "The week after [Reference Date]".
                               - "yesterday"/specific day => compute exact date.
                            5. If question asks duration, answer "X years/months/days".

                            Short answer:
                            """
            temperature = 0.0

        elif category == 3:
            temperature = 0.0
            q_lower = question.lower()
            sid = str(sample_id) if sample_id is not None else ""

            cat3_group_a = {"0", "1", "2", "3", "4"}
            cat3_group_b = {"5", "6", "7", "8", "9"}
            cat3_group_prompts = {
                "A": {
                    "choice": """1. Answer with the chosen option. 2. Use exact phrases from context for reason. 3. Short answer (1-3 words). 4. Plain text only.""",
                    "yes_no": """1. Use "Likely no" when uncertain; "No" when definitive. 2. Use exact context words. 3. For hypothetical "if she hadn't..." questions, answer ONLY "Likely no" when no reason needed. 4. Plain text only.""",
                    "other": """1. Plain text only. 2. Answer only what asked. 3. Use exact proper nouns from context. 4. For "why": "Answer, because reason".""",
                },
                "B": {
                    "choice": """1. One word only for "X or Y" when possible. 2. Use exact names from context.""",
                    "yes_no": """1. Use "Likely yes"/"Presumably not"/"Most likely yes" when uncertain. 2. Use "Yes"/"No" when clear. 3. Add concise reason with exact context words when needed.""",
                    "other": """1. Use exact names/terms from context. 2. Use state/country names precisely.""",
                },
            }

            is_choice = " or " in q_lower and "?" in q_lower
            is_yes_no = q_lower.startswith(
                ("is ", "are ", "do ", "does ", "did ", "can ", "could ", "would ", "will ", "should ", "was ", "were ", "has ", "have ")
            )

            if sid in cat3_group_a:
                sp = cat3_group_prompts["A"]
            elif sid in cat3_group_b:
                sp = cat3_group_prompts["B"]
            else:
                sp = None

            if sp is not None:
                if is_choice:
                    cat3_instruction = sp["choice"]
                elif is_yes_no:
                    cat3_instruction = sp["yes_no"]
                else:
                    cat3_instruction = sp["other"]
            else:
                if is_choice:
                    cat3_instruction = "1. Chosen option. 2. Add reason with exact context phrases if available."
                elif is_yes_no:
                    cat3_instruction = "1. Yes/No/Likely no. 2. Add '; reason' using exact context words when clear."
                else:
                    cat3_instruction = "1. Very short phrase (1-5 words). 2. Use exact words from context."

            user_prompt = f"""
                            Based on the context provided below:
                            {context}

                            Question: {question}

                            Instructions:
                            {cat3_instruction}

                            Short answer:
                            """

        elif category == 1:
            temperature = 0.0
            q_lower = question.lower()
            is_yes_no = q_lower.startswith(("is ", "are ", "do ", "does ", "did ", "can ", "could ", "would ", "will ", "should "))
            is_count = q_lower.startswith(("how many", "how much"))

            if is_yes_no:
                cat1_specific_instruction = """
                - Answer with EXACTLY ONE word from: Yes, No, Likely, Unlikely.
                - Do not add explanation or punctuation.
                """
            elif is_count:
                cat1_specific_instruction = """
                - Answer with only the number.
                - Do not add units/extra words.
                """
            else:
                cat1_specific_instruction = """
                - Give the shortest adequate answer.
                - Use exact words/proper nouns from context.
                - Resolve references to concrete names.
                - If multiple items are asked, list all explicit items with commas.
                """

            user_prompt = f"""
                            Based on the context provided below:
                            {context}

                            Question: {question}

                            Instructions:
                            1. This is a Multi-Hop question. Extract and connect information.
                            2. {cat1_specific_instruction}
                            3. Output short phrase or comma-separated list, not full sentence.

                            Short answer:
                            """

        else:
            q_lower = question.lower()
            is_yes_no = q_lower.startswith(("did ", "do ", "does ", "is ", "are ", "was ", "were ", "has ", "have ", "can ", "will ", "would "))
            if is_yes_no:
                cat4_specific = """
                - Answer with ONLY Yes or No (or Likely/Unlikely if uncertain).
                - No explanation.
                """
            else:
                cat4_specific = """
                - Give the shortest adequate answer.
                - Use exact words from context.
                - Prefer proper names when asked for person/artist.
                """
            user_prompt = f"""
                            Based on the context provided below:
                            {context}

                            Question: {question}

                            Instructions:
                            1. Write a short phrase, not a complete sentence.
                            2. {cat4_specific}

                            Short answer:
                            """
            temperature = 0.0

        prediction = ""
        for attempt in range(self.empty_answer_retries + 1):
            raw_response = ""
            error_msg = None
            try:
                raw_response = self.llm_controller.llm.get_completion(
                    user_prompt,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "response",
                            "schema": {
                                "type": "object",
                                "properties": {"answer": {"type": "string"}},
                                "required": ["answer"],
                                "additionalProperties": False,
                            },
                            "strict": True,
                        },
                    },
                    temperature=temperature,
                )
                prediction = self._extract_prediction(raw_response)
            except Exception as e:
                error_msg = str(e)

            append_jsonl(
                self.raw_response_dump_path,
                {
                    "ts": datetime.now().isoformat(),
                    "model": self.model,
                    "backend": self.backend,
                    "sample_id": sample_id,
                    "question_id_in_sample": question_id,
                    "question": question,
                    "attempt": attempt + 1,
                    "category": category,
                    "prediction": prediction,
                    "prediction_is_empty": (not bool((prediction or "").strip())),
                    "error": error_msg,
                    "raw_response": raw_response,
                },
            )

            if prediction and prediction.strip():
                break
        return prediction, user_prompt

def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger("locomo_eval_origin")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    logger.handlers.clear()
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

def evaluate_dataset(
    dataset_path: str,
    model: str,
    output_path: Optional[str] = None,
    ratio: float = 1.0,
    backend: str = "openai",
    api_key: Optional[str] = None,
    max_context_tokens: Optional[int] = None,
    sample_id: Optional[str] = None,
    empty_answer_retries: int = 2,
    raw_response_dump_path: Optional[str] = None,
):
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    model_name_sanitized = model.replace("/", "_").replace(":", "_")
    log_filename = f"eval_origin_{model_name_sanitized}_{backend}_ratio{ratio}_{timestamp}.log"
    log_path = os.path.join(os.path.dirname(__file__), "logs", log_filename)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = setup_logger(log_path)
    effective_context_tokens = resolve_context_window(model, max_context_tokens)
    logger.info(f"Loading dataset from {dataset_path}")
    logger.info(
        f"Context token budget: {effective_context_tokens} (auto model-aware default applied: {max_context_tokens is None})"
    )

    samples = load_locomo_dataset(dataset_path)
    logger.info(f"Loaded {len(samples)} samples")
    if sample_id is not None:
        samples = [s for s in samples if str(s.sample_id) == str(sample_id)]
        if not samples:
            raise ValueError(f"sample_id={sample_id} not found in dataset")
        logger.info(f"Filtered to sample_id={sample_id}, remaining samples: {len(samples)}")
    if ratio < 1.0:
        num_samples = max(1, int(len(samples) * ratio))
        samples = samples[:num_samples]
        logger.info(f"Using {num_samples} samples ({ratio * 100:.1f}% of dataset)")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if not os.path.exists(output_path):
            with open(output_path, "w", encoding="utf-8") as f:
                meta_info = {
                    "type": "metadata",
                    "model": model,
                    "backend": backend,
                    "dataset": str(dataset_path),
                    "ratio": ratio,
                    "retrieval_strategy": "direct_context",
                    "max_context_tokens": effective_context_tokens,
                    "empty_answer_retries": empty_answer_retries,
                    "raw_response_dump_path": raw_response_dump_path,
                    "evaluation_start_time": datetime.now().isoformat(),
                }
                f.write(json.dumps(meta_info, ensure_ascii=False) + "\n")
        else:
            logger.info(f"Found existing results file. Will resume and append to {output_path}")

    all_metrics = []
    all_categories = []
    total_questions = 0
    category_counts = defaultdict(int)
    error_num = 0
    allow_categories = [1, 2, 3, 4]

    for sample in samples:
        agent = DirectContextAgent(
            model=model,
            backend=backend,
            api_key=api_key,
            max_context_tokens=effective_context_tokens,
            empty_answer_retries=empty_answer_retries,
            raw_response_dump_path=raw_response_dump_path,
        )
        full_context = build_full_context(sample)
        truncated_context = truncate_keep_latest(full_context, effective_context_tokens)

        logger.info(f"Answering questions for sample {sample.sample_id}")
        processed_questions_set = set()
        if output_path and os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get("type") == "qa_result" and str(entry.get("sample_id")) == str(sample.sample_id):
                            processed_questions_set.add(entry.get("question", "").strip())
                    except json.JSONDecodeError:
                        continue

        qas_in_sample = [qa for qa in sample.qa if int(qa.category) in allow_categories]
        qas_to_process_with_index = []
        for idx, qa in enumerate(qas_in_sample):
            if qa.question.strip() not in processed_questions_set:
                qas_to_process_with_index.append((idx, qa))

        if qas_to_process_with_index:
            initial_progress = len(qas_in_sample) - len(qas_to_process_with_index)
            with tqdm(total=len(qas_in_sample), initial=initial_progress, desc=f"Answering questions for sample {sample.sample_id}") as pbar:
                for original_idx, qa in qas_to_process_with_index:
                    global_question_id = original_idx + 1
                    total_questions += 1
                    category_counts[qa.category] += 1

                    prediction, _ = agent.answer_question(
                        question=qa.question,
                        category=qa.category,
                        sample_id=sample.sample_id,
                        context=truncated_context,
                        question_id=global_question_id,
                    )

                    try:
                        if isinstance(prediction, str) and prediction.strip().startswith("{"):
                            pred_json = json.loads(prediction)
                            prediction = pred_json.get("answer", prediction)
                    except Exception:
                        pass

                    prediction = str(prediction)
                    logger.info(f"\nQuestion {global_question_id} (Sample {sample.sample_id}): {qa.question}")
                    logger.info(f"Prediction: {prediction}")
                    logger.info(f"Reference: {qa.final_answer}")

                    metrics = calculate_metrics(prediction, qa.final_answer) if qa.final_answer else {
                        "exact_match": 0,
                        "f1": 0.0,
                        "rouge1_f": 0.0,
                        "rouge2_f": 0.0,
                        "rougeL_f": 0.0,
                        "bleu1": 0.0,
                        "bleu2": 0.0,
                        "bleu3": 0.0,
                        "bleu4": 0.0,
                        "bert_f1": 0.0,
                        "meteor": 0.0,
                        "sbert_similarity": 0.0,
                    }

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
                        "retrieved_nodes": [],
                    }
                    if output_path:
                        with open(output_path, "a", encoding="utf-8") as f:
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
            "category_distribution_in_this_run": {str(cat): count for cat, count in category_counts.items()},
            "aggregate_metrics_for_this_run": aggregate_results,
            "run_end_time": datetime.now().isoformat(),
        }
        if output_path:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(run_summary, ensure_ascii=False) + "\n")
        logger.info(f"Total questions evaluated in this run: {total_questions}")
    else:
        logger.info("No new questions were processed in this run.")

def main():
    parser = argparse.ArgumentParser(description="Evaluate direct-context baseline on LoComo dataset")
    parser.add_argument("--dataset", type=str, default="data/locomo10.json", help="Path to dataset")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model to use")
    parser.add_argument("--output", type=str, default=None, help="Path to save evaluation results")
    parser.add_argument("--ratio", type=float, default=1.0, help="Ratio of dataset to evaluate")
    parser.add_argument("--backend", type=str, default="openai", help="Backend to use")
    parser.add_argument("--api-key", type=str, default=None, help="API key")
    parser.add_argument("--sample-id", type=str, default=None, help="Run only one sample_id (e.g. 0)")
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=None,
        help="Context budget override. If unset, gpt-5o/gpt-5o-mini use 32000, others use 12000.",
    )
    parser.add_argument(
        "--empty-answer-retries",
        type=int,
        default=2,
        help="Retry count when extracted answer is empty.",
    )
    parser.add_argument(
        "--raw-response-dump",
        type=str,
        default=None,
        help="Path to debug JSONL for raw model responses (all attempts).",
    )
    args = parser.parse_args()

    if args.ratio <= 0.0 or args.ratio > 1.0:
        raise ValueError("Ratio must be between 0.0 and 1.0")
    if args.max_context_tokens is not None and args.max_context_tokens <= 0:
        raise ValueError("max-context-tokens must be > 0")
    if args.empty_answer_retries < 0:
        raise ValueError("empty-answer-retries must be >= 0")

    dataset_path = os.path.join(os.path.dirname(__file__), args.dataset)
    output_path = args.output if args.output else None
    raw_response_dump_path = args.raw_response_dump
    if raw_response_dump_path is None and output_path:
        root, _ = os.path.splitext(output_path)
        raw_response_dump_path = f"{root}.raw_responses.jsonl"
    evaluate_dataset(
        dataset_path=dataset_path,
        model=args.model,
        output_path=output_path,
        ratio=args.ratio,
        backend=args.backend,
        api_key=args.api_key,
        max_context_tokens=args.max_context_tokens,
        sample_id=args.sample_id,
        empty_answer_retries=args.empty_answer_retries,
        raw_response_dump_path=raw_response_dump_path,
    )

if __name__ == "__main__":
    main()

