
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"
OUTPUT_DIR="${OUTPUT_DIR:-output/locomo/model_test}"
DATASET="${DATASET:-data/locomo10.json}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-}"
EMPTY_ANSWER_RETRIES="${EMPTY_ANSWER_RETRIES:-2}"
SAMPLE_ID="${SAMPLE_ID:-0}"

MODELS=(
  "gpt-5"
  "gpt-5-mini"
  "gpt-4o"
  "gpt-4o-mini"
)

TAGS=(
  "gpt5"
  "gpt5_mini"
  "gpt4o"
  "gpt4o_mini"
)

mkdir -p "${OUTPUT_DIR}"

echo "==> Using python: ${PYTHON_BIN}"
echo "==> Output dir: ${OUTPUT_DIR}"
echo "==> Dataset: ${DATASET}"
echo "==> Sample ID: ${SAMPLE_ID}"

for i in "${!MODELS[@]}"; do
  model="${MODELS[$i]}"
  tag="${TAGS[$i]}"

  jsonl_path="${OUTPUT_DIR}/origin_${tag}.jsonl"
  eval_log_path="${OUTPUT_DIR}/origin_${tag}.eval.log"
  summary_log_path="${OUTPUT_DIR}/origin_${tag}.log"
  raw_dump_path="${OUTPUT_DIR}/origin_${tag}.raw_responses.jsonl"

  echo
  echo "============================================================"
  echo "==> Running model: ${model}"
  echo "============================================================"

  rm -f "${jsonl_path}" "${eval_log_path}" "${summary_log_path}" "${raw_dump_path}"

  cmd=(
    "${PYTHON_BIN}" scripts/eval/origin_test_locomo.py
    --dataset "${DATASET}"
    --model "${model}"
    --backend openai
    --sample-id "${SAMPLE_ID}"
    --output "${jsonl_path}"
    --empty-answer-retries "${EMPTY_ANSWER_RETRIES}"
    --raw-response-dump "${raw_dump_path}"
  )

  if [[ -n "${MAX_CONTEXT_TOKENS}" ]]; then
    cmd+=(--max-context-tokens "${MAX_CONTEXT_TOKENS}")
  fi
  if [[ -n "${API_KEY}" ]]; then
    cmd+=(--api-key "${API_KEY}")
  fi

  "${cmd[@]}" 2>&1 | tee "${eval_log_path}"

  "${PYTHON_BIN}" scripts/analysis/summarize_results.py "${jsonl_path}" \
    2>&1 | tee "${summary_log_path}"
done

echo
echo "All done. Results and logs are under: ${OUTPUT_DIR}"

