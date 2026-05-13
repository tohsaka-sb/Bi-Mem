

python scripts/eval/test_longmem.py --backend probex --model qwen2.5-7b-instruct \
    --api-key "${API_KEY}" \
    --dataset data/longmemeval_s_cleaned.json --ratio 0.1 \
    --output output/longmem/qwen/longmem_results.jsonl

python scripts/eval/test_longmem.py --backend probex --model qwen2.5-7b-instruct \
    --api-key "${API_KEY}" \
    --dataset data/longmemeval_s_cleaned.json --ratio 0.1 \
    --output output/longmem/qwen/wo_calibration_longmem_results.jsonl
