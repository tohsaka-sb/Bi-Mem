python scripts/eval/test_advanced.py \
    --model gpt-4o-mini \
    --backend openai \
    --ratio 0.1 \
    --dataset data/locomo10.json \
    --output output/gpt/without_refinement/g_results.jsonl \
    --retrieval-strategy graph_traversal 
python scripts/analysis/summarize_results.py output/gpt/without_refinement/g_results.jsonl 2>&1 | tee output/gpt/without_refinement/g_results.log

python scripts/eval/test_advanced.py \
    --model gpt-4o-mini \
    --backend openai \
    --ratio 0.1 \
    --dataset data/locomo10_noise10.json \
    --output output/gpt/noise10_without_refinement/g_results.jsonl \
    --retrieval-strategy graph_traversal 
python scripts/analysis/summarize_results.py output/gpt/noise10_without_refinement/g_results.jsonl 2>&1 | tee output/gpt/noise10_without_refinement/g_results.log

python scripts/eval/test_advanced.py \
    --model gpt-4o-mini \
    --backend openai \
    --ratio 0.1 \
    --dataset data/locomo10_noise20.json \
    --output output/gpt/noise20_without_refinement/g_results.jsonl \
    --retrieval-strategy graph_traversal 
python scripts/analysis/summarize_results.py output/gpt/noise20_without_refinement/g_results.jsonl 2>&1 | tee output/gpt/noise20_without_refinement/g_results.log

python scripts/eval/test_advanced.py \
    --model gpt-4o-mini \
    --backend openai \
    --ratio 0.1 \
    --dataset data/locomo10_noise30.json \
    --output output/gpt/noise30_without_refinement/g_results.jsonl \
    --retrieval-strategy graph_traversal 
python scripts/analysis/summarize_results.py output/gpt/noise30_without_refinement/g_results.jsonl 2>&1 | tee output/gpt/noise30_without_refinement/g_results.log
