#可修改参数：
1、每个语义级节点连接多少边？0.2
2、top-down检索每层检索多少个节点top-k-levels 2 3 5

#
python test_advanced.py \
    --model gpt-4o-mini \
    --backend openai \
    --ratio 0.1 \
    --output output/gpt/t_results.json \
    --retrieval-strategy top_down \
    --top-k-levels 2 3 5 \
    2>&1 | tee output/gpt/error.log

