"""
鍣０琛ュ厖瀹為獙锛氬湪 sample 0 鐨勫璇濅腑娉ㄥ叆鏃犲叧璇彞锛岀敤浜庢娴?QA 鎬ц兘涓嬮檷鏇茬嚎銆?
璋冪敤 GPT-4o 鐢熸垚涓庡璇濅富棰樻棤鍏崇殑璇彞浣滀负鍣０锛屾寜 noise_rate 姣斾緥鎻掑叆銆?杈撳嚭淇濆瓨涓烘柊鏂囦欢锛堜笉瑕嗙洊鍘熷 locomo10.json锛夈€?
浣跨敤绀轰緥锛?  python noise_generate.py --noise-rate 0.1
  python noise_generate.py --noise-rate 0.2 --output data/locomo10_noise20.json
"""

import json
import argparse
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("璇峰畨瑁?openai: pip install openai")

BATCH_SIZE = 15

def load_locomo(data_path: str) -> List[Dict]:
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_locomo(data: List[Dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_sample0_turns(conv: Dict) -> List[tuple]:
    """
    鏀堕泦 sample 0 鐨勬墍鏈?turns锛岃繑鍥?[(session_key, turn_idx, turn_dict), ...]
    """
    turns_flat = []
    session_keys = sorted(
        [k for k in conv.keys() if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else 0,
    )
    for sk in session_keys:
        session_data = conv.get(sk, [])
        if not isinstance(session_data, list):
            continue
        for idx, turn in enumerate(session_data):
            if isinstance(turn, dict) and turn.get("text"):
                turns_flat.append((sk, idx, turn))
    return turns_flat

def _parse_statements_from_response(raw: str, max_n: int) -> List[str]:
    parsed = []
    clean = re.sub(r"^```(?:json)?\s*", "", raw)
    clean = re.sub(r"\s*```\s*$", "", clean)
    try:
        data = json.loads(clean)
        stmts = data.get("statements", [])
        for s in stmts[:max_n]:
            if isinstance(s, str) and s.strip():
                parsed.append(s.strip())
        return parsed
    except json.JSONDecodeError:
        pass
    m = re.search(r'"statements"\s*:\s*\[(.*)\]', raw, re.DOTALL)
    if m:
        try:
            arr = json.loads("[" + m.group(1) + "]")
            for s in arr[:max_n]:
                if isinstance(s, str) and s.strip():
                    parsed.append(s.strip())
        except json.JSONDecodeError:
            pass
    return parsed

def generate_noise_statements(
    client: OpenAI,
    num_to_generate: int,
    conv_summary: str,
) -> List[str]:
    """璋冪敤 GPT-4o 鐢熸垚涓庡璇濇棤鍏崇殑鐭彞銆?""
    all_statements = []
    conv_preview = conv_summary[:800] if len(conv_summary) > 800 else conv_summary

    prompt_template = """You are generating IRRELEVANT chat messages for a noise injection experiment.

Context: A conversation between two people about adoption, LGBTQ community, art, counseling, family activities.

{conv_summary}

Your task: Generate exactly {n} SHORT, NATURAL-SOUNDING chat messages that are COMPLETELY IRRELEVANT. They must:
- Be 1-2 sentences each, casual chat style
- Cover random topics: weather, sports, food, tech, movies - NOTHING about adoption, art, LGBTQ, counseling, or the characters
- Sound like normal filler (e.g., "Did you see the game?", "This weather is crazy")

Output JSON only: {{"statements": ["msg1", "msg2", ...]}}"""

    while len(all_statements) < num_to_generate:
        n = min(BATCH_SIZE, num_to_generate - len(all_statements))
        prompt = prompt_template.format(conv_summary=conv_preview, n=n)

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You respond with a JSON object: {\"statements\": [list of strings]}."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8,
                max_tokens=600,
            )
            raw = (response.choices[0].message.content or "").strip()
            stmts = _parse_statements_from_response(raw, n)
            all_statements.extend(stmts)
            if not stmts:
                for i in range(n):
                    all_statements.append(f"[Noise fallback {len(all_statements)+1}]")
        except Exception as e:
            print(f"[WARN] API error: {e}")
            for i in range(n):
                all_statements.append(f"[Noise placeholder {len(all_statements)+1}]")

    return all_statements[:num_to_generate]

def build_conv_summary(conv: Dict) -> str:
    """鏋勫缓瀵硅瘽鎽樿锛屼緵 GPT 鐞嗚В涓婚銆?""
    speaker_a = conv.get("speaker_a", "A")
    speaker_b = conv.get("speaker_b", "B")
    turns = []
    for k in sorted(conv.keys(), key=lambda x: (0, x) if "date" in x else (1, x)):
        if k.startswith("session_") and not k.endswith("_date_time"):
            arr = conv.get(k, [])
            if isinstance(arr, list):
                for t in arr[:3]:
                    if isinstance(t, dict) and t.get("text"):
                        turns.append(t["text"])
    sample = " ".join(turns[:20])
    return f"Speakers: {speaker_a}, {speaker_b}. Sample turns: {sample[:500]}..."

def inject_noise(
    data: List[Dict],
    sample_idx: int,
    noise_rate: float,
    client: OpenAI,
) -> None:
    """
    鍦?data[sample_idx] 鐨?conversation 涓敞鍏ュ櫔澹般€?    鐩存帴淇敼 data 鍘熷璞°€?    """
    sample = data[sample_idx]
    conv = sample.get("conversation", {})
    if not conv:
        raise ValueError("Sample 0 has no conversation")

    turns_flat = get_sample0_turns(conv)
    total_turns = len(turns_flat)
    num_noise = max(1, int(total_turns * noise_rate))

    speaker_a = conv.get("speaker_a", "Caroline")
    speaker_b = conv.get("speaker_b", "Melanie")

    conv_summary = build_conv_summary(conv)
    noise_statements = generate_noise_statements(client, num_noise, conv_summary)

    session_keys = [
        k for k in sorted(conv.keys(), key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else 0)
        if k.startswith("session_") and not k.endswith("_date_time")
    ]

    noise_insertions = []
    for i, stmt in enumerate(noise_statements):
        sk = random.choice(session_keys)
        session_list = conv.get(sk, [])
        pos = random.randint(0, len(session_list)) if session_list else 0
        noise_insertions.append((sk, pos, stmt))

    for sk, pos, stmt in noise_insertions:
        by_session[sk].append((pos, stmt))
    for sk in by_session:
        by_session[sk].sort(key=lambda x: -x[0])

    for sk, items in by_session.items():
        session_list = conv[sk]
        for pos, stmt in items:
            speaker = speaker_a if len(session_list) % 2 == 0 else speaker_b
            dia_id = f"NOISE_{random.randint(10000,99999)}"
            new_turn = {"speaker": speaker, "dia_id": dia_id, "text": stmt}
            session_list.insert(pos, new_turn)

def main():
    parser = argparse.ArgumentParser(description="鍦?sample 0 瀵硅瘽涓敞鍏ユ棤鍏宠鍙ヤ綔涓哄櫔澹?)
    parser.add_argument("--noise-rate", type=float, required=True, help="鍣０姣斾緥锛屽 0.1 琛ㄧず 10%%")
    parser.add_argument("--input", type=str, default="data/locomo10.json", help="杈撳叆 JSON 璺緞")
    parser.add_argument("--output", type=str, default=None, help="杈撳嚭璺緞锛岄粯璁?data/locomo10_noise{rate}.json")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API Key")

    args = parser.parse_args()
    if not (0.0 < args.noise_rate <= 1.0):
        raise ValueError("noise-rate 蹇呴』鍦?(0, 1] 鑼冨洿鍐?)

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("璇疯缃?OPENAI_API_KEY 鎴栦娇鐢?--api-key")

    client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_API_BASE"))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = args.input if os.path.isabs(args.input) else os.path.join(script_dir, args.input)

    if not os.path.exists(data_path):
        print(f"Error: 杈撳叆鏂囦欢涓嶅瓨鍦?{data_path}")
        return

    data = load_locomo(data_path)
    if not data:
        print("Error: 鏁版嵁涓虹┖")
        return

    rate_pct = int(args.noise_rate * 100)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.output:
        output_path = args.output if os.path.isabs(args.output) else os.path.join(script_dir, args.output)
    else:
        output_path = os.path.join(script_dir, "data", f"locomo10_noise{rate_pct}.json")

    conv = data[0].get("conversation", {})
    total = sum(len(conv.get(k, [])) for k in conv if k.startswith("session_") and not k.endswith("_date_time") and isinstance(conv.get(k), list))
    num_noise = max(1, int(total * args.noise_rate))

    print(f"Sample 0 鍘熷 turns: {total}")
    print(f"鍣０姣斾緥: {args.noise_rate*100:.0f}% -> 娉ㄥ叆 {num_noise} 鏉℃棤鍏宠鍙?)

    inject_noise(data, 0, args.noise_rate, client)

    save_locomo(data, output_path)
    print(f"宸蹭繚瀛? {output_path}")

if __name__ == "__main__":
    main()

