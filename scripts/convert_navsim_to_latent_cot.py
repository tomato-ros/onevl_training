"""Convert veomni navsim dataset to ms-swift latent CoT format.

Reads the original JSON dataset with <think>...</think> in assistant responses,
replaces think content with latent tokens, and outputs JSONL for ms-swift.
"""

import argparse
import json
import re
import sys


def extract_and_replace_think(text, c_thought=6, max_latent_stage=1):
    """Extract think content and replace with latent tokens."""
    num_latent = c_thought * max_latent_stage
    latent_block = (
        '<|start-latent|>'
        + '<|latent|>' * num_latent
        + '<|end-latent|>\n'
    )

    think_pattern = r'<think>(.*?)</think>\s*'
    match = re.search(think_pattern, text, flags=re.DOTALL)
    think_content = match.group(1).strip() if match else ''
    new_text = re.sub(think_pattern, latent_block, text, flags=re.DOTALL)
    return new_text, think_content


def convert_sample(item, c_thought=6, max_latent_stage=1):
    """Convert a single sample to ms-swift format."""
    conversations = item.get('conversations', [])
    messages = []

    think_steps = ''
    for conv in conversations:
        role_map = {'human': 'user', 'gpt': 'assistant'}
        role = role_map.get(conv['from'], conv['from'])
        content = conv['value']

        if role == 'assistant':
            content, think_content = extract_and_replace_think(
                content, c_thought, max_latent_stage)
            if think_content:
                think_steps = think_content

        messages.append({'role': role, 'content': content})

    result = {'messages': messages}

    if item.get('images'):
        result['images'] = item['images']

    if think_steps:
        result['think_steps'] = think_steps

    future_tokens = item.get('future_image_tokens', '')
    if future_tokens:
        result['future_image_tokens'] = future_tokens

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input JSON file')
    parser.add_argument('--output', required=True, help='Output JSONL file')
    parser.add_argument('--c_thought', type=int, default=6)
    parser.add_argument('--max_latent_stage', type=int, default=1)
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit number of samples for debugging')
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    if args.max_samples:
        data = data[:args.max_samples]

    count = 0
    with open(args.output, 'w') as out:
        for item in data:
            converted = convert_sample(
                item, args.c_thought, args.max_latent_stage)
            out.write(json.dumps(converted, ensure_ascii=False) + '\n')
            count += 1

    print(f'Converted {count} samples -> {args.output}')

    with open(args.output) as f:
        first = json.loads(f.readline())
    print('Sample output:')
    print(f"  Messages[0] role: {first['messages'][0]['role']}")
    print(f"  Messages[0] content[:200]: {first['messages'][0]['content'][:200]}")
    print(f"  Messages[1] role: {first['messages'][1]['role']}")
    print(f"  Messages[1] content[:300]: {first['messages'][1]['content'][:300]}")
    print(f"  images: {first.get('images', [])[:1]}")
    print(f"  think_steps[:200]: {first.get('think_steps', '')[:200]}")
    print(f"  future_image_tokens[:200]: {first.get('future_image_tokens', '')[:200]}")


if __name__ == '__main__':
    main()
