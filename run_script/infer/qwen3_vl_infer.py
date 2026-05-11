from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
import torch.nn.functional as F
import json
from tqdm import tqdm
import argparse
from PIL import Image
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--test_set_path", type=str,default="projects/ms-swift/data/navsim_test_cot_full_idx_trainfmt.json")
    parser.add_argument("--output_path", type=str,default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--add_assistant_prefix", action="store_true")
    args = parser.parse_args()

    model_path = args.model_path
    test_set_path = args.test_set_path
    output_path = args.output_path
    device = args.device


    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    MAX_IMAGE_SIZE = 1792
    processor.image_processor.max_pixels = MAX_IMAGE_SIZE * MAX_IMAGE_SIZE
    processor.image_processor.size["longest_edge"] = MAX_IMAGE_SIZE * MAX_IMAGE_SIZE
    print(f"[INFO] image_processor.size = {processor.image_processor.size}")


    if test_set_path.endswith(".jsonl"):
        test_set = []
        with open(test_set_path, 'r') as f:
            for line in f:
                test_set.append(json.loads(line))
    else:
        with open(test_set_path, 'r') as f:
            test_set = json.load(f)

    output_list = []
    for item in tqdm(test_set):

        output_dict = {}

        prompt = item["messages"][0]["content"].replace("<image>", "")
        test_image_path = item["images"][0]

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": test_image_path,
                    },
                    {"type": "text", "text": prompt},
                ],
            },]  


        # Preparation for inference
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        assistant_prefix = "<|start-latent|><|latent|><|end-latent|><answer>["
        if args.add_assistant_prefix:
            text += assistant_prefix
        print(text)
        output_dict["messages"] = messages

        inputs = processor(
            text=[text],
            images=[Image.open(test_image_path).convert("RGB")],
            return_tensors="pt",
            padding=False,
        ).to(device)

        ## print full input ids and tokenizer.decode
        print(inputs.input_ids[0])
        # print(processor.tokenizer.decode(inputs.input_ids[0]))

        # Inference: Generation of the output
        # 【修改 1】：增加 return_dict_in_generate 和 output_scores
        torch.cuda.synchronize()
        start_time = time.time()
        print(f"[INFO] Generating output...")
        outputs = model.generate(
            **inputs, 
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,  # 返回包含序列和分数的字典
            output_scores=True             # 输出每一步生成的 logits
        )
        torch.cuda.synchronize()
        latency = time.time() - start_time
        print(f"[INFO] Generation latency: {latency} seconds")
        output_dict["latency"] = latency
        # 提取生成的 IDs
        generated_ids = outputs.sequences

        # Trim 掉 prompt 部分
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        print("=== Predicted Trajectory ===")
        print(output_text[0])
        if "GT" in item:
            output_dict["GT"] = item["GT"]
        else:
            output_dict["GT"] = item["messages"][1]["content"]
        output_dict["output_text"] = output_text[0]

        # ==========================================
        # 【修改 2】：计算预测的 熵 (Entropy) 和 得分 (Score)
        # ==========================================

        # outputs.scores 是一个 tuple，长度为生成的新 token 的数量。
        # 每一个元素是形状为 (batch_size, vocab_size) 的 tensor，代表这一步预测时的 logits。
        scores = outputs.scores
        batch_size = generated_ids.shape[0]

        # 1. 计算每个生成步的香农熵 (Shannon Entropy)
        entropies =[]
        for step_logits in scores:
            # 将 logits 转为概率分布 (由于模型在 bfloat16 下运行，转换为 float32 计算更稳定)
            probs = F.softmax(step_logits.float(), dim=-1)
            # 计算熵: -sum(p * log(p)), 加上 1e-10 防止 log(0) 报错
            step_entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            entropies.append(step_entropy)

        # 将 list of tensors 转换为 tensor: (batch_size, generated_length)
        entropies_tensor = torch.stack(entropies).transpose(0, 1)

        # 计算该序列的平均预测熵 (Mean Entropy)
        avg_entropy = entropies_tensor.mean(dim=1)

        # 2. 计算模型对生成序列的确信度得分 (Transition Scores / Log Probabilities)
        transition_scores = model.compute_transition_scores(
            generated_ids, scores, normalize_logits=True
        )
        # 去除掉可能因为 pad 产生的无效 token 分数（对生成单个序列影响不大，但如果是批量推荐做掩盖）
        avg_log_prob = transition_scores.mean(dim=1)  # 每个 token 的平均对数概率
        seq_confidence = torch.exp(avg_log_prob)      # 转换回 0~1 的置信度概率

        print("\n=== Generation Metrics ===")
        for i in range(batch_size):
            print(f"Sample {i}:")
            print(f"  - Average Entropy  : {avg_entropy[i].item():.4f} (越低表示模型预测越自信/确定)")
            print(f"  - Average Log Prob : {avg_log_prob[i].item():.4f} (接近0越好，负数)")
            print(f"  - Sequence Confidence: {seq_confidence[i].item():.2%} (整体序列置信度概率)")

        output_dict["avg_entropy"] = avg_entropy.item()
        output_dict["avg_log_prob"] = avg_log_prob.item()
        output_dict["seq_confidence"] = seq_confidence.item()

        output_list.append(output_dict)

        with open(output_path, 'w') as f:
            json.dump(output_list, f, indent=4)

if __name__ == "__main__":
    main()