"""
OneVL (Latent CoT) inference script for Qwen3-VL.

Supports:
- Standard latent-token inference (generate answer from latent prefix)
- Optional aux decoder explain: decode the latent hidden states into
  explicit reasoning text using the trained auxiliary decoder
- Optional visual aux decoder explain: decode latent states into
  future visual tokens

Hyperparameters are passed via argparse flags.

The trained checkpoint stores all weights (base model + aux decoder + projections)
in the same safetensors files. This script extracts sub-module weights by prefix.
"""
import sys
import os
import json
import argparse
import glob
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoConfig
from safetensors.torch import load_file


def collect_state_dict_from_safetensors(ckpt_dir, prefix):
    """Load weight tensors matching a given prefix from all safetensors in ckpt_dir."""
    result = {}
    for sf in sorted(glob.glob(os.path.join(ckpt_dir, '*.safetensors'))):
        sd = load_file(sf)
        for k, v in sd.items():
            if k.startswith(prefix):
                result[k[len(prefix):]] = v
    return result


def build_aux_decoder_from_checkpoint(ckpt_dir, prefix, aux_base_model_path, device, dtype):
    """Build an aux decoder model and load its weights from the checkpoint."""
    config = AutoConfig.from_pretrained(aux_base_model_path, trust_remote_code=True)
    model_type = getattr(config, 'model_type', '')
    if 'qwen3_vl' in model_type:
        from transformers import Qwen3VLForConditionalGeneration as Cls
    elif 'qwen2_vl' in model_type:
        from transformers import Qwen2VLForConditionalGeneration as Cls
    else:
        from transformers import AutoModelForCausalLM as Cls

    print(f"[INFO] Building aux decoder architecture from {aux_base_model_path}")
    model = Cls.from_pretrained(aux_base_model_path, dtype=dtype, trust_remote_code=True)

    sd = collect_state_dict_from_safetensors(ckpt_dir, prefix)
    if sd:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[INFO] Loaded {len(sd)} weights with prefix '{prefix}' "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        print(f"[WARN] No weights found with prefix '{prefix}' in {ckpt_dir}")

    model.to(device).eval()
    return model


def build_projection_from_checkpoint(ckpt_dir, prefix, in_dim, out_dim, device, dtype):
    """Build the latent projection and load its weights from the checkpoint."""
    proj = nn.Sequential(
        nn.Linear(in_dim, in_dim),
        nn.GELU(),
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
    )
    sd = collect_state_dict_from_safetensors(ckpt_dir, prefix)
    if sd:
        proj.load_state_dict(sd)
        print(f"[INFO] Loaded projection weights with prefix '{prefix}' ({len(sd)} tensors)")
    else:
        print(f"[WARN] No projection weights found with prefix '{prefix}', using random init")
    proj.to(device=device, dtype=dtype).eval()
    return proj


def get_aux_input_embeddings(aux_decoder):
    if hasattr(aux_decoder, 'model') and hasattr(aux_decoder.model, 'get_input_embeddings'):
        return aux_decoder.model.get_input_embeddings()
    return aux_decoder.get_input_embeddings()


def call_aux_decoder_lm(aux_decoder, inputs_embeds, use_cache=False):
    """Call the aux decoder's language model directly, bypassing the VL wrapper.

    Qwen3VLForConditionalGeneration.forward -> Qwen3VLModel.forward calls
    get_rope_index(input_ids, ...) which crashes when input_ids is None.
    By calling the inner language_model + lm_head directly, we avoid this.
    For non-VL models (AutoModelForCausalLM), we fall back to the normal call.
    """
    if (hasattr(aux_decoder, 'model')
            and hasattr(aux_decoder.model, 'language_model')
            and hasattr(aux_decoder, 'lm_head')):
        lm_out = aux_decoder.model.language_model(
            inputs_embeds=inputs_embeds, use_cache=use_cache)
        hidden = lm_out[0]
        logits = aux_decoder.lm_head(hidden)

        class _AuxOut:
            pass
        out = _AuxOut()
        out.logits = logits
        return out
    return aux_decoder(inputs_embeds=inputs_embeds, use_cache=use_cache)


def extract_visual_embeds(student_embeds, input_ids, image_token_id, video_token_id=None):
    vis_mask = (input_ids == image_token_id)
    if video_token_id is not None:
        vis_mask = vis_mask | (input_ids == video_token_id)
    if not vis_mask.any():
        return None
    return student_embeds[vis_mask]


@torch.no_grad()
def decode_latent_with_aux(
    model, aux_decoder, latent_proj, input_ids, hidden_states,
    processor, c_thought, device,
    use_visual_condition=False, image_token_id=None, video_token_id=None,
    max_explain_tokens=512,
):
    """Use the auxiliary decoder to generate explicit reasoning from latent hidden states."""
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    latent_token_id = tokenizer.convert_tokens_to_ids('<|latent|>')
    last_hidden = hidden_states[-1]

    batch_size = input_ids.size(0)
    results = []

    aux_embedding = get_aux_input_embeddings(aux_decoder)

    for b in range(batch_size):
        positions = (input_ids[b] == latent_token_id).nonzero(as_tuple=True)[0].tolist()
        if not positions:
            results.append("")
            continue

        latent_embeds = last_hidden[b, positions, :]
        if latent_proj is not None:
            latent_embeds = latent_proj(latent_embeds)

        parts = []
        if use_visual_condition and image_token_id is not None:
            embed_fn = (model.model.get_input_embeddings()
                        if hasattr(model, 'model') else model.get_input_embeddings())
            student_embeds = embed_fn(input_ids[b])
            vis_embeds = extract_visual_embeds(
                student_embeds, input_ids[b], image_token_id, video_token_id)
            if vis_embeds is not None:
                parts.append(vis_embeds)
        parts.append(latent_embeds)
        combined = torch.cat(parts, dim=0).unsqueeze(0)

        generated_ids = []
        cur_embeds = combined
        for _ in range(max_explain_tokens):
            out = call_aux_decoder_lm(aux_decoder, cur_embeds, use_cache=False)
            logits = out.logits if hasattr(out, 'logits') else out[0]
            next_id = logits[:, -1, :].argmax(dim=-1)
            generated_ids.append(next_id.item())
            if next_id.item() == tokenizer.eos_token_id:
                break
            next_embed = aux_embedding(next_id).unsqueeze(1)
            cur_embeds = torch.cat([cur_embeds, next_embed], dim=1)

        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        results.append(text)

    return results


@torch.no_grad()
def decode_latent_with_visual_aux(
    model, visual_aux_decoder, visual_latent_proj, input_ids, hidden_states,
    processor, c_thought_visual, device,
    use_visual_condition=False, image_token_id=None, video_token_id=None,
    max_visual_tokens=512,
):
    """Use the visual auxiliary decoder to generate future visual tokens from latent states."""
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    latent_token_id = tokenizer.convert_tokens_to_ids('<|latent|>')
    last_hidden = hidden_states[-1]

    vis_aux_tokenizer = None
    try:
        from transformers import AutoTokenizer
        base_path = getattr(visual_aux_decoder.config, '_name_or_path', None)
        if base_path:
            vis_aux_tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    except Exception:
        pass
    if vis_aux_tokenizer is None:
        vis_aux_tokenizer = tokenizer

    batch_size = input_ids.size(0)
    results = []
    aux_embedding = get_aux_input_embeddings(visual_aux_decoder)

    for b in range(batch_size):
        positions = (input_ids[b] == latent_token_id).nonzero(as_tuple=True)[0].tolist()
        if not positions:
            results.append("")
            continue

        latent_embeds = last_hidden[b, positions, :]
        n_latent = latent_embeds.shape[0]
        n_use = (n_latent // c_thought_visual) * c_thought_visual
        if n_use == 0:
            results.append("")
            continue
        latent_embeds = latent_embeds[:n_use].view(
            -1, c_thought_visual, latent_embeds.size(-1)).mean(dim=1)

        if visual_latent_proj is not None:
            latent_embeds = visual_latent_proj(latent_embeds)

        parts = []
        if use_visual_condition and image_token_id is not None:
            embed_fn = (model.model.get_input_embeddings()
                        if hasattr(model, 'model') else model.get_input_embeddings())
            student_embeds = embed_fn(input_ids[b])
            vis_embeds = extract_visual_embeds(
                student_embeds, input_ids[b], image_token_id, video_token_id)
            if vis_embeds is not None:
                parts.append(vis_embeds)
        parts.append(latent_embeds)
        combined = torch.cat(parts, dim=0).unsqueeze(0)

        generated_ids = []
        cur_embeds = combined
        eos_id = vis_aux_tokenizer.eos_token_id
        for _ in range(max_visual_tokens):
            out = call_aux_decoder_lm(visual_aux_decoder, cur_embeds, use_cache=False)
            logits = out.logits if hasattr(out, 'logits') else out[0]
            next_id = logits[:, -1, :].argmax(dim=-1)
            generated_ids.append(next_id.item())
            if next_id.item() == eos_id:
                break
            next_embed = aux_embedding(next_id).unsqueeze(1)
            cur_embeds = torch.cat([cur_embeds, next_embed], dim=1)

        text = vis_aux_tokenizer.decode(generated_ids, skip_special_tokens=True)
        results.append(text)

    return results


def main():
    parser = argparse.ArgumentParser(description="OneVL (Latent CoT) inference for Qwen3-VL")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained checkpoint (contains base + aux weights)")
    parser.add_argument("--test_set_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--num_latent", type=int, default=6,
                        help="Number of latent tokens in the prefix")
    parser.add_argument("--max_new_tokens", type=int, default=1024)

    parser.add_argument("--decoder_explain", action="store_true",
                        help="Enable text aux decoder to explain latent reasoning")
    parser.add_argument("--aux_model_path", type=str, default=None,
                        help="Base architecture path for aux decoder "
                             "(e.g. Qwen3-VL-4B-Instruct-latent). "
                             "Weights are loaded from the main checkpoint.")
    parser.add_argument("--aux_visual_condition", action="store_true",
                        help="Condition aux decoder on visual tokens")
    parser.add_argument("--c_thought", type=int, default=6,
                        help="Latent tokens per thought group for aux decoder")
    parser.add_argument("--max_explain_tokens", type=int, default=512)

    parser.add_argument("--visual_decoder_explain", action="store_true",
                        help="Enable visual aux decoder to decode future visual tokens")
    parser.add_argument("--visual_aux_model_path", type=str, default=None,
                        help="Base architecture path for visual aux decoder. "
                             "Weights are loaded from the main checkpoint.")
    parser.add_argument("--c_thought_visual", type=int, default=6)
    parser.add_argument("--max_visual_tokens", type=int, default=512)
    parser.add_argument("--add_assistant_prefix", action="store_true",
                        help="Add assistant prefix to the input text")

    args = parser.parse_args()
    device = args.device
    dtype = torch.bfloat16

    # ---- Load main model (only base model weights are consumed) ----
    print(f"[INFO] Loading base model from {args.model_path}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, dtype=dtype, trust_remote_code=True)
    model.to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    MAX_IMAGE_SIZE = 1792
    processor.image_processor.max_pixels = MAX_IMAGE_SIZE * MAX_IMAGE_SIZE
    processor.image_processor.size["longest_edge"] = MAX_IMAGE_SIZE * MAX_IMAGE_SIZE
    print(f"[INFO] image_processor.size = {processor.image_processor.size}")

    image_token_id = getattr(model.config, 'image_token_id', None)
    video_token_id = getattr(model.config, 'video_token_id', None)

    # ---- Resolve hidden sizes ----
    base_hidden = (model.config.text_config.hidden_size
                   if hasattr(model.config, 'text_config')
                   else model.config.hidden_size)

    # ---- Build latent prefix ----
    latent_block = "<|start-latent|>" + "<|latent|>" * args.num_latent + "<|end-latent|>"
    if args.add_assistant_prefix:
        assistant_prefix = latent_block
    else:
        assistant_prefix = ""
    print(f"[INFO] assistant_prefix = {repr(assistant_prefix)}")

    # ---- Load aux decoder + projection from checkpoint ----
    aux_decoder = None
    latent_proj = None
    if args.decoder_explain:
        if not args.aux_model_path:
            raise ValueError("--aux_model_path required when --decoder_explain is set")

        aux_decoder = build_aux_decoder_from_checkpoint(
            args.model_path, '_latent_cot_aux_decoder.',
            args.aux_model_path, device, dtype)

        aux_cfg = AutoConfig.from_pretrained(args.aux_model_path, trust_remote_code=True)
        aux_hidden = (aux_cfg.text_config.hidden_size
                      if hasattr(aux_cfg, 'text_config') else aux_cfg.hidden_size)

        latent_proj = build_projection_from_checkpoint(
            args.model_path, '_latent_cot_latent_proj.',
            base_hidden, aux_hidden, device, dtype)

    visual_aux_decoder = None
    visual_latent_proj = None
    if args.visual_decoder_explain:
        if not args.visual_aux_model_path:
            raise ValueError("--visual_aux_model_path required when --visual_decoder_explain")

        visual_aux_decoder = build_aux_decoder_from_checkpoint(
            args.model_path, '_latent_cot_visual_aux_decoder.',
            args.visual_aux_model_path, device, dtype)

        vis_cfg = AutoConfig.from_pretrained(args.visual_aux_model_path, trust_remote_code=True)
        vis_hidden = (vis_cfg.text_config.hidden_size
                      if hasattr(vis_cfg, 'text_config') else vis_cfg.hidden_size)

        visual_latent_proj = build_projection_from_checkpoint(
            args.model_path, '_latent_cot_visual_latent_proj.',
            base_hidden, vis_hidden, device, dtype)

    # ---- Load test set ----
    with open(args.test_set_path, 'r') as f:
        test_set = json.load(f)
    print(f"[INFO] Loaded {len(test_set)} test samples from {args.test_set_path}")

    # ---- Inference loop ----
    output_list = []
    need_hidden = (aux_decoder is not None or visual_aux_decoder is not None)

    for idx, item in enumerate(tqdm(test_set, desc="Inference")):
        output_dict = {}

        prompt = item["messages"][0]["content"].replace("<image>", "")
        test_image_path = item["images"][0]

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": test_image_path},
                {"type": "text", "text": prompt},
            ],
        }]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        text += assistant_prefix

        try:
            img = Image.open(test_image_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Skipping sample {idx}: cannot open image {test_image_path}: {e}")
            continue

        inputs = processor(
            text=[text], images=[img], return_tensors="pt", padding=True
        ).to(device)

        if need_hidden:
            fwd_out = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = fwd_out.hidden_states

            if aux_decoder is not None:
                explains = decode_latent_with_aux(
                    model, aux_decoder, latent_proj,
                    inputs['input_ids'], hidden_states, processor,
                    args.c_thought, device,
                    use_visual_condition=args.aux_visual_condition,
                    image_token_id=image_token_id,
                    video_token_id=video_token_id,
                    max_explain_tokens=args.max_explain_tokens,
                )
                if explains and explains[0]:
                    output_dict["decoder_explain"] = explains[0]

            if visual_aux_decoder is not None:
                vis_explains = decode_latent_with_visual_aux(
                    model, visual_aux_decoder, visual_latent_proj,
                    inputs['input_ids'], hidden_states, processor,
                    args.c_thought_visual, device,
                    use_visual_condition=args.aux_visual_condition,
                    image_token_id=image_token_id,
                    video_token_id=video_token_id,
                    max_visual_tokens=args.max_visual_tokens,
                )
                if vis_explains and vis_explains[0]:
                    output_dict["visual_decoder_explain"] = vis_explains[0]

            del fwd_out, hidden_states
            torch.cuda.empty_cache()

        torch.cuda.synchronize()
        start_time = time.time()

        # Generate answer
        gen_outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
        torch.cuda.synchronize()
        latency = time.time() - start_time
        print(f"[INFO] Generation latency: {latency} seconds")

        generated_ids = gen_outputs.sequences
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

        output_dict["latency"] = latency
        output_dict["messages"] = messages
        output_dict["GT"] = item.get("GT", "")
        output_dict["output_text"] = output_text[0]

        scores = gen_outputs.scores
        batch_size = generated_ids.shape[0]

        entropies = []
        for step_logits in scores:
            probs = F.softmax(step_logits.float(), dim=-1)
            step_entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            entropies.append(step_entropy)

        entropies_tensor = torch.stack(entropies).transpose(0, 1)
        avg_entropy = entropies_tensor.mean(dim=1)

        transition_scores = model.compute_transition_scores(
            generated_ids, scores, normalize_logits=True)
        avg_log_prob = transition_scores.mean(dim=1)
        seq_confidence = torch.exp(avg_log_prob)

        output_dict["avg_entropy"] = avg_entropy.item()
        output_dict["avg_log_prob"] = avg_log_prob.item()
        output_dict["seq_confidence"] = seq_confidence.item()

        if idx < 3 or idx % 100 == 0:
            print(f"\n=== Sample {idx} ===")
            print(f"  Output: {output_text[0][:]}")
            print(f"  Entropy: {avg_entropy.item():.4f}, Confidence: {seq_confidence.item():.2%}")
            if output_dict.get("decoder_explain"):
                print(f"  Explain: {output_dict['decoder_explain'][:]}")
            if output_dict.get("visual_decoder_explain"):
                print(f"  VisExplain: {output_dict['visual_decoder_explain'][:]}")

        output_list.append(output_dict)

        with open(args.output_path, 'w') as f:
            json.dump(output_list, f, indent=4, ensure_ascii=False)

    print(f"\n[INFO] Done. {len(output_list)} results saved to {args.output_path}")


if __name__ == "__main__":
    main()
