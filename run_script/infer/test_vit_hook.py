#!/usr/bin/env python3
"""Quick test: verify the forward pre-hook captures real ViT embeddings
(diverse per-position) instead of placeholder embeddings (identical per-position).

Usage:
    python test_vit_hook.py \
        --model_path <checkpoint> \
        --image_path <any_jpg> \
        [--device cuda:0]
"""
import argparse, torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    dtype = torch.bfloat16
    device = args.device

    print(f"[1/5] Loading model from {args.model_path} ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, dtype=dtype, trust_remote_code=True)
    model.to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    image_token_id = getattr(model.config, 'image_token_id', None)
    print(f"  image_token_id = {image_token_id}")

    print(f"[2/5] Processing image {args.image_path} ...")
    img = Image.open(args.image_path).convert("RGB")
    text = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n"
    inputs = processor(text=[text], images=[img], return_tensors="pt", padding=True).to(device)
    input_ids = inputs['input_ids']
    print(f"  input_ids shape = {input_ids.shape}")

    vis_mask = (input_ids[0] == image_token_id)
    n_img = vis_mask.sum().item()
    print(f"  n_image_tokens = {n_img}")
    if n_img == 0:
        print("[ERROR] No image tokens found in input_ids. Check image_token_id config.")
        return

    print("[3/5] Getting PLACEHOLDER embeddings (old approach) ...")
    embed_fn = model.model.get_input_embeddings()
    placeholder_embeds = embed_fn(input_ids[0]).detach()
    placeholder_at_img = placeholder_embeds[vis_mask].float()

    print(f"  placeholder shape at img positions: {placeholder_at_img.shape}")
    print(f"  std across positions (mean over dims): {placeholder_at_img.std(dim=0).mean().item():.8f}")
    print(f"  norms of first 5 positions: {placeholder_at_img[:5].norm(dim=-1).tolist()}")
    all_same = torch.allclose(placeholder_at_img[0:1].expand_as(placeholder_at_img),
                              placeholder_at_img, atol=1e-5)
    print(f"  All image positions identical? {all_same}")

    print("[4/5] Getting ViT-INJECTED embeddings (new hook approach) ...")
    captured = {}
    def hook_fn(module, args_, kwargs):
        ie = kwargs.get('inputs_embeds')
        if ie is not None:
            captured['embeds'] = ie.detach()
        return None

    handle = model.model.language_model.register_forward_pre_hook(hook_fn, with_kwargs=True)
    with torch.no_grad():
        _ = model(**inputs, output_hidden_states=False, return_dict=True)
    handle.remove()

    vit_embeds = captured.get('embeds')
    if vit_embeds is None:
        print("[ERROR] Hook did NOT capture inputs_embeds!")
        return

    vit_at_img = vit_embeds[0][vis_mask].float()

    print(f"  vit_embeds shape at img positions: {vit_at_img.shape}")
    print(f"  std across positions (mean over dims): {vit_at_img.std(dim=0).mean().item():.8f}")
    print(f"  norms of first 5 positions: {vit_at_img[:5].norm(dim=-1).tolist()}")
    all_same_vit = torch.allclose(vit_at_img[0:1].expand_as(vit_at_img),
                                  vit_at_img, atol=1e-5)
    print(f"  All image positions identical? {all_same_vit}")

    print("\n[5/5] COMPARISON:")
    print(f"  Placeholder std: {placeholder_at_img.std(dim=0).mean().item():.8f}")
    print(f"  ViT-injected std: {vit_at_img.std(dim=0).mean().item():.8f}")
    diff = (vit_at_img - placeholder_at_img).norm(dim=-1)
    print(f"  L2 diff between ViT and placeholder (first 5): {diff[:5].tolist()}")
    print(f"  Mean L2 diff: {diff.mean().item():.6f}")

    if all_same and not all_same_vit:
        print("\n  >>> SUCCESS: Placeholder embeddings are identical (old bug), "
              "ViT embeddings are diverse (fix works)!")
    elif not all_same_vit:
        print("\n  >>> SUCCESS: ViT embeddings are diverse (fix works)!")
    else:
        print("\n  >>> WARNING: ViT embeddings are also identical -- "
              "investigate further.")


if __name__ == "__main__":
    main()
