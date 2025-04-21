import os
import sys
import math
import torch
import numpy as np
import traceback
from PIL import Image

# Add the ComfyUI directory to the path
comfy_dir = r"E:\LAB\new_folder\ComfyUI_windows_portable\ComfyUI"
sys.path.insert(0, comfy_dir)

# Import ComfyUI modules
import folder_paths
import comfy.sd
import comfy.samplers
import comfy.sample
import comfy.utils
import comfy.model_management
import comfy.controlnet
import latent_preview

# Set cache directories
folder_paths.add_model_folder_path("checkpoints", os.path.join(comfy_dir, "models", "checkpoints"))
folder_paths.add_model_folder_path("vae", os.path.join(comfy_dir, "models", "vae"))
folder_paths.add_model_folder_path("loras", os.path.join(comfy_dir, "models", "loras"))
folder_paths.add_model_folder_path("controlnet", os.path.join(comfy_dir, "models", "controlnet"))
folder_paths.add_model_folder_path("clip", os.path.join(comfy_dir, "models", "clip"))


def setup_model_and_load_checkpoint(checkpoint_name):
    """Load the stable diffusion model from a checkpoint."""
    checkpoint_path = folder_paths.get_full_path("checkpoints", checkpoint_name)
    if checkpoint_path is None:
        raise FileNotFoundError(f"Checkpoint {checkpoint_name} not found")

    print(f"Loading checkpoint: {checkpoint_path}")
    out = comfy.sd.load_checkpoint_guess_config(
        ckpt_path=checkpoint_path,
        output_vae=True,
        output_clip=True,
        output_clipvision=False,
        embedding_directory=None
    )
    return out[:3]  # model, clip, vae


def load_lora(model, clip, lora_name, strength_model=1.0, strength_clip=1.0):
    """Load a LoRA model and apply it to the SD model and CLIP."""
    lora_path = folder_paths.get_full_path("loras", lora_name)
    if lora_path is None:
        raise FileNotFoundError(f"LoRA {lora_name} not found")

    print(f"Loading LoRA: {lora_path}")
    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
    out = comfy.sd.load_lora_for_models(
        model=model,
        clip=clip,
        lora=lora,
        strength_model=strength_model,
        strength_clip=strength_clip
    )
    return out[:2]  # model_lora, clip_lora


def load_controlnet(controlnet_name):
    """Load a ControlNet model."""
    controlnet_path = folder_paths.get_full_path("controlnet", controlnet_name)
    if controlnet_path is None:
        raise FileNotFoundError(f"ControlNet {controlnet_name} not found")

    print(f"Loading ControlNet: {controlnet_path}")
    return comfy.controlnet.load_controlnet(controlnet_path)


def apply_controlnet(positive, negative, controlnet, image_tensor, strength=1.0, start_percent=0.0, end_percent=1.0):
    """Apply ControlNet to the conditioning."""
    if not isinstance(image_tensor, torch.Tensor):
        raise ValueError("Image must be a torch tensor (BCHW format)")

    print(f"Applying ControlNet with strength: {strength}")
    pos = comfy.controlnet.apply_controlnet(
        positive=positive,
        image=image_tensor,
        strength=strength,
        start_percent=start_percent,
        end_percent=end_percent,
        control_net=controlnet
    )
    neg = comfy.controlnet.apply_controlnet(
        positive=negative,
        image=image_tensor,
        strength=strength,
        start_percent=start_percent,
        end_percent=end_percent,
        control_net=controlnet
    )
    return pos, neg


def pil_to_tensor_rgb(img: Image.Image):
    """Convert PIL RGB → torch tensor BCHW, float in [0,1]."""
    img = img.convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0  # H, W, C
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1, 3, H, W


def pil_to_tensor_mask(mask: Image.Image):
    """Convert PIL mask → torch tensor BCHW (0 or 1)."""
    m = mask.convert("L")
    arr = (np.array(m) > 128).astype(np.float32)  # threshold to binary
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # 1, 1, H, W

def prepare_masked_latent(vae, pil_image: Image.Image, pil_mask: Image.Image, grow_mask_by: int = 6):
    """
    Given a PIL image and a PIL mask, returns:
      ({"samples": latent_tensor, "noise_mask": noise_mask},)
    
    This function properly handles masks that are smaller than the image by
    placing the mask in the center or at the correct position and only masking
    the specified region. The mask can be optionally grown by a specified number
    of pixels to ensure smooth transitions.
    
    Args:
        vae: The VAE model for encoding
        pil_image: The input PIL image
        pil_mask: The PIL mask (can be smaller than the image)
        grow_mask_by: Number of pixels to grow the mask by (default: 6)
    
    Returns:
        Dictionary with latent samples and the noise mask
    """
    # 1) Ensure mask is properly sized relative to the image
    img_width, img_height = pil_image.size
    mask_width, mask_height = pil_mask.size
    
    # Create a full-sized mask if needed
    if mask_width != img_width or mask_height != img_height:
        full_mask = Image.new('L', (img_width, img_height), 0)  # 0 = not masked
        # Center the mask on the image
        offset_x = (img_width - mask_width) // 2
        offset_y = (img_height - mask_height) // 2
        full_mask.paste(pil_mask, (offset_x, offset_y))
        pil_mask = full_mask
    
    # 2) Grow mask if specified

    
    # 3) Load into tensors
    pixels = pil_to_tensor_rgb(pil_image)  # [1,3,H,W]
    mask = pil_to_tensor_mask(pil_mask)    # [1,1,H,W]
    
    # 4) Crop H and W to multiples of downscale_ratio
    ds = vae.downscale_ratio
    _, _, H, W = pixels.shape
    Hc = (H // ds) * ds
    Wc = (W // ds) * ds
    
    if Hc != H or Wc != W:
        # Center crop
        ho = (H - Hc) // 2
        wo = (W - Wc) // 2
        pixels = pixels[..., ho:ho+Hc, wo:wo+Wc]
        mask = mask[..., ho:ho+Hc, wo:wo+Wc]
    
    # 5) Apply mask in pixel space (channel-first)
    # Only mask where mask value is 1 (white), leave other areas untouched
    inv_mask = (1.0 - mask).repeat(1, 3, 1, 1)  # [1,3,Hc,Wc]
    masked_pixels = (pixels - 0.5) * inv_mask + 0.5
    
    # 6) Encode into latent
    latent_input = masked_pixels.permute(0, 2, 3, 1)
    latent = vae.encode(latent_input)  # [1,4,Hc/ds,Wc/ds]
    
    # 7) Downsample mask to latent resolution with anti-aliasing for smoother boundaries
    noise_mask = torch.nn.functional.interpolate(
        mask, 
        size=latent.shape[-2:], 
        mode="bilinear",  # Using bilinear instead of nearest for smoother transitions
        align_corners=False
    )
    # Ensure binary mask with proper threshold
    noise_mask = (noise_mask > 0.5).float()
    
    return {"samples": latent, "noise_mask": noise_mask}


def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    """Common sampling function matching the simple workflow."""
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    if disable_noise:
        noise = torch.zeros_like(latent_image)
    else:
        batch_inds = latent.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = latent.get("noise_mask", None)
    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    samples = comfy.sample.sample(
        model=model,
        noise=noise,
        steps=steps,
        cfg=cfg,
        sampler_name=sampler_name,
        scheduler=scheduler,
        positive=positive,
        negative=negative,
        latent_image=latent_image,
        denoise=denoise,
        disable_noise=disable_noise,
        start_step=start_step,
        last_step=last_step,
        force_full_denoise=force_full_denoise,
        noise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed
    )

    out = latent.copy()
    out["samples"] = samples
    return (out,)


def generate_image(
    model,
    clip,
    vae,
    prompt,
    negative_prompt="",
    width=512,
    height=512,
    steps=20,
    cfg=7.0,
    sampler_name="euler_ancestral",
    scheduler="normal",
    seed=None
):
    """Text2Image: generate purely from prompt (no mask)."""
    if seed is None:
        seed = int(torch.randint(0, 2**64 - 1, (1,)).item())
    print(f"Using seed: {seed}")

    device = comfy.model_management.get_torch_device()
    latent = {"samples": torch.zeros([1, 4, height // 8, width // 8], device=device)}

    # encode prompts
    tokens = clip.tokenize(prompt)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    positive = [[cond, {"pooled_output": pooled}]]

    tokens = clip.tokenize(negative_prompt)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    negative = [[cond, {"pooled_output": pooled}]]

    # run sampling
    out_latent, = common_ksampler(
        model=model,
        seed=seed,
        steps=steps,
        cfg=cfg,
        sampler_name=sampler_name,
        scheduler=scheduler,
        positive=positive,
        negative=negative,
        latent=latent
    )

    # decode
    decoded = vae.decode(out_latent["samples"])
    x_samples = decoded.detach().numpy()
    x_samples = np.clip(x_samples * 255.0, 0, 255).astype(np.uint8)
    print(x_samples.shape)
    # For RGB image, properly extract and transpose
    if len(x_samples.shape) == 4:  # [batch, channel, height, width]
        sample_img = x_samples[0] # Convert to [height, width, channel]
        image = Image.fromarray(sample_img)
    else:
        # Handle unexpected shapes
        print(f"Unexpected tensor shape: {x_samples.shape}")
        # Try to reshape if possible or use a fallback
    return image


def generate_image_from_latent(
    model,
    clip,
    vae,
    prompt,
    negative_prompt,
    latent_dict,
    steps=20,
    cfg=7.0,
    sampler_name="euler_ancestral",
    scheduler="normal",
    seed=None
):
    """Img2Img with precomputed latent+mask from prepare_masked_latent."""
    if seed is None:
        seed = int(torch.randint(0, 2**64 - 1, (1,)).item())
    print(f"Using seed: {seed}")

    # encode prompts
    tokens = clip.tokenize(prompt)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    positive = [[cond, {"pooled_output": pooled}]]

    tokens = clip.tokenize(negative_prompt)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    negative = [[cond, {"pooled_output": pooled}]]

    # run sampling
    out_latent, = common_ksampler(
        model=model,
        seed=seed,
        steps=steps,
        cfg=cfg,
        sampler_name=sampler_name,
        scheduler=scheduler,
        positive=positive,
        negative=negative,
        latent=latent_dict
    )

    # decode
    decoded = vae.decode(out_latent["samples"])
    x_samples = decoded.detach().numpy()
    x_samples = np.clip(x_samples * 255.0, 0, 255).astype(np.uint8)
    print(x_samples.shape)
    # For RGB image, properly extract and transpose
    if len(x_samples.shape) == 4:  # [batch, channel, height, width]
        sample_img = x_samples[0] # Convert to [height, width, channel]
        image = Image.fromarray(sample_img)
    else:
        # Handle unexpected shapes
        print(f"Unexpected tensor shape: {x_samples.shape}")
        # Try to reshape if possible or use a fallback
    return image
    

def tensor_to_pil(tensor):
    """Convert a CHW or BCHW tensor to a PIL Image."""
    t = tensor.cpu()
    
    if t.ndim == 4:
        t = t.permute(3,0,1,2)
        t = t.squeeze(0).squeeze(0)
    print(t.shape)
    arr = (t.detach().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def save_image(image, output_path):
    """Save the image to disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)
    print(f"Image saved to: {output_path}")


def generate_sd_image(
    checkpoint_name,
    lora_name,
    prompt,
    negative_prompt,
    width, 
    height,
    steps,
    cfg,
    sampler_name,
    scheduler,
    seed,
    mask_dir,
    image_dir,
    output_dir
):
    """Main function to run the workflow."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs("outputs", exist_ok=True)

    try:
        # load base SDXL model
        model, clip, vae = setup_model_and_load_checkpoint(checkpoint_name)
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if lora_path:
            model_lora, clip_lora = load_lora(model, clip, lora_name, 0.7, 0.7)

            # load the image and its mask
            src = image_dir
            msk = mask_dir  # provide your b&w mask here

            # prepare the latent+noise_mask
            latent_dict = prepare_masked_latent(vae, src, msk, grow_mask_by=6)

            # generate with the masked latent
            img2 = generate_image_from_latent(
                model=model_lora,
                clip=clip_lora,
                vae=vae,
                prompt=prompt + ", detailed",
                negative_prompt=negative_prompt,
                latent_dict=latent_dict,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                seed=seed
            )
            save_image(img2,output_dir)
            return img2
            
        else:
            print(f"Skipping Example 2: LoRA {lora_name} not found")

        print("\nAll examples completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()