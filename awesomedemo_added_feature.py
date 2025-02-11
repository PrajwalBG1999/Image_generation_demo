# %%
from share import *

import config
import cv2
import einops
import numpy as np
import random
import torch

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import imageio
import numpy as np
import matplotlib.pyplot as plt


apply_canny = CannyDetector()

model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


# %%
# image processing functions
def rgb2lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

def lab2rgb(lab: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def rgb2yuv(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)

def yuv2rgb(yuv: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

def srgb2lin(s):
    s = s.astype(float) / 255.0
    return np.where(
        s <= 0.0404482362771082, s / 12.92, np.power(((s + 0.055) / 1.055), 2.4)
    )

def lin2srgb(lin):
    return 255 * np.where(
        lin > 0.0031308, 1.055 * (np.power(lin, (1.0 / 2.4))) - 0.055, 12.92 * lin
    )

def get_luminance(
    linear_image: np.ndarray, luminance_conversion=[0.2126, 0.7152, 0.0722]
):
    return np.sum([[luminance_conversion]] * linear_image, axis=2)

def take_luminance_from_first_chroma_from_second(luminance, chroma, mode="lab", s=1):
    assert luminance.shape == chroma.shape, f"{luminance.shape=} != {chroma.shape=}"
    if mode == "lab":
        lab = rgb2lab(chroma)
        lab[:, :, 0] = rgb2lab(luminance)[:, :, 0]
        return lab2rgb(lab)
    if mode == "yuv":
        yuv = rgb2yuv(chroma)
        yuv[:, :, 0] = rgb2yuv(luminance)[:, :, 0]
        return yuv2rgb(yuv)
    if mode == "luminance":
        lluminance = srgb2lin(luminance)
        lchroma = srgb2lin(chroma)
        return lin2srgb(
            np.clip(
                lchroma
                * ((get_luminance(lluminance) / (get_luminance(lchroma) + 1e-10)) ** s)[
                    :, :, np.newaxis
                ],
                0,
                1,
            )
        )

# %%
# Adaptive threshold function for Canny edge detection
def adaptive_threshold(image: np.ndarray) -> tuple:
    """
    Calculate adaptive thresholds for Canny edge detection based on the image's mean and standard deviation.
    
    Args:
        image (np.ndarray): Input image in RGB format.
    
    Returns:
        tuple: Low and high thresholds for Canny edge detection.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mean = np.mean(gray_image)
    std_dev = np.std(gray_image)
    
    # Set thresholds based on mean and standard deviation
    low_threshold = max(0, mean - std_dev)
    high_threshold = min(255, mean + std_dev)
    
    return int(low_threshold), int(high_threshold)

# utils functions
def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        # Calculate adaptive thresholds
        low_threshold, high_threshold = adaptive_threshold(img)
        
        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [255 - detected_map] + results


# %%
# Load the PNG image into a numpy array
input_image = imageio.imread('test_imgs//bird.png')

# Print the shape of the array
print(input_image.shape)

plt.imshow(input_image)

# %%
# Removed manual thresholding, adaptive thresholds are now used in process function

prompt = "bird"
num_samples = 1
image_resolution = 512
strength = 1.0
guess_mode = False
ddim_steps = 10
scale = 9.0
seed = 1
eta = 0.0
a_prompt = 'good quality' # 'best quality, extremely detailed'
n_prompt = 'animal, drawing, painting, vivid colors, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

# Process the image
result = process(input_image=input_image, 
                 prompt=prompt, 
                 a_prompt=a_prompt, 
                 n_prompt=n_prompt,
                 num_samples=num_samples, 
                 image_resolution=image_resolution, 
                 ddim_steps=ddim_steps, 
                 guess_mode=guess_mode, 
                 strength=strength, 
                 scale=scale, 
                 seed=seed, 
                 eta=eta)

for res in result:
    plt.imshow(res)
    plt.axis(False)
    plt.show()
    plt.savefig('result_added_feature')

# %%
index = -1
test = take_luminance_from_first_chroma_from_second(resize_image(HWC3(input_image), image_resolution), result[index], mode="luminance")

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(input_image)
axs[1].imshow(result[index])
axs[2].imshow(test)

axs[0].axis(False)
axs[1].axis(False)
axs[2].axis(False)

plt.show()
# %%