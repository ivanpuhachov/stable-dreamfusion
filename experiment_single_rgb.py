"""
https://github.com/ashawkey/stable-dreamfusion/issues/96
"""

import math
from tqdm import tqdm
import torch
import torch.nn as nn
from nerf.sd import StableDiffusion, seed_everything
from torch.optim.lr_scheduler import LambdaLR

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps,
        num_training_steps,
        num_cycles: float = 0.5,
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(opt, lr_lambda, -1)


if __name__ == '__main__':
    device = 'cuda:0'
    guidance = StableDiffusion(device)
    # limited memory here, don't need the decoder
    guidance.vae.decoder = None

    prompt = 'pineapples'
    text_embeddings = guidance.get_text_embeds(prompt, '')
    guidance.text_encoder.to('cpu')
    torch.cuda.empty_cache()

    seed_everything(42)
    # put parameters approximately in range(0, 1) since this is what `encode_imgs` expects
    rgb = nn.Parameter(torch.randn(1, 3, 512, 512, device=device) / 2 + .5)
    optimizer = torch.optim.AdamW([rgb], lr=1e-1, weight_decay=0)
    num_steps = 1000
    scheduler = get_cosine_schedule_with_warmup(
        opt=optimizer,
        num_warmup_steps=100,
        num_training_steps=int(num_steps * 1.5),
    )
    
    frames = []

    for step in tqdm(range(num_steps)):
        optimizer.zero_grad()
        guidance.train_step(text_embeddings, rgb, guidance_scale=100)
        optimizer.step()
        scheduler.step()
        rgb.data = rgb.data.clip(0, 1)
        if step % 20 == 0:
            frames.append(((rgb.squeeze(0).permute(1,2,0).detach().cpu().numpy() + 0.5) * 255).astype(np.uint8))

    frames.append(((rgb.squeeze(0).permute(1,2,0).detach().cpu().numpy() + 0.5) * 255).astype(np.uint8))
    frames = [Image.fromarray(frame) for frame in frames]
    frames[0].save(
                f"training.gif",
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=5,
                loop=0,
            )

    plt.imshow(rgb.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu())
    plt.axis('off')
    plt.show()