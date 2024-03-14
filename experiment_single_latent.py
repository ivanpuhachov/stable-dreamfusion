'''
https://github.com/ashawkey/stable-dreamfusion/issues/96
'''

import math
from tqdm import tqdm
import torch
import torch.nn as nn
from nerf.sd import StableDiffusion, seed_everything
from torch.optim.lr_scheduler import LambdaLR

import matplotlib.pyplot as plt
from experiment_single_rgb import get_cosine_schedule_with_warmup


if __name__ == '__main__':
    device = 'cuda:0'
    guidance = StableDiffusion(device)
    guidance.vae.encoder = None

    prompt = 'pineapple'
    text_embeddings = guidance.get_text_embeds(prompt, '')
    guidance.text_encoder.to('cpu')
    torch.cuda.empty_cache()

    seed_everything(42)
    latents = nn.Parameter(torch.randn(1, 4, 64, 64, device=device))
    optimizer = torch.optim.AdamW([latents], lr=1e-1, weight_decay=0)
    num_steps = 1000
    scheduler = get_cosine_schedule_with_warmup(optimizer, 100, int(num_steps*1.5))

    for step in tqdm(range(num_steps)):
        optimizer.zero_grad()

        t = torch.randint(guidance.min_step, guidance.max_step + 1, [1], dtype=torch.long, device=guidance.device)
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = guidance.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = guidance.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + 100 * (noise_pred_text - noise_pred_uncond)

        w = (1 - guidance.alphas[t])
        grad = w * (noise_pred - noise)

        latents.backward(gradient=grad, retain_graph=True)

        optimizer.step()
        scheduler.step()

        if step > 0 and step % 100 == 0:
            rgb = guidance.decode_latents(latents)
            img = rgb.detach().squeeze(0).permute(1,2,0).cpu().numpy()
            print('[INFO] save image', img.shape, img.min(), img.max())
            plt.imsave(f'tmp_lat_img_{step}.jpg', img)