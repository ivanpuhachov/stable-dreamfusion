import torch
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
from nerf.sd import StableDiffusion, seed_everything
from experiment_single_rgb import get_cosine_schedule_with_warmup
from encoding import get_encoder


class Implicit2D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        n_hidden_neurons = 256
        self.encoder, self.enc_dim = get_encoder('frequency_torch', input_dim=2, log2_hashmap_size=4, desired_resolution=256)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.enc_dim, n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
            torch.nn.Linear(n_hidden_neurons, n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
            torch.nn.Linear(n_hidden_neurons, n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
            torch.nn.Linear(n_hidden_neurons, 3),
            # torch.nn.Sigmoid(),
            # To ensure that the colors correctly range between [0-1],
            # the layer is terminated with a sigmoid layer.
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.mlp(x)

    def render(self, size=64, clip=False, device="cuda"):
        xs = torch.linspace(-1, 1, steps=size)
        ys = torch.linspace(-1, 1, steps=size)
        x, y = torch.meshgrid(xs, ys, indexing='xy')
        xy = torch.stack([x, y], dim=2).view(-1, 2).to(device)
        rgb = self.forward(xy).view(size, size, 3)
        if clip:
            rgb = torch.clip(rgb, 0.0, 1.0)
        return rgb


class Implicit2DTrainer:
    def __init__(
            self,
            model: Implicit2D,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

    @staticmethod
    def save_frames_as_gif(file_name, frames):
        frames = [Image.fromarray(frame) for frame in frames]
        out_dir = os.path.join(os.getcwd(), "renders")
        os.makedirs(out_dir, exist_ok=True)
        frames[0].save(
            file_name,
            save_all=True,
            append_images=frames[1:],
            optimize=False,
            duration=5 * len(frames) / 20,
            loop=0,
        )

    def fit_image(
            self,
            image: torch.Tensor,
            n_steps=100,
            save_every=10,
    ):
        assert image.shape[0] == image.shape[1]
        assert image.shape[2] == 3
        image = image.to(self.device)
        imsize = image.shape[0]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4)
        frames = []
        for i_step in (pbar := tqdm(range(n_steps))):
            optimizer.zero_grad()
            output = self.model.render(size=imsize)
            loss = torch.nn.MSELoss()(output, image)
            loss.backward()
            optimizer.step()
            pbar.set_description(f"{loss.item():.5f}")
            if (i_step % save_every == (save_every-1)) or (i_step == 0) or (i_step == n_steps):
                frames.append((output.detach().cpu().numpy() * 255).astype(np.uint8))

        self.save_frames_as_gif("demo.gif", frames)
        plt.figure()
        plt.imshow(frames[-1])
        plt.show()
    
    def optimize_sds(
            self,
            prompt="pineapple",
            n_steps=500,
            save_every=10,
    ):
        frames = []
        guidance = StableDiffusion(self.device)
        guidance.vae.decoder = None
        text_embeddings = guidance.get_text_embeds(prompt, '')
        guidance.text_encoder.to('cpu')
        torch.cuda.empty_cache()
        seed_everything(42)
        # rgb = torch.nn.Parameter(torch.rand(1, 3, 128, 128, device=self.device) / 2 + .5)
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=0)
        # optimizer = torch.optim.Adam([rgb], lr=0.01)
        # optimizer = torch.optim.AdamW([rgb], lr=1e-1, weight_decay=0)
        scheduler = get_cosine_schedule_with_warmup(
            opt=optimizer,
            num_warmup_steps=100,
            num_training_steps=int(n_steps * 1.5),
        )
        for i_step in (pbar := tqdm(range(n_steps))):
            optimizer.zero_grad()
            output = self.model.render(size=128, clip=True)
            guidance.train_step(text_embeddings, output.unsqueeze(0).permute(0,3,1,2), guidance_scale=100)
            # guidance.train_step(text_embeddings, rgb, guidance_scale=100)
            optimizer.step()
            scheduler.step()
            # rgb.data = rgb.data.clip(0, 1)
            if (i_step % save_every == (save_every-1)) or (i_step == 0) or (i_step == n_steps):
                frames.append((output.detach().cpu().numpy() * 255).astype(np.uint8))
                # frames.append((rgb.squeeze(0).permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8))
                Image.fromarray(frames[-1]).save(
                    f"last.png",
                )
        self.save_frames_as_gif("demo.gif", frames)


def test_implicit_2d():
    model = Implicit2D()
    out = model.render(size=4)
    print(out.shape)


def test_implicit_2d_trainer():
    height, width = 128, 128
    gt_image = torch.ones((height, width, 3)) * 1.0
    gt_image[: height // 2, : width // 2, :] = torch.tensor([1.0, 0.0, 0.0])
    gt_image[height // 2:, width // 2:, :] = torch.tensor([0.0, 0.0, 1.0])
    mymodel = Implicit2D()
    mytrainer = Implicit2DTrainer(model=mymodel)
    # mytrainer.fit_image(image=gt_image, n_steps=1800, save_every=20)
    mytrainer.optimize_sds()
    # print(mytrainer.model.enc_dim)
    # print(mymodel.encoder)


if __name__ == '__main__':
    test_implicit_2d_trainer()