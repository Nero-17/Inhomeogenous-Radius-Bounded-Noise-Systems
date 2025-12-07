## Installation

In Colab or a Python environment with git:

```bash
pip install --no-cache-dir git+https://github.com/Nero-17/Inhomeogenous-Radius-Bounded-Noise-Systems.git


import torch
from irbns import phi_dynamic_gpu_v7

def f(x):
    A = torch.tensor([[0.9, -0.1],
                      [0.1,  0.9]], device=x.device, dtype=x.dtype)
    return x @ A.T

def ep(x):
    return torch.full(x.shape[:-1], 0.1, device=x.device, dtype=x.dtype)

A_hist, B_hist = phi_dynamic_gpu_v7(
    f, ep,
    L=6,
    init_diameter=1.0,
    init_pos=(0.0, 0.0),
    row=2, column=3,
)
