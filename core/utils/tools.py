import json
import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil
import math
from torch_geometric.data import Data

def RemoveDir(filepath):
    '''
    If the folder doesn't exist, create it; and if it exists, clear it.
    '''
    if not os.path.exists(filepath):
        os.makedirs(filepath,exist_ok=True)
    else:
        shutil.rmtree(filepath)
        os.makedirs(filepath, exist_ok=True)


class Config:
    def __init__(self) -> None:
        pass
    def __setattr__(self, __name: str, __value) -> None:
        self.__dict__[__name] = __value


def parse_config(file='config.json'):
    configs = Config() 
    if not os.path.exists(file):
        return configs
    with open(file, 'r') as f:
        data = json.load(f)
        for k, v in data.items():
            config = Config()
            if isinstance(v, dict):
                for k1, v1 in v.items():
                    config.setattr(k1, v1)
            else:
                raise TypeError
            configs[k] = config
    return configs[k]


def modelTrainer(config):
    model = config.model
    graph = config.graph
    opt = config.optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=config.lrstep, gamma=0.99)

    # Precompute node features: [x, y, f]
    x = graph.pos[:, 0:1]
    y = graph.pos[:, 1:2]
    f = (2*math.pi*torch.cos(math.pi*y)*torch.sin(math.pi*x)
         + 2*math.pi*torch.cos(math.pi*x)*torch.sin(math.pi*y)
         + (x + y)*torch.sin(math.pi*x)*torch.sin(math.pi*y)
         - 2*(math.pi**2)*(x + y)*torch.sin(math.pi*x)*torch.sin(math.pi*y))

    # For plotting/logging
    train_hist = {"epoch": [], "residual_loss": [], "relL2_exact": []}

    # (optional) closed-form exact u for Helmholtz test
    u_exact = (x + y) * torch.sin(math.pi * x) * torch.sin(math.pi * y)

    for epoch in range(1, config.epchoes + 1):
        graph.x = torch.cat([graph.pos, f], dim=-1)   # [N,3]
        u_raw = model(graph)
        u = config.bc1(graph, u_raw, lb=config.lb, ru=config.ru)

        # PDE residual (your pde() returns strong-form residual at nodes)
        res = config.pde(graph, values_this=u)  # shape [N,1] or [N]
        loss = torch.mean(res**2)               # MSE of residual (more standard than ||·||)

        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

        # logging (every epoch)
        with torch.no_grad():
            # relative L2 to exact (optional but nice for Helmholtz section)
            relL2 = torch.linalg.norm(u - u_exact) / torch.linalg.norm(u_exact)
            train_hist["epoch"].append(epoch)
            train_hist["residual_loss"].append(loss.item())
            train_hist["relL2_exact"].append(relL2.item())

        if epoch % 500 == 0:
            print(f"[Epoch {epoch:5d}] residual_MSE={loss.item():.3e}  relL2={relL2.item():.3e}")

    # save history for plotting
    import pandas as pd
    pd.DataFrame(train_hist).to_csv("helmholtz_training_history.csv", index=False)

    model.save_model(opt)
    print(f"Training completed. Final residual_MSE={loss.item():.3e}, relL2={relL2.item():.3e}")
        
@torch.no_grad()
def modelTester(config):
    """
    Single‐shot evaluation of the trained steady‐state GNN.
    Returns:
      u_pred (numpy array [N,1]): Predicted solution at each mesh node.
    """
    # 1) Move model and graph to the right device
    model = config.model.to(config.device)
    graph = config.graph.to(config.device)

    # 2) Build the fixed node features [x, y, f] exactly as in training
    x = graph.pos[:, 0:1]
    y = graph.pos[:, 1:2]
    f = (
        2 * math.pi * torch.cos(math.pi * y) * torch.sin(math.pi * x)
      + 2 * math.pi * torch.cos(math.pi * x) * torch.sin(math.pi * y)
      + (x + y) * torch.sin(math.pi * x) * torch.sin(math.pi * y)
      - 2 * (math.pi**2) * (x + y) * torch.sin(math.pi * x) * torch.sin(math.pi * y)
    )
    graph.x = torch.cat([x, y, f], dim=-1)

    # 3) Forward pass + boundary enforcement
    u_raw  = model(graph)               # shape [N,1]
    u_pred = config.bc1(graph, u_raw, lb=config.lb, ru=config.ru)   # apply ansatz/hard clamp

    return u_pred.cpu().numpy()


def compute_steady_error(u_pred, u_exact, config):
    # 1) Convert predictions to NumPy
    if isinstance(u_pred, torch.Tensor):
        u_pred_np = u_pred.detach().cpu().numpy()
    else:
        u_pred_np = np.array(u_pred, copy=False)

    # 2) Convert exact to NumPy
    if isinstance(u_exact, torch.Tensor):
        u_exact_np = u_exact.detach().cpu().numpy()
    else:
        u_exact_np = np.array(u_exact, copy=False)

    # 3) Flatten both to 1D arrays
    u_pred_flat  = u_pred_np.reshape(-1)
    u_exact_flat = u_exact_np.reshape(-1)

    # 4) Compute relative L2 norm
    num   = np.linalg.norm(u_pred_flat - u_exact_flat)
    denom = np.linalg.norm(u_exact_flat)
    rel_l2 = num / (denom + 1e-16)  # small eps to avoid div0

    return rel_l2

def render_results(u_pred, u_exact, graph, filename="steady_results.png"):
    """
    Scatter‐plot Exact, Predicted, and Absolute Error on the mesh nodes.
    """
    pos = graph.pos.cpu().numpy()
    x, y = pos[:,0], pos[:,1]
    error = np.abs(u_exact - u_pred)

    fig, axes = plt.subplots(1, 3, figsize=(18,5))

    # 1) Exact
    sc0 = axes[0].scatter(x, y, c=u_exact.flatten(), cmap='viridis', s=5)
    axes[0].set_title("Exact Solution")
    plt.colorbar(sc0, ax=axes[0], shrink=0.7)

    # 2) Predicted
    sc1 = axes[1].scatter(x, y, c=u_pred.flatten(), cmap='viridis', s=5)
    axes[1].set_title("GNN Prediction")
    plt.colorbar(sc1, ax=axes[1], shrink=0.7)

    # 3) Absolute Error
    sc2 = axes[2].scatter(x, y, c=error.flatten(), cmap='magma', s=5)
    axes[2].set_title("Absolute Error")
    plt.colorbar(sc2, ax=axes[2], shrink=0.7)

    for ax in axes:
        ax.set_xlabel("x"); ax.set_ylabel("y")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)


