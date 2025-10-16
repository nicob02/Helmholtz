# run_seeds.py
import os, json, argparse, statistics, random
import numpy as np
import torch

from core.utils.tools import parse_config, modelTester, compute_steady_error, modelTrainer
from core.models import msgPassing
from core.geometry import ElectrodeMesh
from functions import ElectroThermalFunc as Func
from NM import run_fem_helmholtz

# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def summary(vals, name):
    if not vals:
        return f"{name}: (no values)"
    m = statistics.mean(vals)
    s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return f"{name}: mean={m:.3e}, std={s:.3e}"

# -------------------------
# Build base config (unique ckpt per seed)
# -------------------------
def build_base(seed: int, epochs: int, density: int, tag: str, device=None):
    device = device or torch.device(0 if torch.cuda.is_available() else "cpu")
    delta_t = 1
    poisson_params = 25.13274  # 8*pi
    out_ndim = 1

    func_main = Func(delta_t=delta_t, params=poisson_params)

    mesh  = ElectrodeMesh(ru=(1.0, 1.0), lb=(0.0, 0.0), density=density)
    graph = mesh.getGraphData().to(device)

    # Unique checkpoint path per seed to avoid overwrite
    ckpt_dir = os.path.join("checkpoint", "reliability")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"sim_{Func.func_name}_seed{seed}{('_'+tag) if tag else ''}.pth")

    model = msgPassing(
        message_passing_num=3,
        node_input_size=out_ndim + 2,
        edge_input_size=3,
        ndim=out_ndim,
        device=device,
        model_dir=ckpt_path
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    lb = torch.tensor((0.0, 0.0), device=device)
    ru = torch.tensor((1.0, 1.0), device=device)

    cfg = parse_config()
    setattr(cfg, 'pde',           func_main.pde)
    setattr(cfg, 'graph_modify',  func_main.graph_modify)
    setattr(cfg, 'delta_t',       delta_t)
    setattr(cfg, 'ic',            func_main.init_condition)
    setattr(cfg, 'bc1',           func_main.boundary_condition)
    setattr(cfg, 'graph',         graph)
    setattr(cfg, 'model',         model)
    setattr(cfg, 'lb',            lb)
    setattr(cfg, 'ru',            ru)
    setattr(cfg, 'optimizer',     optimizer)
    setattr(cfg, 'train_steps',   1)
    setattr(cfg, 'NodeTypesRef',  ElectrodeMesh.node_type_ref)
    setattr(cfg, 'step_times',    1)
    setattr(cfg, 'ndim',          out_ndim)
    setattr(cfg, 'lrstep',        100)
    setattr(cfg, 'device',        device)
    setattr(cfg, 'epchoes',       epochs)

    return cfg, func_main, mesh, graph, model, ckpt_path

# -------------------------
# Post-hoc train residual loss (no trainer changes needed)
# -------------------------
@torch.no_grad()
def compute_train_residual_loss(cfg, func_main):
    """
    Rebuild node features [x,y,f], forward, apply BC, compute PDE residual norm / sqrt(N).
    This approximates a comparable "train loss" after training.
    """
    model = cfg.model.to(cfg.device)
    graph = cfg.graph.to(cfg.device)

    x = graph.pos[:, 0:1]
    y = graph.pos[:, 1:2]
    f = (
        2 * np.pi * torch.cos(np.pi * y) * torch.sin(np.pi * x)
      + 2 * np.pi * torch.cos(np.pi * x) * torch.sin(np.pi * y)
      + (x + y) * torch.sin(np.pi * x) * torch.sin(np.pi * y)
      - 2 * (np.pi ** 2) * (x + y) * torch.sin(np.pi * x) * torch.sin(np.pi * y)
    )
    graph.x = torch.cat([x, y, f], dim=-1)

    u_raw = model(graph)
    u     = cfg.bc1(graph, u_raw, lb=cfg.lb, ru=cfg.ru)

    res = cfg.pde(graph, values_this=u)           # [N,1] or [N]
    loss = torch.norm(res) / (res.numel() ** 0.5) # normalize by sqrt(N)
    return float(loss.item())

# -------------------------
# Seed run: train → eval
# -------------------------
def run_one_seed(seed: int, epochs: int, density: int, tag: str, device=None):
    set_seed(seed)
    cfg, func_main, mesh, graph, model, ckpt_path = build_base(seed, epochs, density, tag, device)

    # Train with your existing trainer (saves to ckpt_path at final epoch)
    modelTrainer(cfg)

    # Load & eval
    model.load_model(ckpt_path)
    model.eval()

    # NN prediction at graph nodes
    with torch.no_grad():
        u_pred = modelTester(cfg)  # numpy [N,1]

    # Analytic exact
    u_exact = func_main.exact_solution(graph).detach().cpu().numpy()

    # FEM baseline sampled at GNN nodes
    coords_fem, u_fem = run_fem_helmholtz(
        mesh=mesh,
        coords=graph.pos.detach().cpu().numpy(),
        eps_val=1.0,
        k_val=1.0,
        lb=(0.0, 0.0),
        ru=(1.0, 1.0)
    )

    # Errors
    rel_gnn_vs_exact = compute_steady_error(u_pred, u_exact, cfg)
    rel_fem_vs_exact = compute_steady_error(u_fem,  u_exact, cfg)
    rel_gnn_vs_fem   = compute_steady_error(u_pred, u_fem,   cfg)

    # Post-hoc "train residual loss"
    train_res_loss = compute_train_residual_loss(cfg, func_main)

    return {
        "seed": seed,
        "epochs": epochs,
        "density": density,
        "ckpt_path": ckpt_path,
        "train_residual_loss": float(train_res_loss),
        "RelL2_gnn_vs_exact": float(rel_gnn_vs_exact),
        "RelL2_fem_vs_exact": float(rel_fem_vs_exact),
        "RelL2_gnn_vs_fem":   float(rel_gnn_vs_fem),
    }

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Multi-seed reliability runner (single file).")
    parser.add_argument("--seeds", type=str, default="111,222,333,444,555",
                        help="Comma-separated seeds, e.g., '111,222,333'")
    parser.add_argument("--epochs", type=int, default=3000, help="Training epochs per seed")
    parser.add_argument("--density", type=int, default=65, help="Mesh density")
    parser.add_argument("--tag", type=str, default="et_8pi", help="Optional tag for filenames")
    parser.add_argument("--save_json", type=str, default="checkpoint/reliability/seeds_report.json",
                        help="Where to write the JSON report")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    device = torch.device(0 if torch.cuda.is_available() else "cpu")

    os.makedirs(os.path.dirname(args.save_json), exist_ok=True)

    results = []
    for s in seeds:
        print(f"\n=== Seed {s} ===")
        res = run_one_seed(s, args.epochs, args.density, args.tag, device=device)
        for k, v in res.items():
            print(f"{k}: {v}")
        results.append(res)

    # Print summaries
    print("\n=== Multi-Seed Reliability Summary ===")
    print(summary([r["train_residual_loss"]  for r in results], "train_residual_loss"))
    print(summary([r["RelL2_gnn_vs_exact"]  for r in results], "RelL2 GNN vs analytic"))
    print(summary([r["RelL2_fem_vs_exact"]  for r in results], "RelL2 FEM vs analytic"))
    print(summary([r["RelL2_gnn_vs_fem"]    for r in results], "RelL2 GNN vs FEM"))

    # Save JSON
    with open(args.save_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved report: {args.save_json}")

if __name__ == "__main__":
    main()
