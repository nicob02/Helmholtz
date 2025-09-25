# test_helmholtz_timed.py

import os, time, statistics, platform
import numpy as np
import torch

from core.utils.tools import parse_config, modelTester, RemoveDir
from core.utils.tools import compute_steady_error, render_results
from core.models import msgPassing
from core.geometry import ElectrodeMesh
from functions import ElectroThermalFunc as Func
from NM import run_fem_helmholtz


# -------------------------
# Timing helpers
# -------------------------
def timeit_repeat(fn, *args, repeats=5, synchronize_cuda=True, desc="op"):
    """Time a callable with optional CUDA sync; returns list of seconds."""
    times = []
    # warmup (esp. for CUDA kernels / JIT graphs)
    _ = fn(*args)
    if torch.cuda.is_available() and synchronize_cuda:
        torch.cuda.synchronize()
    for _ in range(repeats):
        if torch.cuda.is_available() and synchronize_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = fn(*args)
        if torch.cuda.is_available() and synchronize_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times

def describe_system():
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    print("=== System / Runtime Info ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {gpu}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"CPU: {platform.processor() or platform.machine()}")
    print(f"Threads (intraop): {torch.get_num_threads()}")
    print("=============================\n")


# -------------------------
# Config & setup
# -------------------------
delta_t = 1
poisson_params = 25.13274  # 8*pi
out_ndim = 1

dens = 150
device = torch.device(0 if torch.cuda.is_available() else "cpu")
ckptpath = 'checkpoint/simulator_%s.pth' % Func.func_name

func_main = Func(delta_t=delta_t, params=poisson_params)

bc1 = func_main.boundary_condition
ic  = func_main.init_condition
lb_tuple = (0.0, 0.0)
ru_tuple = (1.0, 1.0)

mesh  = ElectrodeMesh(ru=ru_tuple, lb=lb_tuple, density=dens)
graph = mesh.getGraphData()
graph = graph.to(device)

model = msgPassing(
    message_passing_num=3,
    node_input_size=out_ndim+2,
    edge_input_size=3,
    ndim=out_ndim,
    device=device,
    model_dir=ckptpath
)
model.load_model(ckptpath)
model.to(device)
model.eval()

test_steps = 30
lb = torch.tensor(lb_tuple, device=device)
ru = torch.tensor(ru_tuple, device=device)

test_config = parse_config()
setattr(test_config, 'poisson_params', poisson_params)
setattr(test_config, 'delta_t', delta_t)
setattr(test_config, 'device', device)
setattr(test_config, 'ic', ic)
setattr(test_config, 'bc1', bc1)
setattr(test_config, 'model', model)
setattr(test_config, 'test_steps', test_steps)
setattr(test_config, 'NodeTypesRef', ElectrodeMesh.node_type_ref)
setattr(test_config, 'lb', lb)
setattr(test_config, 'ru', ru)
setattr(test_config, 'ndim', out_ndim)
setattr(test_config, 'graph_modify', func_main.graph_modify)
setattr(test_config, 'graph', graph)
setattr(test_config, 'density', dens)

describe_system()

N_nodes = graph.num_nodes
print(f"Nodes: {N_nodes}, density: {dens}\n")

# -------------------------
# 1) NN inference timing
# -------------------------
print('************* model test starts! ***********************')

with torch.no_grad():
    nn_times = timeit_repeat(lambda: modelTester(test_config),
                             repeats=7, synchronize_cuda=True, desc="NN inference")

nn_t_median = statistics.median(nn_times)
nn_t_mean   = statistics.mean(nn_times)
nn_t_std    = statistics.pstdev(nn_times)

# actual predictions to use below
with torch.no_grad():
    predicted_results = modelTester(test_config)

print("NN inference time over {} repeats:".format(len(nn_times)))
print("  mean   = {:.6f} s".format(nn_t_mean))
print("  median = {:.6f} s".format(nn_t_median))
print("  std    = {:.6f} s".format(nn_t_std))
print("  per-node (median) = {:.3f} µs/node".format(1e6 * nn_t_median / N_nodes))
print()

# -------------------------
# 2) FEM solve timing
# -------------------------
def fem_call():
    return run_fem_helmholtz(
        mesh=mesh,
        coords=graph.pos.detach().cpu().numpy(),  # sample at GNN nodes
        eps_val=1.0,
        k_val=1.0,
        lb=lb_tuple,
        ru=ru_tuple
    )

fem_times = timeit_repeat(lambda: fem_call(), repeats=5, synchronize_cuda=False, desc="FEM solve")
fem_t_median = statistics.median(fem_times)
fem_t_mean   = statistics.mean(fem_times)
fem_t_std    = statistics.pstdev(fem_times)

coords_fem, u_fem = fem_call()  # actual FEM output to use below

print("FEM solve time over {} repeats:".format(len(fem_times)))
print("  mean   = {:.6f} s".format(fem_t_mean))
print("  median = {:.6f} s".format(fem_t_median))
print("  std    = {:.6f} s".format(fem_t_std))
print("  per-node (median) = {:.3f} µs/node".format(1e6 * fem_t_median / N_nodes))
print()

# -------------------------
# 3) Accuracy vs analytic + vs FEM
# -------------------------
u_exact = func_main.exact_solution(graph)
u_exact_np = u_exact.detach().cpu().numpy()

err_gnn_vs_exact = compute_steady_error(predicted_results, u_exact_np, test_config)
err_fem_vs_exact = compute_steady_error(u_fem,            u_exact_np, test_config)
err_gnn_vs_fem   = compute_steady_error(predicted_results, u_fem,      test_config)

print(f"Relative L2 error (GNN vs analytic): {err_gnn_vs_exact:.3e}")
print(f"Relative L2 error (FEM vs analytic): {err_fem_vs_exact:.3e}")
print(f"Relative L2 error (GNN vs FEM):      {err_gnn_vs_fem:.3e}")

# -------------------------
# 4) Render (FEM vs exact) at graph nodes
# -------------------------
pred_1d  = np.asarray(predicted_results).reshape(-1)
exact_1d = np.asarray(u_exact_np).reshape(-1)
fem_1d   = np.asarray(u_fem).reshape(-1)

# This utility expects three inputs: field_pred, field_exact, graph, filename
render_results(fem_1d, exact_1d, graph, filename="helmholtz_steady.png")
print("Saved: helmholtz_steady.png")

