import torch
from core.utils.tools import parse_config, modelTester, RemoveDir
from core.utils.tools import compute_steady_error, render_results
from core.models import msgPassing
from core.geometry import ElectrodeMesh
from functions import ElectroThermalFunc as Func
import os
from NM import run_fem_helmholtz



delta_t = 1 # Mess around with this
#poisson_params = 6.28318
#poisson_params = 12.56637  #4pi, initial test case, change later.
poisson_params = 25.13274  #8pi, initial test case, change later.
#func_name = 'rfa'
out_ndim = 1

dens=65
ckptpath = 'checkpoint/simulator_%s.pth' % Func.func_name    #FIGURE THIS OUT
device = torch.device(0)

func_main = Func(delta_t=delta_t, params=poisson_params)

bc1 = func_main.boundary_condition
ic = func_main.init_condition
lb=(0, 0)
ru=(1, 1)
mesh = ElectrodeMesh(ru=ru, lb=lb, density=65)
graph = mesh.getGraphData()
model = msgPassing(message_passing_num=3, node_input_size=out_ndim+2, 
                   edge_input_size=3, ndim=out_ndim, device=device, model_dir=ckptpath)
model.load_model(ckptpath)
model.to(device)
model.eval()
test_steps = 20
lb = torch.tensor((0.0, 0.0), device=device)
ru = torch.tensor((1.0, 1.0), device=device)
test_config = parse_config()

#model = kwargs['model'] # Extracts the model's dictioanry with the weights and biases values
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

      

#-----------------------------------------

print('************* model test starts! ***********************')
predicted_results = modelTester(test_config)

# Analytic (exact) field on graph nodes
u_exact = func_main.exact_solution(graph)  
u_exact_np  = u_exact.detach().cpu().numpy()

coords_fem, u_fem = run_fem_helmholtz(
    mesh=mesh,
    coords=graph.pos.cpu().numpy(),  # sample FEM on the same points as the GNN
    eps_val=1.0,
    k_val=1.0,
    lb=lb,                                  # <<< set these to your Helmholtz domain
    ru=ru 
)

# --- Errors vs analytic (report BOTH, as reviewers asked) ---
err_gnn_vs_exact = compute_steady_error(predicted_results, u_exact_np, test_config)
err_fem_vs_exact = compute_steady_error(u_fem,            u_exact_np, test_config)

print(f"Relative L2 error (GNN vs analytic): {err_gnn_vs_exact:.3e}")
print(f"Relative L2 error (FEM vs analytic): {err_fem_vs_exact:.3e}")

# (Optional) also show how close GNN is to FEM:
err_gnn_vs_fem = compute_steady_error(predicted_results, u_fem, test_config)
print(f"Relative L2 error (GNN vs FEM):      {err_gnn_vs_fem:.3e}")
# 3) Render the three‐panel result
render_results(predicted_results, u_exact_np, graph, filename="helmholtz_steady.png")





