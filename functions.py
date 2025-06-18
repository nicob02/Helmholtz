
import torch
from core.pde import laplacian, grad
import numpy as np
import math

class ElectroThermalFunc(): 

    func_name = 'Helmholtz'
    def __init__(self, delta_t, params) -> None:
        self.delta_t = delta_t
        self.params = params
        self.laplacianop = laplacian()
        self.gradop = grad()
                    

    def graph_modify(self, graph, value_last, **argv)->None:
        
        x = graph.pos[:, 0:1]
        y = graph.pos[:, 1:2]
        #freq = self.params
        #f = -2*freq*freq*torch.sin(freq*x)*torch.sin(freq*y)
    
        # f(x,y) = 2π cos(πy) sin(πx)
        #         + 2π cos(πx) sin(πy)
        #         + (x+y) sin(πx) sin(πy)
        #         - 2π² (x+y) sin(πx) sin(πy)
        f = (
            2 * math.pi * torch.cos(math.pi * y) * torch.sin(math.pi * x)
            + 2 * math.pi * torch.cos(math.pi * x) * torch.sin(math.pi * y)
            + (x + y) * torch.sin(math.pi * x) * torch.sin(math.pi * y)
            - 2 * (math.pi ** 2) * (x + y) * torch.sin(math.pi * x) * torch.sin(math.pi * y)
        )

        graph.x = torch.cat((graph.pos, value_last, f), dim=-1)

        return graph    

    def init_condition(self, pos):
        
        # Assume pos is a tensor of shape [N, 2] with (x,y) coordinates.
        # Compute the boundary extremes:
        min_x = pos[:, 0].min()
        max_x = pos[:, 0].max()
        min_y = pos[:, 1].min()
        max_y = pos[:, 1].max()
        
        # Compute the distances from each node to the four boundaries:
        d_left   = pos[:, 0:1] - min_x          # distance to left boundary
        d_right  = max_x - pos[:, 0:1]          # distance to right boundary
        d_bottom = pos[:, 1:2] - min_y          # distance to bottom boundary
        d_top    = max_y - pos[:, 1:2]          # distance to top boundary
        
        # For each node, the minimal distance to any boundary is used:
        d = torch.min(torch.cat([d_left, d_right, d_bottom, d_top], dim=1), dim=1)[0].unsqueeze(1)
        
        # Determine the maximum distance across all nodes (typically the center)
        max_d = d.max()
        
        # Choose a random maximum voltage value. For example, pick uniformly between 0.8 and 1.2.
        rand_val = torch.rand(1, device=pos.device) * 0.25 + 1.5
        
        # The initial voltage is set to zero at the boundaries (d=0) and increases linearly to rand_val at the center (d=max_d)
        volt = rand_val * (d / max_d)
        
        return volt


    def boundary_condition(self, graph, predicted):
        
        #volt = torch.full_like(graph.pos[:, 0:1], 0)  # Create a tensor filled with 0s for the B.C. voltage
        
        pos = graph.pos           # [N,2]
        x = pos[:,0:1];  y = pos[:,1:2]
        lb_x, lb_y = pos.min(dim=0).values  # tensor([ℓ_x, ℓ_y])
        ru_x, ru_y = pos.max(dim=0).values  # tensor([r_x, r_y])
        
        # normalized coords in [0,1]
        ξ = (x - lb_x)/(ru_x - lb_x)
        η = (y - lb_y)/(ru_y - lb_y)
        '''
        # PDE‐enforcing ansatz: zero on all four boundaries
        ansatz = torch.tanh(math.pi * ξ) \
          * torch.tanh(math.pi * (1-ξ)) \
          * torch.tanh(math.pi * η) \
          * torch.tanh(math.pi * (1-η))
        '''
        # Ansatz that is zero on x=0,1 and y=0,1
        ansatz = (torch.tanh(np.pi * x)
                  * torch.tanh(np.pi * (1.0 - x))
                  * torch.tanh(np.pi * y)
                  * torch.tanh(np.pi * (1.0 - y)))
        
        # Multiply raw network output by ansatz
        return ansatz * predicted
        

    def exact_solution(self, graph):
        x = graph.pos[:, 0:1]
        y = graph.pos[:, 1:2]
        
        # Compute (x + y) * sin(pi*x) * sin(pi*y)
        u = (x + y) * torch.sin(math.pi * x) * torch.sin(math.pi * y)
        
        return u


    def laplacian_ad(self, graph, u):
        pos = graph.pos
        # Make sure pos is a leaf with grad enabled
        if not pos.requires_grad:
            pos.requires_grad_()
    
        # 1) First derivatives ∂u/∂pos
        grad_u = torch.autograd.grad(
            outputs=u,
            inputs=pos,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,      # <— keep this graph around
        )[0]
    
        lap = torch.zeros_like(u)
        # 2) Second derivatives ∂²u/∂x², ∂²u/∂y²
        for i in range(pos.shape[1]):
            grad2 = torch.autograd.grad(
                outputs=grad_u[:, i],
                inputs=pos,
                grad_outputs=torch.ones_like(grad_u[:, i]),
                create_graph=True,
                retain_graph=True,    # <— **also** retain here
            )[0][:, i:i+1]          # keep as column vector
            lap = lap + grad2
    
        return lap

    
    def pde(self, graph, values_this, **argv):

        """
        PDE: -Δu + u = f(x,y)
        with ε=1, k=1, and
        f(x,y) = 2π cos(πy) sin(πx)
               + 2π cos(πx) sin(πy)
               + (x + y) sin(πx) sin(πy)
               - 2 π² (x + y) sin(πx) sin(πy).
        """
    
        # Extract node positions
        x = graph.pos[:, 0:1]
        y = graph.pos[:, 1:2]
    
        # "values_this" is our predicted u at the current iteration
        volt_this = values_this[:, 0:1]
    
        # Compute the Laplacian of volt_this
        lap_volt = self.laplacian_ad(graph, volt_this)
        #lap_value = self.laplacianop(graph, volt_this)
        #lap_volt = lap_value[:, 0:1]
    
        # Define the forcing function f(x,y)
        f = (
            2 * math.pi * torch.cos(math.pi * y) * torch.sin(math.pi * x)
            + 2 * math.pi * torch.cos(math.pi * x) * torch.sin(math.pi * y)
            + (x + y) * torch.sin(math.pi * x) * torch.sin(math.pi * y)
            - 2 * (math.pi ** 2) * (x + y) * torch.sin(math.pi * x) * torch.sin(math.pi * y)
        )
    
        # PDE residual:  Δu + u - f = 0
        # so the "loss" (residual) is
        loss_volt = lap_volt + volt_this - f
    
        # Optional: print statements for debugging
        print("graph.pos")
        print(graph.pos)
        print("graph.x")
        print(graph.x)
        print("lap_volt")
        print(lap_volt)
        print("losses_volt")
        print(loss_volt)
                
        return loss_volt
        

    
    

    
    
