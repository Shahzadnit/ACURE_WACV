import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

    
class Neg_corr(nn.Module):
    def __init__(self):
        super(Neg_corr, self).__init__()

    def forward(self, x,y):
        x = x - x.mean(dim=-1, keepdim=True)
        y = y - y.mean(dim=-1, keepdim=True)
        x_norm = F.normalize(x, p=2, dim=-1)
        y_norm = F.normalize(y, p=2, dim=-1)
        corr = (x_norm * y_norm).sum(dim=-1)
        return -corr.mean()


class NegCorrLoss(nn.Module):
    def __init__(self):
        super(NegCorrLoss, self).__init__()

    def forward(self, x, y):
        # Calculate mean centered predictions and true values
        x = x - x.mean(dim=-1, keepdim=True)
        y = y - y.mean(dim=-1, keepdim=True)
        
        x_norm = F.normalize(x, p=2, dim=-1)
        y_norm = F.normalize(y, p=2, dim=-1)
        
        corr = (x_norm * y_norm).sum(dim=-1)
        return -corr.mean()

def calculate_continuity_residual(rho: torch.Tensor) -> torch.Tensor:
    """
    Calculate the continuity residual (R) from a 3-channel tensor representing density and 2D velocity components.
    
    Args:
        rho (torch.Tensor): Tensor of shape (batch, 3, temp_dim, H, W)
            - rho[:, 0, :, :, :] → Density field (ρ)
            - rho[:, 1, :, :, :] → Velocity field in x-direction (u)
            - rho[:, 2, :, :, :] → Velocity field in y-direction (v)
    
    Returns:
        continuity_residual (torch.Tensor): Tensor of shape (batch, temp_dim, H, W) representing continuity residual.
    """
    # Extract density and velocity components
    density = rho[:, 0, :, :, :]  # Density field (ρ)
    u = rho[:, 1, :, :, :]        # Velocity in x (u)
    v = rho[:, 2, :, :, :]        # Velocity in y (v)
    
    # --- Step 1: Temporal Derivative (∂ρ/∂t) ---
    d_rho_dt = torch.gradient(density, dim=1)[0]  # Temporal derivative along T (time axis)
    
    # --- Step 2: Divergence of Mass Flux (∇⋅(ρV)) ---
    rho_u = density * u  # Mass flux in x-direction
    rho_v = density * v  # Mass flux in y-direction
    
    # Spatial derivatives
    drho_u_dx = torch.gradient(rho_u, dim=3)[0]  # Spatial derivative along W (x-axis)
    drho_v_dy = torch.gradient(rho_v, dim=2)[0]  # Spatial derivative along H (y-axis)
    
    # Divergence of mass flux
    div_rho_V = drho_u_dx + drho_v_dy
    
    # --- Step 3: Continuity Residual (R) ---
    continuity_residual = d_rho_dt + div_rho_V

    continuity_error = torch.mean(torch.abs(continuity_residual))

    return continuity_error




# Define the custom loss function as described in the paper
class SpO2Loss(nn.Module):
    def __init__(self, alpha=0.1):
        super(SpO2Loss, self).__init__()
        self.mse = nn.MSELoss()
        self.neg_corr = NegCorrLoss()
        self.alpha = alpha

    def forward(self, y_pred, y_true, X_DC_pred, X_DC_true, X_AC_pred, X_AC_true):
        # Calculate L_SpO2
        l_spo2 = self.mse(y_pred, y_true) + self.neg_corr(y_pred, y_true)
        
        # Calculate the additional terms for the end-to-end loss
        l_dc = self.mse(X_DC_pred, X_DC_true)
        l_ac = self.mse(X_AC_pred, X_AC_true)

        
        # Combine to form L_EndToEnd
        l_end_to_end = l_spo2 + self.alpha * (l_dc + l_ac) +  calculate_continuity_residual(X_AC_pred)   #pde_residual(X_AC_pred)           #naive_pde_residual(X_AC_pred)

        return l_end_to_end
