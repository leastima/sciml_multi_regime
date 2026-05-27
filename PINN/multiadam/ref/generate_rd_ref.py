import numpy as np
import os

def generate_ref_data(alpha=5.0, tau=4.0, zeta=1.0 / (2.0 * (np.pi / 4.0) ** 2), L=2*np.pi, nx=256, nt=101, filename='ref/reaction_diffusion.dat'):
    x = np.linspace(0, L, nx, endpoint=False)
    dx = x[1] - x[0]
    
    # Second derivative matrix (dense for simplicity)
    D2 = np.zeros((nx, nx))
    for i in range(nx):
        D2[i, (i-1) % nx] = -1 / dx**2
        D2[i, i] = 2 / dx**2
        D2[i, (i+1) % nx] = -1 / dx**2
    
    u0 = np.exp(-zeta * (x - np.pi)**2)
    
    # Stability-limited dt for diffusion
    dt = 0.5 * dx**2 / tau  # Conservative factor 0.5 for safety
    total_steps = int(1 / dt) + 1
    dt = 1 / total_steps  # Adjust dt to exactly reach t=1
    step_save = total_steps // (nt - 1)
    
    u_hist = [u0.copy()]
    
    u = u0.copy()
    
    for i in range(1, total_steps + 1):
        du = tau * (D2 @ u) + alpha * u * (1 - u)
        u += dt * du
        if i % step_save == 0:
            u_hist.append(u.copy())
    
    # Stack and save
    sol_y = np.stack(u_hist)  # (nt, nx)
    X, T = np.meshgrid(x, np.linspace(0, 1, nt))
    data = np.column_stack((X.ravel(), T.ravel(), sol_y.ravel()))
    
    # Create ref/ if needed
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savetxt(filename, data, fmt='%.8e', header='x t u', comments='% ')  # Prefix header with '% ' to make it skippable
    
    print(f"Success! Saved {filename} with shape {data.shape}. dt={dt}, total_steps={total_steps}")

# Loop over alphas with fixed tau=2
alphas = [5, 10, 20, 40, 80]
tau = 2.0  # Fixed as per your setup
for alpha in alphas:
    filename = f'ref/rd_alpha{alpha:.1f}_tau{tau:.1f}.dat'  # Use :.1f to ensure .0 for integers like 5.0
    generate_ref_data(alpha=alpha, tau=tau, filename=filename)
