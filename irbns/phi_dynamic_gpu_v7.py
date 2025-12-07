from typing import Union, Tuple, Optional
import torch
import numpy as np
import matplotlib.pyplot as plt



# ────────────────── 离散 Hausdorff (同 v6) ────────────────
def _hausdorff_cpu(mask_a, mask_b, dx):
    if not _has_scipy:
        raise RuntimeError("SciPy 未安装；安装或改用 hausdorff_mode='gpu'")
    if mask_a.sum() == 0 or mask_b.sum() == 0:
        return np.inf
    dt_b = ndi.distance_transform_edt(~mask_b, sampling=dx)
    d_ab = dt_b[mask_a].max()
    dt_a = ndi.distance_transform_edt(~mask_a, sampling=dx)
    d_ba = dt_a[mask_b].max()
    return float(max(d_ab, d_ba))


def _hausdorff_gpu(mask_a: torch.Tensor, mask_b: torch.Tensor, dx: float):
    if mask_a.sum() == 0 or mask_b.sum() == 0:
        return float("inf")
    def edges(m):
        k = torch.tensor([[0,1,0],[1,1,1],[0,1,0]], device=m.device, dtype=torch.bool)
        neigh = torch.nn.functional.conv2d(m[None,None].float(), k[None,None].float(),
                                           padding=1).bool()[0,0]
        return m & ~neigh
    ea, eb = edges(mask_a), edges(mask_b)
    ya, xa = ea.nonzero(as_tuple=True); yb, xb = eb.nonzero(as_tuple=True)
    pts_a = torch.stack((ya, xa), 1).float() * dx
    pts_b = torch.stack((yb, xb), 1).float() * dx
    d = torch.cdist(pts_a, pts_b)
    return float(torch.max(d.min(1).values.max(), d.min(0).values.max()))



# —— Jacobian / gradient (batched 2×2) ——
def jacobian_batch(f, x, h: float = 1e-6):
    e1 = torch.tensor([h, 0.0], device=x.device, dtype=x.dtype)
    e2 = torch.tensor([0.0, h], device=x.device, dtype=x.dtype)
    col1 = (f(x + e1) - f(x - e1)) / (2 * h)
    col2 = (f(x + e2) - f(x - e2)) / (2 * h)
    return torch.stack((col1, col2), dim=2)          # (..., 2, 2)

def gradient_batch(ep, x, h: float = 1e-6):
    e1 = torch.tensor([h, 0.0], device=x.device, dtype=x.dtype)
    e2 = torch.tensor([0.0, h], device=x.device, dtype=x.dtype)
    g1 = (ep(x + e1) - ep(x - e1)) / (2 * h)
    g2 = (ep(x + e2) - ep(x - e2)) / (2 * h)
    return torch.stack((g1, g2), dim=1)              # (..., 2)

def boundary_map_update_batch(f, ep, pts, normals):
    """
    更新绿色散点的位置/法向，并得到对应红点。
    输入:
        pts     : (M, 2)     —— 上一轮绿色点坐标
        normals : (M, 2)     —— 上一轮法向（单位向量）
    输出:
        z, n_z  : (M, 2)     —— 新一轮绿色点和法向
    """
    fx  = f(pts)                                     # f(·)
    J   = jacobian_batch(f, pts)                     # (M, 2, 2)
    invT = torch.linalg.inv(J).transpose(1, 2)       # (M, 2, 2)
    n_fx = (invT @ normals.unsqueeze(-1)).squeeze(-1)
    n_fx = n_fx / n_fx.norm(dim=1, keepdim=True).clamp_min(1e-12)

    grad_ep_fx = gradient_batch(ep, fx)              # (M, 2)
    dot   = (n_fx * grad_ep_fx).sum(dim=1, keepdim=True)
    normg = grad_ep_fx.norm(dim=1, keepdim=True)
    ep_fx = ep(fx).unsqueeze(1)

    sqrt_term = torch.sqrt(torch.clamp(dot**2 - normg**2 + 1, min=0.0))
    lam = 2 * ep_fx * (dot + sqrt_term)

    z   = fx - ep_fx * grad_ep_fx + 0.5 * lam * n_fx
    n_z = 0.5 * lam * n_fx - ep_fx * grad_ep_fx
    return z.detach(), n_z.detach()



def phi_dynamic_gpu_v7(
    f, ep,
    *,
    L: int,
    init_diameter: float = 1.0,
    init_pos: Union[Tuple[float, float], torch.Tensor] = (0.0, 0.0),
    row: int = 3,
    column: int = 3,
    plot_set_valued: bool = True,
    plot_boundary_map: bool = False,
    bm_num_points: int = 72,
    safety_factor: float = 1.2,
    chunk_size: int = 128,
    save_download: bool = False,
    stop_tol_factor: Optional[float] = None,
    hausdorff_mode: str = "cpu",
    device: Optional[Union[str, torch.device]] = None,
    # ---- grid & window controls ----
    show_grid: bool = True,
    grid_kw: Optional[dict] = None,
    window_mode: str = "square",  # 'square' or 'rect'
    # ---- NEW: axis labels ----
    x_label: str = "S",
    y_label: str = "I",
):
    """
    Dynamic-window version (no forced symmetry around origin).

    window_mode:
      - 'square': centered square window with side = 2*max(halfwidths)
      - 'rect'  : rectangular window fitting bbox with padding

    x_label / y_label set axis labels on all frames/subplots.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype  = torch.float32
    hausdorff_mode = hausdorff_mode.lower()
    if hausdorff_mode not in ("cpu", "gpu"):
        raise ValueError("hausdorff_mode must be 'cpu' or 'gpu'")
    if grid_kw is None:
        grid_kw = {'linestyle': '--', 'alpha': 0.3, 'linewidth': 0.6}
    window_mode = window_mode.lower()
    if window_mode not in ("square", "rect"):
        raise ValueError("window_mode must be 'square' or 'rect'")

    # —— 包装 NumPy 回调 ——
    def _ensure(fn):
        def wrap(x):
            if isinstance(x, torch.Tensor):
                return torch.as_tensor(fn(x.cpu().numpy()), device=x.device, dtype=x.dtype)
            return fn(x)
        return wrap
    try:
        _ = f(torch.zeros((1,2), device=device)); _ = ep(torch.zeros((1,2), device=device))
    except Exception:
        f, ep = _ensure(f), _ensure(ep)

    # —— 初值 & 初始窗口（以 init_pos 为中心） ——
    init_pos = torch.as_tensor(init_pos, dtype=dtype, device=device).view(-1)
    N        = 2**L + 1
    num_iter = row*column - 1

    cx = float(init_pos[0]); cy = float(init_pos[1])
    if init_diameter <= 0:
        init_diameter = float(max(1.0, init_pos.abs().max()))
    half_wx = float(init_diameter); half_wy = float(init_diameter)

    def build_grid(cx, cy, half_wx, half_wy):
        gx = torch.linspace(cx - half_wx, cx + half_wx, N, device=device, dtype=dtype)
        gy = torch.linspace(cy - half_wy, cy + half_wy, N, device=device, dtype=dtype)
        Gx, Gy = torch.meshgrid(gx, gy, indexing='xy')
        return gx, gy, torch.stack((Gx, Gy), dim=-1)

    grid_x, grid_y, G = build_grid(cx, cy, half_wx, half_wy)

    A = torch.zeros((N,N), dtype=torch.bool, device=device)
    B = torch.zeros_like(A)

    step_x = (grid_x[-1] - grid_x[0])/(N-1)
    step_y = (grid_y[-1] - grid_y[0])/(N-1)
    idx_x = ((init_pos[0] - grid_x[0]) / step_x).round().long().clamp(0, N-1)
    idx_y = ((init_pos[1] - grid_y[0]) / step_y).round().long().clamp(0, N-1)
    A[idx_y, idx_x] = True
    B[idx_y, idx_x] = True

    f_init   = f(init_pos.unsqueeze(0))[0]
    eps_init = ep(f_init.unsqueeze(0))[0]

    def stash(mask, gx, gy):
        return (mask.detach().cpu().numpy(),
                gx.detach().cpu().numpy(),
                gy.detach().cpu().numpy())

    A_hist, B_hist = ([], [])
    if plot_set_valued:
        A_hist.append(stash(A, grid_x, grid_y))
        B_hist.append(stash(B, grid_x, grid_y))

    # —— 图像框架 ——
    axes = None
    if plot_set_valued or plot_boundary_map:
        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(column*6, row*6))
        sub = gridspec.GridSpec(row, column, wspace=0.3, hspace=0.3)
        axes = [fig.add_subplot(sub[i,j]) for i in range(row) for j in range(column)]
        ax0  = axes[0]
        if plot_set_valued:
            A0, gx0, gy0 = stash(A, grid_x, grid_y)
            U,V = np.meshgrid(gx0, gy0, indexing='xy')
            ax0.pcolormesh(U, V, A0, cmap='Reds',  shading='auto')
            ax0.pcolormesh(U, V, A0, cmap='Blues', shading='auto', alpha=0.5)
        ax0.set_title('Iteration 0')
        ax0.set_aspect('equal')
        ax0.set_xlabel(x_label); ax0.set_ylabel(y_label)   # <<< NEW
        ax0.grid(show_grid, **grid_kw)

    bm_pts = bm_nrm = None
    B_prev = B.clone() if stop_tol_factor is not None else None

    for t in range(1, num_iter+1):
        active_pts = G[B]
        if active_pts.numel() == 0:
            break

        fx = f(active_pts)
        r_max = ep(fx).max()
        x_min, _ = fx[:,0].min(dim=0)
        x_max, _ = fx[:,0].max(dim=0)
        y_min, _ = fx[:,1].min(dim=0)
        y_max, _ = fx[:,1].max(dim=0)
        pad = safety_factor * float(r_max)

        x_lo = float(x_min) - pad; x_hi = float(x_max) + pad
        y_lo = float(y_min) - pad; y_hi = float(y_max) + pad
        cx   = 0.5*(x_lo + x_hi); cy = 0.5*(y_lo + y_hi)

        if window_mode == "square":
            half_w = 0.5*max(x_hi - x_lo, y_hi - y_lo)
            if half_w < 1e-8:
                half_w = init_diameter
            half_wx = half_wy = float(half_w)
        else:
            half_wx = max(0.5*(x_hi - x_lo), 1e-8)
            half_wy = max(0.5*(y_hi - y_lo), 1e-8)

        grid_x, grid_y, G = build_grid(cx, cy, half_wx, half_wy)
        A.zero_(); B.zero_()
        step_x = (grid_x[-1] - grid_x[0])/(N-1)
        step_y = (grid_y[-1] - grid_y[0])/(N-1)

        ix = ((fx[:,0] - grid_x[0]) / step_x).round().long().clamp(0, N-1)
        iy = ((fx[:,1] - grid_y[0]) / step_y).round().long().clamp(0, N-1)
        A[iy, ix] = True

        if A.any():
            pts, radii = G[A], ep(G[A])
            Gx, Gy = G[...,0], G[...,1]
            for s in range(0, pts.shape[0], chunk_size):
                pb, rb = pts[s:s+chunk_size], radii[s:s+chunk_size]
                dx = pb[:,0,None,None] - Gx
                dy = pb[:,1,None,None] - Gy
                B |= (dx**2 + dy**2 <= (rb[:,None,None]+1e-12)**2).any(0)

        if stop_tol_factor is not None:
            dxy = float(max(step_x, step_y))
            if hausdorff_mode == "cpu":
                dH = _hausdorff_cpu(B_prev.cpu().numpy(), B.cpu().numpy(), dxy)
            else:
                dH = _hausdorff_gpu(B_prev, B, dxy)
            if dH < stop_tol_factor * max(half_wx, half_wy):
                print(f"[Early-Stop] iter={t}, H={dH:.3e} < tol={stop_tol_factor*max(half_wx,half_wy):.3e}")
                break
            B_prev = B.clone()

        if plot_boundary_map:
            if t == 1:
                theta = torch.linspace(0, 2*np.pi, bm_num_points+1, device=device, dtype=dtype)[:-1]
                bm_pts = f_init + eps_init * torch.stack((torch.cos(theta), torch.sin(theta)), 1)
                bm_nrm = torch.stack((torch.cos(theta), torch.sin(theta)), 1)
                red_pts = None
            else:
                red_pts = f(bm_pts)
                bm_pts, bm_nrm = boundary_map_update_batch(f, ep, bm_pts, bm_nrm)

        if axes and t < len(axes):
            ax = axes[t]; ax.cla()
            if plot_set_valued:
                A_cpu, gx_cpu, gy_cpu = stash(A, grid_x, grid_y)
                B_cpu, _, _ = stash(B, grid_x, grid_y)
                U,V = np.meshgrid(gx_cpu, gy_cpu, indexing='xy')
                ax.pcolormesh(U, V, A_cpu, cmap='Reds', shading='auto')
                ax.pcolormesh(U, V, B_cpu, cmap='Blues', shading='auto', alpha=0.5)
            if plot_boundary_map and bm_pts is not None:
                if red_pts is not None:
                    ax.scatter(red_pts[:,0].cpu(), red_pts[:,1].cpu(), c='red', s=0.5)
                ax.scatter(bm_pts[:,0].cpu(), bm_pts[:,1].cpu(), c='green', s=0.5, marker='x')
            ax.set_title(f'Iteration {t}')
            ax.set_aspect('equal')
            ax.set_xlabel(x_label); ax.set_ylabel(y_label)   # <<< NEW
            ax.grid(show_grid, **grid_kw)

        if plot_set_valued:
            A_hist.append(stash(A, grid_x, grid_y))
            B_hist.append(stash(B, grid_x, grid_y))

    if axes:
        plt.tight_layout(); plt.show()
        if save_download:
            from google.colab import files
            fig.savefig("/content/phi_dynamic_gpu_result.png", dpi=300)
            files.download("/content/phi_dynamic_gpu_result.png")

    return A_hist, B_hist
