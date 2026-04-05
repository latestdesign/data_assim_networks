import torch
import matplotlib.pyplot as plt
import manage_exp


@torch.no_grad()
def simulate_sample(net, prop, obs, x0, horizon, h_dim):
    """Generate x_t, y_t and mu_a_t for one sample trajectory."""
    ha = manage_exp.get_ha0(1, h_dim)
    x = x0.clone()

    xs, ys, mus = [], [], []
    for _ in range(horizon + 1):
        y = obs(x).sample()
        hb = net.b(ha)
        ha = net.a(torch.cat([hb, y], dim=1))
        pdf_a = net.c(ha)

        xs.append(x.squeeze(0).detach().cpu())
        ys.append(y.squeeze(0).detach().cpu())
        mus.append(pdf_a.mean.squeeze(0).detach().cpu())

        x = prop(x).sample()

    return torch.stack(xs), torch.stack(ys), torch.stack(mus)


def plot_trajectories(xt, yt, mua, title, dims=None):
    """Plot trajectories in 2D state space (x_dim==2) or per dimension over time.
    Red: true state x_t, Green: observations y_t, Blue: analysis mean mu_a_t.
    """
    x_dim = xt.shape[1]
    xt_np = xt.cpu().numpy()
    yt_np = yt.cpu().numpy()
    mua_np = mua.cpu().numpy()

    color_true = "#DC2626"
    color_obs = "#16A34A"
    color_analysis = "#2563EB"

    if x_dim == 2:
        # 2D state space plot
        fig, ax = plt.subplots(figsize=(8, 8))

        # Points in the state space
        ax.scatter(xt_np[:, 0], xt_np[:, 1], s=26, c=color_true, marker="o", edgecolors="none", label="true x_t")
        ax.scatter(yt_np[:, 0], yt_np[:, 1], s=22, c=color_obs, marker="x", edgecolors="none", alpha=0.80, label="obs y_t")
        ax.scatter(mua_np[:, 0], mua_np[:, 1], s=26, c=color_analysis, marker="o", edgecolors="none", label="analysis mu_a_t")

        # Trajectory lines
        ax.plot(xt_np[:, 0], xt_np[:, 1], color=color_true, alpha=0.22, lw=1.0)
        ax.plot(yt_np[:, 0], yt_np[:, 1], color=color_obs, alpha=0.18, lw=1.0)
        ax.plot(mua_np[:, 0], mua_np[:, 1], color=color_analysis, alpha=0.22, lw=1.0)

        # First and last true states
        ax.scatter(xt_np[0, 0], xt_np[0, 1], s=80, c=color_true, marker="^", edgecolors="black", linewidths=0.5, label="start x_0")
        ax.scatter(xt_np[-1, 0], xt_np[-1, 1], s=80, c=color_true, marker="X", edgecolors="black", linewidths=0.5, label="end x_T")

        ax.set_xlabel("dim 0")
        ax.set_ylabel("dim 1")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)

        for spine in ["top", "right"]: # Nicer axes
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_alpha(0.35)
        ax.spines["bottom"].set_alpha(0.35)

        ax.legend(loc="best", frameon=True, framealpha=0.9, fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold")

    else:
        # Per dimension over time
        if dims is None:
            dims = list(range(0, x_dim, x_dim // 4))[:4]

        fig, axes = plt.subplots(len(dims), 1, figsize=(8, 3 * len(dims)), sharex=True)
        if len(dims) == 1:
            axes = [axes]
        ts = range(xt_np.shape[0])

        for ax, d in zip(axes, dims):
            ax.plot(ts, xt_np[:, d], color=color_true, lw=1.2, label="true x_t")
            ax.scatter(ts, yt_np[:, d], s=8, c=color_obs, marker="x", alpha=0.5, label="obs y_t")
            ax.plot(ts, mua_np[:, d], color=color_analysis, lw=1.2, alpha=0.85, label="analysis mu_a_t")
            ax.set_ylabel(f"dim {d}")
            ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)

            for spine in ["top", "right"]: # Nicer axes
                ax.spines[spine].set_visible(False)
            ax.spines["left"].set_alpha(0.35)
            ax.spines["bottom"].set_alpha(0.35)

        axes[0].legend(loc="best", frameon=True, framealpha=0.9, fontsize=9)
        axes[0].set_title(title, fontsize=12, fontweight="bold")
        axes[-1].set_xlabel("t")

    plt.tight_layout()
    plt.show()
