import numpy as np
import matplotlib.pyplot as plt
import GPy
import io
import contextlib
import re
from matplotlib import patches
import textwrap


def print_model_summary(m):
    """
    Print a concise summary of an optimised GPy regression model.

    Parameters
    ----------
    m : Trained and optimised GPy regression model.
    """
    print("MODEL SUMMARY")
    print(f"NEGATIVE LOG MARGINAL LIKELIHOOD: {-float(m.log_likelihood()):.6g}")

    def show_kernel(k, indent="  "):
        name = getattr(k, "name", type(k).__name__).lower()

        # Composite kernel (sum or product)
        if hasattr(k, "parts") and len(k.parts) > 0:
            op = "sum" if isinstance(k, GPy.kern.Add) else "product" if isinstance(k, GPy.kern.Prod) else "composite"
            #print(f"{indent}{op.capitalize()} of kernels:")
            for i, part in enumerate(k.parts, 1):
                print(f"{indent}  Component {i}:")
                show_kernel(part, indent + "    ")
            return

        # Base kernel
        label = (
            "SE" if "rbf" in name else
            "Periodic" if "periodic" in name else
            type(k).__name__
        )

        # Lengthscale(s)
        if hasattr(k, "lengthscale"):
            ell = np.asarray(getattr(k.lengthscale, "values", k.lengthscale)).ravel()
            if len(ell) == 1:
                print(rf"{indent}• lengthscale (ℓ): {ell[0]:.6g}")
            else:
                print(f"{indent}• lengthscales (ℓ, ARD): {ell.tolist()}")

        # Variance
        if hasattr(k, "variance"):
            sf = float(np.sqrt(np.asarray(getattr(k.variance, "values", k.variance))))
            print(f"{indent}• signal std (σ_f): {sf:.6g}")

        # Period
        if hasattr(k, "period"):
            p = float(np.asarray(getattr(k.period, "values", k.period)))
            print(f"{indent}• period (p): {p:.6g}")

    # Print kernel structure
    print("OPTIMISED HYPERPARAMETERS:")
    show_kernel(m.kern, indent="  ")

    # Noise std
    sn = float(np.sqrt(np.asarray(getattr(m.likelihood.variance, "values", m.likelihood.variance))))
    print(f"  • noise std (σ_n): {sn:.6g}") if not(hasattr(m.kern, "parts") and len(m.kern.parts) > 0) else print(f"    Noise std (σ_n): {sn:.6g}")


def plot_predictive_error_bars(m, X, y, code_snippet, save_path=None):
    """
    Plot the GP predictive mean and 95% predictive error bars for 1D input data.

    Parameters
    ----------    
    m : Trained GPy regression model (after optimisation).
    X : Training input data.
    y : Training targets.
    save_path : If provided, the plot will be saved to this path.
    """
    # Build grid for predictions
    Xs = np.linspace(
        X.min() - 0.1 * (X.max() - X.min()),
        X.max() + 0.1 * (X.max() - X.min()),
        400
    )[:, None]

    # Predictive mean and variances
    mu_f, var_f = m.predict(Xs, full_cov=False)
    var_y = var_f + m.likelihood.variance
    sd_y = np.sqrt(var_y)

    summary_text = None
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_model_summary(m)
    summary_text = buf.getvalue().rstrip()
    summary_text = summary_text.replace("ℓ", r"$\ell$")
    summary_text = summary_text.replace("MODEL SUMMARY", r"$\mathbf{MODEL\ SUMMARY}$"
    )

    def _round_num(match):
        num = float(match.group())
        return f"{num:.3f}".rstrip('0').rstrip('.') if abs(num) > 1e-3 else match.group()

    summary_text = re.sub(r"-?\d+\.\d+(?:[eE][-+]?\d+)?", _round_num, summary_text)

    fig = plt.figure(figsize=(12, 8))
    outer = fig.add_gridspec(2, 1, height_ratios=(6, 3.5), hspace=0.28)
    ax = fig.add_subplot(outer[0])

    bottom = outer[1].subgridspec(1, 2, width_ratios=(2.5, 1.5), wspace=0.1)
    ax_code = fig.add_subplot(bottom[0, 0])
    ax_txt  = fig.add_subplot(bottom[0, 1])

    ax_txt.axis("off")
    ax_txt.text(
        0.0, 0.88, summary_text,
        va="top", ha="left",
        family="sans-serif",
        fontsize=10,
        linespacing=1.8
    )

    # Plot
    kernel_name = "SE" if type(m.kern).__name__ == "RBF" else "Periodic"
    ax.plot(X, y, "x", ms=6, color="red", label="Training data")
    ax.plot(Xs, mu_f, "b", lw=2, label="Predictive mean")
    ax.fill_between(
        Xs[:, 0],
        mu_f[:, 0] - 1.96 * sd_y[:, 0],
        mu_f[:, 0] + 1.96 * sd_y[:, 0],
        alpha=0.2,
        color="gray",
        label="95% CI"
    )
    ax.legend(loc='best')
    ax.set_title(f"{kernel_name} Kernel — 95% predictive error bars")
    ax.set_xlabel(r"Input, $x$")
    ax.set_ylabel(r"Output, $y$")

    ax_code.axis("off")
    x0, y0, w, h = 0.02, 0.08, 0.96, 0.84
    box = patches.FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0.012,rounding_size=8",
        linewidth=1.0, edgecolor="#cccccc", facecolor="#f7f7f7",
        transform=ax_code.transAxes, clip_on=True
    )
    ax_code.add_patch(box)

    # optional title
    ax_code.text(
        0.02, 0.88, 'CODE',
        transform=ax_code.transAxes,
        va="top", ha="left",
        fontsize=10, fontweight="bold", family="sans-serif"
    )

    fig.canvas.draw()                          
    bbox = ax_code.get_window_extent(fig.canvas.get_renderer())

    left_pad  = 0.025                          
    right_pad = 0.04                           
    content_px = bbox.width * (1 - left_pad - right_pad)

    code_fs = 9                                 
    avg_char_px = (code_fs * fig.dpi / 72.0) * 0.62
    wrap_cols = max(20, int(content_px / avg_char_px))

    wrapped = "\n".join(
        textwrap.fill(
            line.expandtabs(4), width=wrap_cols,
            break_long_words=False, break_on_hyphens=False
        )
        for line in code_snippet.splitlines()
    )

    # render code (monospace) inside the box
    ax_code.text(
        left_pad, 0.84, wrapped,
        transform=ax_code.transAxes,
        va="top", ha="left",
        fontfamily="monospace",
        fontsize=9,
        linespacing=1.75,
    )
    
    if save_path is not None:
        plt.savefig(save_path, dpi=600)

    
    plt.show()


def plot_sampled_functions(X, F, code_snippet, n_draws=3, save_path=None):
    """
    Plot functions sampled from a GP prior with the specified kernel.

    Parameters
    ----------
    X : Input grid (N x 1) at which the functions are evaluated
    F : Sampled function values (N x n_draws).
    save_path : If provided, saves the figure to this path.
    """

    fig = plt.figure(figsize=(10, 7.0), constrained_layout=False)
    outer = fig.add_gridspec(2, 1, height_ratios=(5,2.5), hspace=0.28)

    ax = fig.add_subplot(outer[0])     # top plot
    ax_code = fig.add_subplot(outer[1])  # bottom code panel

    for j in range(n_draws):
        ax.plot(X[:, 0], F[:, j], lw=1.5, alpha=0.9, label=f"Sample {j+1}")
    ax.set_title("Samples from GP with composite kernel = Periodic × SE")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$f(x)$")
    ax.grid(True, alpha=0.2)
    ax.legend()

    ax_code.axis("off")
    # full-bleed rounded background
    bg = patches.FancyBboxPatch(
        (0.02, 0.08), 0.96, 0.84,
        boxstyle="round,pad=0.012,rounding_size=8",
        linewidth=1.0, edgecolor="#cccccc", facecolor="#f7f7f7",
        transform=ax_code.transAxes, clip_on=True, zorder=0
    )
    ax_code.add_patch(bg)

    title_y   = 0.88
    text_y    = 0.84
    left_pad  = 0.04
    right_pad = 0.04
    code_fs   = 9

    # heading
    ax_code.text(
        left_pad, title_y, 'CODE',
        transform=ax_code.transAxes, va="top", ha="left",
        fontsize=10, fontweight="bold", family="sans-serif"
    )

    # wrap code to available width (respects left/right pads)
    fig.canvas.draw()
    bbox = ax_code.get_window_extent(fig.canvas.get_renderer())
    content_px = bbox.width * (1 - left_pad - right_pad)

    # rough monospace glyph width in px
    avg_char_px = (code_fs * fig.dpi / 72.0) * 0.62
    wrap_cols = max(24, int(content_px / avg_char_px))

    wrapped = "\n".join(
        textwrap.fill(
            line.expandtabs(4),
            width=wrap_cols,
            break_long_words=False,
            break_on_hyphens=False
        )
        for line in code_snippet.splitlines()
    )

    ax_code.text(
        left_pad, text_y, wrapped,
        transform=ax_code.transAxes, va="top", ha="left",
        fontfamily="monospace", fontsize=code_fs, linespacing=1.25
    )

    if save_path is not None:
        plt.savefig(save_path, dpi=600)
    plt.show()


def print_model_marginal_likelihood(m):
    """
    Print the log marginal likelihood decomposition for an optimised GPy regression model.

    Parameters
    ----------
    m : Trained and optimised GPy regression model.
    """
    # Extract training data and parameters
    X = m.X
    y = m.Y
    N = X.shape[0]
   
    if getattr(m, "mean_function", None) is None:
        mu = np.zeros_like(y)
    else:
        mu = m.mean_function.f(X)
    
    K = m.kern.K(X, X)
    sigma_n2 = float(m.likelihood.variance)
    Ky = K + sigma_n2 * np.eye(N)
    
    L = np.linalg.cholesky(Ky)
    r = y - mu
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, r))

    data_fit = -0.5 * float(r.T @ alpha)
    complexity = 0.5 * np.linalg.slogdet(Ky)[1] 
    constant = -0.5 * N * np.log(2 * np.pi)
    log_marginal = data_fit - complexity + constant

    # Print results
    print(f"LML: {log_marginal:.6f}")
    print(f"  • Data fit term: {data_fit:.6f}")
    print(f"  • Complexity penalty: {complexity:.6f}")
    print(f"  • Constant term: {constant:.6f}")


def plot_2d_predictive(mA, mB, codeA=None, codeB=None, save_path=None):
    """
    Plot 2D GP predictive surfaces for two models side by side.
    Parameters
    ----------
    mA : First trained GPy regression model.
    mB : Second trained GPy regression model.
    codeA : Code snippet string for model A.
    codeB : Code snippet string for model B.
    save_path : If provided, saves the figure to this path.
    """
    NGRID = 101
    VIEW_ELEV, VIEW_AZIM = 15, 225
    DP = 3
    TITLE_LEFT  = "Simple SE-ARD covariance"
    TITLE_RIGHT = "Sum of SE-ARD covariance functions"

    X = np.asarray(mA.X); y = np.asarray(mA.Y).ravel()
    assert X.shape[1] >= 2, "plot_2d_predictive expects at least 2D inputs."
    u1d = np.linspace(X[:,0].min(), X[:,0].max(), NGRID)
    u2d = np.linspace(X[:,1].min(), X[:,1].max(), NGRID)
    x1g, x2g = np.meshgrid(u1d, u2d)
    Xgrid = np.column_stack([x1g.ravel(), x2g.ravel()])

    muA, _ = mA.predict(Xgrid);  MuA = muA.reshape(x1g.shape)
    muB, _ = mB.predict(Xgrid);  MuB = muB.reshape(x1g.shape)

    def _capture_print(callable_):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            callable_()
        return buf.getvalue().rstrip()

    summaryA = _capture_print(lambda: print_model_summary(mA))
    summaryB = _capture_print(lambda: print_model_summary(mB))
    lmlA     = _capture_print(lambda: print_model_marginal_likelihood(mA))
    lmlB     = _capture_print(lambda: print_model_marginal_likelihood(mB))

    def strip_header_and_next(txt, n_after=2, headers=("MODEL SUMMARY")):
        lines = txt.splitlines()
        if not lines:
            return txt
        first = lines[0].strip().upper()
        start = 1 + n_after if first.startswith(headers) else n_after
        start = min(start, len(lines))
        return "\n".join(lines[start:]).lstrip()

    summaryA = strip_header_and_next(summaryA, n_after=1)
    summaryB = strip_header_and_next(summaryB, n_after=1)

    def _round_num(m):
        x = float(m.group())
        return f"{x:.{DP}f}".rstrip('0').rstrip('.') if abs(x) > 1e-3 else m.group()
    num_pat = r"-?\d+\.\d+(?:[eE][-+]?\d+)?"

    def _process(txt):
        txt = txt.replace("ℓ", r"$\ell$")
        if 'OPTIMISED HYPERPARAMETERS' in txt:
            txt = txt.replace('OPTIMISED HYPERPARAMETERS', r"$\mathbf{OPTIMISED\ HYPERPARAMETERS}$")
        return re.sub(num_pat, _round_num, txt)

    summaryA = _process(summaryA)
    summaryB = _process(summaryB)
    lmlA     = _process(lmlA)
    lmlB     = _process(lmlB)

    fig = plt.figure(figsize=(15, 15), constrained_layout=False)

    outer = fig.add_gridspec(nrows=7, ncols=1, hspace=0.0)
    outer.set_height_ratios([
        4.0,   0.01,   
        1.25,  0.01,  
        1.75,  0.01,   
        1.25           
    ])
    row1 = outer[0].subgridspec(1, 2, wspace=0.08)  # 3D pair: wider gutter
    row2 = outer[2].subgridspec(1, 2, wspace=0.04)  # CODE pair: tighter
    row3 = outer[4].subgridspec(1, 2, wspace=0.06)  # SUMMARY pair
    row4 = outer[6].subgridspec(1, 2, wspace=0.04)  # LML pair

    ax1 = fig.add_subplot(row1[0, 0], projection='3d')
    ax2 = fig.add_subplot(row1[0, 1], projection='3d')

    ax_code_A = fig.add_subplot(row2[0, 0]); ax_code_B = fig.add_subplot(row2[0, 1])
    ax_sum_A  = fig.add_subplot(row3[0, 0]); ax_sum_B  = fig.add_subplot(row3[0, 1])
    ax_lml_A  = fig.add_subplot(row4[0, 0]); ax_lml_B  = fig.add_subplot(row4[0, 1])

    def draw_panel(ax, title, body, boxed=True, monospace=False, title_y=0.88, text_y=0.78,
                   left_pad=0.02, right_pad=0.02, face="#f7f7f7", edge="#cccccc",
                   fsize=10, code_fsize=9, linespacing=1.25):
        ax.axis("off")
        if boxed:
            ax.add_patch(patches.FancyBboxPatch(
                (0.02, 0.08), 0.96, 0.84,
                boxstyle="round,pad=0.012,rounding_size=8",
                linewidth=1.0, edgecolor=edge, facecolor=face,
                transform=ax.transAxes, clip_on=True, zorder=0
            ))
        if title:
            ax.text(left_pad, title_y, title, transform=ax.transAxes,
                    va="top", ha="left", fontsize=fsize, fontweight="bold", family="sans-serif")

        fig.canvas.draw()
        bbox = ax.get_window_extent(fig.canvas.get_renderer())
        fs = code_fsize if monospace else fsize
        content_px = bbox.width * (1 - left_pad - right_pad)
        avg_char_px = (fs * fig.dpi / 72.0) * 0.62
        cols = max(24, int(content_px / avg_char_px))
        wrapped = "\n".join(
            textwrap.fill(line.expandtabs(4), width=cols, break_long_words=False, break_on_hyphens=False)
            for line in (body or "").splitlines()
        )
        ax.text(left_pad, text_y, wrapped, transform=ax.transAxes,
                va="top", ha="left",
                fontfamily=("monospace" if monospace else "sans-serif"),
                fontsize=fs, linespacing=linespacing)

    ax1.plot_wireframe(x1g, x2g, MuA, rstride=1, cstride=1, linewidth=0.5, color='C0', alpha=0.85)
    ax1.scatter(X[:,0], X[:,1], y, c='r', s=18, depthshade=True, label='Training data')
    ax1.set_title(TITLE_LEFT, y=0.92)
    ax1.set_xlabel(r'$x_1$'); ax1.set_ylabel(r'$x_2$'); ax1.set_zlabel(r'$y$')
    ax1.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    ax1.set_box_aspect(None, zoom=0.9)

    ax2.plot_wireframe(x1g, x2g, MuB, rstride=1, cstride=1, linewidth=0.5, color='C0', alpha=0.85, label='Predictive mean')
    ax2.scatter(X[:,0], X[:,1], y, c='r', s=18, depthshade=True, label='Training data')
    ax2.set_title(TITLE_RIGHT, y=0.92)
    ax2.set_xlabel(r'$x_1$'); ax2.set_ylabel(r'$x_2$'); ax2.set_zlabel(r'$y$')
    ax2.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    ax2.set_box_aspect(None, zoom=0.9)
    ax2.legend(loc='upper right')

    draw_panel(ax_code_A, "CODE", codeA, monospace=True)
    draw_panel(ax_sum_A,  None, summaryA, boxed=False, monospace=False, linespacing=1.6, title_y=0.88, text_y=0.78)
    draw_panel(ax_lml_A,  "LML DECOMPOSITION", lmlA, boxed=False, monospace=False, linespacing=1.6, title_y=0.88, text_y=0.75)

    draw_panel(ax_code_B, "CODE", codeB, monospace=True)
    draw_panel(ax_sum_B,  None, summaryB, boxed=False, monospace=False, linespacing=1.6, title_y=0.88, text_y=0.78)
    draw_panel(ax_lml_B,  "LML DECOMPOSITION", lmlB, boxed=False, monospace=False, linespacing=1.6, title_y=0.88, text_y=0.75)

    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()