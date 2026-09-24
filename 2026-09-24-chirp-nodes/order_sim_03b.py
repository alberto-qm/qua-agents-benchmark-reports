"""03b chirp: frequency innermost, upward only vs alternating direction per shot iteration."""
import numpy as np
import order_sim as o

up = lambda F, D: (np.arange(F)[None, :] * D + np.arange(D)[:, None]).ravel()
down = lambda F, D: (np.arange(F)[::-1][None, :] * D + np.arange(D)[:, None]).ravel()

def sim(ts, noise=0.03, reps=40):
    dfs = np.round(-40e6 + 2.5e6 * np.arange(29)); phi20 = 0.01; D = 21
    flux = np.linspace(-1.5 * phi20, 1.5 * phi20, D); step = flux[1] - flux[0]; F = dfs.size
    orders = {"frequency inner, upward": lambda n: up(F, D),
              "frequency inner, up/down": lambda n: up(F, D) if n % 2 == 0 else down(F, D)}
    for t in ts:
        r = 0.0 if t is None else np.exp(-t)
        for name, fn in orders.items():
            ex, ef = [], []
            for k in range(reps):
                g = np.random.default_rng(300 + k)
                x0 = g.uniform(-0.3, 0.3) * phi20; f0 = g.uniform(-8e6, 8e6)
                fq = f0 - 20e6 * ((flux - x0) / phi20) ** 2
                y = o.run(fn, o.P(dfs[:, None] - fq[None, :]).ravel(), r, 12).reshape(F, D)
                res = o.analyse_flux_map(dfs, flux, y + g.normal(0, noise, y.shape), g.normal(0, noise, y.shape) * 0.1, o.BAND, "chirp")
                ex.append((res["x0"] - x0) / step); ef.append((res["f0"] - f0) / 1e6)
            ex, ef = np.array(ex), np.array(ef)
            lab = "no memory" if t is None else f"{t:g} T1"
            print(f"  {lab:9s} {name:26s} x0 {np.nanmean(ex):+6.3f} ({np.nanstd(ex):5.3f}) steps   "
                  f"f0 {np.nanmean(ef):+5.2f} ({np.nanstd(ef):4.2f}) MHz  fails {int(np.isnan(ex).sum())}")

sim([None, 5, 2, 1, 0.5])
