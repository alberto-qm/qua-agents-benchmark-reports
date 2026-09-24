"""Shot-order memory: leftover excitation from a short wait, per scan order, fitted with the nodes' own code.

A chirp inside its box inverts: post = p_in + P (1 - 2 p_in); the next shot starts with r * post,
r = exp(-t) for a wait of t T1. Populations propagate exactly (linear in p), then the node's fit runs.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "code/QM/qua-libs-chirp/qualibration_graphs/superconducting"))
from calibration_utils.qubit_spectroscopy_chirp.boxes import box_model, find_boxes  # noqa: E402
from calibration_utils.qubit_spectroscopy_vs_flux_chirp.analysis import analyse_flux_map  # noqa: E402

BAND, EDGE = 20e6, 1.0e6
rng0 = np.random.default_rng(1)


def P(d):
    return box_model(d, 0.0, 1.0, BAND, EDGE, 0.0)


def run(order_fn, Pgrid, r, shots):
    """order_fn(n) -> flat point indices for shot iteration n; returns mean post per point."""
    acc = np.zeros(Pgrid.size)
    cnt = np.zeros(Pgrid.size)
    post = 0.0
    for n in range(shots):
        for i in order_fn(n):
            p_in = r * post
            post = p_in + Pgrid[i] * (1 - 2 * p_in)
            acc[i] += post
            cnt[i] += 1
    return acc / cnt


def orders_1d(N, rng):
    perms = [rng.permutation(N) for _ in range(8)]
    fixed = rng.permutation(N)
    up = np.arange(N)
    return {
        "frequency inner, upward (now)": lambda n: up,
        "up/down alternating": lambda n: up if n % 2 == 0 else up[::-1],
        "one fixed shuffle": lambda n: fixed,
        "8 shuffles, cycled": lambda n: perms[n % 8],
        "shots inner (repeat each point)": None,
    }


def sim_03a(ts, noise=0.02, reps=40):
    x = np.round(5e6 * (np.arange(49) - 24))
    rows = {}
    for t in ts:
        r = 0.0 if t is None else np.exp(-t)
        for name in orders_1d(49, rng0):
            errs, heights, tails = [], [], []
            for k in range(reps):
                rng = np.random.default_rng(100 + k)
                fq = rng.uniform(-40e6, 40e6)
                Pg = P(x - fq)
                if name.startswith("shots inner"):
                    order = np.repeat(np.arange(49), 100)
                    y = run(lambda n: order, Pg, r, 1)
                else:
                    y = run(orders_1d(49, rng)[name], Pg, r, 20)
                yn = y + rng.normal(0, noise, y.size)
                bs = find_boxes(x, yn, noise, BAND)
                if not bs:
                    errs.append(np.nan)
                    continue
                b = max(bs, key=lambda b: b.snr)
                errs.append(b.centre - fq)
                heights.append(b.height)
                outside = np.abs(x - fq) > BAND / 2 + 3 * EDGE
                tails.append(np.max(y[outside]))
            errs = np.array(errs)
            rows[(t, name)] = (np.nanmean(errs) / 1e6, np.nanstd(errs) / 1e6, np.mean(heights), np.mean(tails),
                               int(np.isnan(errs).sum()))
    return rows


def sim_03b(ts, noise=0.03, reps=20):
    dfs = np.round(-40e6 + 2.5e6 * np.arange(29))
    phi20 = 0.01
    D = 21
    flux = np.linspace(-1.5 * phi20, 1.5 * phi20, D)
    step = flux[1] - flux[0]
    F = dfs.size
    rows = {}
    for t in ts:
        r = 0.0 if t is None else np.exp(-t)
        rng = np.random.default_rng(7)
        perms = [rng.permutation(F * D) for _ in range(8)]
        orders = {
            "flux inner, upward (now)": lambda n: np.arange(F * D),                          # index f*D + j
            "flux inner, snake": lambda n: np.concatenate([np.arange(D) + f * D if f % 2 == 0 else
                                                           np.arange(D)[::-1] + f * D for f in range(F)]),
            "frequency inner, upward": lambda n: (np.arange(F)[None, :] * D + np.arange(D)[:, None]).ravel(),
            "8 shuffles, cycled": lambda n: perms[n % 8],
        }
        for name, fn in orders.items():
            errs, ferrs = [], []
            for k in range(reps):
                g = np.random.default_rng(300 + k)
                x0 = g.uniform(-0.3, 0.3) * phi20
                f0 = g.uniform(-8e6, 8e6)
                fq = f0 - 20e6 * ((flux - x0) / phi20) ** 2                                     # (D,)
                Pg = P(dfs[:, None] - fq[None, :]).ravel()                                      # (F*D,)
                y = run(fn, Pg, r, 12).reshape(F, D)
                I = y + g.normal(0, noise, y.shape)
                Q = g.normal(0, noise, y.shape) * 0.1
                res = analyse_flux_map(dfs, flux, I, Q, BAND, "chirp")
                errs.append((res["x0"] - x0) / step)
                ferrs.append((res["f0"] - f0) / 1e6)
            errs, ferrs = np.array(errs), np.array(ferrs)
            rows[(t, name)] = (np.nanmean(errs), np.nanstd(errs), np.nanmean(ferrs), int(np.isnan(errs).sum()))
    return rows, step / phi20


if __name__ == "__main__":
    ts = [None, 5, 3, 2, 1, 0.5]
    lab = lambda t: "no memory" if t is None else f"{t:g} T1"
    print("03a chirp, 1D cut, 5 MHz steps, band 20 MHz; centre error [MHz] mean +- sd, box height, max tail outside")
    a = sim_03a(ts)
    for (t, name), (m, s, h, tail, nf) in a.items():
        print(f"  {lab(t):9s} {name:34s} bias {m:+6.2f}  sd {s:5.2f}  height {h:4.2f}  tail {tail:4.2f}  fails {nf}")
    print("\n03b chirp, 29 x 21 map; sweet-spot error in flux steps (mean +- sd), apex frequency error [MHz]")
    b, st = sim_03b(ts)
    for (t, name), (m, s, fm, nf) in b.items():
        print(f"  {lab(t):9s} {name:26s} x0 bias {m:+6.3f} steps  sd {s:5.3f}   f0 bias {fm:+5.2f}  fails {nf}")
