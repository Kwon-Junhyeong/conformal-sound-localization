"""Adaptive conformal prediction over a stream of maps. The threshold is calibrated on one maps file and the stream is
one or more maps files in order; FILE:dev or FILE:eval keeps only the windows of that LOCATA split. With --hold-out, a
random share of the calibration recordings is kept out of the calibration and streamed first, so the stream starts in
the calibration conditions. For every part of the stream, the share of windows whose region holds the true direction
and the median region size are compared between the fixed threshold and adaptive conformal inference.
  python run_stream.py --cal maps_locata.npz:dev --stream maps_locata.npz:eval maps_locata_0dB.npz
  python run_stream.py --cal maps_sim.npz --hold-out 0.5 --stream maps_box.npz
  [--alpha 0.1] [--gamma 0.02]"""
import argparse

import numpy as np

import conformal as cp


def load(spec):
    path, _, split = spec.partition(':')
    z = np.load(path)
    keep = z['split'] == split if split else np.ones(len(z['true_idx']), bool)
    P = cp.norm01(z['maps'][keep].astype(np.float32))
    return P, cp.nonconformity(P, z['true_idx'][keep]), z['err'][keep], z['grid'], z['rec'][keep]


def parts(a):
    """Calibration scores and the stream as a list of (name, maps, scores, errors)."""
    P, s, err, grid, rec = load(a.cal)
    held = np.zeros(len(s), bool)
    if a.hold_out > 0:
        recs = np.random.default_rng(a.seed).permutation(np.unique(rec))
        held = np.isin(rec, recs[:int(round(len(recs) * a.hold_out))])
    stream = [('%s, held out' % a.cal, P[held], s[held], err[held])] if held.any() else []
    for spec in a.stream:
        Ps, ss, es, _, _ = load(spec)
        stream.append((spec, Ps, ss, es))
    return s[~held], stream, grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cal', required=True)
    ap.add_argument('--stream', nargs='+', required=True)
    ap.add_argument('--hold-out', type=float, default=0.0)
    ap.add_argument('--alpha', type=float, default=0.1)
    ap.add_argument('--gamma', type=float, default=0.02)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    s_cal, stream, grid = parts(a)
    w = cp.solid_angle_weights(grid)
    q0 = cp.conformal_tau(s_cal, a.alpha)
    aci = cp.AdaptiveCP(s_cal, a.alpha, a.gamma)
    print('calibration %s: %d windows, alpha %g, gamma %g' % (a.cal, len(s_cal), a.alpha, a.gamma))
    print('part                         windows  peak error  held: fixed  adaptive   median region: fixed  adaptive'
          '   whole-grid regions')
    for name, P, s, err in stream:
        q = np.empty(len(s)); hit = np.empty(len(s), bool)
        for t in range(len(s)):
            q[t] = aci.threshold(); hit[t] = aci.update(s[t])
        size_fixed = cp.region_size(P, q0, w)
        size_aci = np.concatenate([cp.region_size(P[t:t + 1], q[t], w) for t in range(len(s))])
        print('%-28s %7d  %7.2f deg  %8.1f %%  %6.1f %%  %14.2f %%  %6.2f %%  %8.1f %%' % (
            name, len(s), np.median(err), (s <= q0).mean() * 100, hit.mean() * 100, np.median(size_fixed),
            np.median(size_aci), np.isinf(q).mean() * 100))


if __name__ == '__main__':
    main()
