"""Split conformal prediction on a maps file (make_maps.py). Half of the recordings calibrate and the other half are
tested, over many random splits. For every risk level alpha: the share of test windows whose region holds the true
direction, and the median region size. Then answer or abstain at one alpha: a test region wider than a quantile
(--keep) of the calibration regions is withheld. For every error bound: the wrong answers among those passed on (FAR),
the right answers withheld (MDR), and the FAR when as many windows are withheld at random.
  python run_split.py MAPS.npz [--alphas 0.2 0.1 0.05 0.01 0.005] [--splits 100] [--abstain-alpha 0.1]
                      [--keep 0.9 0.8 0.7 0.6 0.5] [--errors 5 10 30]"""
import argparse

import numpy as np

import conformal as cp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('maps')
    ap.add_argument('--alphas', type=float, nargs='+', default=[0.2, 0.1, 0.05, 0.01, 0.005])
    ap.add_argument('--splits', type=int, default=100)
    ap.add_argument('--abstain-alpha', type=float, default=0.1)
    ap.add_argument('--keep', type=float, nargs='+', default=[0.9, 0.8, 0.7, 0.6, 0.5])
    ap.add_argument('--errors', type=float, nargs='+', default=[5, 10, 30])
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    z = np.load(a.maps)
    P = cp.norm01(z['maps'].astype(np.float32)); s = cp.nonconformity(P, z['true_idx']); err = z['err']; rec = z['rec']
    w = cp.solid_angle_weights(z['grid'])
    recs = np.unique(rec); rng = np.random.default_rng(a.seed); rng_random = np.random.default_rng(a.seed + 1)
    print('%d windows from %d recordings, point error median %.2f deg' % (len(s), len(recs), np.median(err)) +
          ''.join(', over %g deg %.1f %%' % (t, (err > t).mean() * 100) for t in a.errors))
    cov = {al: [] for al in a.alphas}; size = {al: [] for al in a.alphas}
    ab = {(t, k): dict(answered=[], far=[], mdr=[], random=[]) for t in a.errors for k in a.keep}
    for _ in range(a.splits):
        cal = np.isin(rec, rng.permutation(recs)[:len(recs) // 2]); te = ~cal
        Pt = P[te]
        for al in a.alphas:
            q = cp.conformal_tau(s[cal], al)
            cov[al].append((s[te] <= q).mean() * 100); size[al].append(np.median(cp.region_size(Pt, q, w)))
        q = cp.conformal_tau(s[cal], a.abstain_alpha)
        size_te, size_cal = cp.region_size(Pt, q, w), cp.region_size(P[cal], q, w); n = int(te.sum())
        for k in a.keep:
            ok = cp.answer(size_te, np.quantile(size_cal, k))
            for t in a.errors:
                big = err[te] > t; r = ab[(t, k)]
                r['answered'].append(ok.mean() * 100); r['far'].append(big[ok].mean() * 100)
                r['mdr'].append((~big & ~ok).sum() / max((~big).sum(), 1) * 100)
                r['random'].append(np.mean([big[rng_random.permutation(n)[:ok.sum()]].mean() for _ in range(200)]) * 100)
    print('alpha  target  held the true direction (mean, min, max)  median region size (% of the sphere)')
    for al in a.alphas:
        print('%-6g %5.1f %%  %5.1f %%  (%.1f to %.1f)  %8.3f %%' % (al, (1 - al) * 100, np.mean(cov[al]), np.min(cov[al]),
                                                               np.max(cov[al]), np.mean(size[al])))
    print('answer or abstain at alpha %g' % a.abstain_alpha)
    for t in a.errors:
        print('errors over %g deg: %.1f %% of all windows' % (t, (err > t).mean() * 100))
        print('  keep   answered   FAR      MDR      FAR at random')
        for k in a.keep:
            r = {m: np.mean(v) for m, v in ab[(t, k)].items()}
            print('  %-5g  %5.1f %%   %5.1f %%  %5.1f %%  %5.1f %%' % (k, r['answered'], r['far'], r['mdr'], r['random']))


if __name__ == '__main__':
    main()
