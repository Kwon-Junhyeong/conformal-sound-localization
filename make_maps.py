"""Direction maps of a windows file (windows.py), the input of run_split.py and run_stream.py.
  python make_maps.py WINDOWS.npz MAPS.npz --table BEM.h5 [--panels 0 3 5 6] [--fstep 6] [--synthetic]
  python make_maps.py WINDOWS.npz MAPS.npz --free-field [--step 2] [--log-weight]
With --table the templates come from a BEM table (--synthetic for scenes synthesized from that table). With
--free-field they come from the mic_pos of the windows file, on a grid up to 80 degrees of elevation. Writes the maps
shifted to a zero minimum (float16), the grid, the true and the estimated direction indices, the point errors and the
labels of the windows."""
import argparse

import numpy as np

import estimator as est
import windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('windows')
    ap.add_argument('out')
    ap.add_argument('--table')
    ap.add_argument('--panels', type=int, nargs='+', default=[0, 3, 5, 6])
    ap.add_argument('--fstep', type=int, default=6)
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--free-field', action='store_true')
    ap.add_argument('--step', type=float, default=2.0)
    ap.add_argument('--log-weight', action='store_true')
    a = ap.parse_args()
    x, d, rec, extra = windows.load(a.windows)
    if a.table:
        table = est.load_bem_table(a.table, a.panels, a.fstep, a.synthetic)
    else:
        freqs = np.linspace(0, est.FS / 2, est.NFFT // 2 + 1).astype('float32')
        table = est.make_ff_steering(extra['mic_pos'], est.doa_grid(a.step), freqs)
    maps = est.direction_maps(x, table, a.log_weight)
    grid = table['doa_grid']
    true_idx = np.concatenate([np.argmax(d[i:i + 256] @ grid.T, 1) for i in range(0, len(d), 256)])
    est_idx = maps.argmax(1)
    err = est.angular_error(grid[est_idx], d)
    maps -= maps.min(1, keepdims=True)
    np.savez(a.out, maps=maps.astype(np.float16), grid=grid,
             true_idx=true_idx.astype(np.int32), est_idx=est_idx.astype(np.int32), err=err.astype(np.float32),
             rec=rec, **{k: v for k, v in extra.items() if k != 'mic_pos'})
    print('%s: %d windows, %d directions, median error %.2f deg, over 10 deg %.1f %%, over 30 deg %.1f %%' % (
        a.out, len(x), len(grid), np.median(err), (err > 10).mean() * 100, (err > 30).mean() * 100))


if __name__ == '__main__':
    main()
