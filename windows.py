"""One-second windows with their true directions, the input of make_maps.py. A windows file holds x (windows,
samples, 4) at 16 kHz, the true direction in the body frame as d (windows, 3) or as az and el in degrees, and a
recording label rec (fid in the box recordings). LOCATA windows also carry split, task and mic_pos.
  python windows.py scenes SCENE_DIR OUT.npz [--list FILES]
  python windows.py locata LOCATA_ROOT OUT.npz [--snr DB]
Scenes are the wav and npz pairs written by the synthesis of robot-body-sound-localization. LOCATA uses the robot
head (benchmark2), microphones 1, 4, 10 and 12, the single-talker tasks 1, 3 and 5, and windows where the talker is
active for at least half of the second. With --snr, white noise at that SNR is added to every eval window."""
import argparse
import glob
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

FS = 16000; WIN = 16000; HOP = 8000  # 1.0 s windows, 0.5 s hop
SUB = {'benchmark2': [0, 3, 9, 11]}


def load(path):
    z = np.load(path)
    x = z['x']
    x = x.astype(np.float32) / 32768.0 if x.dtype == np.int16 else x.astype(np.float32)
    x = np.ascontiguousarray(np.transpose(x, (0, 2, 1)))
    if 'd' in z.files:
        d = z['d'].astype(np.float32)
    else:
        az, el = np.radians(z['az'].astype(np.float64)), np.radians(z['el'].astype(np.float64))
        d = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], 1).astype(np.float32)
    rec = z['rec'] if 'rec' in z.files else z['fid']
    return x, d, rec, {k: z[k] for k in ('split', 'task', 'mic_pos') if k in z.files}


def scenes(folder, names=None):
    names = names or sorted(os.path.basename(f) for f in glob.glob(os.path.join(folder, '*.wav')))
    x, d = [], []
    for n in names:
        wav, fs = sf.read(os.path.join(folder, n))
        assert fs == FS, n
        x.append(wav.astype(np.float32)); d.append(np.load(os.path.join(folder, n[:-4] + '.npz'))['d_unit'])
    return dict(x=np.stack(x), d=np.stack(d).astype(np.float32), rec=np.array(names))


def read_pos(path, nmax=None):
    with open(path) as f:
        hdr = f.readline().strip().split('\t')
        rows = []
        for i, ln in enumerate(f):
            if nmax and i >= nmax: break
            ln = ln.strip()
            if ln: rows.append([float(x) for x in ln.split('\t')])
    a = np.array(rows); return {h: a[:, i] for i, h in enumerate(hdr)}


def abss(d, i=None):
    h, m, s = d['hour'], d['minute'], d['second']
    return (h * 3600 + m * 60 + s) if i is None else (h[i] * 3600 + m[i] * 60 + s[i])


def process(rec_dir, array='benchmark2', snr=None, rng=None):
    arr = Path(rec_dir) / array
    af = arr / f'audio_array_{array}.wav'
    if not af.exists(): return None
    audio, fs = sf.read(af)
    sub = SUB[array]
    a16 = resample_poly(audio[:, sub], FS, fs, axis=0).astype(np.float32)
    t0 = abss(read_pos(arr / f'audio_array_timestamps_{array}.txt', nmax=1), 0)
    pa = read_pos(arr / f'position_array_{array}.txt')
    center = np.stack([pa['x'], pa['y'], pa['z']], 1)
    T = center.shape[0]; R = np.empty((T, 3, 3))
    for r in range(3):
        for c in range(3): R[:, r, c] = pa[f'rotation_{r+1}{c+1}']
    treq = abss(pa)
    rt = read_pos(arr / 'required_time.txt'); valid = rt['valid_flag'].astype(bool)
    # microphones in the array frame, local = R^T (global - centre)
    locs = []
    for m in sub:
        mg = np.stack([pa[f'mic{m+1}_x'], pa[f'mic{m+1}_y'], pa[f'mic{m+1}_z']], 1)
        locs.append(np.einsum('tcr,tc->tr', R, mg - center).mean(0))
    mic_local = np.stack(locs).astype(np.float32)
    srcs = sorted(glob.glob(str(arr / 'position_source_*.txt')))
    assert len(srcs) == 1, srcs
    S = read_pos(srcs[0]); S = np.stack([S['x'], S['y'], S['z']], 1)
    nm = os.path.basename(srcs[0]).replace('position_source_', '').replace('.txt', '')
    vad = np.loadtxt(arr / f'VAD_source_{nm}.txt', skiprows=1).astype(np.float32)
    nlen = a16.shape[0]; wavs = []; dirs = []
    for c in range(WIN // 2, nlen - WIN // 2, HOP):
        tc = t0 + c / FS; r = int(np.argmin(np.abs(treq - tc)))
        if not valid[r]: continue
        c48 = int(round((tc - t0) * fs))
        v = float(vad[max(0, c48 - fs // 2):c48 + fs // 2].mean()) if len(vad) else 0.0
        if v < 0.5: continue
        rel = S[r] - center[r]; loc = R[r].T @ rel
        w = a16[c - WIN // 2:c + WIN // 2].copy()
        if snr is not None:
            ps = float((w ** 2).mean()); w = (w + rng.standard_normal(w.shape) * np.sqrt(ps / 10 ** (snr / 10))).astype(np.float32)
        wavs.append(w); dirs.append(loc / np.linalg.norm(loc))
    return wavs, dirs, mic_local


def locata(root, snr=None):
    rng = np.random.default_rng(20261117)
    out = dict(x=[], d=[], rec=[], split=[], task=[]); mic_pos = None
    for split in (['eval'] if snr is not None else ['dev', 'eval']):
        for task in ['task1', 'task3', 'task5']:
            for rd in sorted((Path(root) / split / task).glob('recording*')):
                r = process(rd, snr=snr, rng=rng)
                if r is None: continue
                wavs, dirs, mic_local = r
                assert mic_pos is None or np.allclose(mic_local, mic_pos, atol=1e-6)
                mic_pos = mic_local
                label = f'{split}/{task}/{rd.name}'
                out['x'] += wavs; out['d'] += dirs; out['rec'] += [label] * len(wavs)
                out['split'] += [split] * len(wavs); out['task'] += [task] * len(wavs)
                print(label, len(wavs), 'windows')
    return dict(x=np.stack(out['x']), d=np.stack(out['d']).astype(np.float32), rec=np.array(out['rec']),
                split=np.array(out['split']), task=np.array(out['task']), mic_pos=mic_pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('source', choices=['scenes', 'locata'])
    ap.add_argument('root')
    ap.add_argument('out')
    ap.add_argument('--list', help='text file with the scene wav names to use, one per line')
    ap.add_argument('--snr', type=float)
    a = ap.parse_args()
    if a.source == 'scenes':
        w = scenes(a.root, [l.strip() for l in open(a.list) if l.strip()] if a.list else None)
    else:
        w = locata(a.root, a.snr)
    np.savez(a.out, **w)
    print(a.out, len(w['x']), 'windows')


if __name__ == '__main__':
    main()
