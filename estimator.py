"""Direction maps for four-microphone windows. The cross-spectra of the six microphone pairs are summed over the frames
of a window, reduced to their phase, and matched against the pair cross-spectra of every direction on a grid. The
templates come from a BEM table of the body or, in the free field, from the microphone positions. Nothing is trained."""
import h5py
import numpy as np
import torch

FS = 16000
NFFT = 512
HOP = 256
DEVICE = torch.device('cpu')


def load_bem_table(fp, panels, fstep=6, synthetic=False):
    """BEM h5 (p: panel x direction x 257 bins up to 8 kHz, doa_grid) -> unit cross-spectra of the microphone pairs.
    The table is in the e^{+jwt} convention. Scenes synthesized from the table share it; recordings need the conjugate."""
    panels_arr = np.asarray(panels, dtype=np.int64)
    sort_idx = np.argsort(panels_arr)
    panels_sorted = panels_arr[sort_idx].tolist()
    inv_sort = np.argsort(sort_idx)
    with h5py.File(fp, 'r') as f:
        p_sorted = f['p'][panels_sorted, :, ::fstep].astype(np.complex64)
        p = p_sorted[inv_sort]
        doa_grid = f['doa_grid'][:].astype(np.float32)

    N, G, Fbin = p.shape
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    P = len(pairs)
    pair_flat = np.empty((G, P * Fbin), dtype=np.complex64)
    for k, (i, j) in enumerate(pairs):
        cc = p[i] * np.conj(p[j])         # (G, F)
        cc /= (np.abs(cc) + 1e-12)
        pair_flat[:, k * Fbin:(k + 1) * Fbin] = cc
    if not synthetic:
        pair_flat = np.conj(pair_flat)
    return {
        'pair_flat_cpu': torch.from_numpy(pair_flat),
        'doa_grid': doa_grid,
        'bins': np.arange(0, NFFT // 2 + 1, fstep),
        'pairs': pairs,
        'G': G, 'P': P, 'F': Fbin,
    }


def make_ff_steering(panel_pos, doa_grid, freqs, c_sound=343.0):
    """Free-field plane-wave templates exp(-jk g.(r_i - r_j)) for microphone positions panel_pos (N, 3) in metres."""
    N = panel_pos.shape[0]
    G = doa_grid.shape[0]
    F = freqs.shape[0]
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    Pn = len(pairs)
    deltas = np.stack([panel_pos[i] - panel_pos[j] for (i, j) in pairs], axis=0)
    gd = doa_grid @ deltas.T   # (G, Pn)
    k = 2 * np.pi * freqs / c_sound                   # (F,)
    phase = np.einsum('gp,f->gpf', gd, k)             # (G, Pn, F)
    pair_unit = np.exp(-1j * phase).astype(np.complex64)
    pair_flat = pair_unit.reshape(G, Pn * F)
    return {
        'pair_flat_cpu': torch.from_numpy(pair_flat),
        'doa_grid': doa_grid.astype(np.float32),
        'bins': np.arange(F),
        'pairs': pairs,
        'G': G, 'P': Pn, 'F': F,
    }


def doa_grid(step=2.0):
    az = np.deg2rad(np.arange(0, 360, step)); el = np.deg2rad(np.arange(-80, 81, step))
    AZ, EL = np.meshgrid(az, el, indexing='ij')
    return np.stack([np.cos(EL) * np.cos(AZ), np.cos(EL) * np.sin(AZ), np.sin(EL)], -1).reshape(-1, 3).astype('float32')


def stft_window(device):
    return torch.hann_window(NFFT, device=device, dtype=torch.float32)


def wav_to_stft(wav, win):
    """wav: (B, N_ch=4, T_samples) -> X: (B, N_ch, F, T_frame) complex64."""
    B, C, Ns = wav.shape
    flat = wav.reshape(B * C, Ns)
    X = torch.stft(flat, n_fft=NFFT, hop_length=HOP, win_length=NFFT,
                   window=win, center=False, return_complex=True)
    Fbin, Tf = X.shape[-2], X.shape[-1]
    return X.reshape(B, C, Fbin, Tf)


def stft_to_obs(X, pairs):
    """X (B, C, F, T) -> obs (B, P, F) complex, summed over the frames."""
    B, C, Fbin, Tf = X.shape
    P = len(pairs)
    obs = torch.zeros(B, P, Fbin, dtype=torch.complex64, device=X.device)
    for k, (i, j) in enumerate(pairs):
        obs[:, k] = (X[:, i] * X[:, j].conj()).sum(dim=-1)
    return obs


def normalize_obs(obs):
    return obs / (obs.abs() + 1e-12)


def match_scores(obs_unit, bem, g_chunk=8192):
    """obs_unit (B, P, F) -> scores (B, G): sum over pairs and bins of Re{obs * template}, largest at the source."""
    device = obs_unit.device
    B = obs_unit.shape[0]
    PF = obs_unit.shape[1] * obs_unit.shape[2]
    G = bem['G']
    obs_flat = obs_unit.reshape(B, PF)
    flat = bem['pair_flat_cpu']                       # (G, PF) CPU
    scores = torch.empty(B, G, dtype=torch.float32, device=device)
    for s in range(0, G, g_chunk):
        e = min(s + g_chunk, G)
        flat_chunk = flat[s:e].to(device, non_blocking=True)         # (Gc, PF)
        sc = obs_flat @ flat_chunk.T                                  # (B, Gc) complex
        scores[:, s:e] = sc.real
        del flat_chunk
    return scores


def direction_maps(wav, table, log_weight=False, batch=64):
    """wav (windows, 4, samples) -> scores (windows, directions). With log_weight each pair is weighted by
    log(1 + sum over frames of |X_i| |X_j|), as in SRP-PHAT."""
    win = stft_window(DEVICE)
    out = []
    with torch.no_grad():
        for b0 in range(0, len(wav), batch):
            wt = torch.from_numpy(np.ascontiguousarray(wav[b0:b0 + batch], np.float32)).to(DEVICE)
            X = wav_to_stft(wt, win)[:, :, table['bins']]
            obs = stft_to_obs(X, table['pairs'])
            if log_weight:
                mag = torch.stack([(X[:, i].abs() * X[:, j].abs()).sum(-1) for (i, j) in table['pairs']], 1)
                obs_norm = normalize_obs(obs) * torch.log1p(mag)
            else:
                obs_norm = normalize_obs(obs)
            out.append(match_scores(obs_norm, table).cpu().numpy())
    return np.concatenate(out)


def angular_error(d_pred, d_true):
    return np.degrees(np.arccos(np.clip((d_pred * d_true).sum(-1), -1, 1)))
