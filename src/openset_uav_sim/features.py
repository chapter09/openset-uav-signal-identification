from __future__ import annotations

import numpy as np


EPS = 1e-12


def extract_features(iq: np.ndarray) -> np.ndarray:
    """Extract compact RF features for prototype and energy baselines."""

    x = np.asarray(iq, dtype=np.complex64)
    if x.ndim != 1:
        raise ValueError("iq must be a one-dimensional complex array.")
    if len(x) < 8:
        raise ValueError("iq must contain at least 8 samples.")

    magnitude = np.abs(x)
    power = magnitude**2
    mean_power = float(np.mean(power) + EPS)
    std_power = float(np.std(power))
    peak_to_avg = float(np.max(power) / mean_power)
    envelope_cv = float(np.std(magnitude) / (np.mean(magnitude) + EPS))
    burstiness = float(np.mean(power > (mean_power * 1.8)))

    real = np.real(x)
    imag = np.imag(x)
    real_var = float(np.var(real) + EPS)
    imag_var = float(np.var(imag) + EPS)
    iq_var_ratio = float((real_var - imag_var) / (real_var + imag_var))
    iq_corr = float(np.mean((real - np.mean(real)) * (imag - np.mean(imag))) / np.sqrt(real_var * imag_var))

    phase_step = np.angle(x[1:] * np.conj(x[:-1]))
    phase_step_mean = float(np.mean(phase_step))
    phase_step_std = float(np.std(phase_step))
    lag1 = float(abs(np.mean(x[1:] * np.conj(x[:-1]))) / mean_power)

    window = np.hanning(len(x)).astype(np.float32)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(x * window))) ** 2
    spectrum = spectrum.astype(np.float64) + EPS
    spectrum_prob = spectrum / np.sum(spectrum)
    freq = np.linspace(-0.5, 0.5, len(x), endpoint=False)
    spectral_centroid = float(np.sum(freq * spectrum_prob))
    spectral_spread = float(np.sqrt(np.sum(((freq - spectral_centroid) ** 2) * spectrum_prob)))
    spectral_entropy = float(-np.sum(spectrum_prob * np.log(spectrum_prob)) / np.log(len(spectrum_prob)))
    spectral_flatness = float(np.exp(np.mean(np.log(spectrum))) / (np.mean(spectrum) + EPS))
    occupied_fraction = float(np.mean(spectrum > (np.max(spectrum) * 0.01)))

    features = np.array(
        [
            np.log10(mean_power),
            std_power / mean_power,
            peak_to_avg,
            envelope_cv,
            burstiness,
            iq_var_ratio,
            iq_corr,
            phase_step_mean,
            phase_step_std,
            lag1,
            spectral_centroid,
            spectral_spread,
            spectral_entropy,
            spectral_flatness,
            occupied_fraction,
        ],
        dtype=np.float64,
    )
    return np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)


def featurize_iq_batch(iq_batch: np.ndarray) -> np.ndarray:
    return np.vstack([extract_features(iq) for iq in iq_batch])

