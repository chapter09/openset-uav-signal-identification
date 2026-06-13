from __future__ import annotations

import math

import numpy as np

from .types import EmitterProfile


EPS = 1e-12


def db_to_linear(value_db: float) -> float:
    return float(10 ** (value_db / 10.0))


def normalize_power(x: np.ndarray) -> np.ndarray:
    power = float(np.mean(np.abs(x) ** 2))
    if power < EPS:
        return x.astype(np.complex64)
    return (x / math.sqrt(power)).astype(np.complex64)


def complex_noise(rng: np.random.Generator, n: int, power: float = 1.0) -> np.ndarray:
    scale = math.sqrt(power / 2.0)
    real = rng.normal(0.0, scale, n)
    imag = rng.normal(0.0, scale, n)
    return (real + 1j * imag).astype(np.complex64)


def qpsk_waveform(
    rng: np.random.Generator,
    n: int,
    sample_rate_hz: float,
    symbol_rate_hz: float,
) -> np.ndarray:
    samples_per_symbol = max(2, int(round(sample_rate_hz / symbol_rate_hz)))
    symbol_count = int(math.ceil(n / samples_per_symbol)) + 2
    constellation = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=np.complex64)
    symbols = constellation[rng.integers(0, len(constellation), symbol_count)] / math.sqrt(2.0)
    x = np.repeat(symbols, samples_per_symbol)[: n + samples_per_symbol]
    pulse = np.hanning(samples_per_symbol * 2 + 1)
    pulse = pulse / np.sum(pulse)
    shaped = np.convolve(x, pulse, mode="same")[:n]
    return normalize_power(shaped)


def fsk_waveform(
    rng: np.random.Generator,
    n: int,
    sample_rate_hz: float,
    symbol_rate_hz: float,
    bandwidth_hz: float,
) -> np.ndarray:
    samples_per_symbol = max(2, int(round(sample_rate_hz / symbol_rate_hz)))
    symbol_count = int(math.ceil(n / samples_per_symbol)) + 2
    tones = np.array([-0.35, -0.12, 0.12, 0.35]) * bandwidth_hz
    freq = np.repeat(tones[rng.integers(0, len(tones), symbol_count)], samples_per_symbol)[:n]
    phase = np.cumsum(2.0 * np.pi * freq / sample_rate_hz)
    return np.exp(1j * phase).astype(np.complex64)


def chirp_waveform(
    rng: np.random.Generator,
    n: int,
    sample_rate_hz: float,
    bandwidth_hz: float,
) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / sample_rate_hz
    sweep = rng.choice([-1.0, 1.0])
    start_hz = -0.45 * bandwidth_hz * sweep
    stop_hz = 0.45 * bandwidth_hz * sweep
    duration_s = max(n / sample_rate_hz, EPS)
    k = (stop_hz - start_hz) / duration_s
    phase = 2.0 * np.pi * (start_hz * t + 0.5 * k * t * t)
    return np.exp(1j * phase).astype(np.complex64)


def ofdm_like_waveform(
    rng: np.random.Generator,
    n: int,
    sample_rate_hz: float,
    bandwidth_hz: float,
    subcarriers: int = 48,
) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / sample_rate_hz
    freqs = np.linspace(-0.45 * bandwidth_hz, 0.45 * bandwidth_hz, subcarriers)
    phases = rng.uniform(0.0, 2.0 * np.pi, subcarriers)
    weights = rng.choice(np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]), subcarriers)
    x = np.zeros(n, dtype=np.complex128)
    for freq, phase, weight in zip(freqs, phases, weights):
        x += weight * np.exp(1j * (2.0 * np.pi * freq * t + phase))
    slow_fade = 0.85 + 0.15 * np.sin(2.0 * np.pi * rng.uniform(200.0, 1200.0) * t)
    return normalize_power(x * slow_fade)


def burst_envelope(
    rng: np.random.Generator,
    n: int,
    duty_cycle: float,
    min_ramp_fraction: float = 0.02,
) -> np.ndarray:
    duty = float(np.clip(duty_cycle + rng.normal(0.0, 0.04), 0.08, 1.0))
    if duty >= 0.98:
        return np.ones(n, dtype=np.float32)

    active = max(8, min(n, int(round(n * duty))))
    start = int(rng.integers(0, max(1, n - active + 1)))
    envelope = np.zeros(n, dtype=np.float32)
    envelope[start : start + active] = 1.0

    ramp = max(2, int(round(active * min_ramp_fraction)))
    ramp = min(ramp, active // 2)
    if ramp > 1:
        window = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, ramp))
        envelope[start : start + ramp] *= window
        envelope[start + active - ramp : start + active] *= window[::-1]
    return envelope


def apply_frequency_offset(x: np.ndarray, sample_rate_hz: float, offset_hz: float) -> np.ndarray:
    if abs(offset_hz) < EPS:
        return x.astype(np.complex64)
    t = np.arange(len(x), dtype=np.float64) / sample_rate_hz
    return (x * np.exp(1j * 2.0 * np.pi * offset_hz * t)).astype(np.complex64)


def apply_phase_noise(
    rng: np.random.Generator,
    x: np.ndarray,
    phase_noise_std: float,
    phase_bias_rad: float,
) -> np.ndarray:
    if phase_noise_std <= 0.0 and abs(phase_bias_rad) < EPS:
        return x.astype(np.complex64)
    noise = np.cumsum(rng.normal(0.0, phase_noise_std, len(x)))
    return (x * np.exp(1j * (phase_bias_rad + noise))).astype(np.complex64)


def apply_iq_imbalance(
    x: np.ndarray,
    gain_imbalance: float,
    phase_imbalance_rad: float,
) -> np.ndarray:
    i = np.real(x) * (1.0 + gain_imbalance)
    q = np.imag(x) * (1.0 - gain_imbalance)
    q_rot = q * math.cos(phase_imbalance_rad) + i * math.sin(phase_imbalance_rad)
    return (i + 1j * q_rot).astype(np.complex64)


def apply_multipath(
    rng: np.random.Generator,
    x: np.ndarray,
    max_delay: int = 12,
    tap_count: int = 3,
) -> np.ndarray:
    taps = np.zeros(max_delay + 1, dtype=np.complex64)
    taps[0] = 1.0 + 0.0j
    for _ in range(tap_count - 1):
        delay = int(rng.integers(1, max_delay + 1))
        magnitude = float(rng.uniform(0.05, 0.28))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        taps[delay] += magnitude * np.exp(1j * phase)
    y = np.convolve(x, taps, mode="full")[: len(x)]
    return normalize_power(y)


def synthesize_clean_waveform(
    rng: np.random.Generator,
    profile: EmitterProfile,
    n: int,
    sample_rate_hz: float,
) -> np.ndarray:
    waveform = profile.waveform.lower()
    if waveform in {"qpsk", "controller_qpsk"}:
        x = qpsk_waveform(rng, n, sample_rate_hz, profile.symbol_rate_hz)
    elif waveform in {"fsk", "telemetry_fsk"}:
        x = fsk_waveform(rng, n, sample_rate_hz, profile.symbol_rate_hz, profile.bandwidth_hz)
    elif waveform in {"chirp", "sweep"}:
        x = chirp_waveform(rng, n, sample_rate_hz, profile.bandwidth_hz)
    elif waveform in {"ofdm", "wifi_like", "uav_video"}:
        x = ofdm_like_waveform(rng, n, sample_rate_hz, profile.bandwidth_hz)
    else:
        raise ValueError(f"Unsupported waveform: {profile.waveform}")

    envelope = burst_envelope(rng, n, profile.burst_duty_cycle)
    return normalize_power(x * envelope)


def apply_emitter_impairments(
    rng: np.random.Generator,
    x: np.ndarray,
    profile: EmitterProfile,
    sample_rate_hz: float,
) -> np.ndarray:
    offset = profile.frequency_offset_hz + float(rng.normal(0.0, max(5.0, abs(profile.frequency_offset_hz) * 0.08)))
    y = apply_frequency_offset(x, sample_rate_hz, offset)
    y = apply_phase_noise(rng, y, profile.phase_noise_std, profile.phase_bias_rad)
    y = apply_iq_imbalance(y, profile.iq_gain_imbalance, profile.iq_phase_imbalance_rad)
    y = apply_multipath(rng, y)
    return normalize_power(y)

