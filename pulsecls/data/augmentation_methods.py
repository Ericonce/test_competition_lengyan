"""PDW adaptations of time-series transformations and same-class pattern fusion.

All three CF/PW/PRI channels participate in transformations. CPU spectral/DTW execution on
Ascend avoids depending on optional complex-number NPU operators.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch.nn import functional as F
from pulsecls.data.noise import gaussian_noise


class MethodRunner:
    def channels(self, x):
        return list(range(x.size(1)))

    def values(self, x, transformed):
        result = x.clone()
        result[:, self.channels(x)] = transformed.to(x.device, x.dtype)
        return result

    @staticmethod
    def resize(x, length):
        return F.interpolate(x, size=length, mode="linear", align_corners=True)

    @staticmethod
    def sample_grid(x, positions):
        positions = positions.clamp(0, x.size(-1) - 1)
        left = positions.floor().long()
        right = (left + 1).clamp_max(x.size(-1) - 1)
        weight = (positions - left).unsqueeze(1)
        index = left.unsqueeze(1).expand(-1, x.size(1), -1)
        other = right.unsqueeze(1).expand_as(index)
        return x.gather(2, index) * (1 - weight) + x.gather(2, other) * weight

    def partner(self, x, labels):
        if labels is None:
            raise ValueError("This augmentation needs class labels; use augment.apply(x, labels)")
        indices = torch.arange(x.size(0), device=x.device)
        for label in labels.unique():
            group = torch.nonzero(labels == label, as_tuple=False).flatten()
            if group.numel() > 1:
                offset = int(torch.randint(1, group.numel(), ()).item())
                indices[group] = group.roll(offset)
        return indices

    def spectral_values(self, x):
        values = x[:, self.channels(x)].float()
        return values.cpu() if x.device.type == "npu" else values

    @staticmethod
    def rank_restore(reference, candidate):
        rank = candidate.argsort(dim=-1).argsort(dim=-1)
        return reference.sort(dim=-1).values.gather(-1, rank)

    def spectral(self, name, x, p, labels):
        v = self.spectral_values(x)
        n = v.size(-1)
        spectrum = torch.fft.rfft(v)
        bins = spectrum.size(-1)
        if name in ("frequency_warping", "vtlp"):
            ratio = max(.01, 1 + (float(torch.rand(())) * 2 - 1) * p["max_ratio"])
            grid = torch.linspace(0, 1, bins, device=v.device)
            if name == "vtlp":
                pivot = min(p["pivot_ratio"], .99)
                mapped_pivot = min(max(pivot * ratio, .01), .99)
                positions = torch.where(grid <= mapped_pivot, grid * pivot / mapped_pivot,
                                        pivot + (grid - mapped_pivot) * (1 - pivot) / (1 - mapped_pivot))
            else:
                positions = (grid / ratio).clamp(0, 1)
            positions = (positions * (bins - 1)).expand(v.size(0), -1)
            spectrum = torch.complex(self.sample_grid(spectrum.real, positions),
                                     self.sample_grid(spectrum.imag, positions))
        elif name == "frequency_slicing":
            width = min(bins - 1, int(round(bins * p["band_ratio"])))
            if width:
                start = int(torch.randint(1, bins - width + 1, ()).item())
                spectrum[..., start:start + width] = 0
        elif name == "sfcc":
            indices = self.partner(x, labels)
            second = self.partner(x, labels)
            donors = [spectrum, torch.fft.rfft(v[indices.to(v.device)]), torch.fft.rfft(v[second.to(v.device)])]
            combined = spectrum.clone()
            edges = np.linspace(1, bins, p["bands"] + 1, dtype=int)
            for a, b in zip(edges[:-1], edges[1:]):
                donor = donors[int(torch.randint(0, 3, ()).item())]
                combined[..., a:b] = donor[..., a:b]
            spectrum = (1 - p["blend"]) * spectrum + p["blend"] * combined
        elif name == "emda":
            indices = self.partner(x, labels).to(v.device)
            delay = int(round(n * p["max_delay_ratio"] * (float(torch.rand(())) * 2 - 1)))
            other = torch.fft.rfft(v[indices].roll(delay, dims=-1))
            gains = torch.exp(torch.randn_like(spectrum.real) * p["gain_std"])
            spectrum = (1 - p["blend"]) * spectrum + p["blend"] * other * gains
        else:
            raise ValueError(name)
        return self.values(x, torch.fft.irfft(spectrum, n=n))

    def surrogate(self, name, x, p):
        v = self.spectral_values(x)
        n = v.size(-1)
        gaussian = torch.randn_like(v).sort(dim=-1).values
        ranks = v.argsort(dim=-1).argsort(dim=-1)
        gaussianized = gaussian.gather(-1, ranks)
        spectrum = torch.fft.rfft(gaussianized)
        phases = torch.rand_like(spectrum.real) * 2 * math.pi
        phases[..., 0] = torch.angle(spectrum[..., 0])
        if n % 2 == 0:
            phases[..., -1] = torch.angle(spectrum[..., -1])
        candidate = torch.fft.irfft(torch.polar(spectrum.abs(), phases), n=n)
        candidate = self.rank_restore(v, candidate)
        if name == "iaaft":
            amplitude = torch.fft.rfft(v).abs()
            for _ in range(p["iterations"]):
                candidate = torch.fft.irfft(torch.polar(amplitude, torch.angle(torch.fft.rfft(candidate))), n=n)
                candidate = self.rank_restore(v, candidate)
        return self.values(x, candidate)

    def spectrogram(self, name, x, p):
        v = self.spectral_values(x)
        shape = v.shape
        n_fft = min(p["n_fft"], shape[-1])
        hop = max(1, n_fft // 4)
        window = torch.hann_window(n_fft, device=v.device)
        spec = torch.stft(v.reshape(-1, shape[-1]), n_fft=n_fft, hop_length=hop,
                          window=window, return_complex=True, pad_mode="constant")
        fw = min(spec.size(1), int(round(spec.size(1) * p["frequency_ratio"])))
        tw = min(spec.size(2), int(round(spec.size(2) * p["time_ratio"])))
        if fw and tw:
            f = int(torch.randint(0, spec.size(1) - fw + 1, ()).item())
            t = int(torch.randint(0, spec.size(2) - tw + 1, ()).item())
            if name == "stft_mask":
                spec[:, f:f + fw, t:t + tw] = 0
            else:
                region = spec[:, f:f + fw, t:t + tw]
                region += gaussian_noise(region, p["snr_db"], dimensions=(-2, -1))
        result = torch.istft(spec, n_fft=n_fft, hop_length=hop, window=window, length=shape[-1])
        return self.values(x, result.reshape(shape))

    @staticmethod
    def dtw(a, b, window_ratio, radius=0):
        def descriptors(v):
            padded = np.pad(v, ((radius, radius), (0, 0)), mode="edge")
            return np.stack([padded[i:i + 2 * radius + 1].reshape(-1) for i in range(len(v))])
        aa, bb = (descriptors(a), descriptors(b)) if radius else (a, b)
        n, m = len(a), len(b)
        window = max(abs(n - m), int(max(n, m) * window_ratio), 1)
        costs = np.full((n + 1, m + 1), np.inf)
        costs[0, 0] = 0
        for i in range(1, n + 1):
            for j in range(max(1, i - window), min(m, i + window) + 1):
                costs[i, j] = np.mean((aa[i - 1] - bb[j - 1]) ** 2) + min(costs[i - 1, j], costs[i, j - 1], costs[i - 1, j - 1])
        i, j, path = n, m, []
        while i > 0 and j > 0:
            path.append((i - 1, j - 1))
            step = int(np.argmin([costs[i - 1, j - 1], costs[i - 1, j], costs[i, j - 1]]))
            i -= int(step in (0, 1))
            j -= int(step in (0, 2))
        return np.asarray(path[::-1], dtype=int), float(costs[n, m])

    def pattern(self, name, x, p, labels):
        indices = self.partner(x, labels)
        n = min(x.size(-1), p["max_points"])
        small = self.resize(x.float(), n).detach().cpu().numpy().transpose(0, 2, 1)
        out = small.copy()
        label_array = labels.detach().cpu().numpy()
        idx = indices.detach().cpu().numpy()
        channels = self.channels(x)
        for row in range(len(small)):
            if idx[row] == row:
                continue
            source = small[row]
            guide = small[idx[row]]
            radius = p.get("shape_radius", 0)
            window = p["window_ratio"]
            if name.startswith("discriminative_guided"):
                candidates = np.flatnonzero(label_array == label_array[row])
                candidates = candidates[candidates != row][:p["candidates"]]
                negatives = np.flatnonzero(label_array != label_array[row])[:p["candidates"]]
                if len(negatives):
                    best, score = guide, -np.inf
                    for k in candidates:
                        _, positive = self.dtw(source[:, channels], small[k][:, channels], window, radius)
                        negative = min(self.dtw(small[k][:, channels], small[q][:, channels], window, radius)[1] for q in negatives)
                        if negative - positive > score:
                            best, score = small[k], negative - positive
                    guide = best
            if name == "spawner":
                split = max(1, min(n - 2, int(n * (.2 + .6 * float(torch.rand(()))))))
                left, _ = self.dtw(source[:split + 1, channels], guide[:split + 1, channels], window)
                right, _ = self.dtw(source[split:, channels], guide[split:, channels], window)
                path = np.concatenate([left, right[1:] + split])
            else:
                path, _ = self.dtw(source[:, channels], guide[:, channels], window, radius)
            if "guided_warp" in name:
                aligned = np.stack([source[path[path[:, 1] == j, 0]].mean(axis=0) for j in range(n)])
            elif name == "dba":
                aligned = source.copy()
                neighbors = np.flatnonzero(label_array == label_array[row])[:p["neighbors"]]
                for _ in range(p["iterations"]):
                    buckets = [[aligned[j]] for j in range(n)]
                    for k in neighbors:
                        pairs, _ = self.dtw(aligned[:, channels], small[k][:, channels], window)
                        for a, b in pairs:
                            buckets[a].append(small[k, b])
                    aligned = np.stack([np.mean(bucket, axis=0) for bucket in buckets])
            else:
                blend = p.get("blend", .5)
                aligned = (1 - blend) * source[path[:, 0]] + blend * guide[path[:, 1]]
            restored = self.resize(torch.from_numpy(aligned.T.copy()).float()[None], n)[0].numpy().T
            out[row] = restored
        result = self.resize(torch.from_numpy(out.transpose(0, 2, 1).copy()).to(x.device, x.dtype), x.size(-1))
        if name == "spawner":
            values = result[:, channels]
            result[:, channels] = values + gaussian_noise(values, p["snr_db"])
        singleton = indices == torch.arange(x.size(0), device=x.device)
        result = torch.where(singleton[:, None, None], x, result)
        return result

    def run(self, name, x, p, labels=None):
        channels = self.channels(x)
        v = x[:, channels]
        batch, _, length = x.shape
        if name == "jittering":
            valid = ~x.eq(0).all(dim=1, keepdim=True)
            return self.values(x, v + gaussian_noise(v, p["snr_db"], mask=valid))
        if name == "scaling":
            return self.values(x, v * (1 + torch.randn(batch, len(channels), 1, device=x.device, dtype=x.dtype) * p["std"]))
        if name == "rotation":
            if len(channels) < 2:
                return x.clone()
            angle = (torch.rand(batch, 1, device=x.device) * 2 - 1) * math.radians(p["max_angle_degrees"])
            result = v.clone()
            result[:, 0] = angle.cos() * v[:, 0] - angle.sin() * v[:, 1]
            result[:, 1] = angle.sin() * v[:, 0] + angle.cos() * v[:, 1]
            return self.values(x, result)
        if name == "magnitude_warping":
            curve = 1 + torch.randn(batch, len(channels), p["knots"] + 2, device=x.device, dtype=x.dtype) * p["std"]
            return self.values(x, v * self.resize(curve, length))
        if name == "white_noise_windows":
            result = v.clone()
            width = min(length, int(round(length * p["window_ratio"])))
            if width:
                start = int(torch.randint(0, length - width + 1, ()).item())
                region = result[..., start:start + width]
                valid = ~x[..., start:start + width].eq(0).all(dim=1, keepdim=True)
                region += gaussian_noise(region, p["snr_db"], mask=valid)
            return self.values(x, result)
        if name == "permutation":
            result = v.clone()
            for row in range(batch):
                pieces = torch.tensor_split(v[row], min(length, p["segments"]), dim=-1)
                order = torch.randperm(len(pieces)).tolist()
                result[row] = torch.cat([pieces[i] for i in order], dim=-1)
            return self.values(x, result)  # Apply the same permutation to all descriptor channels.
        if name in ("slicing", "window_slicing"):
            keep = min(length, max(2, int(round(length * p["keep_ratio"]))))
            start = int(torch.randint(0, length - keep + 1, ()).item())
            return self.resize(x[..., start:start + keep], length)
        if name == "time_warping":
            speeds = torch.exp(torch.randn(batch, 1, p["knots"] + 2, device=x.device) * p["std"])
            steps = self.resize(speeds, length)[:, 0]
            positions = steps.cumsum(-1)
            positions = positions - positions[:, :1]
            positions = positions / positions[:, -1:].clamp_min(1e-6) * (length - 1)
            return self.sample_grid(x, positions)
        if name == "window_warping":
            width = min(length, max(1, int(round(length * p["window_ratio"]))))
            start = int(torch.randint(0, length - width + 1, ()).item())
            scale = p["scales"][int(torch.randint(0, len(p["scales"]), ()).item())]
            middle = self.resize(x[..., start:start + width], max(1, int(round(width * scale))))
            return self.resize(torch.cat([x[..., :start], middle, x[..., start + width:]], -1), length)
        if name == "time_shift":
            maximum = int(round(length * p["ratio"]))
            shift = int(torch.randint(-maximum, maximum + 1, ()).item())
            return self.values(x, v.roll(shift, -1))
        if name == "time_flip":
            return x.flip(-1)
        if name in ("frequency_warping", "vtlp", "frequency_slicing", "sfcc", "emda"):
            return self.spectral(name, x, p, labels)
        if name in ("aaft", "iaaft"):
            return self.surrogate(name, x, p)
        if name in ("stft_mask", "random_noise_boxes"):
            return self.spectrogram(name, x, p)
        if name == "rim":
            result = v.clone()
            width = min(length - 2, int(round(length * p["mask_ratio"])))
            if width > 0:
                start = int(torch.randint(1, length - width, ()).item())
                left, right = v[..., start - 1:start], v[..., start + width:start + width + 1]
                alpha = torch.linspace(0, 1, width + 2, device=x.device, dtype=x.dtype)[1:-1]
                result[..., start:start + width] = left * (1 - alpha) + right * alpha
            return self.values(x, result)
        if name in ("interpolation", "statistical_style", "sample_combining"):
            partner = self.partner(x, labels)
            other = v[partner]
            if name == "interpolation":
                return (1 - p["blend"]) * x + p["blend"] * x[partner]
            if name == "statistical_style":
                styled = (v - v.mean(-1, keepdim=True)) / v.std(-1, keepdim=True, unbiased=False).clamp_min(1e-5)
                styled = styled * other.std(-1, keepdim=True, unbiased=False) + other.mean(-1, keepdim=True)
                return self.values(x, v * (1 - p["blend"]) + styled * p["blend"])
            device = "cpu" if x.device.type == "npu" else x.device
            signal = v.to(device).float()
            # Orthonormal DCT-II and its transpose implement the inverse DCT.
            k = torch.arange(length, device=device).view(-1, 1)
            t = torch.arange(length, device=device).view(1, -1)
            matrix = torch.cos(math.pi / length * (t + .5) * k) * math.sqrt(2 / length)
            matrix[0] /= math.sqrt(2)
            coefficients = signal @ matrix.T
            mixed = coefficients.clone()
            donor = coefficients[partner.to(device)]
            edges = np.linspace(1, length, p["bands"] + 1, dtype=int)
            for a, b in zip(edges[:-1], edges[1:]):
                if bool(torch.rand(()) < .5):
                    mixed[..., a:b] = donor[..., a:b]
            restored = ((1 - p["blend"]) * coefficients + p["blend"] * mixed) @ matrix
            return self.values(x, restored)
        if name in ("spawner", "random_guided_warp", "random_guided_warp_shape", "discriminative_guided_warp", "discriminative_guided_warp_shape", "dtw_mix", "dba"):
            return self.pattern(name, x, p, labels)
        if name in ("pulse_mask", "feature_mask"):
            shape = (batch, 1, length) if name == "pulse_mask" else (batch, len(channels), 1)
            mask = torch.rand(*shape, device=x.device) < p["ratio"]
            return self.values(x, v.masked_fill(mask, 0))
        raise ValueError(f"No implementation for {name}")
