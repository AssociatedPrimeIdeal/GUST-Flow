from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.collections import PatchCollection
from matplotlib.patches import Ellipse
import numpy as np
import torch

def gaussian_slice_sections(centers, scales, rotations, shape, slice_index, sigma=1.0):
    centers = np.asarray(centers, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    factor = (np.asarray(shape, dtype=np.float64) - 1) / 2
    mu = (centers + 1) * factor
    transform = factor[None, :, None] * rotations * scales[:, None, :]
    cov = transform @ transform.transpose(0, 2, 1)
    zz = cov[:, 2, 2]
    dz = float(slice_index) - mu[:, 2]
    remaining = sigma ** 2 - dz ** 2 / np.maximum(zz, np.finfo(float).tiny)
    ids = np.flatnonzero((zz > 0) & np.isfinite(remaining) & (remaining > 0))
    if not len(ids):
        return (ids, np.empty((0, 2)), np.empty((0, 2)), np.empty(0))
    cov = cov[ids]
    cross = cov[:, :2, 2]
    xy = mu[ids, :2] + cross * (dz[ids] / zz[ids])[:, None]
    conditional = cov[:, :2, :2] - cross[:, :, None] * cross[:, None, :] / zz[ids, None, None]
    section = conditional[:, ::-1, ::-1] * remaining[ids, None, None]
    values, vectors = np.linalg.eigh(section)
    valid = np.all(values > 0, axis=1)
    ids, xy, values, vectors = (ids[valid], xy[valid], values[valid], vectors[valid])
    diameters = 2 * np.sqrt(values[:, ::-1])
    major = vectors[:, :, 1]
    angle = np.degrees(np.arctan2(major[:, 1], major[:, 0]))
    return (ids, xy[:, ::-1], diameters, angle)

def gaussian_plane_projections(centers, scales, rotations, shape, sigma=1.0):
    centers = np.asarray(centers, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    factor = (np.asarray(shape, dtype=np.float64) - 1) / 2
    mu = (centers + 1) * factor
    transform = factor[None, :, None] * rotations * scales[:, None, :]
    cov = transform @ transform.transpose(0, 2, 1)
    values, vectors = np.linalg.eigh(cov[:, :2, :2][:, ::-1, ::-1])
    valid = np.all(values > 0, axis=1)
    values = values[valid]
    vectors = vectors[valid]
    xy = mu[valid, :2][:, ::-1]
    diameters = 2 * sigma * np.sqrt(values[:, ::-1])
    major = vectors[:, :, 1]
    angle = np.degrees(np.arctan2(major[:, 1], major[:, 0]))
    return (xy, diameters, angle)

class SliceTrainingGif:

    def __init__(self, channel=0, frame=3, slice_index=None, ellipse_opacity=0.18, ellipse_sigma=1.0, show_all_projected=True, projection_opacity=0.025):
        self.channel = channel
        self.frame = frame
        self.slice_index = slice_index
        self.frames = []
        if not 0 <= ellipse_opacity <= 1 or not 0 <= projection_opacity <= 1 or ellipse_sigma <= 0:
            raise ValueError('Invalid ellipse opacity or sigma radius')
        self.ellipse_opacity = ellipse_opacity
        self.ellipse_sigma = ellipse_sigma
        self.show_all_projected = bool(show_all_projected)
        self.projection_opacity = projection_opacity

    def __call__(self, iteration, recovered, history):
        c, t, z = (self.channel, self.frame, self.slice_index)
        kernel = history['kernel_snapshots'][-1]
        if kernel['iteration'] != iteration:
            raise ValueError('Velocity and Gaussian snapshots must be from the same iteration')
        self.frames.append({'iteration': int(iteration), 'velocity': recovered[c, t, :, :, z].detach().cpu().numpy().copy(), 'centers': kernel['centers'].numpy().copy(), 'scales': kernel['scales'].numpy().copy(), 'rotations': kernel['rotations'].numpy().copy()})

    def save(self, result, *, wrapped_phase, gt_velocity, venc, segmask=None, spacing=(1.0, 1.0, 1.0), path='gustflow_training.gif', fps=8):
        if not self.frames:
            raise ValueError('Pass this recorder as fit(callback=..., eval_every=25) first')
        c, t, z = (self.channel, self.frame, self.slice_index)
        shape = np.asarray(wrapped_phase.shape[-3:])
        final = result.recovered[c, t, :, :, z].detach().cpu().numpy()
        np.testing.assert_array_equal(self.frames[-1]['velocity'], final)
        final_centers = torch.tanh(result.model.center_raw[:result.model.active_primitives]).detach().cpu().numpy()
        np.testing.assert_allclose(self.frames[-1]['centers'], final_centers, atol=1e-06)
        if self.frames[-1]['iteration'] != result.history['iterations_completed']:
            raise ValueError('The final GIF frame must be the returned final iteration')
        wrapped = np.asarray(wrapped_phase)[c, t, :, :, z]
        gt = None if gt_velocity is None else np.asarray(gt_velocity)[c, t, :, :, z]
        mask = np.ones(tuple(shape[:2]), bool) if segmask is None else np.asarray(segmask).reshape(tuple(shape))[:, :, z] > 0
        limit = max(float(np.nanpercentile(np.abs(gt), 99.5)) if gt is not None else float(np.max(np.abs(final))), float(np.asarray(venc).reshape(-1)[c]), 1.0)
        error_limit = max(0.1 * limit, 1.0)
        aspect = float(spacing[0]) / float(spacing[1])
        fig, axes = plt.subplots(1, 5, figsize=(20, 4.3))
        common = dict(origin='lower', aspect=aspect, interpolation='nearest')
        axes[0].imshow(wrapped, cmap='gray', vmin=-np.pi, vmax=np.pi, **common)
        axes[0].set_title('Wrapped phase (rad)')
        if gt is not None:
            axes[1].imshow(gt, cmap='gray', vmin=-limit, vmax=limit, **common)
        axes[1].set_title('GT velocity' if gt is not None else 'GT unavailable')
        prediction = axes[2].imshow(final, cmap='gray', vmin=-limit, vmax=limit, **common)
        axes[2].set_title('GUST-Flow velocity')
        error = axes[3].imshow(np.zeros_like(final), cmap='coolwarm', vmin=-error_limit, vmax=error_limit, **common)
        axes[3].set_title('Error (evaluation mask)' if segmask is not None else 'Error')
        gaussian_background = axes[4].imshow(final, cmap='gray', vmin=-limit, vmax=limit, **common)
        ellipses = PatchCollection([], facecolor='#ff922b', edgecolor='#ff922b', linewidths=0.55, alpha=self.ellipse_opacity, zorder=3)
        projected = PatchCollection([], facecolor='#ff922b', edgecolor='none', linewidths=0, alpha=self.projection_opacity, zorder=2, rasterized=True)
        if self.show_all_projected:
            axes[4].add_collection(projected)
        axes[4].add_collection(ellipses)
        for axis in axes:
            axis.set_xlim(-0.5, shape[1] - 0.5)
            axis.set_ylim(-0.5, shape[0] - 0.5)
            axis.axis('off')
        fig.colorbar(prediction, ax=axes[1:3], fraction=0.02, pad=0.02, label='cm/s')
        fig.colorbar(error, ax=axes[3], fraction=0.046, pad=0.03, label='cm/s')
        title = fig.suptitle('')
        fig.subplots_adjust(left=0.01, right=0.96, bottom=0.05, top=0.78, wspace=0.28)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = PillowWriter(fps=fps, metadata={'artist': 'GUST-Flow'})
        try:
            with writer.saving(fig, str(path), dpi=110):
                for snapshot in self.frames:
                    ids, xy, diameters, angle = gaussian_slice_sections(snapshot['centers'], snapshot['scales'], snapshot['rotations'], shape, z, self.ellipse_sigma)
                    patches = [Ellipse(center, width=size[0], height=size[1], angle=degrees) for center, size, degrees in zip(xy, diameters, angle)]
                    if self.show_all_projected:
                        pxy, pdiameters, pangle = gaussian_plane_projections(snapshot['centers'], snapshot['scales'], snapshot['rotations'], shape, self.ellipse_sigma)
                        projected.set_paths([Ellipse(center, width=size[0], height=size[1], angle=degrees) for center, size, degrees in zip(pxy, pdiameters, pangle)])
                    ellipses.set_paths(patches)
                    gaussian_background.set_data(snapshot['velocity'])
                    suffix = f"{len(snapshot['centers'])} projected + {len(ids)} exact sections" if self.show_all_projected else f'{len(ids)} exact sections'
                    axes[4].set_title(f'GUST velocity + {self.ellipse_sigma:g}σ Gaussian geometry\n{suffix} | slice {z}', fontsize=9)
                    prediction.set_data(snapshot['velocity'])
                    error.set_data(np.where(mask, snapshot['velocity'] - gt, np.nan) if gt is not None else np.full_like(final, np.nan))
                    title.set_text(f"GUST-Flow | iteration {snapshot['iteration']} | encoding {c}, frame {t}, slice {z}")
                    writer.grab_frame()
        finally:
            plt.close(fig)
        return path
