import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


@dataclass
class RenderResult:
    views: Dict[str, np.ndarray]

    @property
    def top(self):
        return self.views.get("top")

    @property
    def side(self):
        return self.views.get("side")

    @property
    def iso(self):
        return self.views.get("iso")

    def __getitem__(self, key: str):
        return self.views[key]


def _split_views(view: str) -> List[str]:
    v = (view or "all").strip().lower()
    if v == "all":
        return ["top", "side", "iso"]
    return [x.strip() for x in v.split(",") if x.strip()]


def _load_images(stat_path: str, template_path: Optional[str], interpolation: str):
    import nibabel as nib
    from nilearn import datasets, image

    stat_img = nib.load(stat_path)
    if template_path:
        template_img = nib.load(template_path)
    else:
        template_img = datasets.load_mni152_template()

    # Resample stat to template and canonicalize to reduce affine rotations
    stat_img = image.resample_to_img(stat_img, template_img, interpolation=interpolation)
    stat_img = nib.as_closest_canonical(stat_img)
    template_img = nib.as_closest_canonical(template_img)
    return stat_img, template_img


def _to_uniform_grid(img):
    import pyvista as pv

    data = np.asarray(img.get_fdata(), dtype=np.float32)
    affine = img.affine
    zooms = img.header.get_zooms()[:3]

    # Use translation only; data are canonicalized (RAS-ish)
    origin = tuple(affine[:3, 3].tolist())
    dims = data.shape

    # Use ImageData for broader pyvista compatibility (UniformGrid may be missing)
    grid = pv.ImageData()
    grid.dimensions = dims
    grid.spacing = zooms
    grid.origin = origin
    grid.point_data["values"] = data.ravel(order="F")
    return grid, data


def _voxel_world_points(img, ijk: np.ndarray) -> np.ndarray:
    import nibabel as nib

    return np.asarray(nib.affines.apply_affine(img.affine, ijk), dtype=np.float32)


def _infer_threshold_and_mode(
    stat_path: str,
    stat_data: np.ndarray,
    thr: Optional[float],
    thr_percentile: Optional[float],
    mode: str,
    p_thr: float,
    corrp_thr: float,
):
    if thr is not None:
        return float(thr), mode

    finite = stat_data[np.isfinite(stat_data)]
    if finite.size == 0:
        return 0.0, mode

    vmin = float(finite.min())
    vmax = float(finite.max())
    stat_name = Path(stat_path).name.lower()

    # Randomise/TBSS corrp maps encode significance as 1-p, so higher values
    # are more significant. Use the filename when available because value-only
    # heuristics are unreliable for dense TFCE corrp images.
    if mode == "auto" and "corrp" in stat_name:
        return float(corrp_thr), "gt"

    # Heuristic for TBSS corrp maps (1-p):
    # values in [0,1] with a sparse high tail (e.g., many zeros, few ~0.95).
    if mode == "auto" and vmin >= 0.0 and vmax <= 1.5:
        p95 = float(np.percentile(finite, 95.0))
        p99 = float(np.percentile(finite, 99.0))
        frac_hi = float(np.mean(finite >= corrp_thr))
        if vmax >= 0.9 and (p99 >= 0.9 or p95 >= 0.8 or frac_hi < 0.1):
            return float(corrp_thr), "gt"
        if p99 == 0.0 and vmax > 0.0:
            # sparse nonzero maps (e.g., filled clusters) -> keep any nonzero voxel
            return 0.0, "gt"
        # otherwise assume p-map (low is significant)
        return float(p_thr), "lt"

    if thr_percentile is None:
        thr_percentile = 95.0
    return float(np.percentile(finite, thr_percentile)), mode


def _mask_from_threshold(stat_data: np.ndarray, thr: float, mode: str):
    mode = (mode or "auto").lower()
    if mode == "lt":
        return stat_data <= thr
    if mode == "gt":
        return stat_data > thr
    return stat_data >= thr


def _brain_surface(tpl_grid, tpl_data: np.ndarray, bg_percentile: float, smooth_iters: int):
    finite = tpl_data[np.isfinite(tpl_data)]
    if finite.size == 0:
        return None

    positive = finite[finite > 0]
    if positive.size == 0:
        positive = finite

    # Try a few percentiles to avoid empty meshes
    for pct in [bg_percentile, 70.0, 50.0, 30.0]:
        level = float(np.percentile(positive, pct))
        mask = (tpl_data >= level).astype(np.float32)
        try:
            from scipy import ndimage as ndi

            labeled, n = ndi.label(mask > 0)
            if n > 1:
                counts = np.bincount(labeled.ravel())
                counts[0] = 0
                keep = counts.argmax()
                mask = (labeled == keep).astype(np.float32)
        except Exception:
            pass

        tpl_grid.point_data["mask"] = mask.ravel(order="F")
        surf = tpl_grid.contour([0.5], scalars="mask")
        if surf.n_points > 0:
            if smooth_iters > 0:
                surf = surf.smooth(n_iter=smooth_iters, relaxation_factor=0.1)
            return surf

    return None


def _camera_for_view(view: str, bounds, zoom: float = 1.0):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    cz = 0.5 * (zmin + zmax)
    span = max(xmax - xmin, ymax - ymin, zmax - zmin)
    zoom = max(float(zoom), 1e-3)
    dist = (span * 2.8) / zoom

    view = view.lower()
    if view == "top":
        position = (cx, cy, cz + dist)
        viewup = (0, 1, 0)
    elif view == "side":
        # Flip so anterior/front is on the right in the rendered image
        position = (cx + dist, cy, cz)
        viewup = (0, 0, 1)
    elif view == "iso":
        position = (cx + dist * 0.52, cy + dist * 0.52, cz + dist * 0.52)
        viewup = (0, 0, 1)
    else:
        position = (cx, cy, cz + dist)
        viewup = (0, 1, 0)

    focal = (cx, cy, cz)
    return [position, focal, viewup]


def render(
    stat_path: str,
    template_path: Optional[str] = None,
    out_dir: Optional[str] = None,
    view: str = "all",
    save: bool = False,
    style: str = "voxels",
    mode: str = "auto",
    p_thr: float = 0.05,
    corrp_thr: float = 0.949,
    thr: Optional[float] = None,
    thr_percentile: Optional[float] = None,
    bg_percentile: float = 40.0,
    color: str = "#c62828",
    cmap: str = "autumn",
    opacity: float = 0.85,
    bg_opacity: float = 0.15,
    zoom: float = 1.0,
    point_size: float = 3.0,
    point_stride: int = 1,
    max_points: Optional[int] = 200000,
    voxel_scale: float = 1.0,
    bg_smooth_iters: int = 20,
    resample: str = "nearest",
    labels: bool = True,
    label_color: str = "#666666",
    label_size: int = 14,
    window_size=(900, 900),
):
    """
    Render TBSS clusters as 3D visuals on a template (MNI by default).

    Returns a RenderResult when save=False, otherwise a list of output file paths.
    """
    import pyvista as pv

    stat_img, template_img = _load_images(stat_path, template_path, resample)
    stat_grid, stat_data = _to_uniform_grid(stat_img)
    tpl_grid, tpl_data = _to_uniform_grid(template_img)

    stat_thr, mode = _infer_threshold_and_mode(stat_path, stat_data, thr, thr_percentile, mode, p_thr, corrp_thr)
    # Extract surfaces
    mask = _mask_from_threshold(stat_data, stat_thr, mode).astype(np.float32)
    stat_grid.point_data["mask"] = mask.ravel(order="F")
    stat_surf = stat_grid.contour([0.5], scalars="mask")

    # Scalars for color mapping
    if mode == "lt":
        scalars_data = -stat_data
    else:
        scalars_data = stat_data
    stat_grid.point_data["scalars"] = scalars_data.ravel(order="F")
    masked_vals = scalars_data[mask > 0.5]
    clim = (float(masked_vals.min()), float(masked_vals.max())) if masked_vals.size else None

    brain_surf = _brain_surface(tpl_grid, tpl_data, bg_percentile, bg_smooth_iters)

    if save:
        out_dir = out_dir or os.getcwd()
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    stem = Path(stat_path).name
    for suffix in [".nii.gz", ".nii"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    outputs: List[str] = []
    images: Dict[str, np.ndarray] = {}
    style = (style or "voxels").lower()
    for v in _split_views(view):
        plotter = pv.Plotter(off_screen=True, window_size=window_size)
        plotter.set_background("white")
        plotter.enable_anti_aliasing()

        if brain_surf is not None:
            plotter.add_mesh(brain_surf, color="#bdbdbd", opacity=bg_opacity, smooth_shading=True)
        if style == "surface":
            if stat_surf.n_points == 0:
                raise ValueError(
                    "No suprathreshold voxels found. "
                    "Try --mode lt with --p-thr 0.05 for p-maps or lower --thr/--thr-percentile."
                )
            plotter.add_mesh(
                stat_surf,
                scalars="scalars",
                cmap=cmap,
                clim=clim,
                opacity=opacity,
                smooth_shading=True,
                show_scalar_bar=False,
            )
        elif style == "wire":
            if stat_surf.n_points == 0:
                raise ValueError(
                    "No suprathreshold voxels found. "
                    "Try --mode lt with --p-thr 0.05 for p-maps or lower --thr/--thr-percentile."
                )
            edges = stat_surf.extract_all_edges()
            plotter.add_mesh(edges, color=color, line_width=1.0)
        elif style == "points":
            ijk = np.column_stack(np.where(mask > 0.5))
            if ijk.size == 0:
                raise ValueError(
                    "No suprathreshold voxels found. "
                    "Try --mode lt with --p-thr 0.05 for p-maps or lower --thr/--thr-percentile."
                )
            if point_stride > 1:
                ijk = ijk[::point_stride]
            if max_points is not None and ijk.shape[0] > max_points:
                step = max(1, int(np.ceil(ijk.shape[0] / max_points)))
                ijk = ijk[::step]

            points = _voxel_world_points(stat_img, ijk)
            scalars = scalars_data[ijk[:, 0], ijk[:, 1], ijk[:, 2]]

            cloud = pv.PolyData(points)
            cloud["scalars"] = scalars
            plotter.add_mesh(
                cloud,
                scalars="scalars",
                cmap=cmap,
                clim=clim,
                opacity=opacity,
                render_points_as_spheres=True,
                point_size=point_size,
                show_scalar_bar=False,
            )
        else:
            # voxels (filled cubes)
            ijk = np.column_stack(np.where(mask > 0.5))
            if ijk.size == 0:
                raise ValueError(
                    "No suprathreshold voxels found. "
                    "Try --mode lt with --p-thr 0.05 for p-maps or lower --thr/--thr-percentile."
                )
            if point_stride > 1:
                ijk = ijk[::point_stride]
            if max_points is not None and ijk.shape[0] > max_points:
                step = max(1, int(np.ceil(ijk.shape[0] / max_points)))
                ijk = ijk[::step]

            spacing = np.array(stat_img.header.get_zooms()[:3], dtype=np.float32)
            points = _voxel_world_points(stat_img, ijk)
            scalars = scalars_data[ijk[:, 0], ijk[:, 1], ijk[:, 2]]

            cube = pv.Cube(
                center=(0, 0, 0),
                x_length=spacing[0] * voxel_scale,
                y_length=spacing[1] * voxel_scale,
                z_length=spacing[2] * voxel_scale,
            )
            cloud = pv.PolyData(points)
            cloud["scalars"] = scalars
            vox = cloud.glyph(geom=cube, scale=False, orient=False)
            plotter.add_mesh(
                vox,
                scalars="scalars",
                cmap=cmap,
                clim=clim,
                opacity=opacity,
                smooth_shading=False,
                show_scalar_bar=False,
            )

        # Frame from the template volume rather than the extracted shell so camera
        # distance stays stable across different background intensities/templates.
        bounds = tpl_grid.bounds if tpl_grid is not None else stat_surf.bounds
        plotter.camera_position = _camera_for_view(v, bounds, zoom=zoom)
        if v in {"top", "side"}:
            # Use orthographic projection for anatomical overview views so framing
            # does not depend on perspective/FOV and stays consistent across spaces.
            plotter.camera.parallel_projection = True
            plotter.camera.parallel_scale = max(
                (bounds[1] - bounds[0]) * 0.6,
                (bounds[3] - bounds[2]) * 0.6,
                (bounds[5] - bounds[4]) * 0.6,
            ) / max(float(zoom), 1e-3)

        if labels and v == "top":
            plotter.add_text("L", position="lower_left", font_size=label_size, color=label_color)
            plotter.add_text("R", position="lower_right", font_size=label_size, color=label_color)

        if save:
            out_path = str(Path(out_dir) / f"{stem}_{v}.png")
            plotter.show(screenshot=out_path, auto_close=True)
            outputs.append(out_path)
        else:
            img = plotter.screenshot(transparent_background=False, return_img=True)
            plotter.close()
            images[v] = img

    return outputs if save else RenderResult(images)
