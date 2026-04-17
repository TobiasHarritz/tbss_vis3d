# tbss_vis3d

Lightweight 3D TBSS visualization helper that creates publication‑ready images from multiple views.

## Install (editable)

```bash
pip install -e .
```

## One‑line usage (Python, in‑memory)

```python
from tbss_vis3d import render; r = render(
    "/Users/tobiasharritz/data/tbss_example/ExBox7/tbss_clustere_corrp_tstat1.nii.gz",
    template_path="/Users/tobiasharritz/data/tbss_example/ExBox7/mean_FA.nii.gz",  # optional; defaults to MNI152
    zoom=1.15,
)
```

## CLI

```bash
tbss-vis3d /Users/tobiasharritz/data/tbss_example/ExBox7/tbss_clustere_corrp_tstat1.nii.gz \
  --template /Users/tobiasharritz/data/tbss_example/ExBox7/mean_FA.nii.gz \
  --zoom 1.15 \
  --save
```

## Views

Use `--view all` (default) or a comma list such as `--view top,side,iso`.

## Notes

- The default background is the MNI152 template.
- The background template defaults to MNI152; pass `template_path` to use `mean_FA.nii.gz` or others.
- If your TBSS image is a corrected **corrp** map (1‑p) in [0,1], significance is **higher** values.
  `mode="auto"` (default) detects sparse high tails and uses `corrp_thr=0.949` with `mode="gt"`.
- If it’s a true p‑map, `mode="auto"` uses `p_thr=0.05` and `mode="lt"`.
- Otherwise a percentile threshold is used (default 95th, `mode="gt"`).
- By default, `render(...)` returns a `RenderResult` object with attributes: `r.top`, `r.side`, `r.iso`.
- Pass `save=True` to write PNGs instead.
- Default `style="voxels"` to show filled voxel cubes. Use `style="surface"` for a continuous mesh or `style="points"` for a sparse cloud.
- Default colormap is `autumn` (red→yellow). Use `cmap="hot"` or any matplotlib colormap name.
- Use `zoom` / `--zoom` to adjust framing. Values `>1` zoom in; values `<1` zoom out.
- L/R labels are on by default; disable with `--no-labels`.
- For a cleaner visualization (like FSLeyes “filled” display), you can pass the filled map:
  `tbss_clustere_corrp_tstat1_filled.nii.gz` (visualization only; skeleton is the valid result).
- Default resampling is `nearest` to avoid smearing sparse cluster maps.
