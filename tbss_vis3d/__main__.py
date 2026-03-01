import argparse
from .render import render


def main():
    p = argparse.ArgumentParser(description="Render TBSS results as 3D figures")
    p.add_argument("stat", help="TBSS statistical map (NIfTI)")
    p.add_argument("--template", default=None, help="Background template (NIfTI). Defaults to MNI152")
    p.add_argument("--out-dir", default=None, help="Output folder for PNGs")
    p.add_argument("--view", default="all", help="Views: all|top|side|iso or comma list")
    p.add_argument("--save", action="store_true", help="Save PNGs (default off)")
    p.add_argument("--style", default="voxels", choices=["voxels", "points", "surface", "wire"], help="Cluster render style")
    p.add_argument("--mode", default="auto", choices=["auto", "lt", "gt"], help="Threshold mode")
    p.add_argument("--p-thr", type=float, default=0.05, help="P-value threshold (p-maps)")
    p.add_argument("--corrp-thr", type=float, default=0.949, help="corrp threshold (1-p maps)")
    p.add_argument("--cmap", default="autumn", help="Colormap for clusters (e.g., autumn, hot)")
    p.add_argument("--point-size", type=float, default=3.0, help="Point size when style=points")
    p.add_argument("--point-stride", type=int, default=1, help="Stride for points downsampling")
    p.add_argument("--max-points", type=int, default=200000, help="Max points to render")
    p.add_argument("--voxel-scale", type=float, default=1.0, help="Voxel cube size scale")
    p.add_argument("--bg-smooth-iters", type=int, default=20, help="Smoothing iterations for brain shell")
    p.add_argument("--resample", default="nearest", choices=["nearest", "continuous"], help="Resample interpolation")
    p.add_argument("--no-labels", action="store_true", help="Disable L/R labels")
    p.add_argument("--thr", type=float, default=None, help="Absolute threshold")
    p.add_argument("--thr-percentile", type=float, default=None, help="Percentile threshold (if --thr not set)")
    p.add_argument("--bg-percentile", type=float, default=60.0, help="Template percentile for brain surface")
    p.add_argument("--color", default="#c62828", help="Cluster color")
    p.add_argument("--opacity", type=float, default=0.85, help="Cluster opacity")
    p.add_argument("--bg-opacity", type=float, default=0.15, help="Template opacity")
    args = p.parse_args()

    render(
        args.stat,
        template_path=args.template,
        out_dir=args.out_dir,
        view=args.view,
        save=args.save,
        style=args.style,
        mode=args.mode,
        p_thr=args.p_thr,
        corrp_thr=args.corrp_thr,
        cmap=args.cmap,
        point_size=args.point_size,
        point_stride=args.point_stride,
        max_points=args.max_points,
        voxel_scale=args.voxel_scale,
        bg_smooth_iters=args.bg_smooth_iters,
        resample=args.resample,
        labels=(not args.no_labels),
        thr=args.thr,
        thr_percentile=args.thr_percentile,
        bg_percentile=args.bg_percentile,
        color=args.color,
        opacity=args.opacity,
        bg_opacity=args.bg_opacity,
    )


if __name__ == "__main__":
    main()
