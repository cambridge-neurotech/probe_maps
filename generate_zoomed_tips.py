#!/usr/bin/env python3
"""
Generate zoomed visualizations of probe tip regions.

Creates PNG images focused on the tip area where V-tips and contacts are located,
making it easier to verify contour correctness at full-scale probes.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import probeinterface as pi
from probeinterface.plotting import plot_probe
from tqdm.auto import tqdm


def generate_zoomed_tip_image(
    sheet_name: str,
    contacts_file: Path,
    contours_file: Path,
    output_dir: Path,
    zoom_margin: float = 100.0,
) -> bool:
    """
    Generate a zoomed image of the tip region for a probe.

    Uses native probeinterface contact_sides support (PR #382) for dual-sided probes.

    Args:
        sheet_name: Name of the probe (Excel sheet name)
        contacts_file: Path to probe_contacts.xlsx
        contours_file: Path to probe_contours.xlsx
        output_dir: Directory to save output images
        zoom_margin: Extra margin around contacts in µm

    Returns:
        True if successful, False otherwise
    """
    try:
        contacts = pd.read_excel(contacts_file, sheet_name=sheet_name)
        contour = pd.read_excel(contours_file, sheet_name=sheet_name)

        # Check if dual-sided probe
        is_dual_sided = (
            "contact_sides" in contacts.columns
            and not contacts["contact_sides"].isna().all()
        )

        # Calculate zoom bounds from all contacts
        all_x = contacts["x"].values
        all_y = contacts["y"].values
        min_y_contacts = np.min(all_y)
        max_y_contacts = np.max(all_y)

        # Also consider contour tip
        min_y_contour = contour["y"].min()

        # Zoom region: from below tip to above highest contacts (limited height)
        y_min = min(min_y_contour, min_y_contacts) - zoom_margin
        y_max = min(max_y_contacts + zoom_margin * 3, min_y_contacts + 800)

        # X bounds with margin
        x_min = np.min(all_x) - zoom_margin
        x_max = np.max(all_x) + zoom_margin

        # Build positions array
        positions = np.column_stack([contacts["x"].values, contacts["y"].values])

        # Build shape params
        if "width" in contacts.columns and "height" in contacts.columns:
            shape_params = [
                {"width": float(w), "height": float(h)}
                for w, h in zip(contacts["width"].values, contacts["height"].values)
            ]
        else:
            shape_params = {"width": 12, "height": 12}

        # Get shapes
        if "contact_shapes" in contacts.columns:
            shapes = contacts["contact_shapes"].values
        else:
            shapes = "rect"

        # Get shank_ids
        if "shank_ids" in contacts.columns:
            shank_ids = contacts["shank_ids"].values
        else:
            shank_ids = None

        # Create single probe with native contact_sides support (PR #382)
        probe = pi.Probe(ndim=2)

        if is_dual_sided:
            # Get contact_sides values
            contact_sides = np.array([
                s if s in ("front", "back") else None
                for s in contacts["contact_sides"].values
            ])

            probe.set_contacts(
                positions=positions,
                shapes=shapes,
                shape_params=shape_params,
                shank_ids=shank_ids,
                contact_sides=contact_sides,  # Native PR #382 support
            )
        else:
            probe.set_contacts(
                positions=positions,
                shapes=shapes,
                shape_params=shape_params,
                shank_ids=shank_ids,
            )

        # Set contact IDs if available
        if "contact_ids" in contacts.columns:
            probe.set_contact_ids(contacts["contact_ids"].values)

        # Set contour
        probe.set_planar_contour(contour.values)

        if is_dual_sided:
            # Create side-by-side plots using plot_probe's side parameter (PR #382)
            fig, axes = plt.subplots(1, 2, figsize=(16, 10))

            for i, side_name in enumerate(["front", "back"]):
                ax = axes[i]
                plot_probe(
                    probe,
                    ax=ax,
                    with_contact_id=False,
                    with_device_index=False,
                    side=side_name,  # Native PR #382 side filtering
                )
                ax.set_title(f"{sheet_name} - {side_name.capitalize()} (Tip Region)")
                ax.set_ylim(y_min, y_max)
                ax.set_xlim(x_min, x_max)
        else:
            # Single-sided probe: single plot
            fig, ax = plt.subplots(figsize=(10, 10))
            plot_probe(probe, ax=ax, with_contact_id=False, with_device_index=False)
            ax.set_title(f"{sheet_name} (Tip Region)")
            ax.set_ylim(y_min, y_max)
            ax.set_xlim(x_min, x_max)

        plt.tight_layout()
        output_path = output_dir / f"{sheet_name}_tip.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return True

    except Exception as e:
        print(f"Error processing {sheet_name}: {e}")
        return False


def main():
    """Generate zoomed tip images for all probes."""
    # Paths
    contacts_file = Path("probe_contacts.xlsx")
    contours_file = Path("probe_contours.xlsx")
    output_dir = Path("sanity_checks_zoomed_tips")

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Get all sheet names
    sheet_names = list(pd.read_excel(contacts_file, sheet_name=None).keys())

    print("=" * 60)
    print("GENERATING ZOOMED TIP VISUALIZATIONS")
    print("=" * 60)
    print(f"Total probes: {len(sheet_names)}")
    print(f"Output directory: {output_dir.absolute()}")
    print()

    success_count = 0
    error_count = 0
    errors = []

    for sheet_name in tqdm(sheet_names, desc="Generating zoomed images"):
        if generate_zoomed_tip_image(
            sheet_name, contacts_file, contours_file, output_dir
        ):
            success_count += 1
        else:
            error_count += 1
            errors.append(sheet_name)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Success: {success_count}")
    print(f"Errors: {error_count}")
    if errors:
        print(f"Failed probes: {errors}")
    print(f"\nOutput saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
