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
        is_dual_sided = not np.all(pd.isna(contacts["contact_sides"]))

        if is_dual_sided:
            # Handle dual-sided probe
            probegroup = pi.ProbeGroup()

            for side in ["front", "back"]:
                side_contacts = contacts[contacts["contact_sides"] == side].copy()
                side_contacts = side_contacts.drop(columns=["contact_sides"], errors="ignore")
                if "z" in side_contacts.columns:
                    side_contacts = side_contacts.drop(columns=["z"])

                if len(side_contacts) == 0:
                    continue

                probe = pi.Probe.from_dataframe(side_contacts)
                probe.set_planar_contour(contour)
                probegroup.add_probe(probe)

            if len(probegroup.probes) == 0:
                return False

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

            # Create plot
            num_probes = len(probegroup.probes)
            fig, axes = plt.subplots(1, num_probes, figsize=(8 * num_probes, 10))

            for i, probe in enumerate(probegroup.probes):
                ax = axes[i] if num_probes > 1 else axes
                plot_probe(probe, ax=ax, with_contact_id=False, with_device_index=False)
                side_label = "Front" if i == 0 else "Back"
                ax.set_title(f"{sheet_name} - {side_label} (Tip Region)")
                ax.set_ylim(y_min, y_max)
                ax.set_xlim(x_min, x_max)

        else:
            # Handle single-sided probe
            contacts_clean = contacts.drop(columns=["contact_sides"], errors="ignore")
            if "z" in contacts_clean.columns:
                contacts_clean = contacts_clean.drop(columns=["z"])

            probe = pi.Probe.from_dataframe(contacts_clean)
            probe.set_planar_contour(contour)

            # Calculate zoom bounds
            all_x = contacts["x"].values
            all_y = contacts["y"].values
            min_y_contacts = np.min(all_y)
            max_y_contacts = np.max(all_y)

            # Also consider contour tip
            min_y_contour = contour["y"].min()

            # Zoom region
            y_min = min(min_y_contour, min_y_contacts) - zoom_margin
            y_max = min(max_y_contacts + zoom_margin * 3, min_y_contacts + 800)

            # X bounds with margin
            x_min = np.min(all_x) - zoom_margin
            x_max = np.max(all_x) + zoom_margin

            # Create plot
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
