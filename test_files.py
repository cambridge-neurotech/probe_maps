import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import probeinterface as pi
from probeinterface.plotting import plot_probe
from pathlib import Path
from tqdm.auto import tqdm


# Create output directory
output_dir = Path("sanity_checks_post_test")
output_dir.mkdir(exist_ok=True)

sheet_names = list(pd.read_excel("probe_contacts.xlsx", sheet_name=None).keys())

wrong_contours = []
sheets_with_issues = []
double_sided_probes = []

for sheet_name in tqdm(sheet_names, "Exporting CN probes"):
    wrong = False
    plot = True
    contacts = pd.read_excel("probe_contacts.xlsx", sheet_name=sheet_name)
    contour = pd.read_excel("probe_contours.xlsx", sheet_name=sheet_name)

    is_dual_sided = not np.all(pd.isna(contacts["contact_sides"]))

    if is_dual_sided:
        double_sided_probes.append(sheet_name)
        plot = False

        # Create ProbeGroup with 2 probes (front and back)
        try:
            probegroup = pi.ProbeGroup()

            for side in ["front", "back"]:
                side_contacts = contacts[contacts["contact_sides"] == side].copy()
                side_contacts.drop(columns=["contact_sides", "z"], inplace=True)

                probe = pi.Probe.from_dataframe(side_contacts)
                probe.manufacturer = "cambridgeneurotech"
                probe.model_name = f"{sheet_name}_{side}"
                probe.set_planar_contour(contour)
                probegroup.add_probe(probe)

            # Check contours for both probes
            for probe in probegroup.probes:
                min_x = np.min(probe.contact_positions[:, 0] - contacts["width"].iloc[0] / 2)
                max_x = np.max(probe.contact_positions[:, 0] + contacts["width"].iloc[0] / 2)

                if min_x < np.min(contour["x"]):
                    print(f"Problem with {sheet_name} on left side: {plot}")
                    wrong_contours.append(sheet_name)
                    wrong = True
                    break
                if max_x > np.max(contour["x"]):
                    print(f"Problem with {sheet_name} on right side: {plot}")
                    wrong_contours.append(sheet_name)
                    wrong = True
                    break

            # Export ProbeGroup to JSON
            json_path = output_dir / f"{sheet_name}.json"
            pi.write_probeinterface(str(json_path), probegroup)

            # Export PNG - dual-sided probes with 2 subplots
            png_path = output_dir / f"{sheet_name}.png"
            num_probes = len(probegroup.probes)
            fig, axes = plt.subplots(1, num_probes, figsize=(10 * num_probes, 12))
            for i, probe in enumerate(probegroup.probes):
                ax = axes[i] if num_probes > 1 else axes
                plot_probe(probe, ax=ax, with_contact_id=False, with_device_index=False)
                side_label = "Front" if i == 0 else "Back"
                ax.set_title(f"{sheet_name} - {side_label}")
            plt.tight_layout()
            plt.savefig(png_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        except Exception as e:
            print(f"Problem loading {sheet_name}: {e}")
            sheets_with_issues.append(sheet_name)

    else:
        # Single-sided probe
        contacts.drop(columns="contact_sides", inplace=True)

        if "z" in contacts.columns:
            contacts.drop(columns=["z"], inplace=True)
        try:
            probe = pi.Probe.from_dataframe(contacts)
            probe.manufacturer = "cambridgeneurotech"
            probe.model_name = sheet_name
            probe.set_planar_contour(contour)

            min_x = np.min(probe.contact_positions[:, 0] - contacts["width"][0] / 2)
            max_x = np.max(probe.contact_positions[:, 0] + contacts["width"][0] / 2)

            if min_x < np.min(contour["x"]):
                print(f"Problem with {sheet_name} on left side: {plot}")
                wrong_contours.append(sheet_name)
                wrong = True
            if max_x > np.max(contour["x"]):
                print(f"Problem with {sheet_name} on right side: {plot}")
                wrong_contours.append(sheet_name)
                wrong = True

            # Export Probe to JSON (wrap in ProbeGroup for consistency)
            probegroup = pi.ProbeGroup()
            probegroup.add_probe(probe)
            json_path = output_dir / f"{sheet_name}.json"
            pi.write_probeinterface(str(json_path), probegroup)

            # Export PNG - single probe
            png_path = output_dir / f"{sheet_name}.png"
            fig, ax = plt.subplots(figsize=(10, 12))
            plot_probe(probe, ax=ax, with_contact_id=False, with_device_index=False)
            ax.set_title(sheet_name)
            plt.tight_layout()
            plt.savefig(png_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        except Exception as e:
            print(f"Problem loading {sheet_name}: {e}")
            sheets_with_issues.append(sheet_name)

print(f"\n=== SUMMARY ===")
print(f"Total probes: {len(sheet_names)}")
print(f"Dual-sided probes: {len(double_sided_probes)}")
print(f"Probes with contour issues: {len(set(wrong_contours))}")
print(f"Probes with loading issues: {len(sheets_with_issues)}")
print(f"\nExported JSON + PNG files to: {output_dir.absolute()}")

if wrong_contours:
    print(f"\nContour issues: {set(wrong_contours)}")
if sheets_with_issues:
    print(f"\nLoading issues: {sheets_with_issues}")
if double_sided_probes:
    print(f"\nDual-sided probes: {double_sided_probes}")
