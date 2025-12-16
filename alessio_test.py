import pandas as pd
import numpy as np
import probeinterface as pi
from probeinterface.plotting import plot_probe
from tqdm.auto import tqdm


sheet_names = list(pd.read_excel("probe_contacts.xlsx", sheet_name=None).keys())

wrong_contours = []
sheets_with_issues = []
double_sided_probes = []

for sheet_name in tqdm(sheet_names, "Exporting CN probes"):
    wrong = False
    plot = True
    contacts = pd.read_excel("probe_contacts.xlsx", sheet_name=sheet_name)
    contour = pd.read_excel("probe_contours.xlsx", sheet_name=sheet_name)

    if np.all(pd.isna(contacts["contact_sides"])):
        contacts.drop(columns="contact_sides", inplace=True)
    else:
        double_sided_probes.append(sheet_name)
        plot = False

    if "z" in contacts.columns:
        contacts.drop(columns=["z"], inplace=True)
    try:
        probe = pi.Probe.from_dataframe(contacts)
        probe.manufacturer = "cambridgeneurotech"
        probe.model_name = sheet_name
        probe.set_planar_contour(contour)
        # plot_probe(probe)
        min_x = np.min(probe.contact_positions[:, 0] - contacts["width"][0] / 2)
        max_x = np.max(probe.contact_positions[:, 0] +  contacts["width"][0] / 2)
    
        if min_x  < np.min(contour["x"]):
            print(f"Problem with {sheet_name} on left side: {plot}")
            wrong_contours.append(sheet_name)
            wrong = True
        if max_x  > np.max(contour["x"]):
            print(f"Problem with {sheet_name} on right side: {plot}")
            wrong_contours.append(sheet_name)
            wrong = True
    
        if wrong and plot:
            plot_probe(probe)

    except Exception as e:
        print(f"Problem loading {sheet_name}: {e}")
        sheets_with_issues.append(sheet_name)
