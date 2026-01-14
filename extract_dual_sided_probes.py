#!/usr/bin/env python3
"""
Extract probe contact and contour data from JSON files with proper Z-coordinate calculation
for dual-sided probes based on shank thickness from database.

Output:
    - probe_contacts.xlsx: Contact coordinates with Z based on shank thickness
    - probe_contours.xlsx: 2D contours for each probe
"""

import argparse
import json
import logging
import socket
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from contact_id_mapper import ContactIdMapper


def setup_logging(script_name: str) -> logging.Logger:
    """
    Configure logging to both console and file.

    Args:
        script_name: Name of the script (used in log filename)

    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path.cwd() / "logs"
    log_dir.mkdir(exist_ok=True)

    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    log_filename = f"{script_name}_{timestamp}.log"
    log_path = log_dir / log_filename

    # Create logger
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatters
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Log file: {log_path}")
    return logger


# Configure logging
logger = setup_logging("extract_dual_sided_probes")


def get_base_path() -> tuple[Path, bool]:
    """
    Get the correct base path, prioritizing local files over Google Drive.

    Returns:
        Tuple of (base_path, is_local) where is_local indicates if using local files
    """
    computer_name = socket.gethostname()
    cwd = Path.cwd()

    # First, check if local files exist in current working directory
    local_library = cwd / "probe_library"
    local_database = cwd / "ProbesDataBase_2Dshanks_2025.csv"

    if local_library.exists() and local_database.exists():
        logger.info(f"Computer: {computer_name}, Using local files in: {cwd}")
        return cwd, True

    # Fall back to Google Drive paths for known computers
    if computer_name == "M-01699":
        base_path = Path(r"G:\My Drive\probeinterface\probe_maps")
    elif computer_name == "D-01643":
        base_path = Path(r"H:\Gdrive\rentAspike\My Drive\probeinterface\probe_maps")
    else:
        # Default to current directory if unknown computer
        logger.warning(f"Unknown computer: {computer_name}. Using current directory.")
        base_path = cwd

    logger.info(f"Computer: {computer_name}, Base path: {base_path}")
    return base_path, False


class ProbeDataExtractor:
    """Extract probe data and calculate proper Z coordinates for dual-sided probes."""

    def __init__(
        self,
        library_path: Path,
        database_path: Path,
        output_dir: Path,
        contact_id_excel_path: Optional[Path] = None,
    ):
        """Initialize extractor with paths."""
        self.library_path = library_path
        self.database_path = database_path
        self.output_dir = output_dir

        # Track validation errors across all probes
        self.validation_errors: List[str] = []

        # Load shank thickness and length database
        self.shank_database, self.length_database = self._load_database()
        logger.info(f"Loaded database with {len(self.shank_database)} thickness entries")
        logger.info(f"Loaded database with {len(self.length_database)} length entries")

        # Load contact ID mapper for ASSY-325D probes
        self.contact_id_mapper: Optional[ContactIdMapper] = None
        if contact_id_excel_path and contact_id_excel_path.exists():
            self.contact_id_mapper = ContactIdMapper(contact_id_excel_path)
            logger.info("Loaded contact ID mapper from Excel")

    def _load_database(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Load shank thickness and length from CSV database."""
        df = pd.read_csv(self.database_path)

        # Create lookup dictionaries for shank thickness and length
        thickness_dict = {}
        length_dict = {}

        for _, row in df.iterrows():
            part_name = row["part"]
            thickness = row["shank_thickness_um"]
            # Note: typo in column name "lenght" instead of "length"
            length = row["shank_lenght_mm"] if pd.notna(row["shank_lenght_mm"]) else None

            # Store by exact part name only
            # Note: Do NOT strip 'double' suffix - these are separate probe types
            # The 'double' variants (E-1double, H7double, etc.) have different shank lengths
            # and are only used for ASSY-325D dual-sided probes
            thickness_dict[part_name] = thickness
            if length is not None:
                length_dict[part_name] = length

        return thickness_dict, length_dict

    def _get_probe_type_from_folder(self, folder_name: str) -> str:
        """
        Extract probe type from folder name.
        Examples:
            ASSY-325D-H7 -> H7
            ASSY-325D-E-1 -> E-1
            ASSY-325D-P-2 -> P-2
        """
        parts = folder_name.split("-")
        # For names like ASSY-325D-E-1, take last 2 parts if last is digit
        if len(parts) >= 2 and parts[-1].isdigit():
            return "-".join(parts[-2:])
        # For names like ASSY-325D-H7, take last part
        elif len(parts) >= 1:
            return parts[-1]
        return folder_name

    def _get_shank_thickness(self, probe_name: str) -> Optional[float]:
        """Get shank thickness for a probe from database."""
        # Try exact match first
        if probe_name in self.shank_database:
            return self.shank_database[probe_name]

        # Extract probe type and try to match
        probe_type = self._get_probe_type_from_folder(probe_name)

        # For 325D dual-sided probes, try with 'double' suffix
        if "325D" in probe_name:
            double_name = f"{probe_type}double"
            if double_name in self.shank_database:
                return self.shank_database[double_name]

        # Try without suffix for regular probes
        if probe_type in self.shank_database:
            return self.shank_database[probe_type]

        return None

    def _get_shank_length(self, probe_name: str) -> Optional[float]:
        """Get shank length for a probe from database."""
        # Try exact match first
        if probe_name in self.length_database:
            return self.length_database[probe_name]

        # Extract probe type and try to match
        probe_type = self._get_probe_type_from_folder(probe_name)

        # For 325D dual-sided probes, try with 'double' suffix
        if "325D" in probe_name:
            double_name = f"{probe_type}double"
            if double_name in self.length_database:
                return self.length_database[double_name]

        # Try without suffix for regular probes
        if probe_type in self.length_database:
            return self.length_database[probe_type]

        return None

    def _is_dual_sided(self, folder_name: str, json_data: Dict = None) -> bool:
        """
        Check if probe is dual-sided.

        Dual-sided probes are ONLY those with "325D" in the connector name.
        These probes have:
        - Multiple probe objects in JSON (2 objects = front and back)
        - "325D" in the folder name (e.g., ASSY-325D-E-1)

        Note: The database entries for "double" probes (E-1double, H7double, etc.)
        are specific to the 325D series only.
        """
        # Only 325D series probes are dual-sided
        # Check for "325D" in the connector name
        if "325D" in folder_name:
            # Verify with JSON structure if available
            if json_data and "probes" in json_data:
                if len(json_data["probes"]) > 1:
                    return True
            # Even without JSON data, 325D probes are dual-sided
            return True

        return False

    def _parse_json_probe(self, json_path: Path) -> Tuple[List[Dict], List[List], Dict]:
        """Parse probe JSON file and return contacts, contours, and metadata.

        Note: Dual-sided probes are stored as ProbeGroup objects in probeinterface,
        containing multiple Probe objects (one for each side). The JSON structure
        reflects this with multiple entries in the 'probes' array.
        """
        with open(json_path, "r") as f:
            data = json.load(f)

        if data.get("specification") != "probeinterface":
            raise ValueError(f"Invalid specification in {json_path}")

        probe_name = json_path.stem
        # Pass JSON data for better dual-sided detection
        is_dual = self._is_dual_sided(probe_name, data)
        thickness = self._get_shank_thickness(probe_name)
        length = self._get_shank_length(probe_name)

        if is_dual and thickness is None:
            logger.warning(f"No shank thickness found for dual-sided probe: {probe_name}")
        if length is None:
            logger.debug(f"No shank length found for probe: {probe_name}")

        # Process all probe objects in JSON
        all_contacts = []
        contours = []
        contour_columns = ["x", "y"]  # Default column names
        contact_id_offset = 0  # Track contact ID offset for multiple probe objects

        probes_list = data.get("probes", [])

        for probe_idx, probe_obj in enumerate(probes_list):
            # Determine side for dual-sided probes
            if is_dual and len(probes_list) > 1:
                side = "front" if probe_idx == 0 else "back"
            else:
                side = None

            # Extract contact positions
            contact_positions = probe_obj.get("contact_positions", [])

            # Handle contact_ids - might be list or might not exist
            contact_ids_raw = probe_obj.get("contact_ids", None)
            device_channels = probe_obj.get("device_channel_indices", [])

            # For dual-sided probes (ASSY-325D), ALWAYS use Excel mapper when available
            # to get spatially-sorted contact IDs, ignoring JSON contact_ids
            if self.contact_id_mapper and is_dual and device_channels:
                # Use Excel mapper for ASSY-325D probes with device_channel_indices
                probe_type = self.contact_id_mapper.extract_probe_type_from_name(probe_name)
                if probe_type:
                    # Create mapping from device channel to contact ID
                    channel_to_id = self.contact_id_mapper.create_contact_id_mapping(
                        probe_type, device_channels
                    )

                    # Check for missing channels and warn
                    missing_channels = [
                        ch for ch in device_channels if ch not in channel_to_id
                    ]
                    if missing_channels:
                        logger.warning(
                            f"Missing Excel mappings for {probe_name}: "
                            f"channels {missing_channels}"
                        )

                    # Map each position's device channel to its contact ID
                    # IMPORTANT: Use fallback IDs that cannot collide with mapped IDs
                    # Mapped IDs are 1 to len(channel_to_id), so fallback starts after
                    max_mapped_id = max(channel_to_id.values()) if channel_to_id else 0
                    fallback_id = max_mapped_id + 1

                    contact_ids = []
                    for i, ch in enumerate(device_channels):
                        if ch in channel_to_id:
                            contact_ids.append(channel_to_id[ch] + contact_id_offset)
                        else:
                            # Use non-colliding fallback ID
                            contact_ids.append(fallback_id + contact_id_offset)
                            fallback_id += 1

                    logger.debug(
                        f"Mapped {len(contact_ids)} contact IDs from Excel for {probe_name} "
                        f"(offset={contact_id_offset}, missing={len(missing_channels)})"
                    )
                else:
                    # Fallback to sequential IDs
                    contact_ids = list(range(
                        contact_id_offset + 1,
                        contact_id_offset + 1 + len(contact_positions)
                    ))
            elif isinstance(contact_ids_raw, list):
                # For non-dual-sided probes, use JSON contact_ids if available
                if probe_idx > 0:
                    contact_ids = [int(cid) + contact_id_offset for cid in contact_ids_raw]
                else:
                    contact_ids = [
                        int(cid) if isinstance(cid, str) else cid
                        for cid in contact_ids_raw
                    ]
            else:
                # Fallback: Generate sequential contact IDs
                contact_ids = list(range(
                    contact_id_offset + 1,
                    contact_id_offset + 1 + len(contact_positions)
                ))

            # Get contact shapes and dimensions
            contact_shapes = probe_obj.get("contact_shapes", None)
            if not isinstance(contact_shapes, list):
                contact_shapes = ["circle"] * len(contact_positions)

            shape_params = probe_obj.get("contact_shape_params", [])
            if isinstance(shape_params, list) and len(shape_params) > 0:
                # List of dicts format: [{"width": 5, "height": 5}, ...]
                widths = [p.get("width", 11) if isinstance(p, dict) else 11 for p in shape_params]
                heights = [p.get("height", 15) if isinstance(p, dict) else 15 for p in shape_params]
            elif isinstance(shape_params, dict):
                # Dict with lists format: {"width": [5, 5, ...], "height": [5, 5, ...]}
                widths = shape_params.get("width", [11] * len(contact_positions))
                heights = shape_params.get("height", [15] * len(contact_positions))
            else:
                widths = [11] * len(contact_positions)
                heights = [15] * len(contact_positions)

            # Get shank IDs if available
            shank_ids_raw = probe_obj.get("shank_ids", None)
            if isinstance(shank_ids_raw, list):
                shank_ids = shank_ids_raw
            else:
                shank_ids = [""] * len(contact_positions)

            # Process each contact
            for i, pos in enumerate(contact_positions):
                x = pos[0]

                # For dual-sided probes: y is vertical position, z represents the side (depth)
                # For single-sided probes: y is vertical, z is shank thickness (negative)
                if is_dual and thickness is not None:
                    # For dual-sided probes with known thickness
                    # y = vertical position along shank (from JSON z coordinate)
                    y = pos[2] if len(pos) > 2 else 0.0
                    # z = side indicator (front=negative, back=positive half-thickness)
                    half_thickness = thickness / 2.0
                    z = -half_thickness if side == "front" else half_thickness
                else:
                    # For single-sided probes, y is vertical position, z is shank thickness (negative)
                    y = pos[1] if len(pos) > 1 else 0.0
                    # Use negative shank thickness as z value for single-sided probes
                    z = -thickness if thickness is not None else 0.0

                contact = {
                    "contact_ids": str(contact_ids[i]) if i < len(contact_ids)
                                  else str(i),
                    "x": x,
                    "y": y,
                    "z": z,
                    "contact_shapes": contact_shapes[i] if i < len(contact_shapes)
                                     else "circle",
                    "width": widths[i] if i < len(widths) else 11,
                    "height": heights[i] if i < len(heights) else 15,
                    "shank_ids": str(shank_ids[i]) if i < len(shank_ids) else "",
                    "contact_sides": side if side else ""
                }
                all_contacts.append(contact)

            # Update contact ID offset for next probe object (for dual-sided probes)
            contact_id_offset += len(contact_positions)

            # Get contour from first probe object only
            if probe_idx == 0 and "probe_planar_contour" in probe_obj:
                contour_data = probe_obj["probe_planar_contour"]
                # Handle both 2D and 3D contours
                if contour_data and len(contour_data) > 0:
                    if len(contour_data[0]) == 2:
                        contours = contour_data
                        contour_columns = ["x", "y"]
                    elif len(contour_data[0]) == 3:
                        # For 3D contours, take x and z coordinates
                        # (y is the depth/side coordinate which stays constant for planar contours)
                        contours = [[p[0], p[2]] for p in contour_data]
                        contour_columns = ["x", "y"]  # Column names in output (extracting x and z from 3D)
                    else:
                        contours = contour_data

        metadata = {
            "name": probe_name,
            "is_dual_sided": is_dual,
            "shank_thickness": thickness,
            "shank_length_mm": length,
            "num_probes": len(probes_list),
            "total_contacts": len(all_contacts),
            "contour_columns": contour_columns,  # Store column names for contours
            "raw_contour_2d": contours,  # Store 2D contour [x, z] for sanity checks
        }

        # Validate contact IDs are unique (CRITICAL CHECK)
        contact_ids = [c["contact_ids"] for c in all_contacts]
        unique_ids = set(contact_ids)
        if len(contact_ids) != len(unique_ids):
            counts = Counter(contact_ids)
            duplicates = {k: v for k, v in counts.items() if v > 1}
            error_msg = f"{probe_name}: NON-UNIQUE contact IDs found: {duplicates}"
            logger.error(f"CRITICAL: {error_msg}")
            self.validation_errors.append(error_msg)
            # Fix duplicates by appending suffix (allows processing to continue)
            seen = {}
            for contact in all_contacts:
                cid = contact["contact_ids"]
                if cid in seen:
                    seen[cid] += 1
                    new_id = f"{cid}_{seen[cid]}"
                    contact["contact_ids"] = new_id
                    logger.warning(f"  Renamed duplicate ID {cid} -> {new_id}")
                else:
                    seen[cid] = 0

        return all_contacts, contours, metadata

    def process_all_probes(self) -> Tuple[Dict, Dict, List]:
        """Process all probe JSON files in library."""
        json_files = sorted(self.library_path.rglob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files to process")

        all_contacts = {}
        all_contours = {}
        metadata_list = []

        for json_file in json_files:
            try:
                probe_name = json_file.stem
                logger.info(f"Processing: {probe_name}")

                contacts, contours, metadata = self._parse_json_probe(json_file)

                # Convert to DataFrames
                contacts_df = pd.DataFrame(contacts)
                contacts_df.set_index("contact_ids", inplace=True)

                # Use appropriate column names based on what was extracted
                if contours:
                    contours_df = pd.DataFrame(contours, columns=metadata["contour_columns"])
                    # Add shank_length_mm as a column
                    if metadata["shank_length_mm"] is not None:
                        contours_df["shank_length_mm"] = metadata["shank_length_mm"]
                    else:
                        contours_df["shank_length_mm"] = None
                else:
                    contours_df = pd.DataFrame(contours, columns=[])

                all_contacts[probe_name] = contacts_df
                all_contours[probe_name] = contours_df
                metadata_list.append(metadata)

                if metadata["is_dual_sided"]:
                    logger.info(
                        f"  ✓ Dual-sided probe: {metadata['total_contacts']} contacts, "
                        f"thickness: {metadata['shank_thickness']} µm"
                    )

            except Exception as e:
                logger.error(f"Failed to process {json_file.name}: {e}")
                continue

        return all_contacts, all_contours, metadata_list

    def _extend_contour_to_shank_length(
        self, contour_df: pd.DataFrame, contacts_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Extend contour Y coordinates to match full shank length from database.

        For single-shank probes only. This function:
        1. Centers the tip on the contacts (fixes off-center tips in source JSON)
        2. Extends the top edge to full shank length

        Args:
            contour_df: DataFrame with x, y columns and shank_length_mm
            contacts_df: Optional DataFrame with contact positions

        Returns:
            Extended contour DataFrame with centered tip and top edge at full shank length
        """
        if contour_df.empty or "shank_length_mm" not in contour_df.columns:
            return contour_df

        shank_length_mm = contour_df["shank_length_mm"].iloc[0]
        if pd.isna(shank_length_mm):
            return contour_df

        # Convert shank length to micrometers
        shank_length_um = shank_length_mm * 1000

        # Get x and y columns
        x_col = "x" if "x" in contour_df.columns else contour_df.columns[0]
        y_col = "y" if "y" in contour_df.columns else contour_df.columns[1]

        x_values = contour_df[x_col].values.copy().astype(float)
        y_values = contour_df[y_col].values.copy().astype(float)

        if len(y_values) == 0:
            return contour_df

        # Check if contacts fit within original contour bounds
        # Only center if contacts are OUTSIDE the contour (indicates misalignment)
        if contacts_df is not None and "x" in contacts_df.columns and len(contacts_df) > 0:
            contact_min_x = contacts_df["x"].min()
            contact_max_x = contacts_df["x"].max()
            contact_width = 11  # Default contact width
            if "width" in contacts_df.columns:
                contact_width = contacts_df["width"].max()

            # Calculate contact bounds with padding
            contact_left = contact_min_x - contact_width / 2 - 2
            contact_right = contact_max_x + contact_width / 2 + 2

            # Get original contour bounds
            contour_min_x = np.min(x_values)
            contour_max_x = np.max(x_values)

            # Only center if contacts extend OUTSIDE the original contour
            # This indicates the source JSON tip is misaligned
            contacts_outside = contact_left < contour_min_x or contact_right > contour_max_x

            if contacts_outside:
                # Find the center X of contacts
                contact_center_x = (contact_min_x + contact_max_x) / 2

                # Find the current tip apex (minimum Y point)
                apex_idx = np.argmin(y_values)
                original_apex_x = x_values[apex_idx]

                # Calculate offset to center the contour on contacts
                x_offset = contact_center_x - original_apex_x

                # Shift all X values to center the tip
                if abs(x_offset) > 1.0:  # Only shift if offset is significant
                    x_values = x_values + x_offset
                    logger.debug(f"Centered contour tip: shifted X by {x_offset:.1f} µm")

        # Find the current max Y in the contour
        current_max_y = np.max(y_values)

        # If contour already extends to or beyond shank length, no change needed
        if current_max_y >= shank_length_um:
            result_df = pd.DataFrame({x_col: x_values, y_col: y_values})
            return result_df

        # Find all points that are at the top edge (at or near max Y)
        top_edge_tolerance = 10.0
        is_top_edge = y_values >= (current_max_y - top_edge_tolerance)

        # Extend top edge points to full shank length
        y_values[is_top_edge] = shank_length_um

        result_df = pd.DataFrame({x_col: x_values, y_col: y_values})

        logger.debug(
            f"Extended contour top edge from {current_max_y:.1f} µm to {shank_length_um:.1f} µm"
        )

        return result_df

    def _extend_multishank_contour(
        self, contour_df: pd.DataFrame, contacts_df: pd.DataFrame = None, probe_name: str = None
    ) -> pd.DataFrame:
        """
        Extend multi-shank contour so each shank extends to near full length.

        For multi-shank probes, extends each individual shank from its current
        top up to near the shank length, with all shanks merging at the very top.

        IMPORTANT: This function uses tip_length from a database based on probe type
        to ensure consistent tip depths regardless of source JSON quality.

        Args:
            contour_df: DataFrame with x, y columns and shank_length_mm
            contacts_df: Optional DataFrame with contact positions and dimensions
            probe_name: Probe name to extract probe type for tip_length lookup

        Returns:
            Extended contour DataFrame with individual shanks extended
        """
        if contour_df.empty or "shank_length_mm" not in contour_df.columns:
            return contour_df[["x", "y"]].copy() if "x" in contour_df.columns else contour_df

        shank_length_mm = contour_df["shank_length_mm"].iloc[0]
        if pd.isna(shank_length_mm):
            return contour_df[["x", "y"]].copy() if "x" in contour_df.columns else contour_df

        shank_length_um = shank_length_mm * 1000
        merge_height = 200  # Height of merged section at top (µm)
        shank_top = shank_length_um - merge_height  # Where individual shanks extend to

        # Get contact half-width for expanding contour bounds
        contact_margin = 2.0  # Additional margin in µm
        if contacts_df is not None and "width" in contacts_df.columns:
            half_width = contacts_df["width"].max() / 2.0 + contact_margin
        else:
            half_width = contact_margin

        x_col = "x" if "x" in contour_df.columns else contour_df.columns[0]
        y_col = "y" if "y" in contour_df.columns else contour_df.columns[1]

        x_values = contour_df[x_col].values.astype(float)
        y_values = contour_df[y_col].values.astype(float)

        current_max_y = np.max(y_values)
        current_min_y = np.min(y_values)  # The tip apex Y from original contour

        # If already extended, just return
        if current_max_y >= shank_length_um:
            return contour_df[[x_col, y_col]].copy()

        # Detect shanks from contact positions (most reliable)
        shanks = []
        if contacts_df is not None and "x" in contacts_df.columns and len(contacts_df) > 0:
            contact_x_values = sorted(contacts_df["x"].unique())
            if len(contact_x_values) > 0:
                x_diffs = np.diff(contact_x_values)
                gap_threshold = 60  # Lower to detect H10/F8/H16 gaps (68-95µm)
                gap_indices = np.where(x_diffs > gap_threshold)[0]

                start_idx = 0
                for gap_idx in gap_indices:
                    shank_contacts_x = contact_x_values[start_idx:gap_idx + 1]
                    left_x = min(shank_contacts_x)
                    right_x = max(shank_contacts_x)
                    tip_x = (left_x + right_x) / 2
                    # Get min Y for this shank's contacts
                    shank_contacts = contacts_df[
                        (contacts_df["x"] >= left_x) & (contacts_df["x"] <= right_x)
                    ]
                    min_contact_y = shank_contacts["y"].min() if len(shank_contacts) > 0 else 0
                    shanks.append((left_x, right_x, tip_x, min_contact_y))
                    start_idx = gap_idx + 1
                if start_idx < len(contact_x_values):
                    shank_contacts_x = contact_x_values[start_idx:]
                    left_x = min(shank_contacts_x)
                    right_x = max(shank_contacts_x)
                    tip_x = (left_x + right_x) / 2
                    shank_contacts = contacts_df[
                        (contacts_df["x"] >= left_x) & (contacts_df["x"] <= right_x)
                    ]
                    min_contact_y = shank_contacts["y"].min() if len(shank_contacts) > 0 else 0
                    shanks.append((left_x, right_x, tip_x, min_contact_y))

        if not shanks:
            return self._extend_contour_to_shank_length(contour_df, contacts_df)

        # Sort shanks by X position (left to right)
        shanks = sorted(shanks, key=lambda s: s[2])

        # Expand shank bounds by half_width to ensure contacts fit inside contour
        shanks = [
            (left - half_width, right + half_width, tip_x, min_y)
            for left, right, tip_x, min_y in shanks
        ]

        # Tip length database based on Cambridge NeuroTech reference image
        # tip_length = how far below lowest contact the apex extends
        TIP_LENGTH_DB = {
            "E": 70,      # E-type: 40 µm tip width, ~70 µm extension
            "E-1": 70,
            "E-2": 70,
            "P": 70,      # P-type: 25 µm tip width, ~70 µm extension
            "P-1": 70,
            "P-2": 70,
            "F": 50,      # F-type: 25 µm tip width, ~50 µm extension
            "Fb": 50,
            "H10": 50,    # H10: 30 µm tip width
            "H6": 65,     # H6: 30 µm tip width
            "H7": 65,     # H7: 50 µm tip width, longer tip
            "H2": 28,     # H2: 25 µm tip width
            "H3": 28,     # H3: 20 µm tip width
            "H5": 30,     # H5: 25 µm tip width
            "H8": 100,    # H8: 60 µm tip width, longer tip
            "H9": 55,     # H9: 45 µm tip width
            "L1": 28,     # L-series: 18 µm tip width
            "L2": 28,
            "L3": 28,
            "M1": 50,     # M-series
            "M2": 50,
            "H1": 40,
            "H4": 28,
            "H12": 28,
            "H13": 28,
            "H14": 20,
            "H15": 20,
            "H16": 20,
            "H20": 20,
        }

        # Get tip length for this probe type
        tip_length = 50  # Default tip length

        # Extract probe type from probe_name (e.g., "ASSY-77-E-1" -> "E-1")
        probe_type = None
        if probe_name:
            probe_type = self._get_probe_type_from_folder(probe_name)

        if probe_type:
            # Try exact match first, then prefix match
            if probe_type in TIP_LENGTH_DB:
                tip_length = TIP_LENGTH_DB[probe_type]
            else:
                # Try to find a matching prefix (e.g., "E-1" matches "E")
                for key in sorted(TIP_LENGTH_DB.keys(), key=len, reverse=True):
                    if probe_type.startswith(key):
                        tip_length = TIP_LENGTH_DB[key]
                        break

        logger.debug(f"Probe {probe_name}: type={probe_type}, tip_length={tip_length}")

        # Calculate apex Y based on lowest contact and tip_length
        # This ensures consistent tip depth regardless of source JSON quality
        # Find the minimum contact Y across all shanks
        if contacts_df is not None and "y" in contacts_df.columns and len(contacts_df) > 0:
            global_min_contact_y = contacts_df["y"].min()
            contact_height = 15
            if "height" in contacts_df.columns:
                contact_height = contacts_df["height"].max()
            # Apex is tip_length below the lowest contact
            apex_y = global_min_contact_y - contact_height / 2 - tip_length
        else:
            # Fallback to original contour minimum
            apex_y = current_min_y

        # PRESERVE original contour shape (tapered geometry) and ADD straight walls above
        # Strategy: Keep all original points, then add new points to extend straight up
        # from the original top to the target shank length.

        # Find the original top Y (where straight walls should start)
        original_top_y = current_max_y

        # Overall bounds for the merged top section
        min_x = min(x_values)
        max_x = max(x_values)

        # Group points by shank (detect gaps in X)
        # Sort points by X to identify shank boundaries
        points_with_idx = [(x, y, i) for i, (x, y) in enumerate(zip(x_values, y_values))]

        # Find the "top edge" X values for each shank (points at or near max Y)
        top_edge_tolerance = 50  # Points within 50µm of max_y are considered "top edge"
        top_edge_points = [(x, y) for x, y in zip(x_values, y_values)
                          if y >= original_top_y - top_edge_tolerance]

        # Build new contour: keep original shape, add extensions above
        new_points = []

        # Start with top-left corner at full shank length
        new_points.append((min_x, shank_length_um))

        # Process original points and insert extension points where needed
        prev_at_top = False
        for i, (x, y) in enumerate(zip(x_values, y_values)):
            at_top = (y >= original_top_y - top_edge_tolerance)

            if at_top and not prev_at_top:
                # Transitioning TO top edge - add extension point first
                new_points.append((x, shank_top))

            # Add original point (preserve taper geometry)
            new_points.append((x, y))

            if not at_top and prev_at_top:
                # Transitioning FROM top edge - already handled by keeping original point
                pass

            prev_at_top = at_top

        # Close with top-right corner at full shank length
        new_points.append((max_x, shank_length_um))

        # Strategy: The contour traces down each shank and jumps horizontally between shanks.
        # For each gap (horizontal jump at top), we need to add vertical extensions on both sides.
        # Original: ... (66.5, 225) → (240, 230) ...  (jump across gap)
        # Extended: ... (66.5, 225) → (66.5, 5800) → (240, 5800) → (240, 230) ...

        top_threshold = 50  # Points within 50µm of max_y are "top" points
        gap_threshold = 60  # X jump > 60µm indicates gap between shanks (H10/F8/H16 have 68-95µm gaps)

        new_points = []
        new_points.append((min_x, shank_length_um))  # Top-left corner of merged section
        new_points.append((min_x, shank_top))  # Extension point on left edge

        n = len(x_values)
        for i in range(n):
            x, y = x_values[i], y_values[i]
            is_top = (y >= original_top_y - top_threshold)

            # Add current point
            new_points.append((x, y))

            # Check if this is followed by a horizontal jump (gap between shanks)
            if i < n - 1:
                next_x, next_y = x_values[i + 1], y_values[i + 1]
                is_next_top = (next_y >= original_top_y - top_threshold)
                x_jump = abs(next_x - x)

                # If both current and next are "top" points with a large X gap, add extensions
                if is_top and is_next_top and x_jump > gap_threshold:
                    # Add vertical extension from current point up
                    new_points.append((x, shank_top))
                    # Add horizontal connection at extended height
                    new_points.append((next_x, shank_top))

        # Add extension on right edge and close at top
        new_points.append((max_x, shank_top))
        new_points.append((max_x, shank_length_um))

        # Remove duplicates while preserving order
        seen = set()
        unique_points = []
        for p in new_points:
            key = (round(p[0], 1), round(p[1], 1))
            if key not in seen:
                seen.add(key)
                unique_points.append(p)

        result_df = pd.DataFrame(unique_points, columns=[x_col, y_col])

        logger.debug(
            f"Extended multi-shank contour: {len(shanks)} shanks, "
            f"individual shanks to {shank_top:.0f} µm, merge at {shank_length_um:.0f} µm"
        )

        return result_df

    def _expand_contour_for_contacts(
        self,
        contour_df: pd.DataFrame,
        contacts_df: pd.DataFrame,
        buffer_margin: float = 1.0
    ) -> pd.DataFrame:
        """
        Expand contour polygon to ensure all contact corners are enclosed.

        The source JSON contours often have V-shaped tips that don't account for
        contact dimensions. This method expands the contour using shapely's buffer
        operation to ensure all contact corners fit inside the polygon.

        Args:
            contour_df: DataFrame with x, y columns defining the contour polygon
            contacts_df: DataFrame with x, y, width, height columns for contacts
            buffer_margin: Additional margin in µm beyond contact corners (default 1.0)

        Returns:
            Expanded contour DataFrame with x, y columns
        """
        from shapely.geometry import Polygon
        from shapely.validation import make_valid

        if contour_df.empty or len(contour_df) < 3:
            return contour_df

        # Get column names
        x_col = "x" if "x" in contour_df.columns else contour_df.columns[0]
        y_col = "y" if "y" in contour_df.columns else contour_df.columns[1]

        # Create polygon from contour
        try:
            coords = list(zip(contour_df[x_col], contour_df[y_col]))
            polygon = Polygon(coords)
            if not polygon.is_valid:
                polygon = make_valid(polygon)
        except Exception as e:
            logger.warning(f"Could not create polygon from contour: {e}")
            return contour_df

        # Get maximum contact half-dimensions
        if contacts_df.empty or "width" not in contacts_df.columns:
            return contour_df

        half_width = contacts_df["width"].max() / 2.0 + buffer_margin
        half_height = contacts_df["height"].max() / 2.0 + buffer_margin

        # Use the larger of the two for buffer distance (conservative expansion)
        buffer_distance = max(half_width, half_height)

        # Expand polygon using buffer
        try:
            expanded_polygon = polygon.buffer(buffer_distance, join_style=2)  # mitre join

            # Extract coordinates from expanded polygon
            if expanded_polygon.is_empty:
                return contour_df

            # Get exterior coordinates
            if hasattr(expanded_polygon, 'exterior'):
                expanded_coords = list(expanded_polygon.exterior.coords)
            else:
                # MultiPolygon - use convex hull
                expanded_coords = list(expanded_polygon.convex_hull.exterior.coords)

            # Remove the closing point (shapely adds it automatically)
            if expanded_coords and expanded_coords[0] == expanded_coords[-1]:
                expanded_coords = expanded_coords[:-1]

            # Create new DataFrame, preserving shank_length_mm if present
            expanded_df = pd.DataFrame(expanded_coords, columns=[x_col, y_col])
            if "shank_length_mm" in contour_df.columns:
                expanded_df["shank_length_mm"] = contour_df["shank_length_mm"].iloc[0]
            return expanded_df

        except Exception as e:
            logger.warning(f"Could not expand contour: {e}")
            return contour_df

    def _is_multi_shank_probe(self, contacts_df: pd.DataFrame) -> bool:
        """
        Detect if a probe has multiple shanks based on contact X positions.

        Multi-shank probes have contacts clustered at different X positions
        separated by large gaps (shank spacing, typically 150-250 µm).
        """
        if contacts_df.empty or "x" not in contacts_df.columns:
            return False

        x_positions = contacts_df["x"].tolist()
        detected_shank_ids = self._detect_shank_ids_from_positions(x_positions)
        num_shanks = len(set(detected_shank_ids))
        return num_shanks > 1

    def write_excel_files(self, contacts_dict: Dict, contours_dict: Dict):
        """Write data to Excel files."""
        contacts_path = self.output_dir / "probe_contacts.xlsx"
        contours_path = self.output_dir / "probe_contours.xlsx"

        # Write contacts
        logger.info(f"Writing contacts to {contacts_path}")
        with pd.ExcelWriter(contacts_path, engine="openpyxl") as writer:
            for probe_name, df in contacts_dict.items():
                # Limit sheet name to 31 characters
                sheet_name = probe_name[:31]
                df.to_excel(writer, sheet_name=sheet_name)

        # Write contours (expanded for contacts first, then extended to full shank length)
        # Order matters: expand for contacts BEFORE extending to shank length
        # because buffer() on elongated shapes creates complex polygons that break
        logger.info(f"Writing contours to {contours_path}")
        with pd.ExcelWriter(contours_path, engine="openpyxl") as writer:
            for probe_name, df in contours_dict.items():
                sheet_name = probe_name[:31]
                contacts_df = contacts_dict.get(probe_name, pd.DataFrame())

                # Check if this is a multi-shank probe
                is_multi_shank = self._is_multi_shank_probe(contacts_df)

                if is_multi_shank:
                    # For multi-shank probes: DON'T use buffer expansion
                    # Use special extension that extends each shank individually
                    # Pass contacts_df to account for contact widths in contour bounds
                    # Pass probe_name to look up tip_length from database
                    extended_df = self._extend_multishank_contour(df, contacts_df, probe_name)
                    logger.debug(f"{probe_name}: Multi-shank probe, extending shanks individually")
                else:
                    # For single-shank probes: expand for contacts, then extend
                    # Pass contacts_df to center the tip on contacts
                    expanded_df = self._expand_contour_for_contacts(df, contacts_df)
                    extended_df = self._extend_contour_to_shank_length(expanded_df, contacts_df)

                # Update the dict with extended contours (for sanity checks)
                contours_dict[probe_name] = extended_df

                extended_df.to_excel(writer, sheet_name=sheet_name, index=False)

    def _detect_shank_ids_from_positions(
        self, x_positions: List[float], shank_spacing_threshold: float = 100.0
    ) -> List[str]:
        """
        Detect shank IDs from contact X positions by clustering.

        Contacts on the same shank have similar X values (within electrode spacing).
        Different shanks are separated by larger gaps (shank spacing, typically 150-250um).

        Args:
            x_positions: List of X coordinates for contacts
            shank_spacing_threshold: Minimum gap between shanks (default 100um)

        Returns:
            List of shank IDs (as strings "0", "1", "2", ...) for each contact
        """
        import numpy as np

        if len(x_positions) == 0:
            return []

        # Find unique X values and sort them
        unique_x = sorted(set(x_positions))

        # Group unique X values into shank clusters
        shank_clusters = []
        current_cluster = [unique_x[0]]

        for x in unique_x[1:]:
            # Check if this X is part of current shank or a new shank
            if x - current_cluster[-1] < shank_spacing_threshold:
                current_cluster.append(x)
            else:
                shank_clusters.append(current_cluster)
                current_cluster = [x]
        shank_clusters.append(current_cluster)

        # Create mapping from X value to shank ID
        x_to_shank = {}
        for shank_idx, cluster in enumerate(shank_clusters):
            for x in cluster:
                x_to_shank[x] = str(shank_idx)

        # Assign shank ID to each contact
        shank_ids = [x_to_shank[x] for x in x_positions]

        logger.debug(
            f"Detected {len(shank_clusters)} shanks from {len(x_positions)} contacts"
        )

        return shank_ids

    def _get_connector_type_from_folder(self, folder_name: str) -> Optional[str]:
        """
        Extract connector type from folder name.
        Examples:
            ASSY-325D-H7 -> ASSY-325D
            ASSY-156-E-1 -> ASSY-156
            ASSY-77-H5 -> ASSY-77
        """
        parts = folder_name.split("-")
        if len(parts) >= 2 and parts[0] == "ASSY":
            # Connector is ASSY-XXX (first two parts)
            return f"{parts[0]}-{parts[1]}"
        return None

    def _get_expected_shank_count(self, probe_name: str) -> Optional[int]:
        """
        Get expected shank count from database for a probe.

        The database has multiple rows for the same probe type (e.g., E-1) with
        different electrode counts. Each row has boolean columns indicating which
        connector types support that configuration (e.g., ASSY-77, ASSY-156).

        To find the correct row:
        1. Extract connector type from probe name (e.g., ASSY-156)
        2. Extract probe type from probe name (e.g., E-1)
        3. Find the row where part matches AND the connector column is TRUE
        """
        df = pd.read_csv(self.database_path)

        probe_type = self._get_probe_type_from_folder(probe_name)
        connector_type = self._get_connector_type_from_folder(probe_name)

        # For 325D probes, look for the "double" variant
        if "325D" in probe_name:
            search_name = f"{probe_type}double"
            connector_col = "ASSY-325D"
        else:
            search_name = probe_type
            connector_col = connector_type

        # Method 1: Match by probe type AND connector column
        if connector_col and connector_col in df.columns:
            for _, row in df.iterrows():
                if row["part"] == search_name:
                    # Check if this connector supports this probe configuration
                    connector_val = row.get(connector_col, False)
                    # Handle various TRUE representations
                    if connector_val in [True, "TRUE", "True", "true", 1, "1"]:
                        shank_count = row.get("shanks_n")
                        if pd.notna(shank_count):
                            logger.debug(
                                f"{probe_name}: Found shank count {int(shank_count)} "
                                f"for {search_name} with connector {connector_col}"
                            )
                            return int(shank_count)

        # Method 2: Fallback - find first matching probe type (legacy behavior)
        for _, row in df.iterrows():
            if row["part"] == search_name:
                shank_count = row.get("shanks_n")
                if pd.notna(shank_count):
                    logger.debug(
                        f"{probe_name}: Fallback to first match for {search_name}, "
                        f"shank count {int(shank_count)}"
                    )
                    return int(shank_count)

        # Method 3: Try exact probe type match without "double" suffix
        for _, row in df.iterrows():
            if row["part"] == probe_type:
                shank_count = row.get("shanks_n")
                if pd.notna(shank_count):
                    return int(shank_count)

        return None

    def write_sanity_checks(
        self, contacts_dict: Dict, metadata_list: List[Dict], contours_dict: Dict = None
    ) -> None:
        """
        Write sanity check files for ALL probes using probeinterface.

        For each probe:
        1. Check for non-unique contact IDs (CRITICAL - must be unique)
        2. Detect shanks from X position clusters
        3. Validate shank count against database
        4. Create probeinterface Probe objects with extended contours
        5. Save as probeinterface JSON format
        6. Plot using probeinterface plotting functions

        Args:
            contacts_dict: Dict of probe name -> contacts DataFrame
            metadata_list: List of metadata dicts for each probe
            contours_dict: Dict of probe name -> extended contours DataFrame (optional)
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import probeinterface as pi
        from probeinterface.plotting import plot_probe

        sanity_dir = self.output_dir / "sanity_checks"
        sanity_dir.mkdir(exist_ok=True)

        # Process ALL probes, not just dual-sided (but log dual-sided ones separately)
        dual_sided = [m for m in metadata_list if m["is_dual_sided"]]
        logger.info(f"Writing sanity checks for {len(metadata_list)} probes to {sanity_dir}")
        logger.info(f"  ({len(dual_sided)} are dual-sided)")

        validation_errors = []

        for metadata in metadata_list:
            probe_name = metadata["name"]
            if probe_name not in contacts_dict:
                continue

            df = contacts_dict[probe_name].reset_index()
            is_dual = metadata["is_dual_sided"]

            # Use extended contours from contours_dict if available, else fall back to raw
            extended_contour = None
            if contours_dict and probe_name in contours_dict:
                contour_df = contours_dict[probe_name]
                if not contour_df.empty and "x" in contour_df.columns and "y" in contour_df.columns:
                    extended_contour = contour_df[["x", "y"]].values.tolist()

            # Fall back to raw contour if no extended contour available
            raw_contour = extended_contour if extended_contour else metadata.get("raw_contour_2d", [])

            # 1. Check for non-unique contact IDs (CRITICAL)
            contact_ids = df["contact_ids"].tolist()
            unique_ids = set(contact_ids)
            has_duplicates = len(contact_ids) != len(unique_ids)

            if has_duplicates:
                counts = Counter(contact_ids)
                duplicates = {k: v for k, v in counts.items() if v > 1}
                error_msg = f"{probe_name}: NON-UNIQUE contact IDs: {duplicates}"
                logger.error(f"  CRITICAL: {error_msg}")
                validation_errors.append(error_msg)
            else:
                logger.info(f"  {probe_name}: All {len(contact_ids)} contact IDs unique")

            # 2. Detect shanks from X positions
            x_positions = df["x"].tolist()
            detected_shank_ids = self._detect_shank_ids_from_positions(x_positions)
            detected_shank_count = len(set(detected_shank_ids))

            # 3. Validate shank count against database
            expected_shank_count = self._get_expected_shank_count(probe_name)
            if expected_shank_count is not None:
                if detected_shank_count != expected_shank_count:
                    warn_msg = (
                        f"{probe_name}: Detected {detected_shank_count} shanks, "
                        f"expected {expected_shank_count} from database"
                    )
                    logger.warning(f"  {warn_msg}")
                else:
                    logger.info(
                        f"  {probe_name}: Shank count validated "
                        f"({detected_shank_count} shanks)"
                    )

            # 4. Create probeinterface Probe object with native contact_sides support
            # Uses PR #382 feature: contact_sides parameter in set_contacts()
            probegroup = pi.ProbeGroup()

            # Create probe with 2D coordinates for visualization
            # For both dual-sided and single-sided probes: x, y (y is vertical position)
            # z is the depth/side indicator for dual-sided, or shank thickness for single-sided
            positions = np.column_stack([
                df["x"].values,
                df["y"].values
            ])

            # Build shape_params as list of dicts (one per contact)
            if "width" in df.columns:
                widths = df["width"].values
            else:
                widths = [12] * len(df)
            if "height" in df.columns:
                heights = df["height"].values
            else:
                heights = [12] * len(df)
            shape_params = [
                {"width": float(w), "height": float(h)}
                for w, h in zip(widths, heights)
            ]

            if "contact_shapes" in df.columns:
                shapes = df["contact_shapes"].values
            else:
                shapes = ["rect"] * len(df)

            # Detect shank IDs from X position clusters
            x_positions = df["x"].tolist()
            shank_ids = self._detect_shank_ids_from_positions(x_positions)

            probe = pi.Probe(ndim=2)

            # For dual-sided probes, use native contact_sides parameter (PR #382)
            if is_dual and "contact_sides" in df.columns:
                # Get contact_sides values, converting empty strings to None
                contact_sides_raw = df["contact_sides"].values
                contact_sides = np.array([
                    s if s in ("front", "back") else None
                    for s in contact_sides_raw
                ])

                probe.set_contacts(
                    positions=positions,
                    shapes=shapes,
                    shape_params=shape_params,
                    shank_ids=np.array(shank_ids),
                    contact_sides=contact_sides,  # Native dual-sided support from PR #382
                )
                logger.debug(f"  {probe_name}: Using native contact_sides parameter")
            else:
                probe.set_contacts(
                    positions=positions,
                    shapes=shapes,
                    shape_params=shape_params,
                    shank_ids=np.array(shank_ids),
                )

            # Set contact IDs
            probe.set_contact_ids(df["contact_ids"].values)

            # 5. Set contour - prefer JSON contour if available, fallback to auto-shape
            # JSON contours preserve special geometries (V-shaped tips in E-1/E-2)
            if raw_contour and len(raw_contour) > 2:
                # Use actual contour from source JSON (preserves V-shaped tips, etc.)
                contour_array = np.array(raw_contour)
                probe.set_planar_contour(contour_array)
            else:
                # Fallback to auto-generated per-shank contours
                probe.create_auto_shape(probe_type="tip")

            probegroup.add_probe(probe)

            # 6. Save as probeinterface JSON
            json_path = sanity_dir / f"{probe_name}_sanity.json"
            pi.write_probeinterface(str(json_path), probegroup)

            # 7. Plot using custom rendering (probeinterface doesn't handle concave contours)
            from matplotlib.patches import Polygon as MplPolygon

            png_path = sanity_dir / f"{probe_name}_sanity.png"

            if len(probegroup.probes) == 0:
                logger.warning(f"  {probe_name}: No probes to plot")
                continue

            probe = probegroup.probes[0]

            # For dual-sided probes, use probeinterface's side parameter (PR #382)
            # to create separate plots for front and back
            if is_dual and "contact_sides" in df.columns:
                fig, axes = plt.subplots(1, 2, figsize=(20, 12))

                for i, side_name in enumerate(["front", "back"]):
                    ax = axes[i]
                    # Draw contour manually FIRST (handles concave polygons correctly)
                    if probe.probe_planar_contour is not None:
                        contour = probe.probe_planar_contour
                        poly = MplPolygon(contour, closed=True, facecolor="lightgreen",
                                          edgecolor="green", alpha=0.5, linewidth=1, zorder=1)
                        ax.add_patch(poly)
                    # Draw contacts using probeinterface with side filter (PR #382)
                    plot_probe(probe, ax=ax, with_contact_id=False, side=side_name,
                               probe_shape_kwargs={"facecolor": "none", "edgecolor": "none", "alpha": 0})
                    title = f"{probe_name} - {side_name.capitalize()}"
                    if has_duplicates:
                        title += "\n⚠️ DUPLICATE IDs"
                    ax.set_title(title)
            else:
                # Single-sided probes: single plot
                fig, ax = plt.subplots(figsize=(10, 12))
                # Draw contour manually FIRST (handles concave polygons correctly)
                if probe.probe_planar_contour is not None:
                    contour = probe.probe_planar_contour
                    poly = MplPolygon(contour, closed=True, facecolor="lightgreen",
                                      edgecolor="green", alpha=0.5, linewidth=1, zorder=1)
                    ax.add_patch(poly)
                # Draw contacts using probeinterface (hide its contour completely)
                plot_probe(probe, ax=ax, with_contact_id=False,
                           probe_shape_kwargs={"facecolor": "none", "edgecolor": "none", "alpha": 0})
                title = f"{probe_name}"
                if has_duplicates:
                    title += "\n⚠️ DUPLICATE IDs"
                ax.set_title(title)

            plt.tight_layout()
            plt.savefig(png_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            logger.info(f"  {probe_name}: JSON + PNG written")

        # Report validation summary
        if validation_errors:
            logger.error(f"\n{'='*60}")
            logger.error(f"VALIDATION ERRORS FOUND: {len(validation_errors)}")
            for err in validation_errors:
                logger.error(f"  - {err}")
            logger.error(f"{'='*60}")

    def generate_summary(self, metadata_list: List[Dict]) -> bool:
        """
        Generate and print summary statistics.

        Returns:
            True if all validations passed, False if there were errors
        """
        total = len(metadata_list)
        dual_sided = [m for m in metadata_list if m["is_dual_sided"]]
        regular = [m for m in metadata_list if not m["is_dual_sided"]]

        print("\n" + "=" * 60)
        print("EXTRACTION SUMMARY")
        print("=" * 60)
        print(f"Total probes processed: {total}")
        print(f"  - Regular probes: {len(regular)}")
        print(f"  - Dual-sided probes: {len(dual_sided)}")

        if dual_sided:
            print("\nDual-sided probes found:")
            for m in dual_sided:
                thickness = m['shank_thickness']
                thickness_str = f"{thickness} µm" if thickness else "MISSING"
                print(f"  - {m['name']}: {m['total_contacts']} contacts, "
                      f"thickness: {thickness_str}")

        print("\nOutput files:")
        print(f"  - {self.output_dir / 'probe_contacts.xlsx'}")
        print(f"  - {self.output_dir / 'probe_contours.xlsx'}")

        # Report validation errors
        if self.validation_errors:
            print("\n" + "!" * 60)
            print(f"VALIDATION ERRORS: {len(self.validation_errors)} probe(s) with issues")
            print("!" * 60)
            for error in self.validation_errors:
                print(f"  ERROR: {error}")
            print("!" * 60)
            logger.error(f"Extraction completed with {len(self.validation_errors)} validation errors")
            return False

        print("\nValidation: All contact IDs are unique")
        print("=" * 60)
        return True


def main() -> int:
    """
    Main entry point with command-line argument support.

    Returns:
        Exit code: 0 if successful, 1 if validation errors occurred
    """
    # Get automatic base path based on computer
    base_path, is_local = get_base_path()
    computer_name = socket.gethostname()

    # Set default paths based on whether using local files or Google Drive
    if is_local:
        # Local files - simple path structure
        default_library = base_path / "probe_library"
        default_database = base_path / "ProbesDataBase_2Dshanks_2025.csv"
        default_output = base_path
    elif computer_name in ["M-01699", "D-01643"]:
        # Google Drive - use full path structure
        default_library = base_path / "probe_maps" / "probe_library"
        default_database = base_path / "ProbesDataBase_2Dshanks_2025.csv"
        default_output = base_path / "probe_maps"
    else:
        # Unknown computer - assume files are in current directory
        default_library = base_path / "probe_library"
        default_database = base_path / "ProbesDataBase_2Dshanks_2025.csv"
        default_output = base_path

    parser = argparse.ArgumentParser(
        description="Extract probe contact and contour data with Z-coordinate calculation"
    )
    parser.add_argument(
        "--library",
        type=Path,
        default=default_library,
        help="Path to probe library folder containing JSON files"
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=default_database,
        help="Path to shank thickness database CSV file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Output directory for Excel files"
    )
    parser.add_argument(
        "--contact-id-excel",
        type=Path,
        default=None,
        help="Path to ProbeMaps_Final2025_SW.xlsx for contact ID mapping"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Use paths from arguments
    library_path = args.library
    database_path = args.database
    output_dir = args.output

    # Auto-detect contact ID Excel if not specified
    contact_id_excel = args.contact_id_excel
    if contact_id_excel is None:
        # Try to find it in common locations
        possible_paths = [
            output_dir / "ProbeMaps_Final2025_SW.xlsx",
            library_path.parent / "ProbeMaps_Final2025_SW.xlsx",
            Path.cwd() / "ProbeMaps_Final2025_SW.xlsx",
        ]
        for p in possible_paths:
            try:
                if p.exists():
                    contact_id_excel = p
                    logger.info(f"Auto-detected contact ID Excel: {contact_id_excel}")
                    break
            except PermissionError:
                logger.warning(
                    f"Permission denied accessing {p} - file may be open in Excel"
                )
                continue

    # Validate paths
    if not library_path.exists():
        raise FileNotFoundError(f"Library path not found: {library_path}")
    if not database_path.exists():
        raise FileNotFoundError(f"Database path not found: {database_path}")

    # Create extractor and run
    extractor = ProbeDataExtractor(
        library_path, database_path, output_dir, contact_id_excel
    )

    # Process all probes
    contacts, contours, metadata = extractor.process_all_probes()

    # Write to Excel
    extractor.write_excel_files(contacts, contours)

    # Write sanity check files for all probes (using extended contours)
    extractor.write_sanity_checks(contacts, metadata, contours)

    # Print summary and check for validation errors
    validation_passed = extractor.generate_summary(metadata)

    # Return exit code based on validation
    return 0 if validation_passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())