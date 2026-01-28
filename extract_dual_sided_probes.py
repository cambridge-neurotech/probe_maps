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
from geometry_database import ProbeGeometryDatabase
from contour_utils import reorder_contour_for_multi_shank, validate_contour


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

        # Load probe geometry database from XML
        xml_path = Path(__file__).parent / "references" / "probe_geometry_database.xml"
        if xml_path.exists():
            self.geometry_db: Optional[ProbeGeometryDatabase] = ProbeGeometryDatabase(xml_path)
            logger.info(f"Loaded probe geometry database from {xml_path}")
        else:
            self.geometry_db = None
            logger.warning(f"Probe geometry database not found at {xml_path}")

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
            ASSY-350-H15_2 -> H15 (underscore variant stripped for database lookup)
        """
        parts = folder_name.split("-")
        # For names like ASSY-325D-E-1, take last 2 parts if last is digit
        if len(parts) >= 2 and parts[-1].isdigit():
            return "-".join(parts[-2:])
        # For names like ASSY-325D-H7, take last part
        elif len(parts) >= 1:
            probe_type = parts[-1]
            # Handle underscore variants like H15_2 -> H15
            # This strips the variant suffix for database lookups
            if "_" in probe_type:
                probe_type = probe_type.split("_")[0]
            return probe_type
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

            # FIX: H15 source data has wrong inter-shank spacing (~150µm instead of 500µm)
            # Correct by shifting shank 1 electrodes to achieve proper 500µm spacing
            probe_type = self._get_probe_type_from_folder(probe_name)
            if probe_type == "H15" and contact_positions:
                # H15 specs: 500µm center-to-center spacing, 76µm shank width
                # Shank 0 electrodes: X = 0-23µm (center ~11.5µm)
                # Shank 1 electrodes: X = 150-173µm (center ~161.5µm) - WRONG
                # Need to shift shank 1 to center at 511.5µm (11.5 + 500)
                # Shift amount: 511.5 - 161.5 = 350µm
                h15_shift = 350.0
                corrected_positions = []
                for pos in contact_positions:
                    x, y = pos[0], pos[1]
                    # Shank 1 electrodes have X >= 100µm in source data
                    if x >= 100.0:
                        x += h15_shift
                    corrected_positions.append([x, y])
                contact_positions = corrected_positions
                logger.info(f"{probe_name}: Applied H15 spacing correction (+{h15_shift}µm to shank 1)")

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
                    missing_channels = [ch for ch in device_channels if ch not in channel_to_id]
                    if missing_channels:
                        logger.warning(
                            f"Missing Excel mappings for {probe_name}: channels {missing_channels}"
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
                    contact_ids = list(
                        range(contact_id_offset + 1, contact_id_offset + 1 + len(contact_positions))
                    )
            elif isinstance(contact_ids_raw, list):
                # For non-dual-sided probes, use JSON contact_ids if available
                if probe_idx > 0:
                    contact_ids = [int(cid) + contact_id_offset for cid in contact_ids_raw]
                else:
                    contact_ids = [
                        int(cid) if isinstance(cid, str) else cid for cid in contact_ids_raw
                    ]
            else:
                # Fallback: Generate sequential contact IDs
                contact_ids = list(
                    range(contact_id_offset + 1, contact_id_offset + 1 + len(contact_positions))
                )

            # Get contact shapes and dimensions
            # First, determine correct defaults from XML database based on probe type
            probe_type_for_dims = self._get_probe_type_from_folder(probe_name)
            if self.geometry_db:
                default_width, default_height, default_shape = (
                    self.geometry_db.get_contact_dimensions(probe_type_for_dims)
                )
            else:
                # Fallback: standard 12×12 square (not E-series 11×15)
                default_width, default_height, default_shape = 12.0, 12.0, "square"

            contact_shapes = probe_obj.get("contact_shapes", None)
            if not isinstance(contact_shapes, list):
                contact_shapes = [default_shape] * len(contact_positions)

            shape_params = probe_obj.get("contact_shape_params", [])
            if isinstance(shape_params, list) and len(shape_params) > 0:
                # List of dicts format: [{"width": 5, "height": 5}, ...]
                widths = [
                    p.get("width", default_width) if isinstance(p, dict) else default_width
                    for p in shape_params
                ]
                heights = [
                    p.get("height", default_height) if isinstance(p, dict) else default_height
                    for p in shape_params
                ]
            elif isinstance(shape_params, dict):
                # Dict with lists format: {"width": [5, 5, ...], "height": [5, 5, ...]}
                widths = shape_params.get("width", [default_width] * len(contact_positions))
                heights = shape_params.get("height", [default_height] * len(contact_positions))
            else:
                widths = [default_width] * len(contact_positions)
                heights = [default_height] * len(contact_positions)

            # Get shank IDs if available
            shank_ids_raw = probe_obj.get("shank_ids", None)
            if isinstance(shank_ids_raw, list):
                shank_ids = shank_ids_raw
            else:
                shank_ids = [""] * len(contact_positions)

            # Process each contact
            for i, pos in enumerate(contact_positions):
                x = pos[0]

                # Handle both 2D and 3D coordinates from JSON
                # For dual-sided probes: JSON has 3D [x, depth, height] where depth encodes side
                # For single-sided probes: JSON has 2D [x, y]
                if len(pos) >= 3:
                    # 3D coordinates: JSON uses [x, depth, height_along_shank]
                    # We need [x, height_along_shank, depth_centered_at_zero]
                    y = pos[2]  # Height along shank is in JSON's Z position
                    if is_dual and thickness is not None:
                        # Center depth around 0: front = +thickness/2, back = -thickness/2
                        z = (thickness / 2.0) if side == "front" else -(thickness / 2.0)
                    else:
                        # Single-sided: use JSON's Y (depth) value directly
                        z = pos[1]
                elif len(pos) >= 2:
                    # 2D coordinates: [x, y]
                    y = pos[1]
                    if is_dual and thickness is not None:
                        # For dual-sided probes, encode side in z
                        z = (thickness / 2.0) if side == "front" else -(thickness / 2.0)
                    else:
                        # For single-sided probes, use negative thickness as depth indicator
                        z = -thickness if thickness is not None else 0.0
                else:
                    # Single coordinate: just x position
                    y = 0.0
                    z = 0.0

                contact = {
                    "contact_ids": str(contact_ids[i]) if i < len(contact_ids) else str(i),
                    "x": x,
                    "y": y,
                    "z": z,
                    "contact_shapes": contact_shapes[i]
                    if i < len(contact_shapes)
                    else default_shape,
                    "width": widths[i] if i < len(widths) else default_width,
                    "height": heights[i] if i < len(heights) else default_height,
                    "shank_ids": str(shank_ids[i]) if i < len(shank_ids) else "",
                    "contact_sides": side if side else "",
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
                        contour_columns = [
                            "x",
                            "y",
                        ]  # Column names in output (extracting x and z from 3D)
                    else:
                        contours = contour_data

        # Reorder contour points for multi-shank probes to fix diagonal polygon fills
        # This ensures contours trace continuously around each shank
        if contours and all_contacts:
            contact_positions = [[c["x"], c["y"]] for c in all_contacts]
            contours = reorder_contour_for_multi_shank(contours, contact_positions)

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

    def _compute_electrode_envelope(
        self, positions: np.ndarray, margin: float = 8.0, y_tolerance: float = 10.0
    ) -> List[Tuple[float, float, float]]:
        """
        Compute a SMOOTH linear taper envelope for E-series electrodes.

        Instead of tracing each electrode level (which creates a zig-zag),
        this computes the overall width at top vs bottom and creates a
        linear taper that smoothly encompasses all electrodes.

        Args:
            positions: Nx2 array of (x, y) electrode positions
            margin: Margin to add on each side (µm)
            y_tolerance: Tolerance for grouping electrodes at same Y level (µm)

        Returns:
            List of (y, left_x, right_x) tuples for smooth linear taper
        """
        if len(positions) == 0:
            return []

        # Find overall bounds
        y_min = float(np.min(positions[:, 1]))
        y_max = float(np.max(positions[:, 1]))
        x_center = float(np.mean(positions[:, 0]))

        # Separate electrodes into top and bottom regions
        y_range = y_max - y_min
        top_threshold = y_max - y_range * 0.25  # Top 25% of electrodes
        bottom_threshold = y_min + y_range * 0.25  # Bottom 25% of electrodes

        top_positions = positions[positions[:, 1] >= top_threshold]
        bottom_positions = positions[positions[:, 1] <= bottom_threshold]

        # Calculate width at top and bottom
        if len(top_positions) > 0:
            top_x_min = float(np.min(top_positions[:, 0]))
            top_x_max = float(np.max(top_positions[:, 0]))
            top_width = top_x_max - top_x_min + 2 * margin
            top_center = (top_x_min + top_x_max) / 2
        else:
            top_width = float(np.max(positions[:, 0]) - np.min(positions[:, 0])) + 2 * margin
            top_center = x_center

        if len(bottom_positions) > 0:
            bottom_x_min = float(np.min(bottom_positions[:, 0]))
            bottom_x_max = float(np.max(bottom_positions[:, 0]))
            bottom_width = bottom_x_max - bottom_x_min + 2 * margin
            bottom_center = (bottom_x_min + bottom_x_max) / 2
        else:
            bottom_width = top_width * 0.5  # Assume 50% taper if no bottom electrodes
            bottom_center = x_center

        # Ensure minimum width for electrodes
        min_width = 2 * margin + 5  # At least margin on each side
        top_width = max(top_width, min_width)
        bottom_width = max(bottom_width, min_width)

        # Create smooth linear taper with just 3 key points: top, middle, bottom
        # This avoids the zig-zag while still following the V-pattern trend
        envelope = []

        # Top point
        envelope.append((y_max, top_center - top_width / 2, top_center + top_width / 2))

        # Middle point (50% of height)
        mid_y = (y_max + y_min) / 2
        mid_width = (top_width + bottom_width) / 2
        mid_center = (top_center + bottom_center) / 2
        envelope.append((mid_y, mid_center - mid_width / 2, mid_center + mid_width / 2))

        # Bottom point
        envelope.append((y_min, bottom_center - bottom_width / 2, bottom_center + bottom_width / 2))

        # Validate: ensure ALL electrodes fit within the envelope at each Y level
        # If any electrode is outside, expand locally
        for pos in positions:
            x, y = pos
            # Find interpolated width at this Y level
            if y >= mid_y:
                # Between top and mid
                t = (y_max - y) / (y_max - mid_y) if (y_max - mid_y) > 0 else 0
                interp_left = (1 - t) * envelope[0][1] + t * envelope[1][1]
                interp_right = (1 - t) * envelope[0][2] + t * envelope[1][2]
            else:
                # Between mid and bottom
                t = (mid_y - y) / (mid_y - y_min) if (mid_y - y_min) > 0 else 0
                interp_left = (1 - t) * envelope[1][1] + t * envelope[2][1]
                interp_right = (1 - t) * envelope[1][2] + t * envelope[2][2]

            # Check if electrode fits with margin
            contact_left = x - margin
            contact_right = x + margin

            if contact_left < interp_left or contact_right > interp_right:
                # Electrode would be outside - need to expand envelope
                # Find which envelope point is closest and expand it
                if y >= mid_y:
                    # Expand top or mid point
                    if abs(y - y_max) < abs(y - mid_y):
                        envelope[0] = (envelope[0][0], min(envelope[0][1], contact_left),
                                       max(envelope[0][2], contact_right))
                    else:
                        envelope[1] = (envelope[1][0], min(envelope[1][1], contact_left),
                                       max(envelope[1][2], contact_right))
                else:
                    # Expand mid or bottom point
                    if abs(y - mid_y) < abs(y - y_min):
                        envelope[1] = (envelope[1][0], min(envelope[1][1], contact_left),
                                       max(envelope[1][2], contact_right))
                    else:
                        envelope[2] = (envelope[2][0], min(envelope[2][1], contact_left),
                                       max(envelope[2][2], contact_right))

        return envelope

    def _generate_e_series_single_shank_contour(
        self, contour_df: pd.DataFrame, contacts_df: pd.DataFrame, probe_name: str
    ) -> pd.DataFrame:
        """
        Generate E-series contour with 3-STAGE SYMMETRIC TAPER.

        E-series contour shape (ref: E_probe_layout.png):
        - STAGE 1: VERTICAL walls from top to 3rd highest electrode
        - STAGE 2: WEAK TAPER from 3rd highest to 2nd lowest electrode
        - STAGE 3: STRONG TAPER from 2nd lowest electrode to tip (25µm)

        Args:
            contour_df: DataFrame with x, y columns and shank_length_mm
            contacts_df: DataFrame with contact positions
            probe_name: Probe name for logging

        Returns:
            E-series contour DataFrame with x, y columns
        """
        if contour_df.empty or contacts_df is None or contacts_df.empty:
            return contour_df

        # Get shank length
        if "shank_length_mm" not in contour_df.columns:
            return contour_df
        shank_length_mm = contour_df["shank_length_mm"].iloc[0]
        if pd.isna(shank_length_mm):
            return contour_df
        shank_length_um = shank_length_mm * 1000

        # Get contact dimensions (E-series: 11x15µm rectangular)
        contact_width = 11.0
        contact_height = 15.0
        if "width" in contacts_df.columns:
            contact_width = contacts_df["width"].max()
        if "height" in contacts_df.columns:
            contact_height = contacts_df["height"].max()

        # E-series tip specs
        tip_extension = 70  # µm below lowest electrode
        tip_width = 25  # µm (from XML database)

        # Margin for electrodes (half contact width + buffer)
        margin = contact_width / 2 + 5

        # Get contact positions
        positions = contacts_df[["x", "y"]].values

        # Find SHANK CENTER (for symmetric contour)
        x_min_global = float(np.min(positions[:, 0]))
        x_max_global = float(np.max(positions[:, 0]))
        shank_center = (x_min_global + x_max_global) / 2

        # Find unique Y levels (sorted low to high)
        unique_y = sorted(set(positions[:, 1]))
        n_levels = len(unique_y)

        # Stage boundaries based on electrode Y levels
        # Stage 1 ends at 3rd highest electrode
        # Stage 2 ends at LOWEST electrode (not 2nd lowest!) to ensure all electrodes fit
        if n_levels >= 4:
            stage1_end_y = unique_y[-3]  # 3rd from top (where taper starts)
            stage2_end_y = unique_y[0]   # LOWEST electrode (stage 3 is tip only)
        elif n_levels >= 2:
            # Fewer electrodes - split at middle
            mid_idx = n_levels // 2
            stage1_end_y = unique_y[mid_idx]
            stage2_end_y = unique_y[0]  # LOWEST electrode
        else:
            # Single electrode - no taper stages
            stage1_end_y = unique_y[0]
            stage2_end_y = unique_y[0]

        # Helper to get width at a Y level from electrode positions
        def get_half_width_at_y(y_level: float) -> float:
            """Get half-width at a specific Y level based on electrode positions."""
            # Find electrodes at or near this Y level
            tolerance = contact_height  # Within one contact height
            mask = np.abs(positions[:, 1] - y_level) <= tolerance
            if np.any(mask):
                x_at_level = positions[mask, 0]
                half_span = np.max(np.abs(x_at_level - shank_center))
                return half_span + margin
            # Fallback to global span
            return (x_max_global - x_min_global) / 2 + margin

        # Calculate widths at stage boundaries
        # Stage 1: Full width at top (widest)
        half_width_top = get_half_width_at_y(unique_y[-1])  # Top electrode level

        # Stage 2 start: Same as stage 1 end (width at 3rd highest electrode)
        half_width_stage2_start = get_half_width_at_y(stage1_end_y)

        # Stage 2 end: Width must accommodate ALL electrodes below stage 2
        # This includes the 2nd lowest AND lowest electrodes
        # Calculate based on the WIDEST electrode span in the bottom 2 levels
        bottom_levels = unique_y[:2] if n_levels >= 2 else unique_y
        bottom_mask = np.isin(positions[:, 1], bottom_levels)
        if np.any(bottom_mask):
            bottom_positions = positions[bottom_mask]
            bottom_half_span = np.max(np.abs(bottom_positions[:, 0] - shank_center))
            half_width_stage2_end = bottom_half_span + margin
        else:
            half_width_stage2_end = get_half_width_at_y(stage2_end_y)

        # Stage 3 end: Tip width
        half_width_tip = tip_width / 2

        # Key Y positions
        top_y = shank_length_um  # Top of shank
        y_min = float(np.min(positions[:, 1]))
        tip_y = y_min - contact_height / 2 - tip_extension  # Tip apex

        # Adjust Y positions for contour (bottom of electrodes at each level)
        stage1_end_contour_y = stage1_end_y - contact_height / 2
        stage2_end_contour_y = stage2_end_y - contact_height / 2

        # Build 3-STAGE CONTOUR (clockwise from top-left)
        contour_points = [
            # STAGE 1: Vertical (left side, top to stage1_end)
            (shank_center - half_width_top, top_y),                    # Top-left
            (shank_center - half_width_stage2_start, stage1_end_contour_y),  # Stage 1/2 boundary left

            # STAGE 2: Weak taper (left side continues down)
            (shank_center - half_width_stage2_end, stage2_end_contour_y),    # Stage 2/3 boundary left

            # STAGE 3: Strong taper to tip
            (shank_center, tip_y),                                     # Tip apex (centered)

            # STAGE 3: Strong taper (right side going up)
            (shank_center + half_width_stage2_end, stage2_end_contour_y),    # Stage 2/3 boundary right

            # STAGE 2: Weak taper (right side continues up)
            (shank_center + half_width_stage2_start, stage1_end_contour_y),  # Stage 1/2 boundary right

            # STAGE 1: Vertical (right side, stage1_end to top)
            (shank_center + half_width_top, top_y),                    # Top-right
        ]

        x_col = "x" if "x" in contour_df.columns else contour_df.columns[0]
        y_col = "y" if "y" in contour_df.columns else contour_df.columns[1]

        result_df = pd.DataFrame(contour_points, columns=[x_col, y_col])

        # Preserve shank_length_mm if present
        if "shank_length_mm" in contour_df.columns:
            result_df["shank_length_mm"] = shank_length_mm

        logger.debug(
            f"{probe_name}: E-series 3-STAGE contour - "
            f"top_width={half_width_top*2:.0f}µm, stage2_end_width={half_width_stage2_end*2:.0f}µm, "
            f"tip_width={tip_width}µm"
        )

        return result_df

    def _generate_h13_tapered_shank_contour(
        self, contour_df: pd.DataFrame, contacts_df: pd.DataFrame, probe_name: str
    ) -> pd.DataFrame:
        """
        Generate a full-shank tapered contour for H13 probes.

        H13 has a LINEAR TAPER from 140µm at top to 20µm at tip over 8mm shank length.
        This is different from E-series where only the tip tapers below electrodes.
        The entire shank tapers continuously, and electrode X positions vary with Y.

        Args:
            contour_df: DataFrame with x, y columns and shank_length_mm
            contacts_df: DataFrame with contact positions
            probe_name: Probe name for logging

        Returns:
            Tapered contour DataFrame with x, y columns
        """
        if contour_df.empty or contacts_df is None or contacts_df.empty:
            return contour_df

        # Get shank dimensions from XML database or use defaults
        if self.geometry_db:
            top_width = self.geometry_db.get_shank_width("H13", "top") or 140.0
            tip_width = self.geometry_db.get_shank_width("H13", "tip") or 20.0
            tip_extension = self.geometry_db.get_tip_extension("H13") or 20
        else:
            top_width = 140.0  # µm at top
            tip_width = 20.0  # µm at tip
            tip_extension = 20  # µm below lowest electrode

        # Get shank length
        if "shank_length_mm" in contour_df.columns:
            shank_length_mm = contour_df["shank_length_mm"].iloc[0]
            if pd.isna(shank_length_mm):
                shank_length_mm = 8.0  # Default H13 shank length
        else:
            shank_length_mm = 8.0

        shank_length_um = shank_length_mm * 1000

        # Get contact dimensions for margin calculation
        contact_height = 12.0  # Default
        if "height" in contacts_df.columns:
            contact_height = contacts_df["height"].max()

        # Get contact bounds and dimensions
        positions = contacts_df[["x", "y"]].values
        y_min = float(np.min(positions[:, 1]))
        x_min = float(np.min(positions[:, 0]))
        x_max = float(np.max(positions[:, 0]))

        # Get contact width for margin calculation
        contact_width = 12.0  # Default
        if "width" in contacts_df.columns:
            contact_width = contacts_df["width"].max()

        # Calculate minimum half-width needed to contain all electrodes
        margin = 5.0  # µm margin around contacts
        min_half_width_for_contacts = max(abs(x_min), abs(x_max)) + contact_width / 2 + margin

        # Calculate half-widths at top and tip
        half_width_top = top_width / 2  # 70µm
        half_width_tip = tip_width / 2  # 10µm

        # Tip apex position
        tip_apex_y = y_min - contact_height / 2 - tip_extension
        tip_apex_x = 0.0  # Center (H13 electrodes are centered)

        # Linear taper formula: at any Y, half_width = half_width_top - taper_rate * (shank_length - y)
        # But we need to ensure all electrodes fit - use two-stage taper if necessary
        # Stage 1: Linear taper from top to bottom electrode level (but not narrower than contacts need)
        # Stage 2: Continue taper to tip (below electrodes, can be as narrow as specified)

        taper_rate = (half_width_top - half_width_tip) / shank_length_um

        # Check if linear taper would be too narrow at electrode level
        half_width_at_bottom_electrode = half_width_top - taper_rate * (shank_length_um - y_min)

        contour_points = []

        # Top edge: start at top-left
        contour_points.append((-half_width_top, shank_length_um))

        # Left edge: taper down - sample points for smooth taper
        # Sample every 200µm for smooth curve
        y_samples = np.arange(shank_length_um, y_min - 10, -200)
        # Ensure we include the point just above the tip
        if y_samples[-1] > y_min - 10:
            y_samples = np.append(y_samples, y_min - 10)

        for y in y_samples:
            # Calculate width at this Y position
            distance_from_top = shank_length_um - y
            half_width_at_y = half_width_top - taper_rate * distance_from_top

            # Ensure width is sufficient to contain electrodes at this Y level
            # Only apply minimum at electrode Y levels
            if y >= y_min and half_width_at_y < min_half_width_for_contacts:
                half_width_at_y = min_half_width_for_contacts

            contour_points.append((-half_width_at_y, float(y)))

        # Tip apex
        contour_points.append((tip_apex_x, tip_apex_y))

        # Right edge: taper up (reverse order)
        for y in reversed(y_samples):
            distance_from_top = shank_length_um - y
            half_width_at_y = half_width_top - taper_rate * distance_from_top

            # Ensure width is sufficient to contain electrodes at this Y level
            if y >= y_min and half_width_at_y < min_half_width_for_contacts:
                half_width_at_y = min_half_width_for_contacts

            contour_points.append((half_width_at_y, float(y)))

        # Top edge: end at top-right
        contour_points.append((half_width_top, shank_length_um))

        # Get column names
        x_col = "x" if "x" in contour_df.columns else contour_df.columns[0]
        y_col = "y" if "y" in contour_df.columns else contour_df.columns[1]

        result_df = pd.DataFrame(contour_points, columns=[x_col, y_col])

        logger.debug(
            f"{probe_name}: Generated H13 tapered contour, "
            f"width {top_width:.0f}µm (top) → {tip_width:.0f}µm (tip)"
        )

        return result_df

    def _generate_m_series_contour(
        self, contour_df: pd.DataFrame, contacts_df: pd.DataFrame, probe_name: str
    ) -> pd.DataFrame:
        """
        Generate a 2-stage ASYMMETRIC contour for M-series (large animal) probes.

        M-series probes have:
        - Fixed 140µm shank width
        - 70µm shank thickness
        - 2-stage ASYMMETRIC taper:
          Stage 1: Both sides vertical until taper start Y level
          Stage 2: Left stays vertical, right tapers to tip (tip at LEFT edge)

        Taper start Y level:
        - M1, M3: 3rd row of sites from bottom
        - M2: 5th site from bottom

        Args:
            contour_df: DataFrame with x, y columns and shank_length_mm
            contacts_df: DataFrame with contact positions
            probe_name: Probe name for logging

        Returns:
            M-series contour DataFrame with x, y columns
        """
        if contour_df.empty or contacts_df is None or contacts_df.empty:
            return contour_df

        # M-series fixed dimensions
        shank_width = 140.0  # µm - fixed width for all M-series
        tip_extension = 150  # µm below lowest electrode (sharpened tip per reference images)

        # Determine probe type for taper start
        probe_type = self._get_probe_type_from_folder(probe_name) if probe_name else None
        is_m2 = probe_type and "M2" in probe_type

        # Get shank length
        if "shank_length_mm" in contour_df.columns:
            shank_length_mm = contour_df["shank_length_mm"].iloc[0]
            if pd.isna(shank_length_mm):
                shank_length_mm = 6.5  # Default M-series shank length
        else:
            shank_length_mm = 6.5

        shank_length_um = shank_length_mm * 1000

        # Get contact dimensions
        contact_height = 12.0  # Default
        if "height" in contacts_df.columns:
            contact_height = contacts_df["height"].max()

        # Get contact bounds and unique Y levels
        positions = contacts_df[["x", "y"]].values
        y_min = float(np.min(positions[:, 1]))
        unique_y = sorted(set(positions[:, 1]))  # Sorted low to high

        # Center the shank on the electrode center
        contact_center_x = (float(np.min(positions[:, 0])) + float(np.max(positions[:, 0]))) / 2

        # Calculate half-width
        half_width = shank_width / 2  # 70µm

        # Determine taper start Y level
        # M1, M3: 3rd row from bottom (index 2)
        # M2: 5th site from bottom (index 4)
        if is_m2:
            taper_start_idx = min(4, len(unique_y) - 1)
        else:
            taper_start_idx = min(2, len(unique_y) - 1)

        taper_start_y = unique_y[taper_start_idx] - contact_height / 2

        # Left edge stays at left throughout (ASYMMETRIC - tip at LEFT edge)
        left_edge_x = contact_center_x - half_width
        right_edge_x = contact_center_x + half_width

        # Tip apex position - at LEFT edge (asymmetric)
        tip_apex_y = y_min - contact_height / 2 - tip_extension
        tip_apex_x = left_edge_x  # Tip at LEFT edge

        # Build 2-stage ASYMMETRIC contour
        # Stage 1: Both sides vertical from top to taper_start_y
        # Stage 2: Left stays vertical, right tapers to meet at tip (LEFT edge)
        # IMPORTANT: Right side must stay wide enough to contain bottom electrodes

        # Calculate where right edge should be at bottom electrode level
        # to ensure no electrodes extrude from the taper
        bottom_electrode_y = y_min - contact_height / 2 - 5  # Small margin below bottom electrode

        # Get rightmost electrode X at the bottom levels (below taper start)
        bottom_mask = positions[:, 1] < unique_y[taper_start_idx]
        if np.any(bottom_mask):
            rightmost_bottom_x = float(np.max(positions[bottom_mask, 0]))
            # Right edge at bottom must contain the rightmost bottom electrode + margin
            right_at_bottom = max(rightmost_bottom_x + contact_height / 2 + 10, contact_center_x)
        else:
            right_at_bottom = right_edge_x

        contour_points = [
            (left_edge_x, shank_length_um),      # Top-left
            (left_edge_x, taper_start_y),        # Stage 1/2 boundary left (still vertical)
            (left_edge_x, bottom_electrode_y),   # Left at bottom electrode level (still vertical)
            (left_edge_x, tip_apex_y),           # Tip at LEFT edge
            (right_at_bottom, bottom_electrode_y),  # Right at bottom electrode level (wide enough for electrodes)
            (right_edge_x, taper_start_y),       # Stage 1/2 boundary right (taper starts here)
            (right_edge_x, shank_length_um),     # Top-right
        ]

        # Get column names
        x_col = "x" if "x" in contour_df.columns else contour_df.columns[0]
        y_col = "y" if "y" in contour_df.columns else contour_df.columns[1]

        result_df = pd.DataFrame(contour_points, columns=[x_col, y_col])

        # Preserve shank_length_mm if present
        if "shank_length_mm" in contour_df.columns:
            result_df["shank_length_mm"] = shank_length_mm

        logger.debug(
            f"{probe_name}: Generated M-series 2-stage asymmetric contour, "
            f"width={shank_width:.0f}µm, taper_start_y={taper_start_y:.0f}µm, "
            f"tip at LEFT edge ({tip_apex_x:.0f}, {tip_apex_y:.0f})"
        )

        return result_df

    def _extend_multishank_contour(
        self, contour_df: pd.DataFrame, contacts_df: pd.DataFrame = None, probe_name: str = None
    ) -> pd.DataFrame:
        """
        Generate a proper multi-shank contour from scratch based on contact positions.

        This completely regenerates the contour to ensure proper shank separation.
        Each shank is traced individually (down left, around tip, up right), with
        horizontal bridges at the top connecting shanks. No diagonal fills.

        IMPORTANT: This function uses tip_length from a database based on probe type
        to ensure consistent tip depths regardless of source JSON quality.

        Args:
            contour_df: DataFrame with x, y columns and shank_length_mm
            contacts_df: DataFrame with contact positions and dimensions
            probe_name: Probe name to extract probe type for tip_length lookup

        Returns:
            New contour DataFrame with proper shank separation
        """
        if contour_df.empty or "shank_length_mm" not in contour_df.columns:
            return contour_df[["x", "y"]].copy() if "x" in contour_df.columns else contour_df

        shank_length_mm = contour_df["shank_length_mm"].iloc[0]
        if pd.isna(shank_length_mm):
            return contour_df[["x", "y"]].copy() if "x" in contour_df.columns else contour_df

        shank_length_um = shank_length_mm * 1000
        merge_height = 200  # Height above shank length where merged section extends (µm)
        shank_top = shank_length_um  # Individual shanks extend to full shank length
        merge_top = shank_length_um + merge_height  # Merged section extends 200µm above shank length

        # IMPORTANT: Ensure shank_top is ABOVE all contacts
        # Some probes (e.g., H15_2) have contacts near the top, causing them to appear
        # in the shared area between shanks if connection height is too low
        if contacts_df is not None and "y" in contacts_df.columns and len(contacts_df) > 0:
            global_max_y = contacts_df["y"].max()
            min_shank_top = global_max_y + 50  # At least 50µm above highest contact
            if shank_top < min_shank_top:
                logger.debug(
                    f"Adjusting shank_top from {shank_top:.0f} to {min_shank_top:.0f} "
                    f"to stay above contacts (max_y={global_max_y:.0f})"
                )
                shank_top = min_shank_top
                # Recalculate merge_top to maintain 200µm above adjusted shank_top
                merge_top = shank_top + merge_height

        # Get contact half-width for expanding contour bounds
        contact_margin = 2.0  # Additional margin in µm
        half_width = contact_margin
        contact_height = 15.0
        if contacts_df is not None and "width" in contacts_df.columns:
            half_width = contacts_df["width"].max() / 2.0 + contact_margin
        if contacts_df is not None and "height" in contacts_df.columns:
            contact_height = contacts_df["height"].max()

        # Validate/expand width against XML database shank width
        if self.geometry_db and probe_name:
            probe_type = self._get_probe_type_from_folder(probe_name)
            expected_shank_width = self.geometry_db.get_shank_width(probe_type, "top")
            if expected_shank_width:
                expected_half_width = expected_shank_width / 2
                calculated_width = half_width * 2  # Convert back to full width
                if calculated_width < expected_shank_width:
                    # Calculated width is narrower than expected - use expected
                    half_width = expected_half_width + contact_margin
                    logger.debug(
                        f"{probe_name}: Expanded contour width from {calculated_width:.0f}µm "
                        f"to {expected_shank_width:.0f}µm (from XML)"
                    )
                elif calculated_width > expected_shank_width * 1.5:
                    # Calculated width is much wider than expected - warn
                    logger.warning(
                        f"{probe_name}: Calculated width {calculated_width:.0f}µm is much wider "
                        f"than expected {expected_shank_width:.0f}µm (from XML)"
                    )

        x_col = "x" if "x" in contour_df.columns else contour_df.columns[0]
        y_col = "y" if "y" in contour_df.columns else contour_df.columns[1]

        # Detect probe type early for shank detection strategy
        early_probe_type = None
        if probe_name:
            early_probe_type = self._get_probe_type_from_folder(probe_name)
        early_is_e_series = early_probe_type and early_probe_type.startswith("E")

        # Detect shanks - prefer using shank_ids from contacts if available
        # EXCEPTION: For E-series, ALWAYS use X-gap detection because shank_ids may be
        # assigned per-electrode-column instead of per-physical-shank, causing narrow contours
        shanks = []
        if contacts_df is not None and "x" in contacts_df.columns and len(contacts_df) > 0:
            # Method 1: Use shank_ids if available (most accurate)
            # Skip for E-series which needs X-gap detection to properly group both electrode columns
            if "shank_ids" in contacts_df.columns and not early_is_e_series:
                unique_shank_ids = contacts_df["shank_ids"].dropna().unique()
                # Filter out empty strings
                unique_shank_ids = [s for s in unique_shank_ids if s != ""]
                if len(unique_shank_ids) > 1:
                    for shank_id in sorted(unique_shank_ids):
                        shank_contacts = contacts_df[contacts_df["shank_ids"] == shank_id]
                        if len(shank_contacts) > 0:
                            left_x = shank_contacts["x"].min()
                            right_x = shank_contacts["x"].max()
                            tip_x = (left_x + right_x) / 2
                            min_contact_y = shank_contacts["y"].min()
                            max_contact_y = shank_contacts["y"].max()
                            shanks.append(
                                {
                                    "left": left_x,
                                    "right": right_x,
                                    "tip_x": tip_x,
                                    "min_y": min_contact_y,
                                    "max_y": max_contact_y,
                                }
                            )
                    if early_is_e_series:
                        logger.debug(f"{probe_name}: E-series - skipping shank_ids, using X-gap detection")

            # Method 2: Fall back to X-gap detection if shank_ids not available or E-series
            if not shanks:
                contact_x_values = sorted(contacts_df["x"].unique())
                if len(contact_x_values) > 0:
                    x_diffs = np.diff(contact_x_values)
                    gap_threshold = 100  # Increased to avoid splitting E-series columns (70µm)
                    gap_indices = np.where(x_diffs > gap_threshold)[0]

                    start_idx = 0
                    for gap_idx in gap_indices:
                        shank_contacts_x = contact_x_values[start_idx : gap_idx + 1]
                        left_x = min(shank_contacts_x)
                        right_x = max(shank_contacts_x)
                        tip_x = (left_x + right_x) / 2
                        shank_contacts = contacts_df[
                            (contacts_df["x"] >= left_x) & (contacts_df["x"] <= right_x)
                        ]
                        min_contact_y = shank_contacts["y"].min() if len(shank_contacts) > 0 else 0
                        max_contact_y = shank_contacts["y"].max() if len(shank_contacts) > 0 else 0
                        shanks.append(
                            {
                                "left": left_x,
                                "right": right_x,
                                "tip_x": tip_x,
                                "min_y": min_contact_y,
                                "max_y": max_contact_y,
                            }
                        )
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
                        max_contact_y = shank_contacts["y"].max() if len(shank_contacts) > 0 else 0
                        shanks.append(
                            {
                                "left": left_x,
                                "right": right_x,
                                "tip_x": tip_x,
                                "min_y": min_contact_y,
                                "max_y": max_contact_y,
                            }
                        )

        if not shanks:
            return self._extend_contour_to_shank_length(contour_df, contacts_df)

        # Sort shanks by X position (left to right)
        shanks = sorted(shanks, key=lambda s: s["tip_x"])

        # Get tip length and width for this probe type
        # Priority: 1) XML database, 2) Fallback hardcoded dict
        tip_length = 50  # Default tip length
        tip_width = 0    # Default tip width (0 = sharp point)
        probe_type = None
        if probe_name:
            probe_type = self._get_probe_type_from_folder(probe_name)

        if probe_type:
            # Try XML database first (most accurate source)
            if self.geometry_db:
                xml_tip_length = self.geometry_db.get_tip_extension(probe_type)
                xml_tip_width = self.geometry_db.get_tip_width(probe_type)
                if xml_tip_length is not None:
                    tip_length = xml_tip_length
                if xml_tip_width is not None:
                    tip_width = xml_tip_width
                    logger.debug(
                        f"Probe {probe_name}: tip_length={tip_length}µm, tip_width={tip_width}µm (from XML)"
                    )
                else:
                    # Fallback to hardcoded dict (updated values from XML)
                    TIP_LENGTH_DB = {
                        "E": 70,
                        "E-1": 70,
                        "E-2": 70,
                        "P": 70,
                        "P-1": 70,
                        "P-2": 70,
                        "F": 50,
                        "Fb": 50,
                        "F8": 15,
                        "F8-0": 15,
                        "F8-1": 15,
                        "F8-2": 15,
                        "H10": 50,
                        "H10-1": 50,
                        "H10-2": 50,
                        "H6": 65,
                        "H7": 65,
                        "H2": 28,
                        "H3": 28,
                        "H5": 30,
                        "H8": 100,
                        "H9": 55,
                        "L1": 28,
                        "L2": 28,
                        "L3": 28,
                        "L13": 90,
                        "L14": 50,
                        "M1": 50,
                        "M2": 50,
                        "M3": 60,
                        "H1": 40,
                        "H4": 28,
                        "H12": 26,
                        "H13": 20,
                        "H14": 25,
                        "H15": 75,
                        "H16": 25,
                        "H20": 30,
                    }
                    if probe_type in TIP_LENGTH_DB:
                        tip_length = TIP_LENGTH_DB[probe_type]
                    else:
                        for key in sorted(TIP_LENGTH_DB.keys(), key=len, reverse=True):
                            if probe_type.startswith(key):
                                tip_length = TIP_LENGTH_DB[key]
                                break
                    logger.debug(f"Probe {probe_name}: tip_length={tip_length}µm (from fallback)")
            else:
                # No XML database - use fallback dict
                TIP_LENGTH_DB = {
                    "E": 70,
                    "E-1": 70,
                    "E-2": 70,
                    "P": 70,
                    "P-1": 70,
                    "P-2": 70,
                    "F": 50,
                    "Fb": 50,
                    "F8": 15,
                    "F8-0": 15,
                    "F8-1": 15,
                    "F8-2": 15,
                    "H10": 50,
                    "H10-1": 50,
                    "H10-2": 50,
                    "H6": 65,
                    "H7": 65,
                    "H2": 28,
                    "H3": 28,
                    "H5": 30,
                    "H8": 100,
                    "H9": 55,
                    "L1": 28,
                    "L2": 28,
                    "L3": 28,
                    "L13": 90,
                    "L14": 50,
                    "M1": 50,
                    "M2": 50,
                    "M3": 60,
                    "H1": 40,
                    "H4": 28,
                    "H12": 26,
                    "H13": 20,
                    "H14": 25,
                    "H15": 75,
                    "H16": 25,
                    "H20": 30,
                }
                if probe_type in TIP_LENGTH_DB:
                    tip_length = TIP_LENGTH_DB[probe_type]
                else:
                    for key in sorted(TIP_LENGTH_DB.keys(), key=len, reverse=True):
                        if probe_type.startswith(key):
                            tip_length = TIP_LENGTH_DB[key]
                            break
                logger.debug(f"Probe {probe_name}: tip_length={tip_length}µm (no XML database)")

        # Detect if this is an E-series probe (has tapered V-shaped shanks)
        is_e_series = probe_type and probe_type.startswith("E")

        # Detect if this is an L-series probe (ASYMMETRIC taper: left vertical, right tapers)
        is_l_series = probe_type and probe_type.startswith("L") and probe_type in ("L13", "L14")

        # Detect if this is an M-series probe (large animal, 140µm fixed shank width)
        is_m_series = probe_type and probe_type.startswith("M") and probe_type in (
            "M1", "M1v1", "M1v2", "M2", "M2v1", "M2v2", "M3"
        )
        m_series_shank_width = 140.0  # µm - fixed width for M-series

        # ============================================================
        # GENERATE MULTI-SHANK CONTOUR FROM SCRATCH
        # ============================================================
        # Strategy: Build a single polygon that traces around ALL shanks properly.
        # The contour starts at top-left, goes down around first shank's tip,
        # up to connection height, horizontal bridge to next shank, repeat,
        # then closes at top-right and back to start.
        #
        # For E-series probes: Create smooth V-tapered contours where the shank
        # width decreases from top to tip (straight diagonal lines from corners
        # to tip apex - no rectangular walls). SYMMETRIC taper on both sides.
        #
        # For L-series probes: ASYMMETRIC taper where the LEFT side is vertical
        # and the RIGHT side has two-stage taper. Tip is at left edge.
        #
        # For other probes: Rectangular walls with V-shaped tips only at bottom.
        # ============================================================

        new_points = []

        # Get contact dimensions for margin calculation (needed for global bounds)
        contact_width = 12.0  # Default
        if contacts_df is not None and "width" in contacts_df.columns:
            contact_width = contacts_df["width"].max()

        # Global bounds for the merged top section
        # Calculate per-shank width for first and last shanks
        first_shank = shanks[0]
        last_shank = shanks[-1]
        first_center = (first_shank["left"] + first_shank["right"]) / 2
        last_center = (last_shank["left"] + last_shank["right"]) / 2

        # Per-shank width for first shank
        first_span = first_shank["right"] - first_shank["left"]
        first_required_half = first_span / 2 + contact_width / 2 + contact_margin
        if is_m_series:
            first_half_width = m_series_shank_width / 2
        else:
            first_half_width = max(half_width, first_required_half)

        # Per-shank width for last shank
        last_span = last_shank["right"] - last_shank["left"]
        last_required_half = last_span / 2 + contact_width / 2 + contact_margin
        if is_m_series:
            last_half_width = m_series_shank_width / 2
        else:
            last_half_width = max(half_width, last_required_half)

        global_left = first_center - first_half_width
        global_right = last_center + last_half_width

        # Start at top-left corner (200µm above shank length)
        new_points.append((global_left, merge_top))

        # Process each shank: down left edge, around V-tip, up right edge
        for i, shank in enumerate(shanks):
            # Calculate per-shank width to ensure ALL electrodes fit
            # The shank bounds (left/right) are electrode CENTER positions
            electrode_center = (shank["left"] + shank["right"]) / 2
            shank_electrode_span = shank["right"] - shank["left"]

            # Required half-width = half electrode span + half contact width + margin
            # This ensures contact edges don't extrude
            required_half_width = shank_electrode_span / 2 + contact_width / 2 + contact_margin

            # M-series: Use fixed 140µm shank width (large animal probes)
            if is_m_series:
                per_shank_half_width = m_series_shank_width / 2
            else:
                # Use the larger of global half_width and per-shank required width
                per_shank_half_width = max(half_width, required_half_width)

            left_x = electrode_center - per_shank_half_width
            right_x = electrode_center + per_shank_half_width
            tip_x = shank["tip_x"]
            min_y = shank["min_y"]

            # Calculate tip apex Y
            tip_apex_y = min_y - contact_height / 2 - tip_length

            if is_e_series:
                # E-series: 3-STAGE SYMMETRIC TAPER
                # STAGE 1: VERTICAL from top to 3rd highest electrode
                # STAGE 2: WEAK TAPER from 3rd highest to 2nd lowest electrode
                # STAGE 3: STRONG TAPER from 2nd lowest to tip (25µm)
                # Reference: E_probe_layout.png

                # Get this shank's contacts for calculating taper stages
                shank_contacts = None
                if contacts_df is not None:
                    shank_contacts = contacts_df[
                        (contacts_df["x"] >= shank["left"] - 5)
                        & (contacts_df["x"] <= shank["right"] + 5)
                    ]

                shank_x_min = shank["left"]
                shank_x_max = shank["right"]
                shank_y_min = shank["min_y"]
                shank_center = (shank_x_min + shank_x_max) / 2

                # Margin for electrodes
                e_margin = contact_width / 2 + 5  # Half contact width + buffer

                # Calculate width at TOP - use FULL electrode span (both columns)
                half_width_top = (shank_x_max - shank_x_min) / 2 + e_margin

                # Tip specs
                tip_width_e = 25  # µm from XML
                half_width_tip = tip_width_e / 2
                sharp_tip_y = shank_y_min - contact_height / 2 - tip_length

                # Find stage boundaries from electrode Y positions
                if shank_contacts is not None and len(shank_contacts) > 0:
                    positions = shank_contacts[["x", "y"]].values
                    unique_y = sorted(set(positions[:, 1]))
                    n_levels = len(unique_y)

                    if n_levels >= 4:
                        stage1_end_y = unique_y[-3]  # 3rd from top
                        stage2_end_y = unique_y[0]   # LOWEST electrode (stage 3 is tip only)
                    elif n_levels >= 2:
                        mid_idx = n_levels // 2
                        stage1_end_y = unique_y[mid_idx]
                        stage2_end_y = unique_y[0]  # LOWEST electrode
                    else:
                        stage1_end_y = unique_y[0]
                        stage2_end_y = unique_y[0]

                    # Calculate widths at each boundary based on electrode positions
                    def get_half_width_at_y_level(y_level: float) -> float:
                        tolerance = contact_height
                        mask = np.abs(positions[:, 1] - y_level) <= tolerance
                        if np.any(mask):
                            x_at_level = positions[mask, 0]
                            half_span = np.max(np.abs(x_at_level - shank_center))
                            return half_span + e_margin
                        return half_width_top

                    half_width_stage2_start = get_half_width_at_y_level(stage1_end_y)

                    # Stage 2 end width must accommodate ALL electrodes in bottom 2 levels
                    bottom_levels = unique_y[:2] if n_levels >= 2 else unique_y
                    bottom_mask = np.isin(positions[:, 1], bottom_levels)
                    if np.any(bottom_mask):
                        bottom_positions = positions[bottom_mask]
                        bottom_half_span = np.max(np.abs(bottom_positions[:, 0] - shank_center))
                        half_width_stage2_end = bottom_half_span + e_margin
                    else:
                        half_width_stage2_end = get_half_width_at_y_level(stage2_end_y)
                else:
                    # Fallback: linear taper
                    stage1_end_y = shank_top - 100
                    stage2_end_y = shank_y_min + 50
                    half_width_stage2_start = half_width_top * 0.9
                    half_width_stage2_end = half_width_top * 0.5

                # Adjust Y for contour (bottom of electrodes)
                stage1_end_contour_y = stage1_end_y - contact_height / 2
                stage2_end_contour_y = stage2_end_y - contact_height / 2

                # E-series 3-STAGE CONTOUR: left side down, tip, right side up
                left_x_top = shank_center - half_width_top
                right_x_top = shank_center + half_width_top

                # STAGE 1: Vertical (left side, top to stage1_end)
                new_points.append((left_x_top, shank_top))
                new_points.append((shank_center - half_width_stage2_start, stage1_end_contour_y))

                # STAGE 2: Weak taper (left side continues down)
                new_points.append((shank_center - half_width_stage2_end, stage2_end_contour_y))

                # STAGE 3: Strong taper to tip
                new_points.append((shank_center, sharp_tip_y))  # Tip apex

                # STAGE 3: Strong taper (right side going up)
                new_points.append((shank_center + half_width_stage2_end, stage2_end_contour_y))

                # STAGE 2: Weak taper (right side continues up)
                new_points.append((shank_center + half_width_stage2_start, stage1_end_contour_y))

                # STAGE 1: Vertical (right side, stage1_end to top)
                new_points.append((right_x_top, shank_top))

                logger.debug(
                    f"{probe_name} shank {i}: E-series 3-STAGE contour - "
                    f"top_width={half_width_top*2:.0f}µm, stage2_end_width={half_width_stage2_end*2:.0f}µm, "
                    f"tip=({shank_center:.0f}, {sharp_tip_y:.0f})"
                )

            elif is_l_series:
                # ============================================================
                # L-series: 3-STAGE ASYMMETRIC TAPER (L13, L14)
                # Reference: L13.png, L14.png
                #
                # Stage 1: Both sides VERTICAL until top site Y level
                # Stage 2: LEFT stays vertical, RIGHT has weak taper (down to below bottom site)
                # Stage 3: BOTH sides taper strongly to tip (tip at LEFT edge)
                #
                #     |-----------|     STAGE 1: Both sides vertical (above top site)
                #     |           |
                #     |-----------|     Y = top site level
                #     |            \    STAGE 2: Left vertical, Right weak taper
                #     |             \   (from top site to below bottom site)
                #     |              \
                #     |--------------\  Y = stage2_end (below bottom site)
                #      \              \ STAGE 3: Both sides taper to tip
                #       \              \
                #        V              Tip at LEFT edge
                # ============================================================

                # Get this shank's contacts for calculating taper points
                shank_contacts = None
                if contacts_df is not None and "shank_ids" in contacts_df.columns:
                    unique_shank_ids = sorted(contacts_df["shank_ids"].dropna().unique())
                    unique_shank_ids = [s for s in unique_shank_ids if s != ""]
                    if i < len(unique_shank_ids):
                        shank_id = unique_shank_ids[i]
                        shank_contacts = contacts_df[contacts_df["shank_ids"] == shank_id]

                if shank_contacts is None or len(shank_contacts) == 0:
                    shank_contacts = contacts_df[
                        (contacts_df["x"] >= shank["left"] - 5)
                        & (contacts_df["x"] <= shank["right"] + 5)
                    ]

                # L-series margin
                l_margin = contact_width / 2 + 5

                if len(shank_contacts) > 0:
                    positions = shank_contacts[["x", "y"]].values
                    shank_y_min = float(np.min(positions[:, 1]))
                    shank_y_max = float(np.max(positions[:, 1]))

                    # L-series electrodes are on the LEFT side of the shank
                    # Left edge = leftmost electrode - margin
                    left_edge_x = float(np.min(positions[:, 0])) - l_margin

                    # Right edge = left edge + shank width (from geometry)
                    # L13: 76µm, L14: 50µm
                    if probe_type == "L13":
                        shank_width_um = 76.0
                    elif probe_type == "L14":
                        shank_width_um = 50.0
                    else:
                        shank_width_um = 76.0  # Default
                    right_edge_x = left_edge_x + shank_width_um

                    # Stage boundaries:
                    # Stage 1 ends at TOP electrode Y level
                    # Stage 2 ends below BOTTOM electrode Y level
                    stage1_end_y = shank_y_max - contact_height / 2  # Bottom of top electrode
                    stage2_end_y = shank_y_min - contact_height / 2 - 10  # 10µm below bottom electrode

                    # Calculate right edge position at stage 2 end
                    # Right side tapers from right_edge_x at stage1_end to a narrower width at stage2_end
                    # Weak taper: about 30-40% narrower
                    right_at_stage2_end = left_edge_x + shank_width_um * 0.6

                    # Tip position: at LEFT edge (asymmetric)
                    tip_y = shank_y_min - contact_height / 2 - tip_length
                    tip_x_left = left_edge_x  # Tip is at left edge

                else:
                    # Fallback values
                    left_edge_x = left_x
                    right_edge_x = right_x
                    stage1_end_y = shank_top - 100
                    stage2_end_y = min_y - 20
                    right_at_stage2_end = right_x - 30
                    tip_y = tip_apex_y
                    tip_x_left = left_x

                # L-series 3-STAGE ASYMMETRIC contour
                # LEFT side: Vertical in stages 1&2, then tapers in stage 3
                # RIGHT side: Vertical in stage 1, weak taper in stage 2, strong taper in stage 3
                new_points.append((left_edge_x, shank_top))              # Top left (stage 1 start)
                new_points.append((left_edge_x, stage1_end_y))           # Stage 1/2 boundary left (still vertical)
                new_points.append((left_edge_x, stage2_end_y))           # Stage 2/3 boundary left (still vertical)
                new_points.append((tip_x_left, tip_y))                   # Tip at LEFT edge
                new_points.append((right_at_stage2_end, stage2_end_y))   # Stage 2/3 boundary right (after weak taper)
                new_points.append((right_edge_x, stage1_end_y))          # Stage 1/2 boundary right (still full width)
                new_points.append((right_edge_x, shank_top))             # Top right (stage 1 start)

                logger.debug(
                    f"{probe_name} shank {i}: L-series 3-STAGE asymmetric contour - "
                    f"left={left_edge_x:.0f}, right={right_edge_x:.0f}, "
                    f"stage1_end_y={stage1_end_y:.0f}, stage2_end_y={stage2_end_y:.0f}, "
                    f"tip=({tip_x_left:.0f}, {tip_y:.0f})"
                )

            else:
                # Non-E/L-series: Rectangular walls with V-tip at bottom
                # All probes have tapered V-tips (no flat tips)

                # Down to connection height on left edge
                new_points.append((left_x, shank_top))
                # Continue down left edge to just above tip
                new_points.append((left_x, min_y - 10))
                # V-tip: straight diagonal to apex
                new_points.append((tip_x, tip_apex_y))
                # V-tip: straight diagonal up to right edge
                new_points.append((right_x, min_y - 10))
                # Up to connection height on right edge
                new_points.append((right_x, shank_top))

            # Horizontal bridge to next shank (if not the last shank)
            if i < len(shanks) - 1:
                next_shank = shanks[i + 1]
                next_center = (next_shank["left"] + next_shank["right"]) / 2
                # Calculate per-shank width for next shank too
                next_shank_span = next_shank["right"] - next_shank["left"]
                next_required_half_width = next_shank_span / 2 + contact_width / 2 + contact_margin
                next_per_shank_half_width = max(half_width, next_required_half_width)
                next_left_x = next_center - next_per_shank_half_width
                # Bridge at connection height (shank_top) - pure horizontal
                new_points.append((next_left_x, shank_top))

        # Close at top-right corner (200µm above shank length)
        new_points.append((global_right, merge_top))

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
            f"Generated multi-shank contour: {len(shanks)} shanks, "
            f"individual shanks to {shank_top:.0f} µm, merge at {merge_top:.0f} µm "
            f"(200µm above shank length of {shank_length_um:.0f} µm)"
        )

        return result_df

    def _expand_contour_for_contacts(
        self, contour_df: pd.DataFrame, contacts_df: pd.DataFrame, buffer_margin: float = 1.0
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
            if hasattr(expanded_polygon, "exterior"):
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

                # Check probe type for special handling
                probe_type = self._get_probe_type_from_folder(probe_name)
                is_e_series = probe_type and probe_type.startswith("E")
                is_h13 = probe_type == "H13"
                is_m_series = probe_type and probe_type.startswith("M") and probe_type in (
                    "M1", "M1v1", "M1v2", "M2", "M2v1", "M2v2", "M3"
                )

                if is_multi_shank:
                    # For multi-shank probes: DON'T use buffer expansion
                    # Use special extension that extends each shank individually
                    # Pass contacts_df to account for contact widths in contour bounds
                    # Pass probe_name to look up tip_length from database
                    extended_df = self._extend_multishank_contour(df, contacts_df, probe_name)
                    logger.debug(f"{probe_name}: Multi-shank probe, extending shanks individually")
                elif is_e_series:
                    # E-series single-shank: Generate V-tapered contour from scratch
                    extended_df = self._generate_e_series_single_shank_contour(
                        df, contacts_df, probe_name
                    )
                    logger.debug(
                        f"{probe_name}: E-series single-shank, generating V-tapered contour"
                    )
                elif is_h13:
                    # H13: Full shank taper 140µm→20µm
                    extended_df = self._generate_h13_tapered_shank_contour(
                        df, contacts_df, probe_name
                    )
                    logger.info(f"{probe_name}: H13, generating full-shank tapered contour")
                elif is_m_series:
                    # M-series: Fixed 140µm shank width (large animal probes)
                    extended_df = self._generate_m_series_contour(
                        df, contacts_df, probe_name
                    )
                    logger.info(f"{probe_name}: M-series, generating 140µm wide contour")
                else:
                    # For single-shank probes: expand for contacts, then extend
                    # Pass contacts_df to center the tip on contacts
                    expanded_df = self._expand_contour_for_contacts(df, contacts_df)
                    extended_df = self._extend_contour_to_shank_length(expanded_df, contacts_df)

                # Update the dict with extended contours (for sanity checks)
                contours_dict[probe_name] = extended_df

                # Remove shank_length_mm column before writing (probeinterface expects only x, y)
                output_df = extended_df[["x", "y"]].copy() if "x" in extended_df.columns else extended_df
                output_df.to_excel(writer, sheet_name=sheet_name, index=False)

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

        logger.debug(f"Detected {len(shank_clusters)} shanks from {len(x_positions)} contacts")

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
            raw_contour = (
                extended_contour if extended_contour else metadata.get("raw_contour_2d", [])
            )

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
                        f"  {probe_name}: Shank count validated ({detected_shank_count} shanks)"
                    )

            # 4. Create probeinterface Probe object with native contact_sides support
            # Uses PR #382 feature: contact_sides parameter in set_contacts()
            probegroup = pi.ProbeGroup()

            # Create probe with coordinates
            # Use 2D coordinates (x, y) - probeinterface handles side differentiation via contact_sides
            x_coords = df["x"].values
            y_coords = df["y"].values
            positions = np.column_stack([x_coords, y_coords])

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
                {"width": float(w), "height": float(h)} for w, h in zip(widths, heights)
            ]

            if "contact_shapes" in df.columns:
                shapes = df["contact_shapes"].values
            else:
                shapes = ["rect"] * len(df)

            # Detect shank IDs from X position clusters
            x_positions = df["x"].tolist()
            shank_ids = self._detect_shank_ids_from_positions(x_positions)

            # Check for duplicate positions (can occur with dual-sided probes)
            unique_positions = np.unique(positions, axis=0)
            has_duplicates = len(unique_positions) < len(positions)

            # Always use 2D probe
            probe = pi.Probe(ndim=2)

            # For dual-sided probes with duplicate positions, skip probeinterface creation
            # These will be handled manually in output files
            if is_dual and has_duplicates:
                logger.warning(
                    f"  {probe_name}: Skipping probeinterface probe (dual-sided with "
                    f"{len(positions)} contacts at {len(unique_positions)} positions). "
                    f"Will use extracted contact data directly."
                )
                # Still save JSON and PNG, just not through probeinterface
                json_path = sanity_dir / f"{probe_name}_sanity.json"
                png_path = sanity_dir / f"{probe_name}_sanity.png"

                # Write extracted contact data as JSON
                contact_data = {
                    "probe_name": probe_name,
                    "contact_count": len(df),
                    "contacts": df.to_dict("records"),
                    "note": "Dual-sided probe with multiple contacts per (x, y) position",
                }
                with open(json_path, "w") as f:
                    json.dump(contact_data, f, indent=2)

                # Generate PNG manually for dual-sided probes
                from matplotlib.patches import Polygon as MplPolygon, Rectangle

                fig, axes = plt.subplots(1, 2, figsize=(20, 12))

                for i, side_name in enumerate(["front", "back"]):
                    ax = axes[i]

                    # Filter contacts by side (column is "contact_sides" not "side")
                    if "contact_sides" in df.columns:
                        side_df = df[df["contact_sides"] == side_name]
                    elif "side" in df.columns:
                        side_df = df[df["side"] == side_name]
                    else:
                        side_df = df

                    # Draw contour from extended contours
                    if raw_contour and len(raw_contour) > 2:
                        contour_array = np.array(raw_contour)
                        poly = MplPolygon(
                            contour_array,
                            closed=True,
                            facecolor="lightgreen",
                            edgecolor="green",
                            alpha=0.5,
                            linewidth=1,
                            zorder=1,
                        )
                        ax.add_patch(poly)

                    # Draw contacts as rectangles
                    for _, contact in side_df.iterrows():
                        cx, cy = contact["x"], contact["y"]
                        w = contact.get("width", 12)
                        h = contact.get("height", 12)
                        rect = Rectangle(
                            (cx - w / 2, cy - h / 2),
                            w,
                            h,
                            facecolor="orange",
                            edgecolor="darkred",
                            alpha=0.7,
                            linewidth=0.5,
                            zorder=2,
                        )
                        ax.add_patch(rect)

                    ax.set_aspect("equal")
                    ax.autoscale_view()
                    ax.set_xlabel("x (µm)")
                    ax.set_ylabel("y (µm)")
                    ax.set_title(f"{probe_name} - {side_name.capitalize()}")

                plt.tight_layout()
                plt.savefig(png_path, dpi=150, bbox_inches="tight")
                plt.close(fig)

                # Save zoomed version - create new figure with explicit limits
                y_coords = df["y"].values
                x_coords = df["x"].values
                y_min, y_max = y_coords.min(), y_coords.max()
                x_min, x_max = x_coords.min(), x_coords.max()
                x_margin = (x_max - x_min) * 0.1  # 10% margin
                xlim = (x_min - x_margin - 20, x_max + x_margin + 20)
                ylim = (y_min - 200, y_max + 200)

                fig_zoom, axes_zoom = plt.subplots(1, 2, figsize=(20, 12))
                for i, side_name in enumerate(["front", "back"]):
                    ax = axes_zoom[i]
                    side_df = df[df["contact_sides"] == side_name]

                    # Draw contour
                    if raw_contour and len(raw_contour) > 2:
                        contour_array = np.array(raw_contour)
                        poly = MplPolygon(
                            contour_array, closed=True,
                            facecolor="lightgreen", edgecolor="green",
                            alpha=0.5, linewidth=1, zorder=1,
                        )
                        ax.add_patch(poly)

                    # Draw contacts
                    for _, contact in side_df.iterrows():
                        cx, cy = contact["x"], contact["y"]
                        w = contact.get("width", 12)
                        h = contact.get("height", 12)
                        rect = Rectangle(
                            (cx - w / 2, cy - h / 2), w, h,
                            facecolor="orange", edgecolor="darkred",
                            alpha=0.7, linewidth=0.5, zorder=2,
                        )
                        ax.add_patch(rect)

                    # Set synchronized limits BEFORE autoscale
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    ax.set_xlabel("x (µm)")
                    ax.set_ylabel("y (µm)")
                    ax.set_title(f"{probe_name} - {side_name.capitalize()} (zoomed)")

                plt.tight_layout()
                zoomed_path = sanity_dir / f"{probe_name}_sanity_zoomed.png"
                plt.savefig(zoomed_path, dpi=150, bbox_inches="tight")
                plt.close(fig_zoom)

                logger.info(f"  {probe_name}: JSON + PNG written (direct contact data)")
                continue

            # For probes with unique positions, create probeinterface probe
            # For dual-sided probes, try to use contact_sides (if probeinterface version supports it)
            if is_dual and "contact_sides" in df.columns:
                # Get contact_sides values, converting empty strings to None
                contact_sides_raw = df["contact_sides"].values
                contact_sides = np.array(
                    [s if s in ("front", "back") else None for s in contact_sides_raw]
                )

                try:
                    probe.set_contacts(
                        positions=positions,
                        shapes=shapes,
                        shape_params=shape_params,
                        shank_ids=np.array(shank_ids),
                        contact_sides=contact_sides,  # Native dual-sided support (if available)
                    )
                except TypeError:
                    # contact_sides not supported in this probeinterface version
                    probe.set_contacts(
                        positions=positions,
                        shapes=shapes,
                        shape_params=shape_params,
                        shank_ids=np.array(shank_ids),
                    )
                logger.debug(f"  {probe_name}: Using contact_sides parameter")
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
                        poly = MplPolygon(
                            contour,
                            closed=True,
                            facecolor="lightgreen",
                            edgecolor="green",
                            alpha=0.5,
                            linewidth=1,
                            zorder=1,
                        )
                        ax.add_patch(poly)
                    # Draw contacts using probeinterface with side filter (PR #382)
                    plot_probe(
                        probe,
                        ax=ax,
                        with_contact_id=False,
                        side=side_name,
                        probe_shape_kwargs={"facecolor": "none", "edgecolor": "none", "alpha": 0},
                    )
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
                    poly = MplPolygon(
                        contour,
                        closed=True,
                        facecolor="lightgreen",
                        edgecolor="green",
                        alpha=0.5,
                        linewidth=1,
                        zorder=1,
                    )
                    ax.add_patch(poly)
                # Draw contacts using probeinterface (hide its contour completely)
                plot_probe(
                    probe,
                    ax=ax,
                    with_contact_id=False,
                    probe_shape_kwargs={"facecolor": "none", "edgecolor": "none", "alpha": 0},
                )
                title = f"{probe_name}"
                if has_duplicates:
                    title += "\n⚠️ DUPLICATE IDs"
                ax.set_title(title)

            plt.tight_layout()
            plt.savefig(png_path, dpi=150, bbox_inches="tight")

            # Save zoomed version focused on electrode sites
            # Y limits: 200µm below lowest site to 200µm above highest site
            # X limits: synchronized across subplots for dual-sided
            contact_positions = probe.contact_positions
            y_coords = contact_positions[:, 1]
            x_coords = contact_positions[:, 0]
            y_min, y_max = y_coords.min(), y_coords.max()
            x_min, x_max = x_coords.min(), x_coords.max()
            x_margin = (x_max - x_min) * 0.1  # 10% margin
            if is_dual and "contact_sides" in df.columns:
                # Dual-sided: set ylim and xlim on both axes (synchronized)
                for ax in axes:
                    ax.set_aspect("auto")  # Disable equal aspect for zoomed view
                    ax.set_ylim(y_min - 200, y_max + 200)
                    ax.set_xlim(x_min - x_margin - 20, x_max + x_margin + 20)
            else:
                # Single-sided: single axis
                ax.set_ylim(y_min - 200, y_max + 200)
            zoomed_path = sanity_dir / f"{probe_name}_sanity_zoomed.png"
            plt.savefig(zoomed_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            logger.info(f"  {probe_name}: JSON + PNG written")

        # Report validation summary
        if validation_errors:
            logger.error(f"\n{'=' * 60}")
            logger.error(f"VALIDATION ERRORS FOUND: {len(validation_errors)}")
            for err in validation_errors:
                logger.error(f"  - {err}")
            logger.error(f"{'=' * 60}")

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
                thickness = m["shank_thickness"]
                thickness_str = f"{thickness} µm" if thickness else "MISSING"
                print(
                    f"  - {m['name']}: {m['total_contacts']} contacts, thickness: {thickness_str}"
                )

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
            logger.error(
                f"Extraction completed with {len(self.validation_errors)} validation errors"
            )
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
        help="Path to probe library folder containing JSON files",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=default_database,
        help="Path to shank thickness database CSV file",
    )
    parser.add_argument(
        "--output", type=Path, default=default_output, help="Output directory for Excel files"
    )
    parser.add_argument(
        "--contact-id-excel",
        type=Path,
        default=None,
        help="Path to ProbeMaps_Final2025_SW.xlsx for contact ID mapping",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

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
                logger.warning(f"Permission denied accessing {p} - file may be open in Excel")
                continue

    # Validate paths
    if not library_path.exists():
        raise FileNotFoundError(f"Library path not found: {library_path}")
    if not database_path.exists():
        raise FileNotFoundError(f"Database path not found: {database_path}")

    # Create extractor and run
    extractor = ProbeDataExtractor(library_path, database_path, output_dir, contact_id_excel)

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
