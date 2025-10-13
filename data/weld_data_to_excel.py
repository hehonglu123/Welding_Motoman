"""
xlsx_writer.py

Write multi-list data into an XLSX workbook using openpyxl.

- Each top-level list becomes one sheet.
- Each sub-list is a "layer" (row), labeled layer0, layer1, ...
- Columns are positions 0, 1, 2, ... multiplied by `dlambda` (e.g., 0, 0.1, 0.2, ...)
- Elements can be scalars or tuples. By default tuples are written as strings "(x, y, z)".
  Set expand_tuples=True to spread tuples across multiple columns per position.
"""

from __future__ import annotations
import os
import math
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union
from pathlib import Path
import shutil, re
import numpy as np

from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import Font, Alignment

Scalar = Union[int, float, str]
CellValue = Union[Scalar, Tuple[Any, ...], List[Any]]


# -----------------------
# Helpers
# -----------------------

def safe_sheet_title(name: str) -> str:
    """Excel sheet titles must be <=31 chars and avoid: : \ / ? * [ ]"""
    bad = set(':\\/?*[]')
    cleaned = ''.join(ch for ch in name if ch not in bad)
    return cleaned[:31] if cleaned else "Sheet"

def format_col_header(idx: int, dlambda: float) -> str:
    """Format column header like 0, 0.1, 0.2 with minimal trailing zeros."""
    val = idx * dlambda
    # Use general format, then strip trailing zeros and dot.
    s = f"{val:.12g}"  # robust general formatting
    # Normalize -0.0 to 0
    if s in ("-0", "-0.0", "-0.00"):
        s = "0"
    return s

def is_iterable_nonstring(x: Any) -> bool:
    return isinstance(x, (list, tuple))

def tuple_arity(seq: Sequence[Any]) -> int:
    """Return max tuple/list length found in seq (ignoring scalars)."""
    arity = 0
    for v in seq:
        if is_iterable_nonstring(v):
            arity = max(arity, len(v))
    return arity

# -----------------------
# Core writer
# -----------------------

def write_list_to_sheet(
    wb: Workbook,
    data: List[List[CellValue]],
    sheet_name: str,
    dlambda: float = 1.0,
    expand_tuples: bool = False,
    tuple_labels: Sequence[str] | None = None,
) -> None:
    """
    Write a 2D ragged list to a worksheet.

    Parameters
    ----------
    wb : Workbook
        Target workbook.
    data : List[List[CellValue]]
        data[layer_idx][position_idx] -> value (scalar or tuple/list)
    sheet_name : str
        Name of the sheet to create.
    dlambda : float
        Spacing for column headers (position * dlambda).
    expand_tuples : bool
        If True, expand tuple/list elements across multiple adjacent columns.
        If False, write them as strings like "(x, y, z)".
    tuple_labels : Sequence[str] | None
        Optional labels for expanded tuple components, e.g., ("x","y","z").
        Only used when expand_tuples=True.
    """
    ws = wb.create_sheet(safe_sheet_title(sheet_name))

    if not data:
        ws["A1"] = "No data"
        return

    num_layers = len(data)
    max_len = max((len(row) for row in data), default=0)

    # Header row
    # A1 is blank; A2..A(N+1) are layer labels. Top row (row=1) B1.. = position headers.
    bold = Font(bold=True)
    center = Alignment(horizontal="center")

    ws.cell(row=1, column=1, value="")  # top-left blank

    if expand_tuples:
        # Determine arity across *all* elements encountered per position to allocate columns.
        # For simplicity, use the maximum tuple arity across the entire sheet.
        max_arity = 0
        for row in data:
            max_arity = max(max_arity, tuple_arity(row))
        if max_arity == 0:
            max_arity = 1  # treat as scalar

        # Build header with component suffixes if arity > 1
        for j in range(max_len):
            base = format_col_header(j, dlambda)
            if max_arity == 1:
                header = base
                ws.cell(row=1, column=2 + j, value=header).font = bold
                ws.cell(row=1, column=2 + j).alignment = center
            else:
                # allocate max_arity columns for this position j
                for k in range(max_arity):
                    comp = tuple_labels[k] if (tuple_labels and k < len(tuple_labels)) else f"c{k}"
                    header = f"{base}_{comp}"
                    col = 2 + j * max_arity + k
                    ws.cell(row=1, column=col, value=header).font = bold
                    ws.cell(row=1, column=col).alignment = center
    else:
        # Simple one-column-per-position header
        for j in range(max_len):
            header = format_col_header(j, dlambda)
            ws.cell(row=1, column=2 + j, value=header).font = bold
            ws.cell(row=1, column=2 + j).alignment = center

    # Write layers
    for i, row_vals in enumerate(data):
        # layer label in column A
        ws.cell(row=2 + i, column=1, value=f"layer{i}").font = bold

        if expand_tuples:
            # Write each element expanded
            max_arity = 0
            # re-compute local arity to map correctly (still write into fixed-width slots)
            # but the header reserved the global max_arity; we need it for indexing.
            for r in data:
                max_arity = max(max_arity, tuple_arity(r))
            if max_arity == 0:
                max_arity = 1

            for j, val in enumerate(row_vals):
                if is_iterable_nonstring(val):
                    comps = list(val)
                else:
                    comps = [val]
                # pad/truncate to max_arity
                if len(comps) < max_arity:
                    comps = comps + [None] * (max_arity - len(comps))
                else:
                    comps = comps[:max_arity]

                for k, v in enumerate(comps):
                    col = 2 + j * max_arity + k
                    ws.cell(row=2 + i, column=col, value=v)
        else:
            # Write values directly (tuples become strings)
            for j, val in enumerate(row_vals):
                ws.cell(row=2 + i, column=2 + j, value=val if not is_iterable_nonstring(val) else str(tuple(val)))

    # Freeze top-left headers and widen the first column
    ws.freeze_panes = "B2"
    ws.column_dimensions["A"].width = 14

    # Auto-size data columns (rough heuristic)
    if expand_tuples:
        max_arity = 0
        for row in data:
            max_arity = max(max_arity, tuple_arity(row))
        if max_arity == 0:
            max_arity = 1
        total_cols = 1 + max_len * max_arity
    else:
        total_cols = 1 + max_len

    for col_idx in range(1, total_cols + 1):
        letter = get_column_letter(col_idx)
        ws.column_dimensions[letter].width = max(ws.column_dimensions[letter].width or 0, 10)


def write_workbook_for_folder(
    folder: str,
    lists_by_name: Dict[str, List[List[CellValue]]],
    out_name: str | None = None,
    dlambda: float = 1.0,
    expand_tuples: bool = False,
    tuple_labels: Sequence[str] | None = None,
    overwrite: bool = True,
) -> str:
    """
    Create an XLSX workbook with one sheet per list.

    Parameters
    ----------
    folder : str
        Target folder to save the workbook.
    lists_by_name : dict
        Mapping like {"list_xyz": list_xyz, "list_feedrate": list_feedrate, ...}
    out_name : str | None
        Output filename. If None, use '<folder_basename>_export.xlsx'.
    dlambda : float
        Spacing for column headers.
    expand_tuples : bool
        Expand tuple/list elements into multiple columns.
    tuple_labels : Sequence[str] | None
        Labels for expanded tuple components (e.g., ("x","y","z")).
    overwrite : bool
        Overwrite existing file if present.

    Returns
    -------
    path : str
        Full path to the saved workbook.
    """
    os.makedirs(folder, exist_ok=True)
    if out_name is None:
        out_name = f"microstructure/process_parameters.xlsx"
    out_path = os.path.join(folder, out_name)

    if (not overwrite) and os.path.exists(out_path):
        raise FileExistsError(f"File exists: {out_path}")

    wb = Workbook()
    # Remove default sheet
    default = wb.active
    wb.remove(default)

    for name, data in lists_by_name.items():
        write_list_to_sheet(
            wb,
            data=data,
            sheet_name=name,
            dlambda=dlambda,
            expand_tuples=expand_tuples,
            tuple_labels=tuple_labels,
        )

    wb.save(out_path)
    return out_path


# -----------------------
# Your preprocessing hook
# -----------------------

def preprocess(folder: str, dlambda: float) -> Dict[str, List[List[CellValue]]]:
    """
    TODO: Replace with your actual preprocessing logic.

    For demonstration, we return example data consistent with your description.
    """
    # Example data (replace these with your computed lists)
    # list_xyz = [[[0, 0, 0], [1.01, 1, 1.21]], [[2.10, 2.2, 2.3]]]
    # list_feedrate = [[100, 150], [200]]
    # list_torchspeed = [[10.31, 7.5], [0.5]]

    list_xyz = []
    list_feedrate = []
    list_torchspeed = []

    # Identify layer folders in weld_data
    target_dir = os.path.join(folder, "weld_data")
    pattern = re.compile(r"layer[_\-]?(\d+)", re.IGNORECASE)
    layer_folders = []
    for d in os.listdir(target_dir):
        full_path = os.path.join(target_dir, d)
        if os.path.isdir(full_path):
            match = pattern.match(d)
            if match:
                layer_num = int(match.group(1))
                layer_folders.append((layer_num, full_path))
    # Sort by the numeric part
    layer_folders.sort(key=lambda x: x[0])
    # Extract only paths in order
    sorted_layer_paths = [p for _, p in layer_folders]

    for layer_path in sorted_layer_paths:
        print(f"Preprocessing layer folder: {layer_path}")
        # Example: read some data files and populate lists
        raw_data_dir = os.path.join(layer_path, "raw_data")
        proc_data_dir = os.path.join(layer_path, "proc_data")
        weld_relative_path = np.loadtxt(os.path.join(proc_data_dir, "weld_relative_exe.csv"), delimiter=',')
        weld_stamps = np.loadtxt(os.path.join(raw_data_dir, "weld_js_exe.csv"), delimiter=',')[:,0]
        weld_cmd = np.loadtxt(os.path.join(raw_data_dir, "weld_cmd.csv"), delimiter=',')

        # find the stamps while welding (after the last "weld on" command)
        stamps_diff_sorted = np.argsort(np.diff(weld_stamps))[::-1]
        for stamp_diff_id in stamps_diff_sorted:
            # make sure to find the time jump after the welding command
            if weld_stamps[stamp_diff_id] > weld_cmd[-1,0] and weld_stamps[stamp_diff_id] < weld_cmd[-1,0]+3:
                weld_split_id = stamp_diff_id
                break
        weld_stamps = weld_stamps[:weld_split_id+1]

        # the relative file paths and timestamps should match in length
        assert len(weld_relative_path) == len(weld_stamps), "Data length mismatch"

        # compute cumulative path length (lambda) along the weld path
        # path_lambda = np.cumsum(np.linalg.norm(np.diff(weld_relative_path, axis=0), axis=1))
        path_lambda = np.cumsum(np.fabs(np.diff(weld_relative_path[:,0])))
        path_lambda = np.insert(path_lambda, 0, 0)  # start at 0
        # Create a uniform sample of path lengths at intervals of dlambda to visualize the weld path
        path_lambda_sample = np.arange(0, path_lambda[-1], dlambda)  # dlambda=1.0

        # compute sampled positions
        sampled_xyz = np.empty((len(path_lambda_sample), 3))
        for i in range(3):
            sampled_xyz[:, i] = np.interp(path_lambda_sample, path_lambda, weld_relative_path[:, i])
            sampled_xyz[:, i] = np.round(sampled_xyz[:, i], decimals=1)
        # compute sampled torchspeed and feedrate
        sample_stamps = np.interp(path_lambda_sample, path_lambda, weld_stamps)
        sampled_torchspeed = []
        sampled_feedrate = []
        for stamp in sample_stamps:
            # find the index of largest timestamp in weld_cmd where timestamp <= stamp, 
            idx = np.searchsorted(weld_cmd[:, 0], stamp, side='right') - 1
            if idx >= 0:
                sampled_torchspeed.append(np.round(weld_cmd[idx, -2], decimals=2))
                sampled_feedrate.append(int(weld_cmd[idx, -1]))
            else:
                sampled_torchspeed.append(np.round(weld_cmd[0, -2], decimals=2))
                sampled_feedrate.append(int(weld_cmd[0, -1]))
        # append to lists
        list_xyz.append(sampled_xyz.tolist())
        list_feedrate.append(sampled_feedrate)
        list_torchspeed.append(sampled_torchspeed)

    return {
        "(x,y,z)": list_xyz,
        "feedrate": list_feedrate,
        "torchspeed": list_torchspeed,
    }


# -----------------------
# Batch over folders
# -----------------------

def export_folders(
    folders: List[str],
    parent_folder: str | None = None,
    dlambda: float = 1.0,
    expand_tuples: bool = False,
    tuple_labels: Sequence[str] | None = None,
) -> List[str]:
    """
    Run preprocessing for each folder and write an XLSX per folder.

    Returns list of saved workbook paths.
    """
    out_paths = []
    for folder in folders:
        if parent_folder:
            folder = os.path.join(parent_folder, folder)
        lists_by_name = preprocess(folder, dlambda=dlambda)
        out_path = write_workbook_for_folder(
            folder=folder,
            lists_by_name=lists_by_name,
            out_name=None,               # default: <folder>_export.xlsx
            dlambda=dlambda,
            expand_tuples=expand_tuples,
            tuple_labels=tuple_labels,
            overwrite=True,
        )
        out_paths.append(out_path)
    return out_paths


# -----------------------
# CLI example
# -----------------------
if __name__ == "__main__":
    # Example usage (replace with your own folder list and real preprocess)
    parent_folder = "wall_weld_test/"
    FOLDERS = ["2025_06_11_17_16_SSWL0_datacollection/"]
    # Example 1: simple (tuple written as "(x, y, z)")
    saved = export_folders(FOLDERS, parent_folder=parent_folder, dlambda=1, expand_tuples=False)
    print("Saved workbooks:", saved)
    print("======================")

    # Example 2: expand tuples into columns named x,y,z
    # saved = export_folders(FOLDERS, parent_folder=parent_folder, dlambda=0.1, expand_tuples=True, tuple_labels=("x", "y", "z"))
    # print("Saved workbooks:", saved)
