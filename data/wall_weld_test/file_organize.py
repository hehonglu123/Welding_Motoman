from pathlib import Path
import shutil, re

# --------- CONFIGURE HERE ----------
RUN_DIRS = ['weld_fujiscan_2025_06_11_16_27_41/','weld_fujiscan_2025_06_11_16_52_36/','weld_fujiscan_2025_06_11_17_16_48/',\
            'weld_fujiscan_2025_06_11_17_49_27/','weld_fujiscan_2025_06_11_18_14_56/','weld_fujiscan_2025_06_12_17_33_24/',\
            'weld_fujiscan_2025_06_12_16_59_09/','weld_fujiscan_2025_06_12_15_33_03/','weld_fujiscan_2025_06_12_15_03_27/',\
            'weld_fujiscan_2025_07_09_14_52_42/','weld_fujiscan_2025_07_09_15_21_35/','weld_fujiscan_2025_07_09_16_16_40/',\
            'weld_fujicontrol_2025_08_13_14_17_58/','weld_fujicontrol_2025_08_13_14_57_52/','weld_fujicontrol_2025_08_14_11_19_59/',\
            'weld_fujicontrol_2025_08_14_12_04_14/','weld_fujicontrol_2025_09_22_17_10_07/','weld_fujicontrol_2025_09_24_12_06_39/',\
            'weld_fujiscan_2025_09_18_12_11_55/','2025_10_08_18_00_23_SSWL0_datacollection_fujicamconfig/','weld_2025_10_03_18_17_44',\
            'weld_2025_10_03_18_49_52'
            ]  # e.g. ['run_01','run_02',...]
NEW_RUN_DIRS_STR = ['datacollection','datacollection','datacollection',\
                    'datacollection','datacollection','datacollection',\
                    'datacollection','datacollection','datacollection',\
                    'datacollection','datacollection','datacollection',\
                    'RNNJacobiControl_StaticShape','Baseline','RNNJacobiControl_StaticShape',\
                    'RNNJacobiControl_AxeShape','LogLogControl_fujicamconfig','LogLogControl_fujicamconfig',\
                    'datacollection_fujicamconfig','datacollection_fujicamconfig','torchorient_Rx',\
                    'torchorient_Ry']  # e.g. ['experiment1','experiment2',...]
assert len(RUN_DIRS) == len(NEW_RUN_DIRS_STR), "RUN_DIRS and NEW_RUN_DIRS_STR must have the same length."

# Files inside each layerXX/baselayersXX that should go to raw_data; others in that folder go to proc_data
LAYER_RAW_FILENAMES = {'scan_exe.pickle','ir_recording.pickle','ir_stamps.csv','js_cmd.csv','scan_exe.pickle',\
                       'weld_cmd.csv','weld_js_exe.csv','welding.csv','current.csv','control_status_log.csv'}  # add more like {'scan_exe.pickle','joint_log.csv',...}

# Top-level files you want under run_X/weld_data (no subfolder)
WELD_DATA_TOP_FILES = {'fujicam.csv','flir.csv','fujicam_0905.csv','torch.csv','torch_calib.csv','Transz0_H.csv'}

# Microstructure top-level files -> microstructure/proc_data or /images
MICRO_PROC_FILES  = {'Porosity_Percentage.xlsx','SDAS_Calculation.xlsx'}
MICRO_IMAGE_FILES = {'layer_1_micro.jpg'}

# Media extensions that go to run_X/medias
MEDIA_EXTS = {'.mp3','.mp4','.jpg','.png','.JPG','.PNG'}
# -----------------------------------

layer_pat = re.compile(r'^(?:base)?layers?_?\d+', re.IGNORECASE)
ts_pat    = re.compile(r'(\d{4}_\d{2}_\d{2}_\d{2}_\d{2})(?:_\d{2})?')  # YYYY_MM_DD_HH_MM[_SS]

def safe_move(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        # print(f"Skip (exists): {dst.name}")
        return
    shutil.move(str(src), str(dst))
    # print(f"Moved: {src} -> {dst}")

def rename_run_dir(old: Path, new_str: str) -> Path:
    m = ts_pat.search(old.name)
    if not m:
        print(f"No timestamp found in '{old.name}', keeping name.")
        return old
    new_name = f"{m.group(1)}_WL0SS_{new_str}"
    new_path = old.with_name(new_name)
    if new_path.exists():
        print(f"Target '{new_path}' exists, keeping original name.")
        return old
    old.rename(new_path)
    print(f"Renamed: {old.name} -> {new_path.name}")
    return new_path

for old_name, new_str in zip(RUN_DIRS, NEW_RUN_DIRS_STR):
    root = Path(old_name)
    print(f"Processing folder: {root}")
    if not root.is_dir():
        print(f"Missing folder: {root} (skipping)")
        continue

    # 0. Rename run folder per rule
    root = rename_run_dir(root, new_str)

    weld_data_dir = root / 'weld_data'
    micro_dir     = root / 'microstructure'
    medias_dir    = root / 'medias'
    # create dirs
    weld_data_dir.mkdir(parents=True, exist_ok=True)
    micro_dir.mkdir(parents=True, exist_ok=True)
    medias_dir.mkdir(parents=True, exist_ok=True)

    # 1–2. Handle layerXX / baselayersXX folders into weld_data/{layer}/(raw_data|proc_data)
    for item in root.iterdir():
        if item.is_dir() and layer_pat.match(item.name):
            layer_dst = weld_data_dir / item.name
            (layer_dst / 'raw_data').mkdir(parents=True, exist_ok=True)
            (layer_dst / 'proc_data').mkdir(parents=True, exist_ok=True)
            for f in item.iterdir():
                if f.is_file():
                    sub = 'raw_data' if f.name in LAYER_RAW_FILENAMES else 'proc_data'
                    safe_move(f, layer_dst / sub / f.name)

    # 3–6. Handle top-level files in run_X
    for f in root.iterdir():
        if not f.is_file():
            continue
        if f.name in WELD_DATA_TOP_FILES:
            safe_move(f, weld_data_dir / f.name)
        elif f.name in MICRO_PROC_FILES:
            safe_move(f, micro_dir / 'proc_data' / f.name)
        elif f.name in MICRO_IMAGE_FILES:
            safe_move(f, micro_dir / 'images' / f.name)
        elif f.suffix in MEDIA_EXTS:
            safe_move(f, medias_dir / f.name)
        else:
            # 6. Leave everything else in run_X
            pass

    # remove the old empty layerXX/baselayersXX folders
    for item in root.iterdir():
        # check there are no files left
        if item.is_dir() and layer_pat.match(item.name) and not any(item.iterdir()):
            item.rmdir()
            # print(f"Removed empty folder: {item}")
        else:
            print(f"Left folder (not empty or not layer folder): {item}")
    
    print("================================\n")