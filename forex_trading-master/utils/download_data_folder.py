import sys
sys.path.append('..')

import os
from pathlib import Path
from tqdm import tqdm
from utils import GCStorage
from constants import CREDENTIAL_PATH

storage = GCStorage('FX_Trading', 'integral_data', CREDENTIAL_PATH)

folder_name = 'joint_300_data'
download_path = Path(f'/sailhome/jingbo/CXR_RELATED/temp_store/{folder_name}')

fs = storage.list_files(folder_name)[1]

for fname in tqdm(fs):
    fpath = download_path  / fname
    cloud_path = Path(folder_name) / fname
    storage.download(fpath, cloud_path)