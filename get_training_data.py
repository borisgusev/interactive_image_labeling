import os
import pandas as pd
import numpy as np

dir_list = [
    "/mnt/external.data/TowbinLab/igheor/20231020_Ti2_10x_rpl22_AID_356_369_20231020_164026_547/",
    "/mnt/external.data/TowbinLab/igheor/20230914_Ti2_10x_vhp-1_338_344_186_160_20230914_172832_008/",
]

dir_of_interest = [
    {'raw': "analysis/ch1_raw_str/", 'mask': "analysis/ch1_il_strS/"},
    {'raw': "analysis/ch1_raw_str/", 'mask': "analysis/ch1_il_strS/"},
]

training_database_size = 1000

# combine list of directories and directories of interest

data_dir_list = [{'raw': os.path.join(dir_list[i], dir_of_interest[i]['raw']), 'mask': os.path.join(dir_list[i], dir_of_interest[i]['mask'])} for i in range(len(dir_list))]

def get_paired_mask_and_raw(raw_dir, mask_dir):
    raw_files = [os.path.join(raw_dir, file) for file in os.listdir(raw_dir) if file.endswith(".tiff")]
    mask_files = [os.path.join(mask_dir, file) for file in os.listdir(mask_dir) if file.endswith(".tiff")]
    raw_files.sort()
    mask_files.sort()
    assert len(raw_files) == len(mask_files)

    return [{'raw' : raw_files[i], 'mask' : mask_files[i]} for i in range(len(raw_files))]

# get a list of all the paired files
paired_files = []
for d in data_dir_list:
    print(d)
    paired_files.extend(get_paired_mask_and_raw(d['raw'], d['mask']))

paired_files_df = pd.DataFrame(paired_files)
# randomly select the number of files you want for training

training_database_df = paired_files_df.sample(n=training_database_size)

# save the training files
training_database_df.to_csv("training_database.csv", index=False)