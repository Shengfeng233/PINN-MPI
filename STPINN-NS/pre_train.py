"""
specify data for each process
子网络数据存储
"""
import numpy as np
import shutil
import os


def clear_dir(filepath):
    if os.path.exists(filepath):
        shutil.rmtree(filepath)
    os.makedirs(filepath, exist_ok=True)
    # del_list = os.listdir(filepath)
    # for f in del_list:
    #     file_path = os.path.join(filepath, f)
    #     if os.path.isfile(file_path):
    #         os.remove(file_path)
    return


def assign_sub_net(filename_raw_data, size_of_process):
    old_path = filename_raw_data
    target_path = './sub_data/sub_read/stack_unsteady_{:02d}.mat'
    for i in range(size_of_process):
        new_path = target_path.format(i)
        shutil.copy(old_path, new_path)
    return


if __name__ == "__main__":
    size_of_process = 40
    filename_select_data = './cylinder_Re3900_ke_25points_100snaps.mat'
    clear_dir(filepath='./sub_data/sub_read')
    clear_dir(filepath='./sub_data/sub_write')
    assign_sub_net(filename_select_data, size_of_process)
