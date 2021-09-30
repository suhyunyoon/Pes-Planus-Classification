from posix import listdir
import pandas as pd
import numpy as np
from pandas.io.parsers import read_csv
from tqdm import tqdm
import argparse
import shutil

import glob


def make_folder(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)


debug = False
dataset_root = "./data"

max_p = 0
target_dataset = dataset_root

# 1_Dynamic_Maximum_Image.txt



class RSDBFile:
    def __init__(self, rsdb_dir=None, name=None, debug=debug):
        self.debug = debug
        self.name = name
        self.file_dir = "{}/1_{}.txt".format(rsdb_dir, name)
        
        if not os.path.isfile(self.file_dir):
            self.file_dir = self.file_dir.replace("val", "train")
        
        
        # self.file_dir = glob.glob("{}/*{}*".format(rsdb_dir, name))[0]
        

        # print("file_dir = {}".format(self.file_dir))
        # self.file = open(self.file_dir, "r", encoding='euc-kr')
        self.file = None
        self.is_end = False
        self.cols = None
        self.rsdb_dir = rsdb_dir
    
    def to_csv_set(self):
        return


    def convert_to_csv_set():
        dataset_name = target_dataset
        dataset_dir = "./data/{}".format(dataset_name)
        df = pd.read_csv('./data/{}_annotation.csv'.format(dataset_name))
        
        rsdb_type = "Dynamic_Maximum_Image"

        n_dataset_dir = "./data/{}_{}".format(dataset_name, rsdb_type)
        make_folder(n_dataset_dir)

        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            uuid = row[0]
            rsdb_dir = "{}/{}/rsdb".format(dataset_dir, uuid)
            n_rsdb_dir = "{}/{}".format(n_dataset_dir, uuid)
            make_folder(n_rsdb_dir)

            convert_Dynamic_Maximum_Image(rsdb_dir=rsdb_dir, n_rsdb_dir=n_rsdb_dir)
        


    def open(self):
        self.file = open(self.file_dir, "r", encoding='euc-kr')
    
    def readlines(self, number):
        lines = []
        for i in range(number):
            lines.append(self.file.readline())
        return lines

    def read_until_enter(self, debug=debug):
        lines = []
        while True:
            l = self.readline()
            if debug:
                print(l)
            lines.append(l)

            if len(l.strip()) == 0:
                # print("l.len = {}".format(len(l)))
                return lines[:-1]  


    def readline(self):
        return self.file.readline()
    
    def read_foot_data_side(self):
        l = self.read_until_find("foot data")[-1]
        if l == "":
            return None

        side = l.split()[0].lower()
        return side

    def read_until_find(self, key, debug=debug):
        lines = []
        while True:
            l = self.readline()
            # print(l)
            if debug:
                print("{}, len = {}".format(l, len(l)))
            
            lines.append(l)

            if l == "":
                self.is_end = True
                return lines
            
            if key in l:
                return lines
    
    def split_after_find(self, key, debug=debug):
        self.read_until_find(key)
        l = self.readline()
        return l.split()

    def read_rectangle_position(self, debug=debug):
        self.read_until_find("rectangle position")
        lines = self.read_until_enter()
        return {
            'Rectangle bottom': int(lines[1]),
            'Rectangle left': int(lines[3]),
            'Rectangle width': int(lines[5]),
            'Rectangle height': int(lines[7])
        }

    def read_frame(self):
        lines = self.read_until_enter()
        frame = []
        for l in lines:
            row = []
            for s in l.split():
                row.append(float(s))
            frame.append(row)
        return frame

    def read_frame_lines(self):
        return self.readlines(self.frame_range)

    def close(self):
        self.file.close()



def convert_rsdb_all(target_dataset="test", data_root="./data"):
    dataset_dir = "{}/{}".format(data_root, target_dataset)
    for uuid in os.listdir(dataset_dir):
        rsdb_dir = "{}/rsdb/{}".format(dataset_dir, uuid)
        convert_rsdb(rsdb_dir=rsdb_dir)


rsdb_data_names = [
    "Center_of_Force_line",
    "Dynamic_Roll_Off",
    "Foot_Dimensions",
    "Static_Image",
    "Axis_angles",
    "Dynamic_Maximum_Image",
    "Pressures_and_Forces",
    "Contactpercentages",
    "Timing_Information",
]


uuid = '0afe6a5b4df8f2e9564a97d514674eb4c5f6bf5c712dac573b4d33b80ab7ecd8'
dataset_name = "test"


import os



def convert_Pressures_and_Forces(rsdb_dir=None):
    Pressures_and_Forces = RSDBFile(rsdb_dir=rsdb_dir, name="Pressures_and_Forces")
    Pressures_and_Forces.open()
    ct = {
        'left': 0,
        'right': 0,
    }
    
    f_cols = ['Toe 1','Toe 2-5','Meta 1','Meta 2','Meta 3','Meta 4','Meta 5','Midfoot Heel','Medial Heel','Lateral', 'Sum', 'Calibration Factor']
    p_cols = ['Toe 1','Toe 2-5','Meta 1','Meta 2','Meta 3','Meta 4','Meta 5','Midfoot Heel','Medial Heel','Lateral', 'Calibration Factor']

    pnf_data_name_set = [
        ('Force (N) graphs zones', 'Force_graphes_zones', f_cols),
        ('Pressure (N/cm2) graphs zones', 'Pressure_graphes_zones', p_cols),
        ('Force (N) graphs cursors', 'Force_graphes_zones', f_cols),
        ('Pressure (N/cm2) graphs cursors', 'Pressure_graphes_zones', p_cols),
    ]

    n_rsdb_dir = "{}/{}".format(rsdb_dir, Pressures_and_Forces.name)
    # shutil.rmtree(n_rsdb_dir)
    make_folder(n_rsdb_dir)

    while True:
        side = Pressures_and_Forces.read_foot_data_side()
        if side is None:
            break
        ct[side] += 1

        for data_name, sn, cols in pnf_data_name_set:
            # print("read {}".format(fn))
            
            Pressures_and_Forces.read_until_find(data_name)
            Pressures_and_Forces.readline()
            lines = Pressures_and_Forces.read_until_enter(debug=False)
            rows = []
            # foot_data['Pressures_and_Forces'][pnf_data_name] = []
            for l in lines:
                row = []
                for s in l.split():
                    # if debug:
                    #     print("col = {}, s = {}".format(col, s))
                    row.append(float(s))
                rows.append(row)
                # foot_data['Pressures_and_Forces'][pnf_data_name].append(row)
            data = np.array(rows)

            df = pd.DataFrame(data, columns=cols)
            
            file_name = '{}_{}_{}.csv'.format(sn, side, ct[side])

            file_dir = '{}/{}'.format(n_rsdb_dir, file_name)
            df.to_csv(file_dir)

            if data_name == 'Pressure (N/cm2) graphs cursors':
                break
        
        # foot_data['Dynamic_Maximum_Image'] = np.array(image)


# def make_dmi_combination(rsdb_dir):
#     dmi_list_map = convert_Dynamic_Maximum_Image(rsdb_dir)
#     for left_foot in dmi_list_map['left']:
#         for right_foot in dmi_list_map['right']:
#             pass
           

def convert_Dynamic_Maximum_Image(rsdb_dir=None, write_csv=False):
    Dynamic_Maximum_Image = RSDBFile(rsdb_dir=rsdb_dir, name='Dynamic_Maximum_Image')
    Dynamic_Maximum_Image.open()

    ct = {
        'left': 0,
        'right': 0
    }

    foot_data_list = []

    dmi_map = {
        'left': [],
        'right': [],
    }


    # max_p = 0

    while True:
        # print("read data {}".format(len(foot_data_list)))
        # Axis_angles
        # foot_data = {}
        l = Dynamic_Maximum_Image.read_until_find("Maximum pressure")[-1]
        
        if l == "":
            break

        # foot_data['side'] = l.split()[-1].lower()
        side = l.split()[-2].lower() 
        ct[side] += 1
        # print("read {} {}".format(side, ct[side]))

        # Dynamic_Maximum_Image.read_rectangle_position()
        Dynamic_Maximum_Image.read_until_enter()
        Dynamic_Maximum_Image.read_until_enter()
        lines = Dynamic_Maximum_Image.read_until_enter()[:-1]
        
        # foot_data['Dynamic_Maximum_Image'] = []

        rows = []
        for l in lines:
            # print(l)
            row = []
            # print(l)
            for s in l.split():
                p = float(s)
         
                row.append(p)
            rows.append(row)
        
        image = np.array(rows, dtype=np.float32)
        


        if write_csv:
            df = pd.DataFrame(image)
            file_name = '{}_{}_Dynamic_Maximum_Image.csv'.format(side, ct[side])
            n_rsdb_dir = "{}/{}".format(rsdb_dir, Dynamic_Maximum_Image.name)
            make_folder(n_rsdb_dir)
            file_dir = '{}/{}'.format(n_rsdb_dir, file_name)
            df.to_csv(file_dir, header=False, index=False)
        
        dmi_map[side].append(image)

        # foot_data['Dynamic_Maximum_Image'] = np.array(image)
    
    return dmi_map

    Dynamic_Maximum_Image.close()



def convert_all_Dynamic_Maximum_Image(target_dataset="test"):
    dataset_name = target_dataset
    dataset_dir = "./data/{}".format(dataset_name)
    df = pd.read_csv('./data/{}_annotation.csv'.format(dataset_name))
    
    rsdb_type = "Dynamic_Maximum_Image"

    n_dataset_dir = "./data/{}_{}".format(dataset_name, rsdb_type)
    make_folder(n_dataset_dir)

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        uuid = row[0]
        rsdb_dir = "{}/{}/rsdb".format(dataset_dir, uuid)
        n_rsdb_dir = "{}/{}".format(n_dataset_dir, uuid)
        make_folder(n_rsdb_dir)

        convert_Dynamic_Maximum_Image(rsdb_dir=rsdb_dir, n_rsdb_dir=n_rsdb_dir)





def convert_rsdb(rsdb_dir=None):
    convert_Dynamic_Maximum_Image(rsdb_dir, write_csv=False)
    # convert_Pressures_and_Forces(rsdb_dir)


def convert_rsdb_all(dataset_dir="./data/test"):
    for uuid in tqdm(os.listdir(dataset_dir)):
        rsdb_dir = "{}/{}/rsdb".format(dataset_dir, uuid)
        convert_rsdb(rsdb_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='rsdb parser')

    for ds_name in ['test', 'train']:
        print("convert dataset {}".format(ds_name))
        convert_rsdb_all(dataset_dir="./data/{}".format(ds_name))

