from posix import listdir
import pandas as pd
import numpy as np
from pandas.io.parsers import read_csv
from tqdm import tqdm
import argparse
import shutil




def make_folder(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)


debug = False
dataset_root = "./data"

max_p = 0
target_dataset = dataset_root

class RSDBFile:
    def __init__(self, rsdb_dir=None, name=None, debug=debug):
        self.debug = debug
        self.name = name
        self.file_dir = "{}/1_{}.txt".format(rsdb_dir, name)
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


class Center_of_Force_line_File(RSDBFile):
    def __init__(self, rsdb_dir=None):
        super.__init__(rsdb_dir=rsdb_dir, name="Center_of_Force_line")
    
    def to_csv_set(self):
        self.open()

        ct = {
            'left': 0,
            'right': 0
        }
        while True:
            l = self.read_until_find("Maximum pressure")[-1]
            
            if l == "":
                break

            side = l.split()[-2].lower() 
            ct[side] += 1
            
            self.read_until_enter()
            self.read_until_enter()
            lines = self.read_until_enter()[:-1]
            
            rows = []
            for l in lines:
                row = []
                for s in l.split():
                    row.append(float(s))
                rows.append(row)
            
            image = np.array(rows, dtype=np.float32)
            
            
            df = pd.DataFrame(image)
            
            file_name = '{}_{}_Dynamic_Maximum_Image.csv'.format(side, ct[side])
            
            file_dir = '{}/{}'.format(n_rsdb_dir, file_name)
            df.to_csv(file_dir, header=False, index=False)
            
            # foot_data['Dynamic_Maximum_Image'] = np.array(image)
        self.close()

class Dynamic_Roll_Off_File(RSDBFile):
    def __init__(self, rsdb_dir=None):
        super.__init__(rsdb_dir=rsdb_dir, name="Dynamic_Roll_Off")
    
    def to_csv_set(self):
        pass

class Foot_Dimensions_File(RSDBFile):
    def __init__(self, rsdb_dir=None):
        super.__init__(rsdb_dir=rsdb_dir, name="Foot_Dimensions")

    def to_csv_set(self):
        pass

class Static_Image_File(RSDBFile):
    def __init__(self, rsdb_dir=None):
        super.__init__(rsdb_dir=rsdb_dir, name="Static_Image")

    def to_csv_set(self):
        pass

class Axis_angles_File(RSDBFile):
    def __init__(self, rsdb_dir=None):
        super.__init__(rsdb_dir=rsdb_dir, name="Axis_angles")

    def to_csv_set(self):
        pass

class Dynamic_Maximum_Image_File(RSDBFile):
    def __init__(self, rsdb_dir=None):
        super.__init__(rsdb_dir=rsdb_dir, name="Dynamic_Maximum_Image")

    def to_csv_set(self):
        pass

class Pressures_and_Forces_File(RSDBFile):
    def __init__(self, rsdb_dir=None):
        super.__init__(rsdb_dir=rsdb_dir, name="Pressures_and_Forces")
        # self.n_rsdb_dir = ""

    def to_csv_set(self, n_rsdb_dir=None):
        if n_rsdb_dir is None:
            n_rsdb_dir = self.rsdb_dir
        # rsdb_dir = self.rsdb_dir

        self.open()
        ct = {
            'left': 0,
            'right': 0,
        }
        
        f_cols = ['Toe 1','Toe 2-5','Meta 1','Meta 2','Meta 3','Meta 4','Meta 5','Midfoot Heel','Medial Heel','Lateral', 'Sum', 'Calibration Factor']
        p_cols = ['Toe 1','Toe 2-5','Meta 1','Meta 2','Meta 3','Meta 4','Meta 5','Midfoot Heel','Medial Heel','Lateral', 'Calibration Factor']

        pnf_data_name_set = [
            ('Force (N) graphs zones', 'fgz', f_cols),
            ('Pressure (N/cm2) graphs zones', 'pgz', p_cols),
            ('Force (N) graphs cursors', 'fgc', f_cols),
            ('Pressure (N/cm2) graphs cursors', 'pgc', p_cols),
        ]


        while True:
            side = self.read_foot_data_side()
            if side is None:
                break
            ct[side] += 1

            for data_name, sn, cols in pnf_data_name_set:
                # print("read {}".format(fn))
                
                self.read_until_find(data_name)
                self.readline()
                lines = self.read_until_enter(debug=False)
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
                
                file_name = '{}_{}_pnf_{}.csv'.format(side, ct[side], sn)
                # file_name = "{}.csv".format()
                file_dir = '{}/{}'.format(n_rsdb_dir, file_name)
                df.to_csv(file_dir)

                if data_name == 'Pressure (N/cm2) graphs cursors':
                    break
                
                # self.read_until_find('Toe')
                # lines = self.read_until_enter()
            
            # foot_data['Dynamic_Maximum_Image'] = np.array(image)

class Contactpercentages_File(RSDBFile):
    def __init__(self, rsdb_dir=None):
        super.__init__(rsdb_dir=rsdb_dir, name="Contactpercentages")

    def to_csv_set(self):
        pass

class Timing_Information_File(RSDBFile):
    def __init__(self, rsdb_dir=None):
        super.__init__(rsdb_dir=rsdb_dir, name="Timing_Information")

    def to_csv_set(self):
        pass





def convert_rsdb(rsdb_dir=None):
    rsdb_files = []
    Axis_angles = Axis_angles_File(rsdb_dir)
    Center_of_Force = Center_of_Force_line_File(rsdb_dir)
    Contactpercentages = Contactpercentages_File(rsdb_dir)
    Dynamic_Maximum_Image = Dynamic_Maximum_Image_File(rsdb_dir)
    Dynamic_Roll_Off = Dynamic_Roll_Off_File(rsdb_dir)
    Foot_Dimensions = Foot_Dimensions_File(rsdb_dir)
    Pressures_and_Forces = Pressures_and_Forces_File(rsdb_dir)
    Static_Image = Static_Image_File(rsdb_dir)
    Timing_Information = Timing_Information_File(rsdb_dir)

    rsdb_files = [
        Axis_angles, 
        Center_of_Force, 
        Contactpercentages,
        Dynamic_Maximum_Image,
        Dynamic_Roll_Off,
        Foot_Dimensions,
        Pressures_and_Forces,
        Static_Image,
        Timing_Information,
        ]

    for rsdb_file in rsdb_files:
        rsdb_file.to_csv_set()

    pass


def convert_rsdb_all(target_dataset="test", data_root="./data"):
    dataset_dir = "{}/{}".format(data_root, target_dataset)
    for uuid in os.listdir(dataset_dir):
        rsdb_dir = "{}/rsdb/{}".format(dataset_dir, uuid)
        convert_rsdb(rsdb_dir=rsdb_dir)



    pass


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

# for rsdb_name in rsdb_data_names:
#     rsdb_files[rsdb_name] = RSDBFile(rsdb_dir=rsdb_dir, filename='1_{}.txt'.format(rsdb_name))



uuid = '0afe6a5b4df8f2e9564a97d514674eb4c5f6bf5c712dac573b4d33b80ab7ecd8'
dataset_name = "test"

# rsdb_dir = "./data/{}/{}/rsdb".format(uuid)

import os

    


def read_rsdb(rsdb_dir=None):
    Center_of_Force_line = RSDBFile(rsdb_dir=rsdb_dir, name='Center_of_Force_line')
    Dynamic_Roll_Off = RSDBFile(rsdb_dir=rsdb_dir, name='Dynamic_Roll_Off')
    Foot_Dimensions = RSDBFile(rsdb_dir=rsdb_dir, name='Foot_Dimensions')
    Static_Image = RSDBFile(rsdb_dir=rsdb_dir, name='Static_Image')
    Axis_angles = RSDBFile(rsdb_dir=rsdb_dir, name='Axis_angles')
    Dynamic_Maximum_Image = RSDBFile(rsdb_dir=rsdb_dir, name='Dynamic_Maximum_Image')
    Pressures_and_Forces = RSDBFile(rsdb_dir=rsdb_dir, name='Pressures_and_Forces')
    Contactpercentages = RSDBFile(rsdb_dir=rsdb_dir, name='Contactpercentages')
    Timing_Information = RSDBFile(rsdb_dir=rsdb_dir, name='Timing_Information')
    rsdb_files = [
        Center_of_Force_line,
        Dynamic_Roll_Off,
        Foot_Dimensions,
        Static_Image,
        Axis_angles,
        Dynamic_Maximum_Image,
        Pressures_and_Forces,
        Contactpercentages,
        Timing_Information
    ]


    def open_files():
        for rsdb_file in rsdb_files:
            rsdb_file.open()

    def close_files():
        for rsdb_file in rsdb_files:
            rsdb_file.close()


    open_files()

    rsdb_files = [
        Center_of_Force_line,
        Dynamic_Roll_Off,
        Foot_Dimensions,
        Static_Image,
        Axis_angles,
        Dynamic_Maximum_Image,
        Pressures_and_Forces,
        Contactpercentages,
        Timing_Information,
        ]



    fn = 0

    foot_data_list = []

        
    pnf_cols = []
    pnf_data_names = [
        'Force (N) graphs zones', 
        'Pressure (N/cm2) graphs zones',
        'Force (N) graphs cursors',
        'Pressure (N/cm2) graphs cursors'
        ]

    pnf_vector_names = ['Toe 1','Toe 2-5','Meta 1','Meta 2','Meta 3','Meta 4','Meta 5','Midfoot Heel','Medial Heel','Lateral Sum', 'Calibration Factor']
    for pnf_data_name in pnf_data_names:
        for pnf_vector_name in pnf_vector_names:
            pnf_cols.append("{}_{}".format(pnf_data_name, pnf_vector_name))
    Pressures_and_Forces.cols = pnf_cols

    ti_cols = []

    keys = ['Initial Foot Contact', 'Initial Metatarsal Contact', 'Initial Metatarsal Contact', 'Initial Forefoot Flat Contact']
    sub_keys = ['ms absolute', 'ms relative', '% relative']

    for k in keys:
        for sk in sub_keys:
            ti_cols.append("{}({})".format(k, sk))
    Timing_Information.cols = ti_cols


    while True:
        # print("read data {}".format(len(foot_data_list)))
        # Axis_angles
        foot_data = {}
        
        # print("read Axis_angles")
        l = Axis_angles.read_until_find("foot data")[-1]
        if l == "":
            break

        foot_data['side'] = l.split()[0].lower()
    

        foot_data['rectangle_position'] = Axis_angles.read_rectangle_position()
        
        sp = Axis_angles.split_after_find('Foot axis')

        foot_data['Axis_angles'] = {
            'Foot axis': float(sp[0]),
            'Subtalar joint flexibility': int(sp[1])
        }

        # print("read Center_of_Force_line")

        Center_of_Force_line.read_until_find("Frame")
        lines = Center_of_Force_line.read_until_enter()
        foot_data['Center_of_force_line'] = []
        for l in lines:
            sp = l.split()
            frame = {
                'Frame': int(sp[0]),
                'ms': float(sp[1]),
                'X (mm)': float(sp[2]),
                'Y (mm)': float(sp[3]),
                'Force (N)': float(sp[4])
            }
            foot_data['Center_of_force_line'].append(frame)

        # print("read Contactpercentages")
        
        sp = Contactpercentages.split_after_find('Surface')
        foot_data['Contactpercentages'] = {
            'Surface rearfoot': float(sp[0]),
            'Surface midfoot': float(sp[1]),
            'Surface forefoot': float(sp[2])
        }

        # print("read Dynamic_Maximum_Image")

        Dynamic_Maximum_Image.read_rectangle_position()

        lines = Dynamic_Maximum_Image.read_until_enter()[:-1]
        
        foot_data['Dynamic_Maximum_Image'] = []

        for l in lines:
            row = []
            # print(l)
            for s in l.split():
                row.append(float(s))
            
            foot_data['Dynamic_Maximum_Image'].append(row)
        
        # print("read Dyanmic Roll Off")
        
        Dynamic_Roll_Off.read_rectangle_position()

        foot_data['Dynamic_Roll_Off'] = []

        while True:
            l = Dynamic_Roll_Off.readline()
            fn = None
            if 'foot data' in l or len(l) == 0:
                break
            elif 'Frame' in l:
                fn = int(l.split()[1])
            lines = Dynamic_Roll_Off.read_until_enter()
            frame = []
            for l in lines:
                row = []
                for s in l.split():
                    row.append(float(s))
                frame.append(row)

            foot_data['Dynamic_Roll_Off'].append(frame)

        # print("read Foot Dimensions")
        
        Foot_Dimensions.read_rectangle_position()
        sp = Foot_Dimensions.split_after_find("Length")
        foot_data['Foot_Dimensions'] = {
            'Length (mm)': int(sp[0]),
            'Width (mm)': int(sp[1])
        }

        foot_data['Pressures_and_Forces'] = {}
        
        # print("read Pressures and Forces")
        
        Pressures_and_Forces.read_rectangle_position()

        Pressures_and_Forces.read_until_find('Toe')
        lines = Pressures_and_Forces.read_until_enter()
        # cols = ['Toe 1','Toe 2-5','Meta 1','Meta 2','Meta 3','Meta 4','Meta 5','Midfoot	Heel','Medial Heel','Lateral Sum', 'Calibration Factor']

        for pnf_data_name in pnf_data_names:
            data = []
            # foot_data['Pressures_and_Forces'][pnf_data_name] = []
            for l in lines:
                if debug:
                    print("l = {}".format(l))
                sp = l.split()
                row = []
                for i, s in enumerate(sp):
                    # if debug:
                    #     print("col = {}, s = {}".format(col, s))
                    row.append(float(s))
                data.append(row)
                # foot_data['Pressures_and_Forces'][pnf_data_name].append(row)
            foot_data['Pressures_and_Forces'][pnf_data_name] = np.array(data)

            if pnf_data_name != 'Pressure (N/cm2) graphs cursors':
                Pressures_and_Forces.read_until_find('Toe')
                lines = Pressures_and_Forces.read_until_enter()
        

        Timing_Information.read_until_find("Initial")
        lines = Timing_Information.read_until_enter()
        # cols = []

        # keys = ['Initial Foot Contact', 'Initial Metatarsal Contact', 'Initial Metatarsal Contact', 'Initial Forefoot Flat Contact']
        # sub_keys = ['ms absolute', 'ms relative', '% relative']

        # for k in keys:
        #     for sk in sub_keys:
        #         cols.append("{}({})".format(k, sk))

        
        foot_data[Timing_Information.name] = []
        
        for l in lines:
            sp = l.split()
            for s in sp:
                foot_data[Timing_Information.name].append(int(s))

        # for sk, l in list(zip(sub_keys, lines)):
        #     sp = l.split()
        #     for k, s in list(zip(keys, sp)):
        #         foot_data["{}({})".format(k, sk)] = int(s)

        foot_data_list.append(foot_data)


    print("foot_data_list.len = {}".format(len(foot_data_list)))

    close_files()


    rsdb_cols = []
    rsdb_df_map = {}

    # time series dataframe list
    foot_data_df_list = [] 

    rsdb_data_names = [

    # 3d time-2d_image data
        "Dynamic_Roll_Off", # 3d(time - image(rectangle size)) foot data list

    # 2d time-vector data
        "Pressures_and_Forces", # 2d(time - vector(11 * 4))  foot data list
        "Center_of_Force_line", # 2d(time - vector(5)) foot data list

    # 2d 2d_image
        "Dynamic_Maximum_Image", # 2d(image(rectangle size)) foot data list

    # 1d data
        "Contactpercentages", # 1d(vector(3)) foot data list
        "Timing_Information", # 1d(vector(5 * 3)) foot data list
        "Foot_Dimensions", # 1d(vector(2)) foot data list
        "Axis_angles", # 1d(vector(2)) foot data list

    # 2d image (no foot data list)
        "Static_Image", # 2d(image) static data
    ]

    ts_cols = []


    time_vector_data_files =[
        Pressures_and_Forces,
        Center_of_Force_line
        ]

    # l_ct = 0
    # r_ct = 0
    count = {
        'left': 0,
        'right': 0
    }

    force_vector_cols = ['Toe 1','Toe 2-5','Meta 1','Meta 2','Meta 3','Meta 4','Meta 5','Midfoot Heel','Medial Heel','Lateral', 'Sum', 'Calibration Factor']
    pressure_vector_names = ['Toe 1','Toe 2-5','Meta 1','Meta 2','Meta 3','Meta 4','Meta 5','Midfoot Heel','Medial Heel','Lateral', 'Calibration Factor']


    # for graph_type in ['zones', 'graphes']:


    import os

    n_rsdb_dir = rsdb_dir.replace("rsdb", "n_rsdb")

    make_folder(n_rsdb_dir)



    for i, foot_data in enumerate(tqdm(foot_data_list)):
        count[foot_data['side']] += 1

        # cols = Pressures_and_Forces.cols + cofl_data.cols
        # time_vector_df = pd.DataFrame(columns=cols)

        pnf_data_list = []
        shorten = {
            'Pressure (N/cm2) graphs zones': 'pgz',
            'Force (N) graphs cursors': 'fgc',
            'Pressure (N/cm2) graphs cursors': 'pgc',
            'Force (N) graphs zones': 'fgz'
        }

        # pnf_folder = "{}/{}

        for key, value in foot_data[Pressures_and_Forces.name].items():
            print("key = {}, value.shape = {}".format(key, value.shape))
            side = foot_data['side']
            
            file_name = "{}_{}_{}_foot_data_{}".format(Pressures_and_Forces.name, shorten[key], side, count[side])
            pnf_df = pd.DataFrame(value[:,:10], columns=force_vector_cols[:10])
            pnf_df.to_csv("{}/{}.csv".format(n_rsdb_dir, file_name))    

    
        # cofl_data = foot_data[Center_of_Force_line.name]



        
            
            
            


    # for i, row in test_df.iterrows():
    #     uuid = row['uuid']


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
        global max_p

        rows = []
        for l in lines:
            # print(l)
            row = []
            # print(l)
            for s in l.split():
                p = float(s)
                if max_p < p:
                    max_p = p

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
    
    print("max_p = {}".format(max_p))

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

class RSDBConverter:
    def __init__(self):
        pass

    def convert_all_Dynamic_Maximum_Image(self, target_dataset="test"):
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




# class Dynam

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='rsdb parser')

    for ds_name in ['test', 'train']:
        print("convert dataset {}".format(ds_name))
        convert_rsdb_all(dataset_dir="./data/{}".format(ds_name))
    # convert_all_Dynamic_Maximum_Image(target_dataset="test")
    # target_dataset="test"
    # dataset_name = target_dataset
    # dataset_dir = "./data/{}".format(dataset_name)
    
    # for uuid in listdir(dataset_dir):
    #     rsdb_dir = "{}/{}/rsdb".format(dataset_dir, uuid)
        
    #     rsdb_type = "Pressures_and_Forces"
        


    #     n_dataset_dir = "./data/{}_{}".format(dataset_name, rsdb_type)
    #     make_folder(n_dataset_dir)

    #     # n_rsdb_dir = "{}/{}".format(n_dataset_dir, uuid)
    #     # make_folder(n_rsdb_dir)
        

    #     convert_Pressures_and_Forces(rsdb_dir=rsdb_dir)



