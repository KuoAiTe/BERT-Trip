import os
import numpy as np
import pandas as pd
import glob
from pathlib import Path
from weeplaces import *
head = 1000
data_in_directory = '/home/aite/Desktop/dptrip/BST/md5/data/raw'
pwd = os.getcwd()
dataset = 'weeplaces'
checkin_file_name = 'checkins.csv'
relationship_file_name = 'friends.csv'
data_out_directory = 'clean'
final_dataset_columns = ['user_id', 'place_id', 'lat', 'lon', 'time', 'cat']
column_sep = '|'
time_series_sep = ','
trajectory_sep = '|||'
user_feature_columns = ['user_id']
time_series_columns = ['place_id', 'lat', 'lon', 'time', 'cat']

def create_dataset_by_poi():
    target_num_top_pois_list = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]#, 1600, 3200, 6400, 12800
    target_num_top_pois_list = [800]#, 1600, 3200, 6400, 12800
    for target_num_top_pois in target_num_top_pois_list:
        create_weeplaces_dataset(target_num_top_pois = target_num_top_pois, target_length = -1, min_length = 3)
def create_weeplaces_dataset(target_num_top_pois = 50, min_length = 3, target_length = -1):
    dataset = 'weeplaces'
    dataset_dir , data_checkin_path, data_relationship_path, final_dataset_path = get_dataset_info('weeplaces')
    processor = WeeplaceDataPreprocessor(dataset_dir, data_checkin_path, data_relationship_path, user_feature_columns, time_series_columns)
    file_name = processor.process(target_num_top_pois = target_num_top_pois, min_length = min_length, target_length = target_length)
    #create_pretrain_files('weeplaces', file_name, expected_size = 50000)
    return file_name
def create_weeplaces_dataset_downstream_task_1(dir, file_name):
    dataset = f'weeplaces'
    dataset_dir , data_checkin_path, data_relationship_path, final_dataset_path = get_dataset_info('weeplaces')
    processor = WeeplaceDataPreprocessor(dataset_dir, data_checkin_path, data_relationship_path, user_feature_columns, time_series_columns)
    #file_name = processor.generate_downstream_task(dir, 'weeplaces_poi_400_length_3-15-numtraj_17835')
    file_name = processor.count_trajectory(dataset_dir, 'weeplaces_poi_12800_length_3-15-numtraj_147287')
    #create_pretrain_files('weeplaces', file_name, expected_size = 50000)
    return file_name

def get_dataset_info(dataset):
    dataset_dir = f'{pwd}/{dataset}'
    data_checkin_path = f'{data_in_directory}/{dataset}/{checkin_file_name}'
    data_relationship_path = f'{data_in_directory}/{dataset}/{relationship_file_name}'
    final_dataset_path = f'{dataset_dir}/{dataset}_data.csv'
    return dataset_dir , data_checkin_path, data_relationship_path, final_dataset_path

create_weeplaces_dataset_downstream_task_1('weeplaces', 'weeplaces_poi_400_length_3-15-numtraj_17835')
def create_vocab_files(dataset, file_name):
    dataset_dir = f'{pwd}/{dataset}/{file_name}'
    file_path = f'{dataset_dir}/data.csv'
    vp = VocabPreprocessor()
    vp.create_vocab_files(dataset_dir, file_name, file_path)

def create_weeplaces_dataset_by_poi_and_length():
    target_num_top_pois_list = [100, 250, 500, 1000]
    target_length_list = [4, 6, 8, 10]
    for target_num_top_pois in target_num_top_pois_list:
        for target_length in target_length_list:
            filename = create_weeplaces_dataset(target_num_top_pois = target_num_top_pois, target_length = target_length)
            print(filename)
            #create_pretrain_files('weeplaces', f'weeplaces_poi_{target_num_top_pois}_length_5-15-numtraj_115701', expected_size = 1000000)

    #convert_dataset(name)

def create_pretrain_files(dataset, file_name, expected_size):
    dataset_dir = f'{pwd}/{dataset}/{file_name}'
    print(f'creating pretrained file ({expected_size}): {dataset_dir}')
    pp = PretrainFileProcessor()
    pp.process(dataset_dir, file_name, expected_size)

def create_weeplaces_dataset_for_selftrip():
    pp = SelfTripDataProcessor('weeplaces', 'weeplaces_poi_25_length_3-15-numtraj_765',)
    pp.gen_dist_metric()
    pp = SelfTripDataProcessor('weeplaces', 'weeplaces_poi_50_length_3-15-numtraj_2134',)
    pp.gen_dist_metric()
    pp = SelfTripDataProcessor('weeplaces', 'weeplaces_poi_100_length_3-15-numtraj_4497',)
    pp.gen_dist_metric()
    pp = SelfTripDataProcessor('weeplaces', 'weeplaces_poi_200_length_3-15-numtraj_7790',)
    pp.gen_dist_metric()
    pp = SelfTripDataProcessor('weeplaces', 'weeplaces_poi_400_length_3-15-numtraj_12288',)
    pp.gen_dist_metric()
def create_pretrain_file():
    create_pretrain_files('flicker', 'toro', expected_size = 50000)
    create_pretrain_files('flicker', 'glas', expected_size = 50000)
    create_pretrain_files('flicker', 'osaka', expected_size = 50000)
    create_pretrain_files('flicker', 'melb', expected_size = 50000)
    create_pretrain_files('flicker', 'edin', expected_size = 50000)
def show_data():
    pwd = os.getcwd()
    file_name = "data.csv"
    joined_files = os.path.join(f"{pwd}/*", file_name)
    joined_list = sorted(glob.glob(joined_files))
    #siamberttrip_df = pd.DataFrame()
    for file_path in joined_list:
        dataset = file_path.replace(pwd, '').replace(file_name, '')[1:-1]
        print(dataset)
        df = pd.read_csv(file_path, sep = '|', names = ['user','traj','ghash','cat','t1','t2','t3','timestamp'])
        trajs = df['traj'].values
        counter = dict()
        for traj in trajs:
            length = len(traj.split(','))
            if length not in counter:
                counter[length] = 0
            counter[length] += 1
        counter = dict(sorted(counter.items()))
        total = 0
        for k, v in counter.items():
            print(f'length: {k} = {v}')
            total += v
        print(f'total = {total}')
        #siamberttrip_df = pd.concat([siamberttrip_df, df], ignore_index=True)

#create_weeplaces_dataset_for_selftrip()

#show_data()
#exit()
create_dataset_by_poi()
#create_vocab_files('flicker', 'toro')
#create_vocab_files('weeplaces', 'weeplaces_poi_10000_length_3-15-numtraj_292009')
#create_pretrain_file()
#create_weeplaces_dataset()
#create_pretrain_files('weeplaces', 'weeplaces_poi_10000_length_5-15-numtraj_115701', expected_size = 1000000)
#create_weeplaces_dataset_downstream_task_1('weeplaces', 'weeplaces_poi_400_length_3-15-numtraj_17835')
#pp = CILPFormatProcessor('flicker', 'toro',)
"""
pp = SelfTripDatProcessor('flicker', 'osaka',)
pp.gen_dist_metric()
pp = SelfTripDatProcessor('flicker', 'glas',)
pp.gen_dist_metric()
pp = SelfTripDatProcessor('flicker', 'toro',)
pp.gen_dist_metric()
pp = SelfTripDatProcessor('flicker', 'edin',)
pp.gen_dist_metric()
pp = SelfTripDatProcessor('flicker', 'melb',)
pp.gen_dist_metric()
"""
"""
pp = PretrainFileProcessor()
pp.crete_test_set('weeplaces', 'weeplaces_poi_25_length_3-15-numtraj_2040')
exit()

exit()
#pp.convert_to_cilp_format()
pp = PretrainFileProcessor()
"""
"""
pp.create_pretrain_file(
'./melb/train.csv',
'./melb/traj-melb.csv',
'./melb/poi_vocab.txt',
'|',
1000
)
"""
"""
pp.create_pretrain_file(
'./weeplaces_poi_400_length_3-15-numtraj_17835/train.csv',
'./weeplaces_poi_400_length_3-15-numtraj_17835/traj-weeplaces_poi_400_length_3-15-numtraj_17835.csv',
'./weeplaces_poi_400_length_3-15-numtraj_17835/poi_vocab.txt',
'|',
1000
)

"""
#dat_suffix = ['toro', 'edin', 'glas', 'melb', 'osaka', 'weeplaces_poi_1000_length_9', 'weeplaces_poi_101_length_9']
#create_weeplaces_dataset_by_poi_and_length()
#clean_dataset()
#clean_dataset_weeplaces()
#create_dataset_by_poi()
#print(df)
#print(load_pois())
#df = df.sort_values(by=['importance'], ascending=False)
#run()

#print(trajectories)
#print(users)
