import os
import numpy as np
import pandas as pd
from pathlib import Path
import dateutil
import math
import util
import time
import datetime
import string
import random
import os

INTERVAL_1_IN_SEC = 86400
INTERVAL_2_IN_SEC = 600
NUM_INTERVAL_2 = INTERVAL_1_IN_SEC / INTERVAL_2_IN_SEC


class DataPreprocessor:
    def __init__(self, dataset_name, data_dir, checkin_path, data_relationship_path, user_feature_columns, time_series_columns):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.data_checkin_path = checkin_path
        self.data_relationship_path = data_relationship_path
        self.user_feature_columns = user_feature_columns
        self.time_series_columns = time_series_columns
        self.columns = self.user_feature_columns + self.time_series_columns
        self.poi_save_path = f'{data_dir}/poi_{dataset_name}.csv'
        self.traj_save_path = f'{data_dir}/traj_{dataset_name}.csv'
        self.column_sep = '|'
        self.time_series_sep = ','
        self.trajectory_sep = '|||'

    def load_pois(self):
        df = pd.read_csv(self.poi_save_path, names = ['place_id', 'place_name', 'lat', 'lon', 'cat', 'visits'], dtype = {'place_id': str, 'place_name': str})
        return df.sort_values(by = ['visits'], ascending = False)

    def load_top_pois(self, num_top_pois):
        df = self.load_pois()
        if num_top_pois != -1:
            df = df[:num_top_pois]
        df = df.set_index('place_id')
        return df
    def save_pois(self, places):
        size = len(places)
        data = {
            'place_id': np.zeros(size, dtype = object),
            'place_name': np.zeros(size, dtype = object),
            'lat': np.zeros(size, dtype = object),
            'lon': np.zeros(size, dtype = object),
            'cat': np.zeros(size, dtype = object),
            'visit': np.zeros(size, dtype = object),
        }
        count = 0
        for place in places.values():
            for key in data:
                data[key][count] = place[key]
            count += 1
        df = pd.DataFrame.from_dict(data)
        df = df.sort_values(by = ['visit'], ascending = False)
        df.to_csv(self.poi_save_path, index = False, header = None)

    def load_trajectories(self):
        return pd.read_csv(self.traj_save_path, sep = self.column_sep, names = self.columns)

    def save_trajectories(self, trajectories):
        count = 0
        num_trajectory = len(trajectories)
        data = {}
        for column in self.columns:
            data[column] = np.zeros(num_trajectory, dtype = object)
        for key in trajectories:
            trajectories[key] = sorted(trajectories[key], key = lambda row: row['time'])
            matches = key.split(self.trajectory_sep)
            user_id = matches[0]
            trajectory = trajectories[key]
            temp = {}
            for column in self.time_series_columns:
                temp[column] = []
            for x in trajectory:
                for column in self.time_series_columns:
                    temp[column].append(str(x[column]))
            data['user_id'][count] = user_id

            for column in self.time_series_columns:
                data[column][count] =  self.time_series_sep.join([str(x) for x in temp[column]])
            count += 1
        result = {}
        for column in self.columns:
            result[column] = data[column]

        df = pd.DataFrame.from_dict(result)
        df.to_csv(self.traj_save_path, index = False, sep = self.column_sep, header = False)
        print(f'output_file: {self.traj_save_path}')
    def convert_dataset(self, dir, file_name, file_path):
        Path(dir).mkdir(parents=True, exist_ok=True)
        print(dir)
        new_poi_file = f'{dir}/poi-{file_name}.csv'
        new_traj_file = f'{dir}/traj-{file_name}.csv'
        columns_mapping = {'place_id': 'poiID', 'cat': 'poiCat', 'lat': 'poiLat', 'lon': 'poiLon'}
        new_columns_order = ['poiID', 'poiCat', 'poiLon', 'poiLat']
        df = pd.read_csv(file_path, names = self.columns, sep = self.column_sep)
        print(df)
        result = {}
        for column in self.columns:
            result[column] = df[column].str.split(",")
        size = len(df.index)
        poi_set = set()
        trajectory_size = 0
        for i in range(size):
            trajectory_size += len(result['place_id'][i])
        traj_df = pd.DataFrame({'userID': np.zeros(trajectory_size, dtype = object), 'trajID': np.zeros(trajectory_size, dtype = object), 'poiID': np.zeros(trajectory_size, dtype = int), 'startTime':np.zeros(trajectory_size, dtype = int), 'endTime': np.zeros(trajectory_size, dtype = int), '#photo': np.zeros(trajectory_size, dtype = int), 'trajLen': np.zeros(trajectory_size, dtype = int), 'poiDuration' : np.zeros(trajectory_size, dtype = int)})

        traj_id = 0
        row = 0
        for i in range(size):
            user_id = result['user_id'][i][0]
            pois = result['place_id'][i]
            traj_len = len(pois)
            for j in range(traj_len):
                poi = pois[j]
                start_time = result['time'][i][j]
                end_time = result['time'][i][j]
                if j < traj_len - 1:
                    end_time = result['time'][i][j + 1]
                traj_df.at[row, 'userID'] = user_id
                traj_df.at[row, 'trajID'] = traj_id
                traj_df.at[row, 'poiID'] = poi
                traj_df.at[row, 'startTime'] = start_time
                traj_df.at[row, 'endTime'] = end_time
                traj_df.at[row, 'poiDuration'] = int((int(end_time) - int(start_time)) / 1000)
                traj_df.at[row, 'trajLen'] = traj_len
                if poi not in poi_set:
                    poi_set.add(poi)
                row += 1
            traj_id += 1
        poi_df = self.load_pois()
        new_poi_df = poi_df[poi_df['place_id'].isin(poi_set)]
        new_poi_df = new_poi_df[columns_mapping.keys()]
        new_poi_df = new_poi_df.rename(columns = columns_mapping)
        new_poi_df = new_poi_df[new_columns_order]

        new_poi_df.to_csv(new_poi_file, index = False)
        traj_df.to_csv(new_traj_file, index = False)
        #print(new_poi_df)
        traj_path = new_traj_file.replace(dir,'')
        new_poi_path = new_poi_file.replace(dir,'')
        print(new_poi_df)
        print(traj_df)
        print()
        print(f'traj:         {traj_path}')
        print(f'new_poi_file: {new_poi_path}')
        print(new_poi_file)
        print()


class WeeplaceDataPreprocessor(DataPreprocessor):
    def __init__(self, data_dir, data_checkin_path, data_relationship_path, user_feature_columns, time_series_columns):
        super().__init__('weeplaces', data_dir, data_checkin_path, data_relationship_path, user_feature_columns, time_series_columns)
    def count_trajectory(self, data_dir, file_name):
        dir = f'{data_dir}/{file_name}'
        file_path = f'{dir}/data.csv'
        df = pd.read_csv(file_path, sep = '|', header = None, index_col = None)
        size = len(df.index)
        sec_dict = {}
        trajs = df.iloc[:, 1].values
        length_count = {}
        for traj in trajs:
            length = len(traj.split(","))
            if length not in length_count:
                length_count[length] = 0
            length_count[length] += 1
        print(length_count)

    def process(self, target_num_top_pois = -1, target_length = -1, min_length = 3, max_length = 15):
        if os.path.isfile(self.poi_save_path) and os.path.isfile(self.traj_save_path):
            print('1. poi and traj file exists. No need to preprocess.')
        else:
            places, trajectories = self.clean_dataset()
            self.save_pois(places)
            self.save_trajectories(trajectories)

        trajectories = self.load_trajectories()
        dir, file_name, file_path = self.generate_dataset(trajectories, target_num_top_pois, target_length, min_length, max_length)
        self.convert_dataset(dir, file_name, file_path)
        vp = VocabPreprocessor()
        vp.create_vocab_files(dir, file_name, file_path)
        return file_name
    def generate_downstream_task(self, data_dir, file_name):
        dir = f'{data_dir}/{file_name}'
        file_path = f'{dir}/data.csv'
        df = pd.read_csv(file_path, sep = '|', header = None, index_col = None)
        size = len(df.index)
        sec_dict = {}
        for i in range(size):
            sequence = np.array(df[1][i].split(','))
            times = np.array(df[7][i].split(','))
            key = f'{sequence[0]}-{sequence[-1]}'
            if key not in sec_dict:
                sec_dict[key] = []
            sec_dict[key].append({'sequence': sequence, 'time': times})

        candidates = {}
        for key, value in sec_dict.items():
            if len(value) > 2:
                candidates[key] = value
        del sec_dict

        print(candidates)
        print(len(candidates))
        #print(df.values[1])
        #print(df)
        pass
    def generate_dataset(self, trajectories, target_num_top_pois, target_length, min_length, max_length):
        poi_count = -1
        num_top_pois = target_num_top_pois
        reach = 0
        while True:
            top_pois = self.load_top_pois(num_top_pois)
            df, poi_count = self.filter_trajectories_weeplaces(trajectories, top_pois, target_length, min_length, max_length)
            difference = poi_count - target_num_top_pois
            num_top_pois -= difference
            print(f'poi_used: {poi_count} / {num_top_pois} target: {target_num_top_pois}')
            # 803 > 800
            if poi_count > target_num_top_pois:
                target_num_top_pois -= difference
                reach += 1
                if reach == 2:
                    break
            if poi_count < target_num_top_pois and reach == 1:
                target_num_top_pois += 1
                reach = 2
        num_trajectories = len(df.index)
        if target_length == -1:
            file_name = f'{self.dataset_name}_poi_{poi_count}_length_{min_length}-{max_length}-numtraj_{num_trajectories}'
        else:
            file_name = f'{self.dataset_name}_poi_{poi_count}_length_{target_length}-numtraj_{num_trajectories}'
        dir = f'{self.data_dir}/{file_name}'
        file_path = f'{dir}/data.csv'
        Path(dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index = False, sep = "|", header = False)
        return dir, file_name, file_path


    def clean_dataset(self):
        users = {}
        users_counter = 0
        places = {}
        places_counter = 1
        categories = {}
        categories_counter = 0
        trajectories = {}
        df = pd.read_csv(self.data_checkin_path, dtype = str)
        df['category'] = df["category"].fillna("Uncategorized")
        for row in df.itertuples():
            username, place_name, datetime, lat, lon, city, cat = row.userid, row.placeid, dateutil.parser.parse(row.datetime), row.lat, row.lon, row.city, row.category
            if username not in users:
                users[username] = users_counter
                users_counter += 1
            user_id = users[username]
            if place_name not in places:
                places[place_name] = {'place_id': places_counter, 'place_name': place_name, 'lat': lat, 'lon': lon, 'city': city, 'cat': cat, 'visit': 0}
                places_counter += 1

            places[place_name]['visit'] += 1
            place_id = places[place_name]['place_id']
            row = {'user_id': user_id, 'place_id': place_id, 'time': str(int(datetime.timestamp())), 'lat': lat, 'lon': lon, 'cat': cat}
            trajectories.setdefault(username, []).append(row)
        count = 0
        return places, trajectories

    def filter_trajectories_weeplaces(self, df, top_pois, target_length, min_length, max_length):
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371.0088
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2) **2
            c = 2 * np.arctan2(a**0.5, (1-a)**0.5)
            d = R * c
            return round(d, 4)

        df['length'] = df['time'].apply(lambda x: len(x.split(',')))
        possible_max_num_trajectory = int(df['length'].sum() / min_length)
        data = {}
        for column in self.columns:
            data[column] = np.zeros(possible_max_num_trajectory, dtype = object)
        count = 0
        poi_used = set()
        traj_unique = set()
        max_timespan = 28800

        for index, user_trajectory in df.iterrows():
            user_id = user_trajectory.user_id
            length = user_trajectory['length']
            feature = {}
            for column in self.time_series_columns:
                feature[column] = user_trajectory[column].split(',')

            candidates = []
            trajectory = []

            this_poi_used = set()

            last_poi_id = -1
            last_time_stamp = -1
            starting_poi_id = -1
            for i in range(length):
                poi_id = feature['place_id'][i]
                timestamp = int(feature['time'][i])
                if last_poi_id == poi_id or poi_id not in top_pois.index or poi_id in this_poi_used: #
                    continue
                # over four hours -> a new traj
                if poi_id == starting_poi_id or (timestamp - last_time_stamp) > max_timespan or i == length - 1:
                    # push current record to the candidate set
                    trajectory_len = len(trajectory)
                    # Added to the candidate only if the length is over the minimum required length.
                    if trajectory_len >= min_length:
                        candidates.append(trajectory)

                    # Reset for new candidate_trajectory
                    this_poi_used = set()
                    starting_poi_id = poi_id
                    trajectory = []

                last_time_stamp = timestamp
                poi_visit_data = {}
                for column in self.time_series_columns:
                    poi_visit_data[column] = str(feature[column][i])
                poi_visit_data['i'] = i
                trajectory.append(poi_visit_data)

                this_poi_used.add(poi_id)
                last_poi_id = poi_id

            index = 0
            final_candidates = []
            for index in range(len(candidates)):
                qualified = True
                candidate = np.array(candidates[index])
                last_poi = candidate[0]
                while True:
                    mask = np.ones(len(candidate), dtype = bool)
                    for i in range(1, len(candidate)):
                        current_poi = candidate[i]
                        timespan = int(current_poi['time']) - int(last_poi['time'])
                        dist_km = haversine(float(last_poi['lat']), float(last_poi['lon']), float(current_poi['lat']), float(current_poi['lon']))
                        speed_m_in_s = 1000 * dist_km / timespan
                        if timespan > max_timespan:
                            mask[i] = False
                        if speed_m_in_s > 40:
                            mask[i] = False
                    if (mask == False).sum() == 0:
                        if len(candidate) >= min_length:
                            final_candidates.append(candidate)
                        break
                    else:
                        candidate = candidate[mask]
            candidates = final_candidates

            for candidate in candidates:
                if target_length != -1:
                    candidate = candidate[:target_length]
                else:
                    candidate = candidate[:max_length]
                new_trajectory_length = len(candidate)
                if (target_length == -1 and new_trajectory_length >= min_length) or (target_length != -1 and new_trajectory_length == target_length):
                    pois = list(map(lambda x:x['place_id'], candidate))
                    key = user_trajectory.user_id + '-'.join(pois)
                    if key not in traj_unique:
                        data['user_id'][count] = f'user-{user_id}'.replace('-', '_')
                        for column in self.time_series_columns:
                            data[column][count] =  ','.join([str(x) for x in list(map(lambda x:x[column], candidate))])
                        for poi_id in pois:
                            poi_used.add(poi_id)
                        count += 1
                        traj_unique.add(key)
        for key in data:
            data[key] = data[key][:count]
        result = {}
        for column in self.columns:
            result[column] = data[column]
        df = pd.DataFrame.from_dict(result)
        poi_count = len(poi_used)
        print(f'poi_count: {poi_count} traj: {len(df)}')
        return df, poi_count
        #random_selection = np.random.rand(len(df.index)) <= 0.85
        #train_data = df[random_selection]
        #test_data = df[~random_selection]
        #train_data.to_csv(f"train_{dataset}_data.csv", index=False, sep="|", header=False)
        #test_data.to_csv(f"test_{dataset}_data.csv", index=False, sep="|", header=False)
        #print(result)

class VocabPreprocessor:
    def create_vocab_files(self, dir, file_name, file_path):

        Path(dir).mkdir(parents = True, exist_ok = True)
        precision = 6
        poi_name = f'{dir}/poi-{file_name}.csv'
        tra_name = f'{dir}/traj-{file_name}.csv'
        op_tdata = open(poi_name, 'r')
        ot_tdata = open(tra_name, 'r')
        #print('To Train',dat_suffix[dat_ix])
        def create_vocab_file(unique):
            prepend = ['[MASK]','[CLS]','[PAD]','[SEP]','[UNK]']
            for key in unique:
                if key == 'poi':
                    data = prepend + list(unique['poi']) + list(unique['time']) + list(unique['user'])
                    print('poi:', len(unique['poi']), 'time:', len(unique['time']), 'user:', len(unique['user']))
                else:
                    data = prepend + list(unique[key])
                d = pd.DataFrame.from_dict(data)
                vocab_path = f'{dir}/{key}_vocab.txt'
                d.to_csv(vocab_path, index = False, header = None)
                #print(f'create_vocab_file [{key}]: {vocab_path}')


        points = util.read_POIs(op_tdata)
        d = []
        for poi_id in points:
            d.append([poi_id, points[poi_id]['lat'],  points[poi_id]['lon']])

        df = pd.DataFrame(d)

        (trajectories, users, poi_count) = util.read_trajectory(ot_tdata)
        trajectories = util.filter_trajectory(trajectories)
        checkins = []
        max_length = 0
        for key in trajectories:
            trajectory_list = trajectories[key]
            length = len(trajectory_list)
            if length > max_length:
                max_length = length
        padding = 0

        unique = {
        'user': set(),
        'poi': set(),
        'time': set(),
        }
        poi_ghash = {}
        poi_cat = {}
        max_length = 0
        for key in trajectories:
            trajectory_list = trajectories[key]
            matches = key.split('-')

            user_id = matches[0]
            traj_id = matches[1]
            input_place_id = []
            sequence_place_id = []
            sequence_category = []
            sequence_timestamp= []
            if len(trajectory_list) > max_length:
                max_length = len(trajectory_list)
            for trajectory in trajectory_list:
                place_id = trajectory[0]
                timestamp = int(trajectory[1])
                dt_object = datetime.datetime.fromtimestamp(timestamp)
                unique['poi'].add(place_id)
                unique['user'].add(user_id)
                timestamp = str(timestamp)

                sequence_place_id.append(place_id)
                sequence_timestamp.append(timestamp)
            sequence_place_id = ','.join(sequence_place_id)
            sequence_timestamp = ','.join(sequence_timestamp)
            checkins.append([user_id, sequence_place_id, sequence_timestamp])
        for i in range(48):
            unique['time'].add(f'time-{i}')
        print(f'{file_name}: {max_length}')
        create_vocab_file(unique)
        df = pd.DataFrame(checkins)
        data_file_path = f"{dir}/data.csv"
        df.to_csv(data_file_path, index = False, sep = "|", header = False)
        print('data_file_path', data_file_path)

class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

class BST:
    def __init__(self):
        self.head = None
        self.unique_values = []

    def insert(self, value):
        n = Node(value)
        if self.head == None:
            self.head = n
            self.unique_values.append(value)
        else:
            parent = self.head
            while parent != None:
                if parent.value < n.value:
                    if parent.right == None:
                        parent.right = n
                        self.unique_values.append(value)
                        break
                    else:
                        parent = parent.right

                elif parent.value > n.value:
                    if parent.left == None:
                        parent.left = n
                        self.unique_values.append(value)
                        break
                    else:
                        parent = parent.left
                else:
                    break



    def hasClosestValueInBST(self, target, tolerance):
        currentNode = self.head
        return self.closest_helper(currentNode, target, tolerance) != None
    def closest_helper(self, currentNode, target, tolerance):
        if currentNode == None:
            return None

        if abs(target - currentNode.value) <= tolerance:
            return currentNode.value
        elif target < currentNode.value:
            return self.closest_helper(currentNode.left, target, tolerance)
        elif target > currentNode.value:
            return self.closest_helper(currentNode.right, target, tolerance)


class PretrainFileProcessor:
    def hi(self, train_data, poi_data, sep):
        import dgl
        from sklearn.feature_extraction.text import TfidfVectorizer
        from scipy.sparse import csr_matrix
        from scipy.sparse import coo_matrix
        poi_df = pd.read_csv(poi_data, header = None)
        poi_corpus = poi_df[0].values
        vectorizer = TfidfVectorizer(norm = 'l1')
        trajectories = pd.read_csv(train_data, sep, header = None).iloc[:, 1].values
        src_ids = []
        dst_ids = []
        eweight = []
        for i in range(len(trajectories)):
            trajectories[i] = set(trajectories[i].split(','))
        for i in range(len(trajectories)):
            for j in range(i + 1, len(trajectories)):
                count = 0
                size = len(trajectories[j])
                for k in trajectories[j]:
                    if k in trajectories[i]:
                        count += 1
                weight = count / size
                if weight != 0:
                    src_ids.append(i)
                    dst_ids.append(j)
                    eweight.append(weight)
                    src_ids.append(j)
                    dst_ids.append(i)
                    eweight.append(weight)
        for i in range(len(trajectories)):
            src_ids.append(i)
            dst_ids.append(i)
            eweight.append(1.0)
        src_ids = np.array(src_ids, dtype = np.int64)
        dst_ids = np.array(dst_ids, dtype = np.int64)
        eweight = np.array(eweight, dtype = np.float32)
        sp_mat = coo_matrix((eweight, (src_ids, dst_ids)))
        print(sp_mat)
        g = dgl.from_scipy(sp_mat, eweight_name ='edge_weight')
        return g
    def create_pretrain_file(self, dataset, train_data, train_data2, poi_data, sep, expected_size = 100):
        self.dataset = dataset
        #N = int(expected_size / 2)
        #df1 = self.repeat_train_file(file_path, sep, N)
        #user_poi, poi_transition_matrix, visiting_time_matrix, poi_visit_time, = self.create_poi_transition_matrix_length_2(train_data2, poi_data)
        user_poi, poi_transition_matrix, visiting_time_matrix, poi_visit_time = {}, {}, {}, {}
        df = self.create_fake_trajectories(train_data, sep, user_poi, poi_transition_matrix, visiting_time_matrix, poi_visit_time, expected_size)
        #df = pd.concat([df1, df2])
        df = df.sample(frac = 1)
        g = None#self.hi(train_data, poi_data, sep)
        return df, g
    def repeat_train_file(self, file_path, sep, expected_size = 100):
        df = pd.read_csv(file_path, sep, header = None)
        df = df.iloc[:, :2]
        #N = int(expected_size / len(df.index)) + 1
        #df = pd.concat([df] * N, ignore_index=True)[:expected_size]
        return df
    def crete_test_set(self, train_data):
        pass
    def create_poi_transition_matrix_length_2(self, file_path, poi_data):
        df = pd.read_csv(file_path)
        poi_df = pd.read_csv(poi_data, header = None)
        pois = [str(i[0]) for i in poi_df.values[5:]]
        t_df = df[df['trajLen'] >= 3].groupby('userID')
        poi_transition_matrix = {}
        trajectory_length = {}
        user_poi = {}
        visiting_time_matrix = {}
        poi_visit_time = {}
        i = 0
        for group_name, df_group in t_df:
            #print(group_name, df_group)
            print(i)
            i += 1
        return user_poi, poi_transition_matrix, visiting_time_matrix, poi_visit_time,
    def create_fake_trajectories(self, file_path, sep, user_poi, poi_transition_matrix, visiting_time_matrix, poi_visit_time, expected_size = 100):
        df = pd.read_csv(file_path, sep, header = None)
        user_ids = df.iloc[:, 0].values
        trajectories = df.iloc[:, 1].values
        trajectory_timestamps = df.iloc[:, 7].values
        for user_id in user_ids:
            user_poi[user_id] = set()
        trajectory_length = {}
        #poi_transition_matrix = {}
        for i in range(len(trajectories)):
            trajectory = trajectories[i]
            user_id = user_ids[i]
            timestamps = trajectory_timestamps[i]
            pois = [str(i) for i in trajectory.split(',')]
            timestamps = [int(i) for i in timestamps.split(',')]
            for j in range(len(timestamps)):
                timestamp  = timestamps[j] = int(timestamps[j])
            length = len(pois)
            if length not in trajectory_length:
                trajectory_length[length] = 0
            trajectory_length[length] += 1
            for i in range(length - 1):
                poi_start = pois[i]
                poi_end = pois[i + 1]
                if poi_start not in poi_transition_matrix:
                    poi_transition_matrix[poi_start] = {}
                if poi_start not in visiting_time_matrix:
                    visiting_time_matrix[poi_start] = {}
                if poi_start not in poi_visit_time:
                    poi_visit_time[poi_start] = BST()
                if poi_end not in poi_visit_time:
                    poi_visit_time[poi_end] = BST()
                if poi_end not in poi_transition_matrix[poi_start]:
                    poi_transition_matrix[poi_start][poi_end] = 0
                if poi_end not in visiting_time_matrix[poi_start]:
                    visiting_time_matrix[poi_start][poi_end] = []
                user_poi[user_id].add(poi_start)
                user_poi[user_id].add(poi_end)
                poi_transition_matrix[poi_start][poi_end] += 1
                visiting_time_matrix[poi_start][poi_end].append(timestamps[i + 1] - timestamps[i])
                poi_visit_time[poi_start].insert(int((timestamps[i] % INTERVAL_1_IN_SEC) / INTERVAL_2_IN_SEC))
                poi_visit_time[poi_end].insert(int((timestamps[i + 1] % INTERVAL_1_IN_SEC) / INTERVAL_2_IN_SEC))
        for key, value in poi_transition_matrix.items():
            count = 0
            keys = list(poi_transition_matrix[key].keys())
            for sub_key in keys:
                count += poi_transition_matrix[key][sub_key]
            p = []
            for sub_key in keys:
                poi_transition_matrix[key][sub_key] /= count
                p.append(poi_transition_matrix[key][sub_key])
            poi_transition_matrix[key] = {'p': p, 'poi': keys}
        """
        for poi_start, matrix in visiting_time_matrix.items():
            #print(matrix)
            for poi_end in matrix:
                matrix[poi_end] = int(sum(matrix[poi_end]) / len(matrix[poi_end])) + 1
        """
        length_choice = list(trajectory_length.keys())
        total_traj_length = sum(trajectory_length.values())
        length_probability = [i / total_traj_length for i in list(trajectory_length.values())]
        return self.create_trajectories_by_transition_matrix(user_poi, poi_transition_matrix, visiting_time_matrix, poi_visit_time, length_choice, length_probability, expected_size)
    def create_trajectories_by_transition_matrix(self, user_poi, poi_transition_matrix, visiting_time_matrix, poi_visit_time, length_choice, length_probability, expected_size):
        def getrange(dataset):
            s = {
            #(length, time-range)
            'melb': (10, 10),
            'glas': (6, 5),
            'osaka': (6, 5),
            }
            result = (6, 5)
            if dataset in s:
                result = s[dataset]
            return result
        fake_traj_length, time_range = getrange(self.dataset)
        poi_id_set = list(poi_transition_matrix.keys())
        max_length = max(length_choice)
        currentSize = 0
        trajectories = {}
        while currentSize < expected_size:
            user_id = np.random.choice(list(user_poi.keys()))
            last_poi = poi_id_set[np.random.choice(len(poi_id_set), 1)[0]]

            while last_poi not in user_poi[user_id]:
                last_poi = poi_id_set[np.random.choice(len(poi_id_set), 1)[0]]

            length_choice_index = np.random.choice(len(length_probability), 1, p = length_probability)[0]
            #fake_traj_length = length_choice[length_choice_index]
            #fake_traj_length = np.random.choice([6])
            sequence_trajectory = [str(last_poi)]
            sequence_timestamp = [INTERVAL_1_IN_SEC * 10 + poi_visit_time[last_poi].unique_values[np.random.choice(len(poi_visit_time[last_poi].unique_values), 1)[0]] * INTERVAL_2_IN_SEC]
            tries = 0
            poi_set = set()
            while len(sequence_trajectory) < fake_traj_length:
                if last_poi not in poi_transition_matrix:
                    break
                t_matrix = poi_transition_matrix[last_poi]
                poi_list, p = t_matrix['poi'], t_matrix['p']
                if len(t_matrix['poi']) > 0:
                    choice_index = np.random.choice(len(poi_list), 1, p = p)[0]
                    next_poi_id = poi_list[choice_index]
                    if next_poi_id in poi_set:
                        continue
                    estimated_visit_time = sequence_timestamp[len(sequence_trajectory) - 1] + np.random.choice(visiting_time_matrix[last_poi][next_poi_id])
                    estimated_visit_interval_ith = int((estimated_visit_time % INTERVAL_1_IN_SEC) / INTERVAL_2_IN_SEC)
                    if poi_visit_time[next_poi_id].hasClosestValueInBST(estimated_visit_interval_ith, time_range):# and next_poi_id in user_poi[user_id]
                        estimated_visit_time += np.random.randint(-INTERVAL_2_IN_SEC * 5, INTERVAL_2_IN_SEC * 5)
                        sequence_trajectory.append(next_poi_id)
                        sequence_timestamp.append(estimated_visit_time)
                        poi_set.add(last_poi)
                        last_poi = next_poi_id
                    else:
                        tries += 1
                        if tries >= 100:
                            break
                        continue
                else:
                    break
            #len(sequence_trajectory) ==
            if len(sequence_trajectory) > 2:
                #user_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
                key = f'{user_id}-{currentSize}'
                trajectories[key] = {'trajectory': sequence_trajectory, 'timestamp': sequence_timestamp}
                currentSize += 1
            if currentSize % 1000 == 0:
                print(currentSize, expected_size)
        checkins = []
        max_length = 0
        count = {}
        for key in trajectories:
            sequence_trajectory = trajectories[key]['trajectory']
            length = len(sequence_trajectory)
            if length not in count:
                count[length] = 0
            count[length] += 1
            if length > max_length:
                max_length = length
        print(count)
        padding = 0
        features = ['poi', 'user']
        unique = {}
        for feature in features:
            unique[feature] = set()
        max_length = 0
        for key in trajectories:
            sequence_trajectory = trajectories[key]['trajectory']
            sequence_timestamp = [str(i) for i in trajectories[key]['timestamp']]
            matches = key.split('-')
            user_id = matches[0]
            traj_id = matches[1]
            sequence_trajectory = ','.join(sequence_trajectory)
            sequence_timestamp = ','.join(sequence_timestamp)
            checkins.append([user_id, sequence_trajectory, '', '', '', '', '', sequence_timestamp])
        df = pd.DataFrame(checkins)
        print(df)
        return df
        #df.to_csv(f"{dir}/pretrain_data.csv", index = False, sep = "|", header = False)
    def process(self, dir, file_name, expected_size = 50000):
        Path(dir).mkdir(parents = True, exist_ok = True)
        precision = 6
        poi_path = f'{dir}/poi-{file_name}.csv'
        traj_path = f'{dir}/traj-{file_name}.csv'
        op_tdata = open(poi_path, 'r')
        points = util.read_POIs(op_tdata)
        Path(dir).mkdir(parents = True, exist_ok = True)
        print("read")
        df = pd.read_csv(traj_path)
        poi_df = pd.read_csv(poi_path)
        poi_id_set = poi_df['poiID'].unique()
        poi_transition_matrix = {}
        for poi_id in poi_id_set:
            poi_transition_matrix[poi_id] = {}
        poi_open_hours = {}
        traj_df = pd.read_csv(traj_path)
        print('check-in', len(traj_df['trajLen'].index))
        print('trajectories', len(traj_df['trajID'].unique()))
        print('average trajectory length', traj_df['trajLen'].mean())
        traj_df = traj_df[traj_df['trajLen'] >= 3]
        print(traj_df)
        last_traj_id = -1
        last_poi = -1
        last_end_time = -1

        for index, row in traj_df.iterrows():
            trajID = row['trajID']
            poi = row['poiID']
            start_time = row['startTime']
            dt_object = datetime.datetime.fromtimestamp(start_time)
            if trajID == last_traj_id:
                if poi not in poi_transition_matrix[last_poi]:
                    poi_transition_matrix[last_poi][poi] = 0
                poi_transition_matrix[last_poi][poi] += 1
                #print('from', last_poi, 'to', poi, 'timetothere', (start_time - last_end_time))
            last_poi = poi
            last_traj_id = trajID
            last_end_time = row['endTime']
        #random.choices(a_list, distribution)
        for key, value in poi_transition_matrix.items():
            count = 0
            keys = list(poi_transition_matrix[key].keys())
            for sub_key in keys:
                count += poi_transition_matrix[key][sub_key]
            p = []
            for sub_key in keys:
                poi_transition_matrix[key][sub_key] /= count
                p.append(poi_transition_matrix[key][sub_key])
            poi_transition_matrix[key] = {'p': p, 'poi': keys}
        # 3, 4, 5, 6
        print(file_name, )
        index = traj_df['trajLen'].value_counts().index
        values = traj_df['trajLen'].value_counts().values
        t = (values / index)
        for i in range(len(index)):
            print('length ', index[i], ':', values[i]/ index[i])


        max_length = np.max(traj_df['trajLen'])
        length_probability = values/values.sum(axis=0,keepdims=1)
        currentSize = 0
        trajectories = {}
        while currentSize < expected_size:
            last_poi = poi_id_set[np.random.choice(len(poi_id_set), 1)[0]]
            length_choice_index = np.random.choice(len(length_probability), 1, p = length_probability)[0]
            fake_traj_length = index[length_choice_index]
            temp = [[str(last_poi), '[MASK]', '[MASK]']]
            while len(temp) < fake_traj_length:
                t_matrix = poi_transition_matrix[last_poi]
                poi_list = t_matrix['poi']
                p = t_matrix['p']
                if len(t_matrix['poi']) > 0:
                    choice_index = np.random.choice(len(poi_list), 1, p = p)[0]
                    next_poi_id = poi_list[choice_index]
                    temp.append([str(next_poi_id), '[MASK]', '[MASK]'])
                    last_poi = next_poi_id
                else:
                    break
            if len(temp) == fake_traj_length:
                user_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
                key = f'{user_id}-{currentSize}'
                trajectories[key] = temp
                currentSize += 1
            if currentSize % 1000 == 0:
                print(file_name, " currentSize: ", currentSize)
        checkins = []
        max_length = 0
        for key in trajectories:
            trajectory_list = trajectories[key]
            length = len(trajectory_list)
            if length > max_length:
                max_length = length
        padding = 0
        features = ['poi', 'user', 'cat', 'ghash', 'week', 'day', 'hour']
        unique = {}
        for feature in features:
            unique[feature] = set()
        poi_ghash = {}
        poi_cat = {}
        max_length = 0
        for key in trajectories:
            trajectory_list = trajectories[key]
            matches = key.split('-')
            user_id = matches[0]
            traj_id = matches[1]
            input_place_id = []
            sequence_place_id = []
            sequence_category = []
            sequence_geohash = []
            sequence_week = []
            sequence_day = []
            sequence_quarter_hour = []
            if len(trajectory_list) > max_length:
                max_length = len(trajectory_list)
            for trajectory in trajectory_list:
                place_id = trajectory[0]
                timestamp = '[MASK]'
                week = '[MASK]'
                day = '[MASK]'
                hour = '[MASK]'
                cat = points[place_id]['cat'].replace(' ', '').replace(',', '-')
                lat = points[place_id]['lat']
                lon = points[place_id]['lon']
                ghash = '#####'
                unique['poi'].add(place_id)
                unique['user'].add(user_id)
                unique['cat'].add(cat)
                unique['ghash'].add(ghash)
                unique['week'].add(week)
                unique['day'].add(day)
                unique['hour'].add(hour)
                poi_ghash[place_id] = ghash
                poi_cat[place_id] = cat

                sequence_category.append(cat)
                sequence_geohash.append(ghash)
                sequence_place_id.append(place_id)
                sequence_week.append(week)
                sequence_day.append(day)
                sequence_quarter_hour.append(hour)
            sequence_place_id = ','.join(sequence_place_id)
            sequence_geohash = ','.join(sequence_geohash)
            sequence_category = ','.join(sequence_category)
            sequence_week = ','.join(sequence_week)
            sequence_day = ','.join(sequence_day)
            sequence_quarter_hour = ','.join(sequence_quarter_hour)
            #, sequence_week, sequence_day, sequence_quarter_hour
            checkins.append([user_id, sequence_place_id, sequence_geohash, sequence_category])
        print(f'{file_name}: {max_length}')
        df = pd.DataFrame(checkins)
        #random_selection = np.random.rand(len(df.index)) <= 0.85
        #train_data = df[random_selection]
        #test_data = df[~random_selection]

        df.to_csv(f"{dir}/pretrain_data.csv", index = False, sep = "|", header = False)
        #df2 = pd.DataFrame.from_dict(poi_ghash, orient='index')
        #df2.to_csv(f"{final_dir}/pretrain_poi_ghash.csv", index = True, sep = "|", header = False)
        #df3 = pd.DataFrame.from_dict(poi_cat, orient='index')
        #df3.to_csv(f"{final_dir}/pretrain_poi_cat.csv", index = True, sep = "|", header = False)
