
import numpy as np
import pandas as pd
import time
def read_POIs(data):
    df = pd.read_csv(data.name)
    df = df.rename(columns = {'poiID': 'id', 'poiCat': 'cat', 'poiLat': 'lat', 'poiLon': 'lon'})
    df['id'] = df['id'].astype('str')
    df = df.set_index('id')
    return df.to_dict('index')

def read_trajectory(data):
    # ['userID', 'trajID', 'poiID', 'startTime', 'endTime', '#photo', 'trajLen', 'poiDuration\n']
    poi_count = {}
    trajectories = []
    users = set()
    for line in data.readlines():
        tokens = line.split(',')
        if tokens[0] == 'userID':
            continue
        poi_id = tokens[2]
        if poi_id not in poi_count:
            poi_count[poi_id] = 0
        poi_count[poi_id] += 1
        trajectory_data = [tokens[i].strip('\n') for i in range(len(tokens))]
        trajectories.append(trajectory_data)
        user = trajectory_data[0]
        if user not in users:
            users.add(user)  # add user id
    users = sorted(list(users))
    poi_count['GO'] = 1
    poi_count['PAD'] = 1
    poi_count['END'] = 1
    poi_count = sorted(poi_count.items(), key = lambda x:x[1], reverse = True)
    #['userID', 'trajID', 'poiID', 'startTime', 'endTime', '#photo', 'trajLen', 'poiDuration\n']
    #print(trajectories)
    return (trajectories, users, poi_count)
'''
    get trajector length > 3
    #the length of the trajectory must over than 3
'''
def filter_trajectory(trajectories):
    data = {}
    for i in range(len(trajectories)):
        trajectory = trajectories[i]
        if(int(trajectory[6]) >= 3):
            id = "{}-{}".format(trajectory[0], trajectory[1])
            data.setdefault(id,[]).append([trajectory[2], trajectory[3], trajectory[4]]) #userID+trajID
    #print("...")
    #print(data)
    return data

def calc_dist_vec(longitudes1, latitudes1, longitudes2, latitudes2):
    """Calculate the distance (unit: km) between two places on earth, vectorised"""
    # convert degrees to radians
    lng1 = np.radians(longitudes1)
    lat1 = np.radians(latitudes1)
    lng2 = np.radians(longitudes2)
    lat2 = np.radians(latitudes2)
    radius = 6371.0088 # mean earth radius, en.wikipedia.org/wiki/Earth_radius#Mean_radius

    # The haversine formula, en.wikipedia.org/wiki/Great-circle_distance
    dlng = np.fabs(lng1 - lng2)
    dlat = np.fabs(lat1 - lat2)
    dist =  2 * radius * np.arcsin( np.sqrt(
                (np.sin(0.5*dlat))**2 + np.cos(lat1) * np.cos(lat2) * (np.sin(0.5*dlng))**2 ))
    return dist

def disc_count(points, trajectories):
    distance_count = []
    for key, trajectory in trajectories.items():
        #print(trajectory)
        #print traj
        for i in range(len(trajectory)):
            starting_poi_id = trajectory[i][0]
            #print(points[starting_poi_id])
            lon1 = float(points[starting_poi_id]['lng'])
            lat1 = float(points[starting_poi_id]['lat'])
            for j in range(i + 1, len(trajectory)):
                ending_poi_id = trajectory[j][0]
                lon2 = float(points[ending_poi_id]['lng'])
                lat2 = float(points[ending_poi_id]['lat'])
                distance_count.append(calc_dist_vec(lon1,lat1,lon2,lat2))
    return distance_count

def generate_train_data(points, trajectories, max_distance):
    train_trajectories = []
    train_users = []
    train_time = []
    train_distances = []
    for keys, user_traj in trajectories.items():
        temp_poi=[]
        temp_time=[]
        temp_dist=[]
        for i in range(len(user_traj)):
            temp_poi.append(user_traj[i][0]) #add poi id
            lon1=float(points[user_traj[i][0]]['lng'])
            lat1=float(points[user_traj[i][0]]['lat'])
            lons=float(points[user_traj[0][0]]['lng'])
            lats=float(points[user_traj[0][0]]['lat'])
            lone=float(points[user_traj[-1][0]]['lng'])
            late=float(points[user_traj[-1][0]]['lat'])
            sd = calc_dist_vec(lon1, lat1, lons, lats)
            ed = calc_dist_vec(lon1, lat1, lone, late)
            value1 = 0.5 * (sd) / max_distance
            value2 = 0.5 * (ed) / max_distance
            #print value
            temp_dist.append([value1,value2]) #lon,lat

            dt = time.strftime("%H:%M:%S", time.localtime(int(user_traj[i][1:][0])))
            #print dt.split(":")[0]
            temp_time.append(int(dt.split(":")[0])) #add poi time
        train_trajectories.append(temp_poi)
        train_users.append(keys)
        train_time.append(temp_time)
        train_distances.append(temp_dist)
    return (train_trajectories, train_users, train_time, train_distances)

def generate_train_dataset(vocab_to_int, train_trajectories):
    train_dataset = []
    for i in range(len(train_trajectories)): #TRAIN
        temp = []
        for j in range(len(train_trajectories[i])):
            temp.append(vocab_to_int[train_trajectories[i][j]])
        train_dataset.append(temp)
    return train_dataset

def write_train_dataset(embedding_name, train_dataset, train_users):
    file_name = 'data/{}_set.dat'.format(embedding_name)
    with open(file_name, 'w') as file:
        for i in range(len(train_dataset)):
            output = [train_users[i], *train_dataset[i], '\n']
            output = [str(output[i]) for i in range(len(output))]
            file.write('\t'.join(output))

def extract_words_vocab(voc_poi):
    int_to_vocab = {idx: word for idx, word in enumerate(voc_poi)}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
    return int_to_vocab, vocab_to_int
def read_embeddings_from_file(embedding_name):
    file_name = 'data/{}vec.dat'.format(embedding_name)
    embeddings = []
    with open(file_name,'r') as file:
        for line in file.readlines():
            tokens = line.split()
            temp = list()
            for i in range(1, len(tokens)):
                temp.append(float(tokens[i]))
            embeddings.append(temp)
    return embeddings
