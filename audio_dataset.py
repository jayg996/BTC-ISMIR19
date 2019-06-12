import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from utils.preprocess import Preprocess, FeatureTypes
import math
from multiprocessing import Pool
from sortedcontainers import SortedList

class AudioDataset(Dataset):
    def __init__(self, config, root_dir='/data/music/chord_recognition', dataset_names=('isophonic',),
                 featuretype=FeatureTypes.cqt, num_workers=20, train=False, preprocessing=False, resize=None, kfold=4):
        super(AudioDataset, self).__init__()

        self.config = config
        self.root_dir = root_dir
        self.dataset_names = dataset_names
        self.preprocessor = Preprocess(config, featuretype, dataset_names, self.root_dir)
        self.resize = resize
        self.train = train
        self.ratio = config.experiment['data_ratio']

        # preprocessing hyperparameters
        # song_hz, n_bins, bins_per_octave, hop_length
        mp3_config = config.mp3
        feature_config = config.feature
        self.mp3_string = "%d_%.1f_%.1f" % \
                          (mp3_config['song_hz'], mp3_config['inst_len'],
                           mp3_config['skip_interval'])
        self.feature_string = "%s_%d_%d_%d" % \
                              (featuretype.value, feature_config['n_bins'], feature_config['bins_per_octave'], feature_config['hop_length'])

        if feature_config['large_voca'] == True:
            # store paths if exists
            is_preprocessed = True if os.path.exists(os.path.join(root_dir, 'result', dataset_names[0]+'_voca', self.mp3_string, self.feature_string)) else False
            if (not is_preprocessed) | preprocessing:
                midi_paths = self.preprocessor.get_all_files()

                if num_workers > 1:
                    num_path_per_process = math.ceil(len(midi_paths) / num_workers)
                    args = [midi_paths[i * num_path_per_process:(i + 1) * num_path_per_process] for i in range(num_workers)]

                    # start process
                    p = Pool(processes=num_workers)
                    p.map(self.preprocessor.generate_labels_features_voca, args)

                    p.close()
                else:
                    self.preprocessor.generate_labels_features_voca(midi_paths)

            # kfold is 5 fold index ( 0, 1, 2, 3, 4 )
            self.song_names, self.paths = self.get_paths_voca(kfold=kfold)
        else:
            # store paths if exists
            is_preprocessed = True if os.path.exists(os.path.join(root_dir, 'result', dataset_names[0], self.mp3_string, self.feature_string)) else False
            if (not is_preprocessed) | preprocessing:
                midi_paths = self.preprocessor.get_all_files()

                if num_workers > 1:
                    num_path_per_process = math.ceil(len(midi_paths) / num_workers)
                    args = [midi_paths[i * num_path_per_process:(i + 1) * num_path_per_process]
                            for i in range(num_workers)]

                    # start process
                    p = Pool(processes=num_workers)
                    p.map(self.preprocessor.generate_labels_features_new, args)

                    p.close()
                else:
                    self.preprocessor.generate_labels_features_new(midi_paths)

            # kfold is 5 fold index ( 0, 1, 2, 3, 4 )
            self.song_names, self.paths = self.get_paths(kfold=kfold)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        instance_path = self.paths[idx]

        res = dict()
        data = torch.load(instance_path)
        res['feature'] = np.log(np.abs(data['feature']) + 1e-6)
        res['chord'] = data['chord']
        return res

    def get_paths(self, kfold=4):
        temp = {}
        used_song_names = list()
        for name in self.dataset_names:
            dataset_path = os.path.join(self.root_dir, "result", name, self.mp3_string, self.feature_string)
            song_names = os.listdir(dataset_path)
            for song_name in song_names:
                paths = []
                instance_names = os.listdir(os.path.join(dataset_path, song_name))
                if len(instance_names) > 0:
                    used_song_names.append(song_name)
                for instance_name in instance_names:
                    paths.append(os.path.join(dataset_path, song_name, instance_name))
                temp[song_name] = paths
        # throw away unused song names
        song_names = used_song_names
        song_names = SortedList(song_names)

        print('Total used song length : %d' %len(song_names))
        tmp = []
        for i in range(len(song_names)):
            tmp += temp[song_names[i]]
        print('Total instances (train and valid) : %d' %len(tmp))

        # divide train/valid dataset using k fold
        result = []
        total_fold = 5
        quotient = len(song_names) // total_fold
        remainder = len(song_names) % total_fold
        fold_num = [0]
        for i in range(total_fold):
            fold_num.append(quotient)
        for i in range(remainder):
            fold_num[i+1] += 1
        for i in range(total_fold):
                fold_num[i+1] += fold_num[i]

        if self.train:
            tmp = []
            # get not augmented data
            for k in range(total_fold):
                if k != kfold:
                    for i in range(fold_num[k], fold_num[k+1]):
                        result += temp[song_names[i]]
                    tmp += song_names[fold_num[k]:fold_num[k + 1]]
            song_names = tmp
        else:
            for i in range(fold_num[kfold], fold_num[kfold+1]):
                instances = temp[song_names[i]]
                instances = [inst for inst in instances if "1.00_0" in inst]
                result += instances
            song_names = song_names[fold_num[kfold]:fold_num[kfold+1]]
        return song_names, result

    def get_paths_voca(self, kfold=4):
        temp = {}
        used_song_names = list()
        for name in self.dataset_names:
            dataset_path = os.path.join(self.root_dir, "result", name+'_voca', self.mp3_string, self.feature_string)
            song_names = os.listdir(dataset_path)
            for song_name in song_names:
                paths = []
                instance_names = os.listdir(os.path.join(dataset_path, song_name))
                if len(instance_names) > 0:
                    used_song_names.append(song_name)
                for instance_name in instance_names:
                    paths.append(os.path.join(dataset_path, song_name, instance_name))
                temp[song_name] = paths
        # throw away unused song names
        song_names = used_song_names
        song_names = SortedList(song_names)

        print('Total used song length : %d' %len(song_names))
        tmp = []
        for i in range(len(song_names)):
            tmp += temp[song_names[i]]
        print('Total instances (train and valid) : %d' %len(tmp))

        # divide train/valid dataset using k fold
        result = []
        total_fold = 5
        quotient = len(song_names) // total_fold
        remainder = len(song_names) % total_fold
        fold_num = [0]
        for i in range(total_fold):
            fold_num.append(quotient)
        for i in range(remainder):
            fold_num[i+1] += 1
        for i in range(total_fold):
                fold_num[i+1] += fold_num[i]

        if self.train:
            tmp = []
            # get not augmented data
            for k in range(total_fold):
                if k != kfold:
                    for i in range(fold_num[k], fold_num[k+1]):
                        result += temp[song_names[i]]
                    tmp += song_names[fold_num[k]:fold_num[k + 1]]
            song_names = tmp
        else:
            for i in range(fold_num[kfold], fold_num[kfold+1]):
                instances = temp[song_names[i]]
                instances = [inst for inst in instances if "1.00_0" in inst]
                result += instances
            song_names = song_names[fold_num[kfold]:fold_num[kfold+1]]
        return song_names, result

def _collate_fn(batch):
    batch_size = len(batch)
    max_len = batch[0]['feature'].shape[1]

    input_percentages = torch.empty(batch_size)  # for variable length
    chord_lens = torch.empty(batch_size, dtype=torch.int64)
    chords = []
    collapsed_chords = []
    features = []
    boundaries = []
    for i in range(batch_size):
        sample = batch[i]
        feature = sample['feature']
        chord = sample['chord']
        diff = np.diff(chord, axis=0).astype(np.bool)
        idx = np.insert(diff, 0, True, axis=0)
        chord_lens[i] = np.sum(idx).item(0)
        chords.extend(chord)
        features.append(feature)
        input_percentages[i] = feature.shape[1] / max_len
        collapsed_chords.extend(np.array(chord)[idx].tolist())
        boundary = np.append([0], diff)
        boundaries.extend(boundary.tolist())

    features = torch.tensor(features, dtype=torch.float32).unsqueeze(1)  # batch_size*1*feature_size*max_len
    chords = torch.tensor(chords, dtype=torch.int64)  # (batch_size*time_length)
    collapsed_chords = torch.tensor(collapsed_chords, dtype=torch.int64)  # total_unique_chord_len
    boundaries = torch.tensor(boundaries, dtype=torch.uint8)  # (batch_size*time_length)

    return features, input_percentages, chords, collapsed_chords, chord_lens, boundaries

class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn
