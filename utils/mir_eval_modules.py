import numpy as np
import librosa
import mir_eval
import torch
import os

idx2chord = ['C', 'C:min', 'C#', 'C#:min', 'D', 'D:min', 'D#', 'D#:min', 'E', 'E:min', 'F', 'F:min', 'F#',
             'F#:min', 'G', 'G:min', 'G#', 'G#:min', 'A', 'A:min', 'A#', 'A#:min', 'B', 'B:min', 'N']

root_list = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
quality_list = ['min', 'maj', 'dim', 'aug', 'min6', 'maj6', 'min7', 'minmaj7', 'maj7', '7', 'dim7', 'hdim7', 'sus2', 'sus4']

def idx2voca_chord():
    idx2voca_chord = {}
    idx2voca_chord[169] = 'N'
    idx2voca_chord[168] = 'X'
    for i in range(168):
        root = i // 14
        root = root_list[root]
        quality = i % 14
        quality = quality_list[quality]
        if i % 14 != 1:
            chord = root + ':' + quality
        else:
            chord = root
        idx2voca_chord[i] = chord
    return idx2voca_chord

def audio_file_to_features(audio_file, config):
    original_wav, sr = librosa.load(audio_file, sr=config.mp3['song_hz'], mono=True)
    currunt_sec_hz = 0
    while len(original_wav) > currunt_sec_hz + config.mp3['song_hz'] * config.mp3['inst_len']:
        start_idx = int(currunt_sec_hz)
        end_idx = int(currunt_sec_hz + config.mp3['song_hz'] * config.mp3['inst_len'])
        tmp = librosa.cqt(original_wav[start_idx:end_idx], sr=sr, n_bins=config.feature['n_bins'], bins_per_octave=config.feature['bins_per_octave'], hop_length=config.feature['hop_length'])
        if start_idx == 0:
            feature = tmp
        else:
            feature = np.concatenate((feature, tmp), axis=1)
        currunt_sec_hz = end_idx
    tmp = librosa.cqt(original_wav[currunt_sec_hz:], sr=sr, n_bins=config.feature['n_bins'], bins_per_octave=config.feature['bins_per_octave'], hop_length=config.feature['hop_length'])
    feature = np.concatenate((feature, tmp), axis=1)
    feature = np.log(np.abs(feature) + 1e-6)
    feature_per_second = config.mp3['inst_len'] / config.model['timestep']
    song_length_second = len(original_wav)/config.mp3['song_hz']
    return feature, feature_per_second, song_length_second

# Audio files with format of wav and mp3
def get_audio_paths(audio_dir):
    return [os.path.join(root, fname) for (root, dir_names, file_names) in os.walk(audio_dir, followlinks=True)
            for fname in file_names if (fname.lower().endswith('.wav') or fname.lower().endswith('.mp3'))]

class metrics():
    def __init__(self):
        super(metrics, self).__init__()
        self.score_metrics = ['root', 'thirds', 'triads', 'sevenths', 'tetrads', 'majmin', 'mirex']
        self.score_list_dict = dict()
        for i in self.score_metrics:
            self.score_list_dict[i] = list()
        self.average_score = dict()

    def score(self, metric, gt_path, est_path):
        if metric == 'root':
            score = self.root_score(gt_path,est_path)
        elif metric == 'thirds':
            score = self.thirds_score(gt_path,est_path)
        elif metric == 'triads':
            score = self.triads_score(gt_path,est_path)
        elif metric == 'sevenths':
            score = self.sevenths_score(gt_path,est_path)
        elif metric == 'tetrads':
            score = self.tetrads_score(gt_path,est_path)
        elif metric == 'majmin':
            score = self.majmin_score(gt_path,est_path)
        elif metric == 'mirex':
            score = self.mirex_score(gt_path,est_path)
        else:
            raise NotImplementedError
        return score

    def root_score(self, gt_path, est_path):
        (ref_intervals, ref_labels) = mir_eval.io.load_labeled_intervals(gt_path)
        ref_labels = lab_file_error_modify(ref_labels)
        (est_intervals, est_labels) = mir_eval.io.load_labeled_intervals(est_path)
        est_intervals, est_labels = mir_eval.util.adjust_intervals(est_intervals, est_labels, ref_intervals.min(),
                                                                   ref_intervals.max(), mir_eval.chord.NO_CHORD,
                                                                   mir_eval.chord.NO_CHORD)
        (intervals, ref_labels, est_labels) = mir_eval.util.merge_labeled_intervals(ref_intervals, ref_labels,
                                                                                    est_intervals, est_labels)
        durations = mir_eval.util.intervals_to_durations(intervals)
        comparisons = mir_eval.chord.root(ref_labels, est_labels)
        score = mir_eval.chord.weighted_accuracy(comparisons, durations)
        return score

    def thirds_score(self, gt_path, est_path):
        (ref_intervals, ref_labels) = mir_eval.io.load_labeled_intervals(gt_path)
        ref_labels = lab_file_error_modify(ref_labels)
        (est_intervals, est_labels) = mir_eval.io.load_labeled_intervals(est_path)
        est_intervals, est_labels = mir_eval.util.adjust_intervals(est_intervals, est_labels, ref_intervals.min(),
                                                                   ref_intervals.max(), mir_eval.chord.NO_CHORD,
                                                                   mir_eval.chord.NO_CHORD)
        (intervals, ref_labels, est_labels) = mir_eval.util.merge_labeled_intervals(ref_intervals, ref_labels,
                                                                                    est_intervals, est_labels)
        durations = mir_eval.util.intervals_to_durations(intervals)
        comparisons = mir_eval.chord.thirds(ref_labels, est_labels)
        score = mir_eval.chord.weighted_accuracy(comparisons, durations)
        return score

    def triads_score(self, gt_path, est_path):
        (ref_intervals, ref_labels) = mir_eval.io.load_labeled_intervals(gt_path)
        ref_labels = lab_file_error_modify(ref_labels)
        (est_intervals, est_labels) = mir_eval.io.load_labeled_intervals(est_path)
        est_intervals, est_labels = mir_eval.util.adjust_intervals(est_intervals, est_labels, ref_intervals.min(),
                                                                   ref_intervals.max(), mir_eval.chord.NO_CHORD,
                                                                   mir_eval.chord.NO_CHORD)
        (intervals, ref_labels, est_labels) = mir_eval.util.merge_labeled_intervals(ref_intervals, ref_labels,
                                                                                    est_intervals, est_labels)
        durations = mir_eval.util.intervals_to_durations(intervals)
        comparisons = mir_eval.chord.triads(ref_labels, est_labels)
        score = mir_eval.chord.weighted_accuracy(comparisons, durations)
        return score

    def sevenths_score(self, gt_path, est_path):
        (ref_intervals, ref_labels) = mir_eval.io.load_labeled_intervals(gt_path)
        ref_labels = lab_file_error_modify(ref_labels)
        (est_intervals, est_labels) = mir_eval.io.load_labeled_intervals(est_path)
        est_intervals, est_labels = mir_eval.util.adjust_intervals(est_intervals, est_labels, ref_intervals.min(),
                                                                   ref_intervals.max(), mir_eval.chord.NO_CHORD,
                                                                   mir_eval.chord.NO_CHORD)
        (intervals, ref_labels, est_labels) = mir_eval.util.merge_labeled_intervals(ref_intervals, ref_labels,
                                                                                    est_intervals, est_labels)
        durations = mir_eval.util.intervals_to_durations(intervals)
        comparisons = mir_eval.chord.sevenths(ref_labels, est_labels)
        score = mir_eval.chord.weighted_accuracy(comparisons, durations)
        return score

    def tetrads_score(self, gt_path, est_path):
        (ref_intervals, ref_labels) = mir_eval.io.load_labeled_intervals(gt_path)
        ref_labels = lab_file_error_modify(ref_labels)
        (est_intervals, est_labels) = mir_eval.io.load_labeled_intervals(est_path)
        est_intervals, est_labels = mir_eval.util.adjust_intervals(est_intervals, est_labels, ref_intervals.min(),
                                                                   ref_intervals.max(), mir_eval.chord.NO_CHORD,
                                                                   mir_eval.chord.NO_CHORD)
        (intervals, ref_labels, est_labels) = mir_eval.util.merge_labeled_intervals(ref_intervals, ref_labels,
                                                                                    est_intervals, est_labels)
        durations = mir_eval.util.intervals_to_durations(intervals)
        comparisons = mir_eval.chord.tetrads(ref_labels, est_labels)
        score = mir_eval.chord.weighted_accuracy(comparisons, durations)
        return score

    def majmin_score(self, gt_path, est_path):
        (ref_intervals, ref_labels) = mir_eval.io.load_labeled_intervals(gt_path)
        ref_labels = lab_file_error_modify(ref_labels)
        (est_intervals, est_labels) = mir_eval.io.load_labeled_intervals(est_path)
        est_intervals, est_labels = mir_eval.util.adjust_intervals(est_intervals, est_labels, ref_intervals.min(),
                                                                   ref_intervals.max(), mir_eval.chord.NO_CHORD,
                                                                   mir_eval.chord.NO_CHORD)
        (intervals, ref_labels, est_labels) = mir_eval.util.merge_labeled_intervals(ref_intervals, ref_labels,
                                                                                    est_intervals, est_labels)
        durations = mir_eval.util.intervals_to_durations(intervals)
        comparisons = mir_eval.chord.majmin(ref_labels, est_labels)
        score = mir_eval.chord.weighted_accuracy(comparisons, durations)
        return score

    def mirex_score(self, gt_path, est_path):
        (ref_intervals, ref_labels) = mir_eval.io.load_labeled_intervals(gt_path)
        ref_labels = lab_file_error_modify(ref_labels)
        (est_intervals, est_labels) = mir_eval.io.load_labeled_intervals(est_path)
        est_intervals, est_labels = mir_eval.util.adjust_intervals(est_intervals, est_labels, ref_intervals.min(),
                                                                   ref_intervals.max(), mir_eval.chord.NO_CHORD,
                                                                   mir_eval.chord.NO_CHORD)
        (intervals, ref_labels, est_labels) = mir_eval.util.merge_labeled_intervals(ref_intervals, ref_labels,
                                                                                    est_intervals, est_labels)
        durations = mir_eval.util.intervals_to_durations(intervals)
        comparisons = mir_eval.chord.mirex(ref_labels, est_labels)
        score = mir_eval.chord.weighted_accuracy(comparisons, durations)
        return score

def lab_file_error_modify(ref_labels):
    for i in range(len(ref_labels)):
        if ref_labels[i][-2:] == ':4':
            ref_labels[i] = ref_labels[i].replace(':4', ':sus4')
        elif ref_labels[i][-2:] == ':6':
            ref_labels[i] = ref_labels[i].replace(':6', ':maj6')
        elif ref_labels[i][-4:] == ':6/2':
            ref_labels[i] = ref_labels[i].replace(':6/2', ':maj6/2')
        elif ref_labels[i] == 'Emin/4':
            ref_labels[i] = 'E:min/4'
        elif ref_labels[i] == 'A7/3':
            ref_labels[i] = 'A:7/3'
        elif ref_labels[i] == 'Bb7/3':
            ref_labels[i] = 'Bb:7/3'
        elif ref_labels[i] == 'Bb7/5':
            ref_labels[i] = 'Bb:7/5'
        elif ref_labels[i].find(':') == -1:
            if ref_labels[i].find('min') != -1:
                ref_labels[i] = ref_labels[i][:ref_labels[i].find('min')] + ':' + ref_labels[i][ref_labels[i].find('min'):]
    return ref_labels

def root_majmin_score_calculation(valid_dataset, config, mean, std, device, model, model_type, verbose=False):
    valid_song_names = valid_dataset.song_names
    paths = valid_dataset.preprocessor.get_all_files()

    metrics_ = metrics()
    song_length_list = list()
    for path in paths:
        song_name, lab_file_path, mp3_file_path, _ = path
        if not song_name in valid_song_names:
            continue
        try:
            n_timestep = config.model['timestep']
            feature, feature_per_second, song_length_second = audio_file_to_features(mp3_file_path, config)
            feature = feature.T
            feature = (feature - mean) / std
            time_unit = feature_per_second

            num_pad = n_timestep - (feature.shape[0] % n_timestep)
            feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
            num_instance = feature.shape[0] // n_timestep

            start_time = 0.0
            lines = []
            with torch.no_grad():
                model.eval()
                feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
                for t in range(num_instance):
                    if model_type == 'btc':
                        encoder_output, _ = model.self_attn_layers(feature[:, n_timestep * t:n_timestep * (t + 1), :])
                        prediction, _ = model.output_layer(encoder_output)
                        prediction = prediction.squeeze()
                    elif model_type == 'cnn' or model_type =='crnn':
                        prediction, _, _, _ = model(feature[:, n_timestep * t:n_timestep * (t + 1), :], torch.randint(config.model['num_chords'], (n_timestep,)).to(device))
                    for i in range(n_timestep):
                        if t == 0 and i == 0:
                            prev_chord = prediction[i].item()
                            continue
                        if prediction[i].item() != prev_chord:
                            lines.append(
                                '%.6f %.6f %s\n' % (
                                    start_time, time_unit * (n_timestep * t + i), idx2chord[prev_chord]))
                            start_time = time_unit * (n_timestep * t + i)
                            prev_chord = prediction[i].item()
                        if t == num_instance - 1 and i + num_pad == n_timestep:
                            if start_time != time_unit * (n_timestep * t + i):
                                lines.append(
                                    '%.6f %.6f %s\n' % (
                                        start_time, time_unit * (n_timestep * t + i), idx2chord[prev_chord]))
                            break
            pid = os.getpid()
            tmp_path = 'tmp_' + str(pid) + '.lab'
            with open(tmp_path, 'w') as f:
                for line in lines:
                    f.write(line)

            root_majmin = ['root', 'majmin']
            for m in root_majmin:
                metrics_.score_list_dict[m].append(metrics_.score(metric=m, gt_path=lab_file_path, est_path=tmp_path))
            song_length_list.append(song_length_second)
            if verbose:
                for m in root_majmin:
                    print('song name %s, %s score : %.4f' % (song_name, m, metrics_.score_list_dict[m][-1]))
        except:
            print('song name %s\' lab file error' % song_name)

    tmp = song_length_list / np.sum(song_length_list)
    for m in root_majmin:
        metrics_.average_score[m] = np.sum(np.multiply(metrics_.score_list_dict[m], tmp))

    return metrics_.score_list_dict, song_length_list, metrics_.average_score

def root_majmin_score_calculation_crf(valid_dataset, config, mean, std, device, pre_model, model, model_type, verbose=False):
    valid_song_names = valid_dataset.song_names
    paths = valid_dataset.preprocessor.get_all_files()

    metrics_ = metrics()
    song_length_list = list()
    for path in paths:
        song_name, lab_file_path, mp3_file_path, _ = path
        if not song_name in valid_song_names:
            continue
        try:
            n_timestep = config.model['timestep']
            feature, feature_per_second, song_length_second = audio_file_to_features(mp3_file_path, config)
            feature = feature.T
            feature = (feature - mean) / std
            time_unit = feature_per_second

            num_pad = n_timestep - (feature.shape[0] % n_timestep)
            feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
            num_instance = feature.shape[0] // n_timestep

            start_time = 0.0
            lines = []
            with torch.no_grad():
                model.eval()
                feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
                for t in range(num_instance):
                    if (model_type == 'cnn') or (model_type == 'crnn') or (model_type == 'btc'):
                        logits = pre_model(feature[:, n_timestep * t:n_timestep * (t + 1), :], torch.randint(config.model['num_chords'], (n_timestep,)).to(device))
                        prediction, _ = model(logits, torch.randint(config.model['num_chords'], (n_timestep,)).to(device))
                    else:
                        raise NotImplementedError
                    for i in range(n_timestep):
                        if t == 0 and i == 0:
                            prev_chord = prediction[i].item()
                            continue
                        if prediction[i].item() != prev_chord:
                            lines.append(
                                '%.6f %.6f %s\n' % (
                                    start_time, time_unit * (n_timestep * t + i), idx2chord[prev_chord]))
                            start_time = time_unit * (n_timestep * t + i)
                            prev_chord = prediction[i].item()
                        if t == num_instance - 1 and i + num_pad == n_timestep:
                            if start_time != time_unit * (n_timestep * t + i):
                                lines.append(
                                    '%.6f %.6f %s\n' % (
                                        start_time, time_unit * (n_timestep * t + i), idx2chord[prev_chord]))
                            break
            pid = os.getpid()
            tmp_path = 'tmp_' + str(pid) + '.lab'
            with open(tmp_path, 'w') as f:
                for line in lines:
                    f.write(line)

            root_majmin = ['root', 'majmin']
            for m in root_majmin:
                metrics_.score_list_dict[m].append(metrics_.score(metric=m, gt_path=lab_file_path, est_path=tmp_path))
            song_length_list.append(song_length_second)
            if verbose:
                for m in root_majmin:
                    print('song name %s, %s score : %.4f' % (song_name, m, metrics_.score_list_dict[m][-1]))
        except:
            print('song name %s\' lab file error' % song_name)

    tmp = song_length_list / np.sum(song_length_list)
    for m in root_majmin:
        metrics_.average_score[m] = np.sum(np.multiply(metrics_.score_list_dict[m], tmp))

    return metrics_.score_list_dict, song_length_list, metrics_.average_score


def large_voca_score_calculation(valid_dataset, config, mean, std, device, model, model_type, verbose=False):
    idx2voca = idx2voca_chord()
    valid_song_names = valid_dataset.song_names
    paths = valid_dataset.preprocessor.get_all_files()

    metrics_ = metrics()
    song_length_list = list()
    for path in paths:
        song_name, lab_file_path, mp3_file_path, _ = path
        if not song_name in valid_song_names:
            continue
        try:
            n_timestep = config.model['timestep']
            feature, feature_per_second, song_length_second = audio_file_to_features(mp3_file_path, config)
            feature = feature.T
            feature = (feature - mean) / std
            time_unit = feature_per_second

            num_pad = n_timestep - (feature.shape[0] % n_timestep)
            feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
            num_instance = feature.shape[0] // n_timestep

            start_time = 0.0
            lines = []
            with torch.no_grad():
                model.eval()
                feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
                for t in range(num_instance):
                    if model_type == 'btc':
                        encoder_output, _ = model.self_attn_layers(feature[:, n_timestep * t:n_timestep * (t + 1), :])
                        prediction, _ = model.output_layer(encoder_output)
                        prediction = prediction.squeeze()
                    elif model_type == 'cnn' or model_type =='crnn':
                        prediction, _, _, _ = model(feature[:, n_timestep * t:n_timestep * (t + 1), :], torch.randint(config.model['num_chords'], (n_timestep,)).to(device))
                    for i in range(n_timestep):
                        if t == 0 and i == 0:
                            prev_chord = prediction[i].item()
                            continue
                        if prediction[i].item() != prev_chord:
                            lines.append(
                                '%.6f %.6f %s\n' % (
                                    start_time, time_unit * (n_timestep * t + i), idx2voca[prev_chord]))
                            start_time = time_unit * (n_timestep * t + i)
                            prev_chord = prediction[i].item()
                        if t == num_instance - 1 and i + num_pad == n_timestep:
                            if start_time != time_unit * (n_timestep * t + i):
                                lines.append(
                                    '%.6f %.6f %s\n' % (
                                        start_time, time_unit * (n_timestep * t + i), idx2voca[prev_chord]))
                            break
            pid = os.getpid()
            tmp_path = 'tmp_' + str(pid) + '.lab'
            with open(tmp_path, 'w') as f:
                for line in lines:
                    f.write(line)

            for m in metrics_.score_metrics:
                metrics_.score_list_dict[m].append(metrics_.score(metric=m, gt_path=lab_file_path, est_path=tmp_path))
            song_length_list.append(song_length_second)
            if verbose:
                for m in metrics_.score_metrics:
                    print('song name %s, %s score : %.4f' % (song_name, m, metrics_.score_list_dict[m][-1]))
        except:
            print('song name %s\' lab file error' % song_name)

    tmp = song_length_list / np.sum(song_length_list)
    for m in metrics_.score_metrics:
        metrics_.average_score[m] = np.sum(np.multiply(metrics_.score_list_dict[m], tmp))

    return metrics_.score_list_dict, song_length_list, metrics_.average_score

def large_voca_score_calculation_crf(valid_dataset, config, mean, std, device, pre_model, model, model_type, verbose=False):
    idx2voca = idx2voca_chord()
    valid_song_names = valid_dataset.song_names
    paths = valid_dataset.preprocessor.get_all_files()

    metrics_ = metrics()
    song_length_list = list()
    for path in paths:
        song_name, lab_file_path, mp3_file_path, _ = path
        if not song_name in valid_song_names:
            continue
        try:
            n_timestep = config.model['timestep']
            feature, feature_per_second, song_length_second = audio_file_to_features(mp3_file_path, config)
            feature = feature.T
            feature = (feature - mean) / std
            time_unit = feature_per_second

            num_pad = n_timestep - (feature.shape[0] % n_timestep)
            feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
            num_instance = feature.shape[0] // n_timestep

            start_time = 0.0
            lines = []
            with torch.no_grad():
                model.eval()
                feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
                for t in range(num_instance):
                    if (model_type == 'cnn') or (model_type == 'crnn') or (model_type == 'btc'):
                        logits = pre_model(feature[:, n_timestep * t:n_timestep * (t + 1), :], torch.randint(config.model['num_chords'], (n_timestep,)).to(device))
                        prediction, _ = model(logits, torch.randint(config.model['num_chords'], (n_timestep,)).to(device))
                    else:
                        raise NotImplementedError
                    for i in range(n_timestep):
                        if t == 0 and i == 0:
                            prev_chord = prediction[i].item()
                            continue
                        if prediction[i].item() != prev_chord:
                            lines.append(
                                '%.6f %.6f %s\n' % (
                                    start_time, time_unit * (n_timestep * t + i), idx2voca[prev_chord]))
                            start_time = time_unit * (n_timestep * t + i)
                            prev_chord = prediction[i].item()
                        if t == num_instance - 1 and i + num_pad == n_timestep:
                            if start_time != time_unit * (n_timestep * t + i):
                                lines.append(
                                    '%.6f %.6f %s\n' % (
                                        start_time, time_unit * (n_timestep * t + i), idx2voca[prev_chord]))
                            break
            pid = os.getpid()
            tmp_path = 'tmp_' + str(pid) + '.lab'
            with open(tmp_path, 'w') as f:
                for line in lines:
                    f.write(line)

            for m in metrics_.score_metrics:
                metrics_.score_list_dict[m].append(metrics_.score(metric=m, gt_path=lab_file_path, est_path=tmp_path))
            song_length_list.append(song_length_second)
            if verbose:
                for m in metrics_.score_metrics:
                    print('song name %s, %s score : %.4f' % (song_name, m, metrics_.score_list_dict[m][-1]))
        except:
            print('song name %s\' lab file error' % song_name)

    tmp = song_length_list / np.sum(song_length_list)
    for m in metrics_.score_metrics:
        metrics_.average_score[m] = np.sum(np.multiply(metrics_.score_list_dict[m], tmp))

    return metrics_.score_list_dict, song_length_list, metrics_.average_score
