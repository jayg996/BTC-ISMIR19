import os
from utils import logger
from btc_model import *
from utils.mir_eval_modules import audio_file_to_features, idx2chord

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

config = HParams.load("run_config.yaml")
model = BTC_model(config=config.model).to(device)

# Load model
if os.path.isfile('test/btc_model.pt'):
    checkpoint = torch.load('test/btc_model.pt')
    mean = checkpoint['mean']
    std = checkpoint['std']
    model.load_state_dict(checkpoint['model'])
    logger.info("restore model")

# Load mp3
# example mp3 file is free music audio from http://www.freemusicarchive.org/
# Monplaisir_-_01_-_Everything_is_true
mp3_file_path = 'test/example.mp3'
feature, feature_per_second, song_length_second = audio_file_to_features(mp3_file_path, config)

# Majmin type chord recognition
feature = feature.T
feature = (feature - mean) / std
time_unit = feature_per_second
n_timestep = config.model['timestep']

num_pad = n_timestep - (feature.shape[0] % n_timestep)
feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
num_instance = feature.shape[0] // n_timestep

start_time = 0.0
lines = []
with torch.no_grad():
    model.eval()
    feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
    for t in range(num_instance):
        self_attn_output, _ = model.self_attn_layers(feature[:, n_timestep * t:n_timestep * (t + 1), :])
        prediction, _ = model.output_layer(self_attn_output)
        prediction = prediction.squeeze()
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

# lab file write
test_result_path = 'test/test.lab'
with open(test_result_path, 'w') as f:
    for line in lines:
        f.write(line)

