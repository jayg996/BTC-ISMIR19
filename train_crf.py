import os
from torch import optim
from utils import logger
from audio_dataset import AudioDataset, AudioDataLoader
from utils.tf_logger import TF_Logger
from btc_model import *
from baseline_models import CNN, CRNN, Crf
from utils.hparams import HParams
import argparse
from utils.pytorch_utils import adjusting_learning_rate
from utils.mir_eval_modules import large_voca_score_calculation_crf, root_majmin_score_calculation_crf
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logger.logging_verbosity(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--index', type=int, help='Experiment Number', default='e')
parser.add_argument('--kfold', type=int, help='5 fold (0,1,2,3,4)',default='e')
parser.add_argument('--voca', type=bool, help='large voca is True', default=False)
parser.add_argument('--model', type=str, default='crf')
parser.add_argument('--pre_model', type=str, help='btc, cnn, crnn', default='e')
parser.add_argument('--dataset1', type=str, help='Dataset', default='isophonic_221')
parser.add_argument('--dataset2', type=str, help='Dataset', default='uspop_185')
parser.add_argument('--dataset3', type=str, help='Dataset', default='robbiewilliams')
parser.add_argument('--restore_epoch', type=int, default=1000)
parser.add_argument('--early_stop', type=bool, help='no improvement during 10 epoch -> stop', default=True)
args = parser.parse_args()

config = HParams.load("run_config.yaml")
if args.voca == True:
    config.feature['large_voca'] = True
    config.model['num_chords'] = 170

config.model['probs_out'] = True

# Result save path
asset_path = config.path['asset_path']
ckpt_path = config.path['ckpt_path']
result_path = config.path['result_path']
restore_epoch = args.restore_epoch
experiment_num = str(args.index)
ckpt_file_name = 'idx_'+experiment_num+'_%03d.pth.tar'
tf_logger = TF_Logger(os.path.join(asset_path, 'tensorboard', 'idx_'+experiment_num))
logger.info("==== Experiment Number : %d " % args.index)

if args.pre_model == 'cnn':
    config.experiment['batch_size'] = 20

# Data loader
train_dataset1 = AudioDataset(config, root_dir=config.path['root_path'], dataset_names=(args.dataset1,), num_workers=20, preprocessing=False, train=True, kfold=args.kfold)
train_dataset2 = AudioDataset(config, root_dir=config.path['root_path'], dataset_names=(args.dataset2,), num_workers=20, preprocessing=False, train=True, kfold=args.kfold)
train_dataset3 = AudioDataset(config, root_dir=config.path['root_path'], dataset_names=(args.dataset3,), num_workers=20, preprocessing=False, train=True, kfold=args.kfold)
train_dataset = train_dataset1.__add__(train_dataset2).__add__(train_dataset3)
valid_dataset1 = AudioDataset(config, root_dir=config.path['root_path'], dataset_names=(args.dataset1,), preprocessing=False, train=False, kfold=args.kfold)
valid_dataset2 = AudioDataset(config, root_dir=config.path['root_path'], dataset_names=(args.dataset2,), preprocessing=False, train=False, kfold=args.kfold)
valid_dataset3 = AudioDataset(config, root_dir=config.path['root_path'], dataset_names=(args.dataset3,), preprocessing=False, train=False, kfold=args.kfold)
valid_dataset = valid_dataset1.__add__(valid_dataset2).__add__(valid_dataset3)
train_dataloader = AudioDataLoader(dataset=train_dataset, batch_size=config.experiment['batch_size'], drop_last=False, shuffle=True)
valid_dataloader = AudioDataLoader(dataset=valid_dataset, batch_size=config.experiment['batch_size'], drop_last=False)

# Model and Optimizer
if args.pre_model == 'cnn':
    pre_model = CNN(config=config.model).to(device)
elif args.pre_model == 'crnn':
    pre_model = CRNN(config=config.model).to(device)
elif args.pre_model == 'btc':
    pre_model = BTC_model(config=config.model).to(device)
else: raise NotImplementedError

if args.pre_model == 'cnn':
    if args.voca == False:
        if args.kfold == 0:
            load_ckpt_file_name = 'idx_0_%03d.pth.tar'
            load_restore_epoch = 10
    else:
        if args.kfold == 0:
            load_ckpt_file_name = 'idx_1_%03d.pth.tar'
            load_restore_epoch = 10
else:
    raise NotImplementedError

if os.path.isfile(os.path.join(asset_path, ckpt_path, load_ckpt_file_name % load_restore_epoch)):
    checkpoint = torch.load(os.path.join(asset_path, ckpt_path, load_ckpt_file_name % load_restore_epoch))
    pre_model.load_state_dict(checkpoint['model'])
    logger.info("restore pre model with %d epochs" % load_restore_epoch)
else:
    raise NotImplementedError

# Fix Pre Model Parameters
for param in pre_model.parameters():
    param.requires_grad = False

# Crf Model and Optimizer
crf = Crf(num_chords=config.model['num_chords'], timestep=config.model['timestep']).to(device)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, crf.parameters()), lr=0.01, weight_decay=config.experiment['weight_decay'], betas=(0.9, 0.98), eps=1e-9)

# Make asset directory
if not os.path.exists(os.path.join(asset_path, ckpt_path)):
    os.makedirs(os.path.join(asset_path, ckpt_path))
    os.makedirs(os.path.join(asset_path, result_path))

# Load model
if os.path.isfile(os.path.join(asset_path, ckpt_path, ckpt_file_name % restore_epoch)):
    checkpoint = torch.load(os.path.join(asset_path, ckpt_path, ckpt_file_name % restore_epoch))
    crf.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    logger.info("restore model with %d epochs" % restore_epoch)
else:
    logger.info("no checkpoint with %d epochs" % restore_epoch)
    restore_epoch = 0

# Global mean and variance calculate
mp3_config = config.mp3
feature_config = config.feature
mp3_string = "%d_%.1f_%.1f" % (mp3_config['song_hz'], mp3_config['inst_len'], mp3_config['skip_interval'])
feature_string = "_%s_%d_%d_%d_" % ('cqt', feature_config['n_bins'], feature_config['bins_per_octave'], feature_config['hop_length'])
z_path = os.path.join(config.path['root_path'], 'result', mp3_string + feature_string + 'mix_kfold_'+ str(args.kfold) +'_normalization.pt')
if os.path.exists(z_path):
    normalization = torch.load(z_path)
    mean = normalization['mean']
    std = normalization['std']
    logger.info("Global mean and std (k fold index %d) load complete" % args.kfold)
else:
    mean = 0
    square_mean = 0
    k = 0
    for i, data in enumerate(train_dataloader):
        features, input_percentages, chords, collapsed_chords, chord_lens, boundaries = data
        features = features.to(device)
        mean += torch.mean(features).item()
        square_mean += torch.mean(features.pow(2)).item()
        k += 1
    square_mean = square_mean / k
    mean = mean / k
    std = np.sqrt(square_mean - mean * mean)
    normalization = dict()
    normalization['mean'] = mean
    normalization['std'] = std
    torch.save(normalization, z_path)
    logger.info("Global mean and std (training set, k fold index %d) calculation complete" % args.kfold)

current_step = 0
best_acc = 0
before_acc = 0
early_stop_idx = 0
pre_model.eval()
for epoch in range(restore_epoch, config.experiment['max_epoch']):
    # Training
    crf.train()
    train_loss_list = []
    total = 0.
    correct = 0.
    second_correct = 0.
    for i, data in enumerate(train_dataloader):
        features, input_percentages, chords, collapsed_chords, chord_lens, boundaries = data
        features, chords = features.to(device), chords.to(device)

        features.requires_grad = True
        features = (features - mean) / std

        # forward
        features = features.squeeze(1).permute(0,2,1)
        optimizer.zero_grad()
        logits = pre_model(features, chords)
        if args.pre_model == 'crnn':
            logits = logits.detach()
            logits.requires_grad = True
        prediction, total_loss = crf(logits, chords)

        # save accuracy and loss
        total += chords.size(0)
        correct += (prediction == chords).type_as(chords).sum()
        train_loss_list.append(total_loss.item())

        # optimize step
        total_loss.backward()
        optimizer.step()

        current_step += 1

    # logging loss and accuracy using tensorboard
    result = {'loss/tr': np.mean(train_loss_list), 'acc/tr': correct.item() / total}
    for tag, value in result.items(): tf_logger.scalar_summary(tag, value, epoch+1)
    logger.info("training loss for %d epoch: %.4f" % (epoch + 1, np.mean(train_loss_list)))
    logger.info("training accuracy for %d epoch: %.4f" % (epoch + 1, (correct.item() / total)))

    # Validation
    with torch.no_grad():
        crf.eval()
        val_total = 0.
        val_correct = 0.
        val_second_correct = 0.
        validation_loss = 0
        n = 0
        for i, data in enumerate(valid_dataloader):
            val_features, val_input_percentages, val_chords, val_collapsed_chords, val_chord_lens, val_boundaries = data
            val_features, val_chords = val_features.to(device), val_chords.to(device)

            val_features = (val_features - mean) / std

            val_features = val_features.squeeze(1).permute(0, 2, 1)
            val_logits = pre_model(val_features, val_chords)
            val_prediction, val_loss = crf(val_logits, val_chords)

            val_total += val_chords.size(0)
            val_correct += (val_prediction == val_chords).type_as(val_chords).sum()
            validation_loss += val_loss.item()

            n += 1

        # logging loss and accuracy using tensorboard
        validation_loss /= n
        result = {'loss/val': validation_loss, 'acc/val': val_correct.item() / val_total}
        for tag, value in result.items(): tf_logger.scalar_summary(tag, value, epoch + 1)
        logger.info("validation loss(%d): %.4f" % (epoch + 1, validation_loss))
        logger.info("validation accuracy(%d): %.4f" % (epoch + 1, (val_correct.item() / val_total)))

        current_acc = val_correct.item() / val_total

        if best_acc < val_correct.item() / val_total:
            early_stop_idx = 0
            best_acc = val_correct.item() / val_total
            logger.info('==== best accuracy is %.4f and epoch is %d' % (best_acc, epoch + 1))
            logger.info('saving model, Epoch %d, step %d' % (epoch + 1, current_step + 1))
            model_save_path = os.path.join(asset_path, 'model', ckpt_file_name % (epoch + 1))
            state_dict = {'model': crf.state_dict(),'optimizer': optimizer.state_dict(),'epoch': epoch}
            torch.save(state_dict, model_save_path)
            last_best_epoch = epoch + 1

        # save model
        elif (epoch + 1) % config.experiment['save_step'] == 0:
            logger.info('saving model, Epoch %d, step %d' % (epoch + 1, current_step + 1))
            model_save_path = os.path.join(asset_path, 'model', ckpt_file_name % (epoch + 1))
            state_dict = {'model': crf.state_dict(),'optimizer': optimizer.state_dict(),'epoch': epoch}
            torch.save(state_dict, model_save_path)
            early_stop_idx += 1
        else:
            early_stop_idx += 1

    if (args.early_stop == True) and (early_stop_idx > 5):
        logger.info('==== early stopped and epoch is %d' % (epoch + 1))
        break
    # learning rate decay
    if before_acc > current_acc:
        adjusting_learning_rate(optimizer=optimizer, factor=0.95, min_lr=5e-6)
    before_acc = current_acc

# Load model
if os.path.isfile(os.path.join(asset_path, ckpt_path, ckpt_file_name % last_best_epoch)):
    checkpoint = torch.load(os.path.join(asset_path, ckpt_path, ckpt_file_name % last_best_epoch))
    crf.load_state_dict(checkpoint['model'])
    logger.info("last best restore model with %d epochs" % last_best_epoch)
else:
    raise NotImplementedError

# score Validation
if args.voca == True:
    score_metrics = ['root', 'thirds', 'triads', 'sevenths', 'tetrads', 'majmin', 'mirex']
    score_list_dict1, song_length_list1, average_score_dict1 = large_voca_score_calculation_crf(valid_dataset=valid_dataset1, config=config, pre_model=pre_model, model=crf, model_type=args.pre_model, mean=mean, std=std, device=device)
    score_list_dict2, song_length_list2, average_score_dict2 = large_voca_score_calculation_crf(valid_dataset=valid_dataset2, config=config, pre_model=pre_model, model=crf, model_type=args.pre_model, mean=mean, std=std, device=device)
    score_list_dict3, song_length_list3, average_score_dict3 = large_voca_score_calculation_crf(valid_dataset=valid_dataset3, config=config, pre_model=pre_model, model=crf, model_type=args.pre_model, mean=mean, std=std, device=device)
    for m in score_metrics:
        average_score = (np.sum(song_length_list1) * average_score_dict1[m] + np.sum(song_length_list2) *average_score_dict2[m] + np.sum(song_length_list3) * average_score_dict3[m]) / (np.sum(song_length_list1) + np.sum(song_length_list2) + np.sum(song_length_list3))
        logger.info('==== %s score 1 is %.4f' % (m, average_score_dict1[m]))
        logger.info('==== %s score 2 is %.4f' % (m, average_score_dict2[m]))
        logger.info('==== %s score 3 is %.4f' % (m, average_score_dict3[m]))
        logger.info('==== %s mix average score is %.4f' % (m, average_score))
else:
    score_metrics = ['root', 'majmin']
    score_list_dict1, song_length_list1, average_score_dict1 = root_majmin_score_calculation_crf(valid_dataset=valid_dataset1, config=config, pre_model=pre_model, model=crf, model_type=args.pre_model, mean=mean, std=std, device=device)
    score_list_dict2, song_length_list2, average_score_dict2 = root_majmin_score_calculation_crf(valid_dataset=valid_dataset2, config=config, pre_model=pre_model, model=crf, model_type=args.pre_model, mean=mean, std=std, device=device)
    score_list_dict3, song_length_list3, average_score_dict3 = root_majmin_score_calculation_crf(valid_dataset=valid_dataset3, config=config, pre_model=pre_model, model=crf, model_type=args.pre_model, mean=mean, std=std, device=device)
    for m in score_metrics:
        average_score = (np.sum(song_length_list1) * average_score_dict1[m] + np.sum(song_length_list2) *average_score_dict2[m] + np.sum(song_length_list3) * average_score_dict3[m]) / (np.sum(song_length_list1) + np.sum(song_length_list2) + np.sum(song_length_list3))
        logger.info('==== %s score 1 is %.4f' % (m, average_score_dict1[m]))
        logger.info('==== %s score 2 is %.4f' % (m, average_score_dict2[m]))
        logger.info('==== %s score 3 is %.4f' % (m, average_score_dict3[m]))
        logger.info('==== %s mix average score is %.4f' % (m, average_score))
