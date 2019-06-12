from utils.hparams import HParams
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from crf_model import CRF

use_cuda = torch.cuda.is_available()

class CNN(nn.Module):
    def __init__(self,config):
        super(CNN, self).__init__()

        self.timestep = config['timestep']
        self.context = 7
        self.pad = nn.ConstantPad1d(self.context, 0)
        self.probs_out = config['probs_out']
        self.num_chords = config['num_chords']

        self.drop_out = nn.Dropout2d(p=0.5)
        self.conv1 = self.cnn_layers(1, 32, kernel_size=(3,3), padding=1)
        self.conv2 = self.cnn_layers(32, 32, kernel_size=(3,3), padding=1)
        self.conv3 = self.cnn_layers(32, 32, kernel_size=(3,3), padding=1)
        self.conv4 = self.cnn_layers(32, 32, kernel_size=(3,3), padding=1)
        self.pool_max = nn.MaxPool2d(kernel_size=(2,1))
        self.conv5 = self.cnn_layers(32, 64, kernel_size=(3, 3), padding=0)
        self.conv6 = self.cnn_layers(64, 64, kernel_size=(3, 3), padding=0)
        self.conv7 = self.cnn_layers(64, 128, kernel_size=(12, 9), padding=0)
        self.conv_linear = nn.Conv2d(128, config['num_chords'], kernel_size=(1,1), padding=0)

    def cnn_layers(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        layers = []
        conv2d = nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size, stride=stride, padding=padding)
        batch_norm = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        layers += [conv2d, batch_norm, relu]
        return nn.Sequential(*layers)

    def forward(self, x, labels):
        x = x.permute(0,2,1)
        x = self.pad(x)
        batch_size = x.size(0)
        for i in range(batch_size):
            for j in range(self.timestep):
                if i == 0 and j == 0:
                    inputs = x[i,:,j : j + self.context *2 + 1].unsqueeze(0)
                else:
                    tmp = x[i, :, j : j + self.context *2 + 1].unsqueeze(0)
                    inputs = torch.cat((inputs,tmp), dim=0)
        # inputs : [batchsize * timestep, feature_size, context]
        inputs = inputs.unsqueeze(1)
        conv = self.conv1(inputs)
        conv = self.conv2(conv)
        conv = self.conv3(conv)
        conv = self.conv4(conv)
        pooled = self.pool_max(conv)
        pooled = self.drop_out(pooled)
        conv = self.conv5(pooled)
        conv = self.conv6(conv)
        pooled = self.pool_max(conv)
        pooled = self.drop_out(pooled)
        conv = self.conv7(pooled)
        conv = self.drop_out(conv)
        conv = self.conv_linear(conv)
        avg_pool = nn.AvgPool2d(kernel_size=(conv.size(2), conv.size(3)))
        logits = avg_pool(conv).squeeze(2).squeeze(2)
        if self.probs_out is True:
            crf_input = logits.view(-1, self.timestep, self.num_chords)
            return crf_input
        log_probs = F.log_softmax(logits, -1)
        topk, indices = torch.topk(log_probs, 2)
        predictions = indices[:,0]
        second = indices[:,1]
        prediction = predictions.view(-1)
        second = second.view(-1)
        loss = F.nll_loss(log_probs.view(-1, self.num_chords), labels.view(-1))
        return prediction, loss, 0, second

class Crf(nn.Module):
    def __init__(self, num_chords, timestep):
        super(Crf, self).__init__()
        self.output_size = num_chords
        self.timestep = timestep
        self.Crf = CRF(self.output_size)

    def forward(self, probs, labels):
        prediction = self.Crf(probs)
        prediction = prediction.view(-1)
        labels = labels.view(-1, self.timestep)
        loss = self.Crf.loss(probs, labels)
        return prediction, loss

class CRNN(nn.Module):
    def __init__(self,config):
        super(CRNN, self).__init__()

        self.feature_size = config['feature_size']
        self.timestep = config['timestep']
        self.probs_out = config['probs_out']
        self.num_chords = config['num_chords']
        self.hidden_size = 128

        self.relu = nn.ReLU(inplace=True)
        self.batch_norm = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 1, kernel_size=(5,5), padding=2)
        self.conv2 = nn.Conv2d(1, 36, kernel_size=(1,self.feature_size))
        self.gru = nn.GRU(input_size=36, hidden_size=self.hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.hidden_size*2, self.num_chords)

    def forward(self, x, labels):
        # x : [batchsize * timestep * feature_size]
        x = x.unsqueeze(1)
        x = self.batch_norm(x)
        conv = self.relu(self.conv1(x))
        conv = self.relu(self.conv2(conv))
        conv = conv.squeeze(3).permute(0,2,1)

        h0 = torch.zeros(4, conv.size(0), self.hidden_size).to(torch.device("cuda" if use_cuda else "cpu"))
        gru, h = self.gru(conv, h0)
        logits = self.fc(gru)
        if self.probs_out is True:
            # probs = F.softmax(logits, -1)
            return logits
        log_probs = F.log_softmax(logits, -1)
        topk, indices = torch.topk(log_probs, 2)
        predictions = indices[:,:,0]
        second = indices[:,:,1]
        prediction = predictions.view(-1)
        second = second.view(-1)
        loss = F.nll_loss(log_probs.view(-1, self.num_chords), labels.view(-1))
        return prediction, loss, 0, second


if __name__ == "__main__":
    config = HParams.load("run_config.yaml")
    device = torch.device("cuda" if use_cuda else "cpu")
    config.model['probs_out'] = True
    batch_size = 2
    timestep = config.model['timestep']
    feature_size = config.model['feature_size']
    num_chords = config.model['num_chords']

    features = torch.randn(batch_size,timestep,feature_size,requires_grad=True).to(device)
    chords = torch.randint(num_chords,(batch_size*timestep,)).to(device)

    model = CNN(config=config.model).to(device)
    crf = Crf(num_chords=config.model['num_chords'], timestep=config.model['timestep']).to(device)

    probs = model(features, chords)
    prediction, total_loss = crf(probs, chords)

    print(total_loss)
