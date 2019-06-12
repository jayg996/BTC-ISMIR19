from utils.transformer_modules import *
from utils.transformer_modules import _gen_timing_signal, _gen_bias_mask
from utils.hparams import HParams

use_cuda = torch.cuda.is_available()

class self_attention_block(nn.Module):
    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads,
                 bias_mask=None, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0, attention_map=False):
        super(self_attention_block, self).__init__()

        self.attention_map = attention_map
        self.multi_head_attention = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth,hidden_size, num_heads, bias_mask, attention_dropout, attention_map)
        self.positionwise_convolution = PositionwiseFeedForward(hidden_size, filter_size, hidden_size, layer_config='cc', padding='both', dropout=relu_dropout)
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, inputs):
        x = inputs

        # Layer Normalization
        x_norm = self.layer_norm_mha(x)

        # Multi-head attention
        if self.attention_map is True:
            y, weights = self.multi_head_attention(x_norm, x_norm, x_norm)
        else:
            y = self.multi_head_attention(x_norm, x_norm, x_norm)

        # Dropout and residual
        x = self.dropout(x + y)

        # Layer Normalization
        x_norm = self.layer_norm_ffn(x)

        # Positionwise Feedforward
        y = self.positionwise_convolution(x_norm)

        # Dropout and residual
        y = self.dropout(x + y)

        if self.attention_map is True:
            return y, weights
        return y

class bi_directional_self_attention(nn.Module):
    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads, max_length,
                 layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):

        super(bi_directional_self_attention, self).__init__()

        self.weights_list = list()

        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  _gen_bias_mask(max_length),
                  layer_dropout,
                  attention_dropout,
                  relu_dropout,
                  True)

        self.attn_block = self_attention_block(*params)

        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  torch.transpose(_gen_bias_mask(max_length), dim0=2, dim1=3),
                  layer_dropout,
                  attention_dropout,
                  relu_dropout,
                  True)

        self.backward_attn_block = self_attention_block(*params)

        self.linear = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, inputs):
        x, list = inputs

        # Forward Self-attention Block
        encoder_outputs, weights = self.attn_block(x)
        # Backward Self-attention Block
        reverse_outputs, reverse_weights = self.backward_attn_block(x)
        # Concatenation and Fully-connected Layer
        outputs = torch.cat((encoder_outputs, reverse_outputs), dim=2)
        y = self.linear(outputs)

        # Attention weights for Visualization
        self.weights_list = list
        self.weights_list.append(weights)
        self.weights_list.append(reverse_weights)
        return y, self.weights_list

class bi_directional_self_attention_layers(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=100, input_dropout=0.0, layer_dropout=0.0,
                 attention_dropout=0.0, relu_dropout=0.0):
        super(bi_directional_self_attention_layers, self).__init__()

        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  max_length,
                  layer_dropout,
                  attention_dropout,
                  relu_dropout)
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.self_attn_layers = nn.Sequential(*[bi_directional_self_attention(*params) for l in range(num_layers)])
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs):
        # Add input dropout
        x = self.input_dropout(inputs)

        # Project to hidden size
        x = self.embedding_proj(x)

        # Add timing signal
        x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

        # A Stack of Bi-directional Self-attention Layers
        y, weights_list = self.self_attn_layers((x, []))

        # Layer Normalization
        y = self.layer_norm(y)
        return y, weights_list

class BTC_model(nn.Module):
    def __init__(self, config):
        super(BTC_model, self).__init__()

        self.timestep = config['timestep']
        self.probs_out = config['probs_out']

        params = (config['feature_size'],
                  config['hidden_size'],
                  config['num_layers'],
                  config['num_heads'],
                  config['total_key_depth'],
                  config['total_value_depth'],
                  config['filter_size'],
                  config['timestep'],
                  config['input_dropout'],
                  config['layer_dropout'],
                  config['attention_dropout'],
                  config['relu_dropout'])

        self.self_attn_layers = bi_directional_self_attention_layers(*params)
        self.output_layer = SoftmaxOutputLayer(hidden_size=config['hidden_size'], output_size=config['num_chords'], probs_out=config['probs_out'])

    def forward(self, x, labels):
        labels = labels.view(-1, self.timestep)
        # Output of Bi-directional Self-attention Layers
        self_attn_output, weights_list = self.self_attn_layers(x)

        # return logit values for CRF
        if self.probs_out is True:
            logits = self.output_layer(self_attn_output)
            return logits

        # Output layer and Soft-max
        prediction,second = self.output_layer(self_attn_output)
        prediction = prediction.view(-1)
        second = second.view(-1)

        # Loss Calculation
        loss = self.output_layer.loss(self_attn_output, labels)
        return prediction, loss, weights_list, second

if __name__ == "__main__":
    config = HParams.load("run_config.yaml")
    device = torch.device("cuda" if use_cuda else "cpu")

    batch_size = 2
    timestep = 108
    feature_size = 144
    num_chords = 25

    features = torch.randn(batch_size,timestep,feature_size,requires_grad=True).to(device)
    chords = torch.randint(25,(batch_size*timestep,)).to(device)

    model = BTC_model(config=config.model).to(device)

    prediction, loss, weights_list, second = model(features, chords)
    print(prediction.size())
    print(loss)


