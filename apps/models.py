import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)


class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.convbn1 = nn.ConvBN(3, 16, 7, 4, device=device, dtype=dtype)
        self.convbn2 = nn.ConvBN(16, 32, 3, 2, device=device, dtype=dtype)
        self.resd1 = nn.Residual(
            nn.Sequential(
                nn.ConvBN(32, 32, 3, 1, device=device, dtype=dtype),
                nn.ConvBN(32, 32, 3, 1, device=device, dtype=dtype)
            )
        )

        self.convbn5 = nn.ConvBN(32, 64, 3, 2, device=device, dtype=dtype)
        self.convbn6 = nn.ConvBN(64, 128, 3, 2, device=device, dtype=dtype)
        self.resd2 = nn.Residual(
            nn.Sequential(
                nn.ConvBN(128, 128, 3, 1, device=device, dtype=dtype),
                nn.ConvBN(128, 128, 3, 1, device=device, dtype=dtype)
            )
        )
        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(128, 128, device=device, dtype=dtype)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 10, device=device, dtype=dtype)

        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        x = self.convbn1(x)
        x = self.convbn2(x)
        x = self.resd1(x)

        x = self.convbn5(x)
        x = self.convbn6(x)
        x = self.resd2(x)
        
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        return self.linear2(x)
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.emb = nn.Embedding(output_size, embedding_size, device=device, dtype=dtype)
        if seq_model == "rnn":
            self.model = nn.RNN(
                embedding_size, hidden_size, num_layers, 
                nonlinearity='tanh', device=device, dtype=dtype
            )
        else:
            self.model = nn.LSTM(
                embedding_size, hidden_size, num_layers, device=device, dtype=dtype
            )
        self.linear_layer = nn.Linear(
            hidden_size, output_size, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using LSTM,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        emb = self.emb(x)
        
        # rnn:  out (seq_len, bs, hidden_size), h_n (num_layers, bs, hidden_size)
        # lstm:  (out, (h_n, c_n))
        #    out: (seq_len, bs, hidden_size)
        #    h_n: (num_layers, bs, hidden_size)
        #    c_n: (num_layers, bs, hidden_size)
        out, h = self.model(emb, h)

        seq_len, bs, hidden_size = out.shape
        out = self.linear_layer(out.reshape((seq_len*bs, hidden_size)))
        return out, h

        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(dataset[1][0].shape)