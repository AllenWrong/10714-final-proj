"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype, requires_grad=True))
        self.use_bias = bias
        if bias:
            self.bias = Parameter(
                            init.kaiming_uniform(
                                out_features, 1, 
                                device=device, 
                                dtype=dtype, 
                                requires_grad=True
                            ).reshape((1, out_features))
                        )
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.use_bias:
            return X @ self.weight + self.bias
        else:
            return X @ self.weight
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        shapes = X.shape[1:]
        prod = 1
        for it in shapes:
            prod *= it
        return ops.reshape(X, shape=(X.shape[0], prod))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.tanh(x)
        ### END YOUR SOLUTION


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        ex = ops.exp(x)
        return ops.divide(ex, ops.add_scalar(ex, 1))
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for m in self.modules:
            x = m(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        z_y = ops.summation(
            logits * init.one_hot(logits.shape[1], y, device=logits.device), 
            axes=1
        )
        return ops.summation(ops.logsumexp(logits, axes=(1,)) - z_y) / logits.shape[0]
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype, requires_grad=False)
        self.running_var = init.ones(dim, device=device, dtype=dtype, requires_grad=False)
        self.device = device
        self.dtype = dtype
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            bs = x.shape[0]
            mux_o = ops.divide_scalar(ops.summation(x, 0), bs)
            mux = ops.reshape(mux_o, shape=(1, self.dim))
            x_sub_mu = x - mux

            power = ops.power_scalar(x_sub_mu, 2)
            varx_o = ops.summation(power, 0) / bs
            varx = ops.reshape(varx_o, (1, self.dim))

            varx = ops.broadcast_to(varx, x.shape)

            std_eq = x_sub_mu / ops.power_scalar(varx + self.eps, 1/2)
            w = ops.broadcast_to(ops.reshape(self.weight, (1, self.dim)), x.shape)
            b = ops.broadcast_to(ops.reshape(self.bias, (1, self.dim)), x.shape)

            self.running_var = self.running_var * (1 - self.momentum) + varx_o.data * self.momentum
            self.running_mean = self.running_mean * (1 - self.momentum) + mux_o.data * self.momentum

            return w * std_eq + b
        else:
            std_x = (x - self.running_mean) / ops.power_scalar(self.running_var + self.eps, 1/2)
            w = ops.broadcast_to(ops.reshape(self.weight, (1, self.dim)), x.shape)
            b = ops.broadcast_to(ops.reshape(self.bias, (1, self.dim)), x.shape)
            return w * std_x + b
        ### END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))
        self.device = device
        self.dtype = dtype
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        mux = ops.divide_scalar(ops.summation(x, 1), self.dim)
        mux = ops.reshape(mux, shape=(mux.shape[0], 1))
        x_sub_mu = x - mux

        power = ops.power_scalar(x_sub_mu, 2)
        varx = ops.summation(power, 1) / self.dim
        varx = ops.reshape(varx, (x.shape[0], 1))

        varx = ops.broadcast_to(varx, x.shape)

        std_eq = x_sub_mu / ops.power_scalar(varx + self.eps, 1 / 2)
        w = ops.broadcast_to(ops.reshape(self.weight, (1, self.dim)), x.shape)
        b = ops.broadcast_to(ops.reshape(self.bias, (1, self.dim)), x.shape)

        return w * std_eq + b
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = init.randb(*x.shape, dtype=x.dtype, p=1-self.p)
            mask = mask / (1 - self.p)
            return x * mask
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        # compute padding
        self.padding = kernel_size // 2
        self.weight = Parameter(
            init.kaiming_uniform(
                self.in_channels * self.kernel_size * self.kernel_size, 
                self.out_channels * self.kernel_size * self.kernel_size, 
                shape = (
                    self.kernel_size,
                    self.kernel_size,
                    self.in_channels,
                    self.out_channels),
                device=device, 
                dtype=dtype
            )
        )
        if bias:
            bound = 1.0/(in_channels * kernel_size * kernel_size)
            bound = bound ** 0.5
            self.bias = Parameter(
                init.rand(
                    out_channels, low=-bound, 
                    high=bound, device=device, 
                    dtype=dtype)
            )
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """input x is NCHW"""
        ### BEGIN YOUR SOLUTION
        # x = x.transpose((0, 2, 3, 1))
        x = x.transpose((1, 3)).transpose((1, 2))
        # NHWC
        out = ops.conv(x, self.weight, stride=self.stride, padding=self.padding)
        
        if self.bias is not None:
            bias = ops.reshape(self.bias, (1, 1, 1, self.out_channels))
            bias = ops.broadcast_to(bias, out.shape)
            out = out + bias # NHWC
            out = ops.transpose(out, (1, 3))
            return ops.transpose(out, (2, 3))
            # return ops.transpose(out, (1, 3))
            # return ops.transpose(out, (0, 3, 1, 2))

        out = ops.transpose(out, (1, 3))
        return ops.transpose(out, (2, 3))
        # return ops.transpose(out, (0, 3, 1, 2))
        ### END YOUR SOLUTION


class ConvBN(Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size,
        stride=1,
        eps=1e-5, 
        momentum=0.1, 
        bias=True, 
        device=None, 
        dtype="float32"):
        
        super().__init__()
        self.conv = Conv(in_channels, out_channels, kernel_size, stride, bias, device, dtype)
        self.bn = BatchNorm2d(out_channels, eps, momentum, device, dtype)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv.forward(x)
        x = self.bn.forward(x)
        return ops.relu(x)


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype

        bound = (1/hidden_size) ** 0.5
        self.W_ih = Parameter(
            init.rand(input_size, hidden_size, low=-bound, high=bound,
                device=device, dtype=dtype, requires_grad=True)
        )
        self.W_hh = Parameter(
            init.rand(hidden_size, hidden_size, low=-bound, high=bound,
                device=device, dtype=dtype, requires_grad=True)
        )
        if bias:
            self.bias = True
            self.bias_hh = Parameter(
                init.rand(hidden_size, low=-bound, high=bound,
                    device=device, dtype=dtype, requires_grad=True)
            )
            self.bias_ih = Parameter(
                init.rand(hidden_size, low=-bound, high=bound,
                    device=device, dtype=dtype, requires_grad=True)
            )
        else:
            self.bias = False
        
        if nonlinearity == "tanh":
            self.activate = Tanh()
        elif nonlinearity == "relu":
            self.activate = ReLU()
        else:
            raise Exception(f"Unsupported activation '{nonlinearity}'")
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]
        if h is None:
            h = init.zeros(bs, self.hidden_size,
                           device=self.device, dtype=self.dtype)
        z = X @ self.W_ih + h @ self.W_hh
        
        if self.bias:
            # add bias
            hidden_size = self.bias_ih.shape[0]
            
            z += ops.reshape(self.bias_ih, (1, hidden_size)).broadcast_to((bs, hidden_size))
            z += ops.reshape(self.bias_hh, (1, hidden_size)).broadcast_to((bs, hidden_size))
        return self.activate(z)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.rnn_cells = [RNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype)]

        for _ in range(1, num_layers):
            self.rnn_cells.append(
                RNNCell(hidden_size, hidden_size, bias, nonlinearity, device, dtype)
            )

        self.device = device
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION        
        # iterate X with seq_len
        seq_len, bs, input_size = X.shape
        
        h_ts = []    # output features from the last layer of RNN
        h_ns = []    # hidden state
        if h0 is None:
            h0 = init.zeros(self.num_layers, bs, self.hidden_size, device=self.device)
        h0 = ops.split(h0, axis=0)
        
        # build the init h_ns
        for it in h0:
            h_ns.append(it)
        assert len(h_ns) == self.num_layers

        xs = ops.split(X, axis=0)

        for t in range(seq_len):
            # pass every layer
            xi = xs[t]
            for i in range(self.num_layers):
                xi = self.rnn_cells[i](xi, h_ns[i])
                h_ns[i] = xi
            
            h_ts.append(xi)

        return ops.stack(tuple(h_ts), axis=0), ops.stack(tuple(h_ns), axis=0)
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION

        bound = (1 / hidden_size) ** 0.5
        def init_param(shape, bound=bound, device=device, dtype=dtype):
            return Parameter(init.rand(
                *shape, low=-bound, high=bound,
                device=device, dtype=dtype, requires_grad=True
            ))

        self.W_ih = init_param((input_size, 4 * hidden_size))
        self.W_hh = init_param((hidden_size, 4 * hidden_size))
        self.bias_ih = init_param((4 * hidden_size, )) if bias else None
        self.bias_hh = init_param((4 * hidden_size, )) if bias else None

        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype

        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs, input_size = X.shape
        if h is None:
            h0 = init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype)
            c0 = init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype)
        else:
            h0, c0 = h
        
        sigmoid = Sigmoid()
        tanh = Tanh()

        out = X @ self.W_ih + h0 @ self.W_hh

        # if using bias, add bias
        if self.bias_ih is not None:
            bias_ih = self.bias_ih.reshape((1, 4 * self.hidden_size))
            bias_ih = ops.broadcast_to(bias_ih, (bs, 4 * self.hidden_size))
            bias_hh = self.bias_hh.reshape((1, 4 * self.hidden_size))
            bias_hh = ops.broadcast_to(bias_hh, (bs, 4 * self.hidden_size))
            out += bias_ih + bias_hh

        i, f, g, o = ops.split(out.reshape((bs, 4, self.hidden_size)), axis=1)
        i = sigmoid(i)
        f = sigmoid(f)
        g = tanh(g)
        o = sigmoid(o)

        c_1 = f * c0 + g * i
        h_1 = o * tanh(c_1)

        return h_1, c_1      
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.device = device
        self.dtype = dtype
        
        self.lstm_cells = [LSTMCell(input_size, hidden_size, bias, device, dtype)]
        for i in range(1, num_layers):
            self.lstm_cells.append(LSTMCell(hidden_size, hidden_size, bias, device, dtype))
        
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, input_size = X.shape
        xs = ops.split(X, axis=0)
        
        output = []
        if h is None:
            h = (init.zeros(self.num_layers, bs, self.hidden_size, device=self.device, dtype=self.dtype),
                 init.zeros(self.num_layers, bs, self.hidden_size, device=self.device, dtype=self.dtype))
        
        h_n = list(ops.split(h[0], axis=0))
        c_n = list(ops.split(h[1], axis=0))

        for t in range(seq_len):
            x = xs[t]
            xl = x

            # pass every layer
            for i in range(self.num_layers):
                xl, cl = self.lstm_cells[i].forward(xl, (h_n[i], c_n[i]))
                h_n[i] = xl
                c_n[i] = cl

            output.append(xl)

        return ops.stack(output, axis=0), (ops.stack(h_n, axis=0), ops.stack(c_n, axis=0))
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.randn(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        self.device = device
        self.dtype = dtype
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        num_embeddings, embedding_dim = self.weight.shape
        seq_len, bs = x.shape
        one_hot = init.one_hot(num_embeddings, x.reshape(
            (seq_len * bs,)), device=self.device, dtype=self.dtype)
        output = one_hot @ self.weight
        return output.reshape((seq_len, bs, embedding_dim))
        ### END YOUR SOLUTION
