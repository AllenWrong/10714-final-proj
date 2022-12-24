"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray


def as_tuple(x):
    if hasattr(x, '__iter__'):
        return tuple(x)
    else:
        return tuple((x,))


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        assert isinstance(self.scalar, Number)
        return a + numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        assert isinstance(self.scalar, Number)
        return a * numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        assert isinstance(self.scalar, Number)
        return a ** numpy.float32(self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, = node.inputs
        return out_grad * self.scalar * power_scalar(lhs, self.scalar-1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        if a.dtype == b.dtype:
            c = a / b 
            return c.astype(a.dtype)
        else:
            return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return divide(out_grad, rhs), out_grad * (divide(-lhs, rhs*rhs))
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        assert isinstance(self.scalar, Number)
        ret = a / numpy.float32(self.scalar)
        return ret.astype(a.dtype)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return Tensor(divide_scalar(out_grad, self.scalar), dtype=out_grad.dtype)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    # def compute(self, a):
    #     ### BEGIN YOUR SOLUTION
    #     return a.permute(self.axes)
    #     ### END YOUR SOLUTION

    # def gradient(self, out_grad, node):
    #     ### BEGIN YOUR SOLUTION
    #     if self.axes is None:
    #         return transpose(out_grad, None)
    #     else:
    #         axes = numpy.argsort(self.axes)
    #         lhs, = node.inputs
    #         print("lhs.shape", lhs.shape)
    #         print("out_grad.shape in trans", out_grad.shape)
    #         print("self.axes", self.axes)
    #         print("axes in trans", axes)
    #         temp = transpose(out_grad, axes)
    #         print("temp shape", temp.shape)
    #         return temp
    #     ### END YOUR SOLUTION
    def compute(self, a):

        index = list(range(len(a.shape)))
        if self.axes is None:
            index[-1], index[-2] = index[-2], index[-1]
        else:
            axis1 = self.axes[0]
            axis2 = self.axes[1]
            index[axis1], index[axis2] = index[axis2], index[axis1]
        return a.permute(tuple(index))

    def gradient(self, out_grad, node):
        return transpose(out_grad, axes=self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a.compact(), self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, = node.inputs
        ret = Reshape(lhs.shape)(out_grad)
        assert(ret.shape == lhs.shape)
        return ret
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape).compact()

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        assert(out_grad.shape == self.shape)
        lhs, = node.inputs
        
        more_len = len(self.shape) - len(lhs.shape)
        for _ in range(more_len):
            out_grad = summation(out_grad, 0)
        
        out_grad_shape = out_grad.shape
        offset = 0
        for i in range(len(lhs.shape)):
            if out_grad_shape[i] != lhs.shape[i]:
                out_grad = summation(out_grad, i - offset)
                offset += 1

        ret = Reshape(lhs.shape)(out_grad)
        assert(ret.shape == lhs.shape)
        return ret
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.sum(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, = node.inputs

        org_shape = lhs.shape
        new_shape = list(out_grad.shape)
        if self.axes == None:
            for _ in range(len(org_shape) - 1):
                new_shape.append(1)
        else:
            self.axes = as_tuple(self.axes)
            for it in self.axes:
                if it == len(org_shape) - 1:
                    new_shape.append(1)
                else:
                    new_shape.insert(it,  1)
        new_shape = tuple(new_shape)
        return broadcast_to(reshape(out_grad, new_shape), lhs.shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        if len(lhs.shape) == len(rhs.shape):
            return out_grad @ transpose(rhs), transpose(lhs) @ out_grad
        elif len(lhs.shape) > len(rhs.shape):
            out = transpose(lhs) @ out_grad
            for _ in range(len(lhs.shape) - len(rhs.shape)):
                out = summation(out, 0)
            return out_grad @ transpose(rhs), out
        else:
            out = out_grad @ transpose(rhs)
            for _ in range(len(rhs.shape) - len(lhs.shape)):
                out = summation(out, 0)
            return out, transpose(lhs) @ out_grad
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return numpy.float32(-1) * a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return numpy.float32(-1) * out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, = node.inputs
        return divide(out_grad, lhs)
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, = node.inputs
        return exp(lhs) * out_grad
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, = node.inputs
        mask = lhs.realize_cached_data() > 0
        return multiply(out_grad, Tensor(mask, dtype=lhs.dtype, device=lhs.device))
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z = Z.max(axis=self.axes, keepdims=True)
        max_z_1 = Z.max(axis=self.axes)
        self.stable_z = Tensor(Z - array_api.broadcast_to(max_z, Z.shape), device=Z.device)
        log_val = array_api.log(array_api.sum(array_api.exp(Z - array_api.broadcast_to(max_z, Z.shape)), axis=self.axes))
        return log_val + max_z_1
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out_grad = Summation(self.axes).gradient(out_grad, node)
        z = self.stable_z

        # softmax opertaion
        exp_z = Exp()(z)
        denominator = Summation(self.axes)(exp_z)
        numerator = exp_z
        denominator = Summation(self.axes).gradient(denominator, node)  # Broadcast function
        softmax_z = EWiseDiv()(numerator, denominator)
        
        return multiply(softmax_z, out_grad)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return (array_api.exp(a) - array_api.exp(-a)) / \
               (array_api.exp(a) + array_api.exp(-a))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, = node.inputs
        return multiply(out_grad, (numpy.float32(1.0) - tanh(lhs) ** 2))
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        ### BEGIN YOUR SOLUTION
        new_shape = list(args[0].shape)
        new_shape.insert(self.axis, len(args))
        new_arr = array_api.empty(shape=new_shape, device=args[0].device)
        
        idxs = []
        for sh in args[0].shape:
            idxs.append(slice(0, sh, 1))
        
        for i in range(len(args)):
            new_idxs = idxs.copy()
            new_idxs.insert(self.axis, i)
            new_arr[tuple(new_idxs)] = args[i]
        
        return new_arr
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        ndim = A.shape[self.axis]

        # build idxs
        idxs = []
        for i, sh in enumerate(A.shape):
            if i != self.axis:
                idxs.append(slice(0, sh, 1))

        ret = []
        for i in range(ndim):
            new_idxs = idxs.copy()
            new_idxs.insert(self.axis, i)
            
            # remove an axis
            it = A[tuple(new_idxs)].compact()
            ret.append(it.sum(self.axis))

        return tuple(ret)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.flip(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)



class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.dilate(a, self.axes, self.dilation)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)

class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # compute idxs
        idxs = []
        a_axes = list(range(a.ndim))
        for axis in a_axes:
            if axis in self.axes:
                idxs.append(slice(0, a.shape[axis], self.dilation + 1))
            else:
                idxs.append(slice(0, a.shape[axis], 1))
        return a[tuple(idxs)]
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        A = A.pad((
            (0, 0), 
            (self.padding, self.padding), 
            (self.padding, self.padding),
            (0, 0)
        ))

        N, H, W, C_in = A.shape
        K, _, _, C_out = B.shape
        Ns, Hs, Ws, Cs = A.strides

        H_new = (H - K) // self.stride + 1
        W_new = (W - K) // self.stride + 1
        
        outer_dim = N * H_new * W_new
        inner_dim = K * K * C_in
        A = A.as_strided(
            shape=(N, H_new, W_new, K, K, C_in),
            strides=(Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs)
        ).compact().reshape((outer_dim, inner_dim))

        out = A @ B.compact().reshape((inner_dim, C_out))

        return out.compact().reshape((N, H_new, W_new, C_out)).compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # A is the data, B is the filter
        # A: (N, H, W, Cin)
        # B: (K, K, Cin, Cout)
        # out_grad: (N, H_new, W_new, Cout)

        A, B = node.inputs
        new_pad = B.shape[0] - self.padding - 1
        grad_a = conv(
            dilate(out_grad, (1,2), self.stride-1), 
            flip(B, (0,1)).transpose((2, 3)), 
            padding=new_pad
        )
        
        # (N, H, W, Cin) -> (Cin, H, W, N)
        per_A = A.transpose((0, 3))
        per_outgrad = dilate(out_grad, (1,2), self.stride-1)
        
        # (N, H_new, W_new, Cout) -> (N, H_new, W_new, Cout)
        # (N, H_new, W_new, Cout) -> (H_new, W_new, N, Cout)
        per_outgrad = per_outgrad.transpose((0, 2)).transpose((0, 1))
        grad_b = conv(per_A, per_outgrad, padding=self.padding)

        # c_in, k1, k2, c_out -> k2, k1, c_in, c_out
        # k2, k1, c_in, c_out -> k1, k2, c_in, c_out 
        grad_b = grad_b.transpose((0, 2)).transpose((0, 1))
        return (grad_a, grad_b)

        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)



