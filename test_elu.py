import random
import unittest

import numpy

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition

from elu import elu


class TestELU(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-3, 3, (5, 4)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-3, 3, (5, 4)).astype(numpy.float32)
        self.alpha = random.random()

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = elu(x, alpha=self.alpha)
        self.assertEqual(y.data.dtype, numpy.float32)

        expected = self.x.copy()
        for i in numpy.ndindex(self.x.shape):
            if self.x[i] < 0:
                expected[i] = self.alpha * (numpy.exp(expected[i]) - 1)

        gradient_check.assert_allclose(expected, y.data)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        x = chainer.Variable(x_data)
        y = elu(x, alpha=self.alpha)
        y.grad = y_grad
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data,))
        gx, = gradient_check.numerical_grad(f, (x.data,), (y.grad,))

        gradient_check.assert_allclose(gx, x.grad)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)
