import numpy as np
from chainer import Variable
import matplotlib.pyplot as plt

from elu import elu


xs = np.arange(-10, 10, 0.01, dtype=np.float32)
ys = elu(Variable(xs)).data

plt.plot(xs, ys)
plt.show()
