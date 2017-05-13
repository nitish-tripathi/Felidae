
import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

import numpy as np
import pylab
from PIL import Image


rng = np.random.RandomState(23455)

# A 4D tensor corresponding to a mini-batch of input images. 
# The shape of the tensor is as follows: [mini-batch size, number of input feature maps, image height, image width].
_input = T.tensor4(name='input')

# A 4D tensor corresponding to the weight matrix W.
# The shape of the tensor is: [number of feature maps at layer m, number of feature maps at layer m-1, filter height, filter width].
w_shp = (2, 3, 9, 9)
w_bound = np.sqrt(3*9*9)
w = theano.shared(np.asarray(rng.uniform(
                            low=-1.0/w_bound,
                            high=1.0/w_bound,
                            size=w_shp), dtype=_input.dtype), name='w')

b_shp = (2,)
b = theano.shared(np.asarray(rng.uniform(
                            low = -0.5,
                            high = 0.5,
                            size=b_shp), dtype=_input.dtype), name='b')

# do convolution of input to weights
conv_out = conv2d(_input, w)
output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))

convolution_fn = theano.function([_input], output)

img = Image.open("Lenna.png")
img = np.asarray(img, dtype='float64') / 256.0
img_shp = (1, img.shape[2], img.shape[0], img.shape[1])

img_ = img.transpose(2, 0, 1).reshape(img_shp)
filtered_img = convolution_fn(img_)
pylab.imshow(filtered_img[0, 0, :, :])
pylab.show()
pylab.imshow(filtered_img[0, 1, :, :])
pylab.show()
        