# Code from Chapter 3 of Machine Learning: An Algorithmic Perspective
# by Stephen Marsland (http://seat.massey.ac.nz/personal/s.r.marsland/MLBook.html)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008

# The Palmerston North Ozone time series example
import matplotlib.pyplot as plt
import numpy as np
import random
from pylab import *
# from numpy import *

PNoz = loadtxt('/Users/rakeshmaharjan/Documents/Machine Learning/MLBook_source-master/3 MLP/PNOz.dat')
print(PNoz)
ion()
plt.plot(np.arange(shape(PNoz)[0]), PNoz[:, 2], '.')
plt.xlabel('Time (Days)')
plt.ylabel('Ozone (Dobson units)')
plt.show()

# Normalise data
PNoz[:, 2] = PNoz[:, 2] - PNoz[:, 2].mean()
PNoz[:, 2] = PNoz[:, 2] / PNoz[:, 2].max()
print("normalized pnoz:")
print(PNoz)
# Assemble input vectors
t = 2
k = 3

lastPoint = shape(PNoz)[0] - t * (k + 1) - 1
print("lastPoint")
print(lastPoint)
inputs = zeros((lastPoint, k))
print("inputs")
print(inputs)
targets = zeros((lastPoint, 1))
print("targets")
print(targets)
for i in range(lastPoint):
    inputs[i, :] = PNoz[i:i + t * k:t, 2]
    targets[i] = PNoz[i + t * (k + 1), 2]

test = inputs[-400:, :]
testtargets = targets[-400:]

# Randomly order the data
inputs = inputs[:-400, :]
targets = targets[:-400]
change = range(shape(inputs)[0])
# random.shuffle(change)
inputs = inputs[change, :]
targets = targets[change, :]

train = inputs[::2, :]
traintargets = targets[::2]
valid = inputs[1::2, :]
validtargets = targets[1::2]

# Train the network
import mlp

net = mlp.mlp(train, traintargets, 3, outtype='linear')
net.earlystopping(train, traintargets, valid, validtargets, 0.25)

test = concatenate((test, -ones((shape(test)[0], 1))), axis=1)
testout = net.mlpfwd(test)

figure()
plt.plot(arange(shape(test)[0]), testout, '.')
plt.plot(arange(shape(test)[0]), testtargets, 'x')
plt.legend(('Predictions', 'Targets'))
print(0.5 * sum((testtargets - testout) ** 2))

plt.show()
