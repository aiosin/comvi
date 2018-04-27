#source:http://code.activestate.com/recipes/578994-discrete-fourier-transform/
# Discrete Fourier Transform (DFT)
# FB - 20141227


# zython@echelon:~$ p
# Python 3.6.4 (default, Jan  5 2018, 02:35:40) 
# [GCC 7.2.1 20171224] on linux
# Type "help", "copyright", "credits" or "license" for more information.
# >>> import numpy as np
# >>> x = np.linspace(0,np.pi*2,100,endpoint=False)
# >>> y = np.sin(x)
# >>> yf = np.fft.rfft(y)
# >>> help(np.set_printoptions)

# >>> np.set_printoptions(suppress=True, precision=3)
# >>> print(yf)
# [-0. +0.j -0.-50.j  0. -0.j -0. -0.j  0. -0.j -0. +0.j -0. -0.j  0. +0.j
#  -0. -0.j  0. -0.j -0. -0.j  0. -0.j  0. +0.j -0. -0.j  0. -0.j -0. +0.j
#  -0. -0.j  0. +0.j -0. -0.j  0. +0.j  0. -0.j  0. -0.j  0. +0.j -0. -0.j
#   0. -0.j -0. +0.j -0. -0.j  0. +0.j -0. -0.j  0. +0.j -0. -0.j  0. -0.j
#   0. +0.j -0. -0.j  0. +0.j -0. -0.j -0. -0.j  0. -0.j -0. -0.j  0. -0.j
#   0. -0.j  0. -0.j  0. -0.j  0. +0.j  0. +0.j  0. +0.j  0. -0.j  0. +0.j
#   0. -0.j  0. +0.j -0. +0.j]
# >>> 


import random
import math
import cmath
pi2 = cmath.pi * 2.0
def DFT(fnList):
    N = len(fnList)
    FmList = []
    for m in range(N):
        Fm = 0.0
        for n in range(N):
            Fm += fnList[n] * cmath.exp(- 1j * pi2 * m * n / N)
        FmList.append(Fm / N)
    return FmList
        
def InverseDFT(FmList):
    N = len(FmList)
    fnList = []
    for n in range(N):
        fn = 0.0
        for m in range(N):
            fn += FmList[m] * cmath.exp(1j * pi2 * m * n / N)
        fnList.append(fn)
    return fnList

# TEST
print "Input Sine Wave Signal:"
N = 360 # degrees (Number of samples)
a = float(random.randint(1, 100))
f = float(random.randint(1, 100))
p = float(random.randint(0, 360))
print "frequency = " + str(f)
print "amplitude = " + str(a)
print "phase ang = " + str(p)
print
fnList = []
for n in range(N):
    t = float(n) / N * pi2
    fn = a * math.sin(f * t + p / 360 * pi2)
    fnList.append(fn)

print "DFT Calculation Results:"
FmList = DFT(fnList)
threshold = 0.001
for (i, Fm) in enumerate(FmList):
    if abs(Fm) > threshold:
        print "frequency = " + str(i)
        print "amplitude = " + str(abs(Fm) * 2.0)
        p = int(((cmath.phase(Fm) + pi2 + pi2 / 4.0) % pi2) / pi2 * 360 + 0.5)
        print "phase ang = " + str(p)
        print

### Recreate input signal from DFT results and compare to input signal
##fnList2 = InverseDFT(FmList)
##for n in range(N):
##    print fnList[n], fnList2[n].real