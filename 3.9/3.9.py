from collections import deque
import numpy
import numpy.random
from scipy.integrate import quad
from scipy.stats import norm
from itertools import cycle
import mpl_toolkits.mplot3d.axes3d as p3
from math import *
from cmath import *
import cmath
import cPickle
from matplotlib import rc, rcParams
import pylab
import matplotlib

def R_rand (N, a=1., theta=None, rand_a=False,zoffset=0):
    """
    returns array of N random vectors (Rx,Ry,Rz)
    |R| = a
    """
    if not rand_a:
        r=a*numpy.ones(N)
    else:
        r=a*numpy.random.random(N)
    phi=2*pi*numpy.random.random(N)
    if theta is None:
        th=numpy.arccos(2*numpy.random.random(N)-1)
    else:
        th = theta*numpy.ones(N)
    xyz=(numpy.array([r*numpy.sin(th)*numpy.cos(phi),
                     r*numpy.sin(th)*numpy.sin(phi),
                     r*numpy.cos(th)+zoffset])).T
    
    return xyz

a_EUT = 0.2693
Rsum=R_rand(1000, a=a_EUT,rand_a=False,zoffset=0)
fig=pylab.figure()
fig.suptitle('Dipolposition', fontsize=15)
ax=p3.Axes3D(fig) 
ax.set_xlabel('x',fontsize=15)
ax.set_ylabel('y',fontsize=15)
ax.set_zlabel('z',fontsize=15)
Rsum=numpy.array(Rsum)
ax.scatter(Rsum.T[0],Rsum.T[1],Rsum.T[2],s=5,c='k')
#ax.w_xaxis.set_ticks([-0.5,0,0.5])
#ax.w_yaxis.set_ticks([-0.5,0,0.5])
#ax.w_zaxis.set_ticks([-0.5,0,0.5])
#ax.scatter(numpy.array([0,0]),numpy.array([0,0]),numpy.array([-5,5]),s=0,c='w')
#pylab.show()
fig.set_size_inches(8, 8)
from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages(r'D:\HIWI\python-script\new_new_results\3.9/result.pdf')
pylab.savefig(pp, format='pdf',dpi=100,bbox_inches='tight')
pp.close()
