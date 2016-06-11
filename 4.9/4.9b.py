#-*- coding: utf-8 -*-
#Import from Python
from collections import deque
import numpy
import numpy.random
from scipy.integrate import quad
from scipy.stats import norm
from ecdf import ECDF
from itertools import cycle
import mpl_toolkits.mplot3d.axes3d as p3
from math import *
from cmath import *
import cmath
import cPickle
from matplotlib import rc, rcParams
from matplotlib.backends.backend_pdf import PdfPages
import pylab
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc, rcParams
c=299792458.
pi=numpy.pi
mu0=4*pi*1e-7
eps0=1./(mu0*c**2)
twopi=2*pi
sin=numpy.sin
cos=numpy.cos
J=0+1j
from matplotlib import rc, rcParams
rc('text', usetex=True)
rc('font', family='serif')
fontsize=20
params = {#'backend': 'ps',
                    'font.size': fontsize,
                    'font.family': 'serif',
                    'font.serif': 'serif', 
                    'axes.titlesize': fontsize,
                    'axes.labelsize': fontsize,
                    'text.fontsize': fontsize,
                    'legend.fontsize': fontsize*0.8,
                    'xtick.labelsize': int(fontsize*0.8),
                    'ytick.labelsize': int(fontsize*0.8),
                    'text.usetex': True,
                    'figure.figsize': (5,5)}
rcParams.update(params)


# see Ma
class Sources(object):
    def __init__ (self, number):
        self.N=number
        self.phis=numpy.random.uniform(0.0,twopi,self.N)
        self.thetas=numpy.arccos(numpy.random.uniform(-1.0,1.0,self.N))
        #self.falsethetas=numpy.random.uniform(0.0,pi,self.N)  # wrong dists as in Wilson et al
        self.Is=numpy.random.uniform(0.0,1.,self.N)
        self.alphas=numpy.random.uniform(0.0,twopi,self.N)
        self.angles=zip(self.thetas,self.phis)
        
    def calc_power_ff(self, ka, th, ph):
        # Kugelcode
        cth=cos(th)
        sth=sin(th)
        cpsi=numpy.array([cth*cos(t)+sth*sin(t)*cos(ph-p) for t,p in self.angles])
        kacpsi=ka*cpsi
        exponent=kacpsi+self.alphas
        intensities=self.Is*numpy.exp(J*exponent)
        intens=sum(intensities)
        power=abs(intens)**2
        return power

def calc_ds (sources, Nrepetitions,k,ophis,othetas):
    results=[]
    for rep in range(Nrepetitions):
        s=Sources(sources)
        pss=[]
        #i=0
        #D=False
        #b=deque(maxlen=10)
        for ph,th in zip(ophis,othetas):
            ps=s.calc_power_ff(k,th,ph)  
            pss.append(ps)  # save to global list
        pav=sum(pss)/len(pss)  # average power
        pmax=max(pss) # max power
        d=pmax/pav # directivity (sampled)
        #i=i+1
        #b.append(d)
        #print i, ka, N, pav, pmax, d, D  # output
        #bav=sum(b)/len(b)
        #if i>b.maxlen and abs(bav-max(b)) < 0.01 and abs(bav-min(b)) < 0.01:
        D=d
        results.append(D)
        #break
    return results

def E_hertz_far (r, p, R, phi, f, t=0, epsr=1.):
    """
    Calculate E field strength of hertzian dipole(s) in the far field
    p: array of dipole moments
    R: array of dipole positions
    r: observation point
    f: frequency
    t: time
    phi: array with dipole phase angles (0..2pi)
    return: field strength at observation point r at time t (3-tuple: Ex, Ey, Ez) 
    """
    N=len(phi)
    rprime=r-R  # r'=r-R
    magrprime=numpy.sqrt(rprime[:,0]**2 + rprime[:,1]**2 + rprime[:,2]**2)
    #magrprime=numpy.sqrt(numpy.array([numpy.vdot(a,a) for a in rprime])) # |r-R|
    w=2*pi*f  # omega
    k=w/c     # wave number
    krp=k*magrprime  # k*|r-R|
    rprime_cross_p = numpy.cross(rprime, p) # (r-R) x p
    rp_c_p_c_rp = numpy.cross(rprime_cross_p, rprime) # ((r-R) x p) x (r-R) 
    expfac=numpy.exp(1j*(w*t-krp+phi))
    Ei = (w**2/(4*pi*eps0*epsr*c**2*magrprime**3) * expfac).reshape(N,1) * rp_c_p_c_rp
    #print Ei
    E=((sum(Ei))) # Ei zeitabheangig aufaddiert, Absolutwertbildung um zeitunabhaengigen Vergleichswert (Amplitude) zu haben, alternativ ist Mittelung ueber eine Periode moeglich (Effektivwert = Amplitude/wurzel2)
    #print 'EI',Ei
    #print 'EI',Ei
    #print type(Ei[0][0])
    #print len(Ei[0])
    #print sum(Ei)
    #print numpy.sum(Ei)
    #print E
    
    #print E
    #print sum(Ei)
    
    return E


def car2sph(xyz):
    ptsnew = numpy.zeros(xyz.shape)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,0] = numpy.sqrt(xy + xyz[:,2]**2)  # r
    ptsnew[:,1] = numpy.arctan2(numpy.sqrt(xy), xyz[:,2]) # tetha
    ptsnew[:,2] = numpy.arctan2(xyz[:,1], xyz[:,0])  # phi
    return ptsnew

def p_rand_auchalt (N, pmax=1., fix_amplitude=None):
    """
    returns array of N random dipole moments (px,py,pz)
    |p| <= pmax
    """
    if not fix_amplitude:
        r=pmax*numpy.random.random(N)
    else:
        r=pmax*numpy.ones(N)
    phi=2*pi*numpy.random.random(N)
    th=pi*numpy.random.random(N)
    #th=numpy.arccos(2*numpy.random.random(N)-1)
    xyz=(numpy.array([r*numpy.sin(th)*numpy.cos(phi),
                     r*numpy.sin(th)*numpy.sin(phi),
                     r*numpy.cos(th)])).T
    #print xyz
    return xyz

def p_rand_alt (N, pmax=1., fix_amplitude=None):
    """
    returns array of N random dipole moments (px,py,pz)
    |p| <= pmax
    """
    if not fix_amplitude:
        r=pmax*numpy.random.random(N)
    else:
        r=pmax*numpy.ones(N)
    phi=2*pi*numpy.random.random(N)
    th=pi*numpy.random.random(N)
    #th=numpy.arccos(2*numpy.random.random(N)-1)
    xyz=(numpy.array([r*numpy.cos(th),r*numpy.sin(th)*numpy.cos(phi),
                     r*numpy.sin(th)*numpy.sin(phi)])).T
    #print xyz
    return xyz


def p_rand (N, pmax=1., fix_amplitude=None):
    """
    returns array of N random dipole moments (px,py,pz)
    |p| <= pmax
    """
    if not fix_amplitude:
        r=pmax*numpy.random.random(N)
    else:
        r=pmax*numpy.ones(N)
    #r=r**0.5 # Gleichverteilung des Dipolmoments
    phi=2*pi*numpy.random.random(N)
    costh=numpy.random.uniform(-1,1,N)
    th=numpy.arccos(costh)
    #th=numpy.arccos(2*numpy.random.random(N)-1)
    xyz=(numpy.array([r*numpy.sin(th)*numpy.cos(phi),
                     r*numpy.sin(th)*numpy.sin(phi),
                     r*numpy.cos(th)])).T
    #print xyz
    
    return xyz

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

def R_notrand (N, a=1., theta=None, rand_a=False,zoffset=0):
    """
    returns array of N random vectors (Rx,Ry,Rz)
    |R| = a
    """
    if not rand_a:
        r=a*numpy.ones(N)
    else:
        r=a*numpy.random.random(N)
    phi=2*pi*numpy.linspace(0,(N-1)/float(N),N)
    #print numpy.linspace(0,(N-1)/float(N),N)
    if theta is None:
        th=numpy.arccos(2*numpy.random.random(N)-1)
    else:
        th = theta*numpy.ones(N)
    #print 'th', th
    #print 'phi',phi
    #wait
    xyz=(numpy.array([r*numpy.sin(th)*numpy.cos(phi),
                     r*numpy.sin(th)*numpy.sin(phi),
                     r*numpy.cos(th)+zoffset])).T
    
    return [xyz,phi,th]

def Ns_hansen_1D(ka):
    return 4*ka+2

def Ns_hansen_3D(ka):
    return 4*ka**2+8*ka

def EDmax_hansen (ka, mu=1., Ns=None):
    if Ns is None:
        Ns=Ns_hansen_3D
    Nska=Ns(ka)
    D=0.577 + numpy.log(Nska) + 0.5/Nska
    return mu*D

def FD_hansen(d, ka, mu=1., Ns=None):
    if Ns is None:
        Ns=Ns_hansen_3D
    Nska=Ns(ka)
    return (1-numpy.exp(-d/mu))**Nska

def calc_ds_MCHansen(ka, N_MC, N_obs, mu=1.0, Ns=None):
    if Ns is None:
        Ns=Ns_hansen_3D
    Nska=int(Ns(ka)+0.5)
    Ds=[]
    for i in range(N_MC):
        Ex=numpy.zeros(Nska)
        Ey=numpy.zeros(Nska)
        Ez=numpy.zeros(Nska)
        for j in range(N_obs):
            Ex+=norm.rvs(size=Nska)
            Ey+=norm.rvs(size=Nska)
            Ez+=norm.rvs(size=Nska)
        magE2=Ex**2+Ey**2+Ez**2
        D=max(magE2)*Nska/sum(magE2)
        Ds.append(D)
        print i, ka, D
    Emean=sum(Ds)/len(Ds)
    Tmean=EDmax_hansen(ka, mu=mu, Ns=Ns)
    Ds=[D*Tmean/Emean for D in Ds]
    return Ds

def fD_hertz_one_cut(d):
    if not ( 1<=d<=2 ):
        return 0.0
    return 1./(pi*d)*sqrt(2/((d-1)*(2-d)))

def FD_hertz_one_cut(d):
    if d<1:
        return 0.0
    if d>2:
        return 1.0
    return 0.5-1./pi*numpy.arcsin(4./d-3)

def FD_hertz_one_cut_costheta(d):
    if d<1:
        return 0.0
    if d>2:
        return 1.0
    return 1-numpy.sqrt((2-d)/(d))
    
def R_Kugel(N,distance,zoffset=0):
    """
    returns array of N random vectors on a sphere
    """
    
    thetaphilist=[]
    thetas=[]
    phis=[]
    thetas.append(0)
    phis.append(0)
    #thetaphilist.append([0,0])
    phiminus1=0.0
    for k in (numpy.array(range(N-2))+2):
    
        hk=-1.0+2.0*(k-1)/(N-1)
        #print hk
        theta=numpy.arccos(hk)
        #print hk

        phi=numpy.mod(phiminus1+3.6/numpy.sqrt(N)/numpy.sqrt(1-hk**2),2*numpy.pi)
        #print phi
        phiminus1=phi
        #thetaphilist.append([phi,theta])
        thetas.append(theta)
        phis.append(phi)
        #thetaphilist.append([0,numpy.pi])    
        #thetas.append(numpy.pi)
        #phis.append(0)
        
    thetas=numpy.array(thetas)
    phis=numpy.array(numpy.real(phis))
        #print thetas,phis
    xyz=(numpy.array([distance*numpy.sin(thetas)*numpy.cos(phis),distance*numpy.sin(thetas)*numpy.sin(phis),distance*numpy.cos(thetas)+zoffset])).T
        #print xyz
    return [xyz,phis,thetas]
def load_padirec(anz):
    f=file(str(anz)+'palist.dmp','r')
    pa=cPickle.load(f)
    f.close()
    f=file(str(anz)+'direction.dmp','r')
    direc=cPickle.load(f)
    f.close()
    f=file(str(anz)+'dpnr.dmp','r')
    dpnr=cPickle.load(f)
    f.close()
    f=file('EUTList.dmp','r')
    EUTlist=cPickle.load(f)
    f.close()
    #print pa,direc,dpnr,EUTlist
    return [pa,direc,dpnr,EUTlist]

def calcEmags2(Es,phi,th):
    
    hfaktor=(numpy.array([-numpy.sin(th)*numpy.sin(phi),
                     numpy.sin(th)*numpy.cos(phi),
                     numpy.cos(th)])).T
    vfaktor=(numpy.array([numpy.cos(th)*numpy.cos(phi),
                     numpy.cos(th)*numpy.sin(phi),
                     numpy.sin(th)])).T
    Eshv=numpy.array([[numpy.vdot(hf,E) for hf,E in zip(hfaktor,Es)],[numpy.vdot(vf,E) for vf,E in zip(vfaktor,Es)]])
    #print 'Eshv',Eshv.T
    Emags2=numpy.array([numpy.vdot(a,a) for a in Eshv.T])  # |E|**2

    return Emags2
def R_rand_oats (N, a=1., theta=None, rand_a=False,zoffset=1):
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
                     r*numpy.cos(th)+zoffset+a])).T
    
    return xyz

def R_Zylinder (N, R,deltah):
    """
    returns array of N random vectors (Rx,Ry,Rz)
    |R| = a
    """
    

    phi=2*pi*numpy.linspace(0,(N[0]-1)/float(N[0]),N[0])
    #print numpy.linspace(0,(N-1)/float(N),N)
    h=numpy.linspace(1,1+deltah,N[1])
    #print 'h', h
    #print 'phi',phi
    xyz=[]
    for z in h:
             
        xyz=xyz+(list(numpy.array(numpy.array([R*numpy.cos(phi),R*numpy.sin(phi),numpy.tile(z,N[0])]).T)))
    
    xyz=numpy.array(xyz)
    #print xyz
    return [xyz,phi,h]

def E_hertz_oats (r, p, R, phi, f, h0=1,t=0, epsr=1.):
    """
    Calculate E field strength of hertzian dipole(s) in the far field
    p: array of dipole moments
    R: array of dipole positions
    p2: array of mirrored dipole moments
    R2: array of mirrored dipole positions
    r: observation point
    f: frequency
    t: time
    phi: array with dipole phase angles (0..2pi)
    return: field strength at observation point r at time t (3-tuple: Ex, Ey, Ez) 
    """
    
    N=len(phi)
    rprime=r-R  # r'=r-R
    magrprime=numpy.sqrt(rprime[:,0]**2 + rprime[:,1]**2 + rprime[:,2]**2)
    #magrprime=numpy.sqrt(numpy.array([numpy.vdot(a,a) for a in rprime])) # |r-R|
    w=2*pi*f  # omega
    k=w/c     # wave number
    krp=k*magrprime  # k*|r-R|
    rprime_cross_p = numpy.cross(rprime, p) # (r-R) x p
    rp_c_p_c_rp = numpy.cross(rprime_cross_p, rprime) # ((r-R) x p) x (r-R) 
    expfac=numpy.exp(1j*(w*t-krp+phi))
    Ei1 = (w**2/(4*pi*eps0*epsr*c**2*magrprime**3) * expfac).reshape(N,1) * rp_c_p_c_rp

    # Gespiegelter Dipol
    R2=numpy.array([R.T[0],R.T[1],-R.T[2]]).T #Dipolpositionen Spiegeln
    p2=numpy.array([p.T[0],p.T[1],-p.T[2]]).T#Dipolrichtung Spiegeln
    rprime=r-R2  # r'=r-R
    magrprime=numpy.sqrt(rprime[:,0]**2 + rprime[:,1]**2 + rprime[:,2]**2)
    #magrprime=numpy.sqrt(numpy.array([numpy.vdot(a,a) for a in rprime])) # |r-R|
    w=2*pi*f  # omega
    k=w/c     # wave number
    krp=k*magrprime  # k*|r-R|
    rprime_cross_p = numpy.cross(rprime, p2) # (r-R) x p
    rp_c_p_c_rp = numpy.cross(rprime_cross_p, rprime) # ((r-R) x p) x (r-R) 
    expfac=numpy.exp(1j*(w*t-krp+phi))
    Ei2 = (w**2/(4*pi*eps0*epsr*c**2*magrprime**3) * expfac).reshape(N,1) * rp_c_p_c_rp    
    
    Ei=Ei1+Ei2
    E=(sum(Ei)) # Ei zeitabheangig aufaddiert, Absolutwertbildung um zeitunabhaengigen Vergleichswert (Amplitude) zu haben, alternativ ist Mittelung ueber eine Periode moeglich (Effektivwert = Amplitude/wurzel2)
    #print 'EI2',Ei2
    #print 'EI1',Ei1
    #print 'EI',Ei
    #print type(Ei[0][0])
    #print E
    #print sum(Ei)

    return E
if __name__ == "__main__":

    Es2_list_av_Z=[]
    distance = 10  # measurement distance
    a_EUT = 0.2693     # radius of EUT
    N_dipole = 10    # number of random dipoles
    N_obs=25
    umfang= 2 * pi * distance #4m / 2 *pi * R
    deltah=3
    #zoffset=1.0
    #print numpy.sqrt(2*pi*distance/deltah*500)
    #upoints=int(round(numpy.sqrt(2*pi*distance/deltah*500)))
    #print upoints
    #hpoints=int(round(float(N_obs)/upoints))
    #N_obs_points=[upoints,hpoints]
    N_obs_points=[5,5] #number of observation points [Winkel,Höhe]
    #print N_obs_points

    N_obs=N_obs_points[0]*N_obs_points[1]
    N_MC=1000      # number of MC runs -> average over different random configurations
    freqs=numpy.array(numpy.linspace(1000,6000,51))*1e6#freqs=numpy.array([1000000000])#,1000000000,1500000000])#range(30,301,30))*1000000#numpy.logspace(10,11,3)  # generate frequencies
    kas=a_EUT*2*pi*freqs/c # vector with k*a values (a: EUT radius)
    deval=numpy.linspace(1,10,100)
    #print deval
  
    n_listen=0                
    #[palist,direc,dpnrlist,EUTlist]=load_padirec(N_dipole*N_MC)
    f=1000*1e6
    ka=a_EUT*2*pi*f/c
    #Ma_Ds=calc_ds (N_dipole, N_MC, ka,
    #               numpy.random.uniform(0.0,twopi,N_obs_points),
    #               0.5*pi*numpy.ones(N_obs_points)) #np.arccos(np.random.uniform(-1.0,1.0,1e5))
    #MCHansen_Ds=calc_ds_MCHansen(ka, N_MC, N_obs=N_obs_points, Ns=Ns)
    [rs,phiwinkel,hoehe]=R_Zylinder(N_obs_points, distance,deltah)#1.5) # generate not random observation points 
    #[rs,phiwinkel,hoehe]=R_notrand(N_obs_points[1], distance)#1.5) # generate not random observation points 
    #rs=R_rand(N_obs_points, distance, theta=0.5*pi,zoffset=1.5) # generate random observation points 
    

    #wait
    #print 'RS=',rs
    Ds=[] # to store the directivities of the MC runs at this freq
    Rsum=[]
    Psum=[]
    n_listen=0
    for mc in range(N_MC): # MC loop
        p=p_rand(N_dipole, pmax=1e-8)   # generate vector with random dipole moments
        R=R_rand_oats(N_dipole, a=a_EUT,rand_a=False,zoffset=1)   # generate random dipole positions on EUT surface
        Rsum.append(numpy.array([R.T[0],R.T[1],-R.T[2]]).T[0])
        Rsum.append(R[0])
        Psum.append(p[0])
        pha=2*pi*numpy.random.random(N_dipole) # generate random phases
        #phase=numpy.zeros(N_dipole)
        #pylab.show()
        #print 'p',p
        Es=numpy.array([E_hertz_oats(r, p, R, pha, f, t=0, epsr=1.) for r in rs]) # calculate sum E-fields at obsevation points

        Emags2=abs(numpy.array([numpy.dot(a,a) for a in Es]))  # |E|**2

        av=sum(Emags2)/(N_obs_points[0]*N_obs_points[1]) # the average of |E|**2
        ma=max(Emags2) # the maximum of |E|**2
        D=ma/av # directivity

        Ds.append(D)
        Es2_list_av_Z.append(av)
        print "Z",mc  

        
    N_obs_points=25    
    [rs,phis,thetas]=R_Kugel(N_obs_points, distance)#1.5) # generate not random observation points 

    Ds_K=[] # to store the directivities of the MC runs at this freq
    Es2_list_av_K=[]
    Rsum=[]
    Psum=[]
    n_listen=0
    #print f/1e9,N_obs_points
    for mc in range(N_MC): # MC loop
        p=p_rand(N_dipole, pmax=1e-8)   # generate vector with random dipole moments
        R=R_rand(N_dipole, a=a_EUT,rand_a=False,zoffset=0)   # generate random dipole positions on EUT surface
        Rsum.append(R[0])
        Psum.append(p[0])
        pha=2*pi*numpy.random.random(N_dipole) # generate random phases

        Es=numpy.array([E_hertz_far(r, p, R, pha, f, t=0, epsr=1.) for r in rs]) # calculate sum E-fields at obsevation points

        Emags2=abs(numpy.array([numpy.dot(a,a) for a in Es]))

        av=sum(Emags2)/N_obs_points # the average of |E|**2
        ma=max(Emags2) # the maximum of |E|**2
        D=ma/av # directivity
        #print mc, ma, av, D
        Ds_K.append(D)
        Es2_list_av_K.append(av)
        print "K",mc        
    Es2_ratio=numpy.divide(Es2_list_av_Z,Es2_list_av_K)
    #ecdfD=ECDF(Es2_ratio)
    fig, ax = plt.subplots()
    #pylab.plot(deval,ecdfD(deval), '%s+-'%clr, label="ECDF (Dipoles)")
    #pylab.plot(deval, [FD_hertz_one_cut(d) for d in deval], label="Theoretical CDF (a=0 m)")
    #pylab.plot(deval, [FD_hertz_one_cut_costheta(d) for d in deval], label="Theoretical CDF cos(theta)(a=0 m)")
    #pylab.axis([deval[0],deval[-1],0,1])
    bins = np.linspace(0,4.8, 24, endpoint=True)
    ax.hist(Es2_ratio, bins,(0,5),facecolor='grey')
    plt.grid()
    #pylab.legend(loc=4)
    plt.xlabel(r"$\frac{E^{Z^{2}}}{E^{K^{2}}}$")
   # pylab.ylabel(r"F($\frac{E^{R^{2}}_{max}}{E^{K^{2}_{max}}}$)")
    plt.title("$N_{dipoles}=%d$, MC runs=%d, $Frequency=%dGHz$, $R=%dm$"%(N_dipole,N_MC,f/1e9,distance))
    #pylab.show()
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(18.5, 10.5)
    pp = PdfPages(r'D:\HIWI\python-script\new_new_results\4.9/result_b.pdf')
    pylab.savefig(pp, format='pdf',dpi=fig.dpi, bbox_inches='tight')
    pp.close()
    output=Es2_ratio
    numpy.savetxt(r"D:\HIWI\python-script\new_new_results\4.9/4.9b.dat", output, fmt=['%.6f'])

