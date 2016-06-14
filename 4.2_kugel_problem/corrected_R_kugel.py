import numpy
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