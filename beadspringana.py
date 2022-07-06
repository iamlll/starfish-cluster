import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.io
import os
from scipy.ndimage import uniform_filter1d

def PlotDisplacementTimelapse(Xmat,Ymat,Type):
    fig,ax = plt.subplots(1,1)
    idx = Type[0]==1
    ax.plot(Xmat[idx,:],Ymat[idx,:],'b.')
    ax.plot(Xmat[~idx,:],Ymat[~idx,:],'r.')
    
    plt.tight_layout()
    plt.show()

def PlotPtclTraj(n,Xmat,Ymat):
    plt.plot(Xmat[n-1,:],label='x traj')
    #plt.plot(Ymat[n-1,:],label='y traj')
    plt.title('Particle %d' %n)
    plt.show()
   
def ParticleFFT(Ux,Uy,X,Y,theta,qvec,logscale,plotting=1):
    Nfr = X.shape[1]
    ws = np.arange(-Nfr/2,Nfr/2)*2*np.pi/Nfr
    Uqxw = np.zeros((len(qvec),Nfr));
    Uqyw = np.zeros((len(qvec),Nfr));
    PSavg = np.zeros((len(qvec),Nfr))

    for j in range(len(qvec)):
        q = qvec[j];
        FT = np.exp(1j*q*(X*np.cos(theta) + Y*np.sin(theta)));
        Uqxt = np.sum(Ux*FT,axis=0); #Spatial Fourier Component of Ux; sum over all ptcls
        Uqyt = np.sum(Uy*FT,axis=0); #Spatial Fourier Component of Uy
        Uqxt = Uqxt
        Uqyt = Uqyt
        Uqxw[j,:] = np.fft.fftshift(np.fft.fft(Uqxt.real)); # time Fourier component of Ux
        Uqyw[j,:] = np.fft.fftshift(np.fft.fft(Uqyt.real)); # Space-time Fourier component of Uy
    PSx = np.abs(Uqxw)**2/Uqxw.size;
    PSy = np.abs(Uqyw)**2/Uqyw.size;
    PS = PSx**2 + PSy**2; # Power spectrum of displacements
    # normalize power spectrum by max value
    PS = PS / np.max(PS);
    PSavg = PSavg + PS;
    if logscale > 0:
        PSavg = np.log10(PSavg); 

    if plotting:
        fig,ax = plt.subplots(1,1)
        im=ax.imshow(PSavg.T,extent = [min(qvec),max(qvec),min(ws),max(ws)]);
        ax.set_title('$\\theta = %.2f$' %theta);
        ax.set_ylabel('$\omega$ (frames$^{-1}$)');
        ax.set_xlabel('$q$ (px$^{-1}$)');
        cb = fig.colorbar(im,ax=ax,orientation='vertical')
        if logscale > 0:
            cb.set_label('Avg power spectrum (log10)');
        else:
            cb.set_label('Avg power spectrum');
        ratio = 1.5 #set aspect ratio
        x_left, x_right = ax.get_xlim()
        y_low, y_high = ax.get_ylim()
        ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
        plt.show() 
    
    return ws,PSavg

def SmoothPosMats(X,Y,smFac, eqFac,plotting=0):
    N = X.shape[0]
    smX = np.zeros(X.shape)
    smY = np.zeros(X.shape)
    eqX = np.zeros(X.shape)
    eqY = np.zeros(X.shape)
    for i in np.arange(N):
        xi = X[i,:]
        yi = Y[i,:]
        idx = np.where(xi != 0)[0]
        smX[i,idx] = uniform_filter1d(xi[idx],size=smFac) 
        smY[i,idx] = uniform_filter1d(yi[idx],size=smFac) 
        eqX[i,idx] = uniform_filter1d(xi[idx],size=eqFac) 
        eqY[i,idx] = uniform_filter1d(yi[idx],size=eqFac) 
    dX = smX - eqX
    dY = smY - eqY

    if plotting > 0:
        # check that this works
        n=70
        plt.plot(X[n,:],label='data')
        plt.plot(eqX[n,:],label='smoothed')
        plt.legend()
        plt.show()

    return smX, smY, eqX, eqY, dX, dY


if __name__ == "__main__":
    import sys
    savedir = sys.argv[1]
    data = scipy.io.loadmat(os.path.join(savedir,'simdata.mat'))
    meta = scipy.io.loadmat(os.path.join(savedir,'meta.mat'))
    Xmat = data['Xmat']
    Ymat = data['Ymat']
    Type = data['Type']
    #PlotDisplacementTimelapse(Xmat,Ymat, Type)
    #PlotPtclTraj(250,Xmat,Ymat)

    smFac = 6; eqFac = 60;
    _,_,_,_,dXmat,dYmat = SmoothPosMats(Xmat,Ymat,smFac, eqFac,plotting=0)
    #Vx = np.diff(Xmat,1,axis=1)
    #Vy = np.diff(Ymat,1,axis=1)
    a=meta['a'][0,0]; logscale=1;
    maxbound = 4*np.pi/(3*a); #max 1 BZ diameter is 8pi/3a
    qvec = np.linspace(-2,2,num=401,endpoint=True)*maxbound
    ParticleFFT(dXmat,dYmat,Xmat,Ymat,0.,qvec,logscale)
    #ParticleFFT(Vx,Vy,Xmat[:,:-1],Ymat[:,:-1],0.,qvec,logscale)
    
    
