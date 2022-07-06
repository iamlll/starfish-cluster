import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.io
import os
from scipy.ndimage import uniform_filter1d

def InitHexagon(Ngrid,a=1,plotting=0):
    ''' initialize triangular lattice in hexagon shape'''
    # Grid vectors 
    e1 = a*[1, 0];
    e2 = a*[np.cos(np.pi/3), np.sin(np.pi/3)];
    e3 = a*[np.cos(-np.pi/3), np.sin(-np.pi/3)];

    # Coordinate arrays
    x1=np.arange(round(Ngrid/2+1))
    x2=np.arange(1,round(Ngrid/2+1))
    A1_1, A2_1 = np.meshgrid(x1,x1)
    A1_2, A2_2 = np.meshgrid(x1,x2);
    A1_3, A2_3 = np.meshgrid(x2,x2);

    # Build coordinates of centered hexagonal reference grid
    X0_1 = e1[0]*A1_1 + e2[0]*A2_1 - a*round(Ngrid/2);
    Y0_1 = e1[1]*A1_1 + e2[1]*A2_1;
    X0_2 = e1[0]*A1_2 + e3[0]*A2_2 - a*round(Ngrid/2);
    Y0_2 = e1[1]*A1_2 + e3[1]*A2_2;
    X0_3 = e2[0]*A1_3 + e3[0]*A2_3;
    Y0_3 = e2[1]*A1_3 + e3[1]*A2_3;
    init_x = np.concatenate((np.ravel(X0_1,'F'),np.ravel(X0_2,'F'),np.ravel(X0_3,'F')),axis=None)
    init_y = np.concatenate((np.ravel(Y0_1,'F'),np.ravel(Y0_2,'F'),np.ravel(Y0_3,'F')),axis=None)
    
    Pos0 = np.array([init_x,init_y]).T
    if plotting > 0:
        plt.scatter(Pos0[:,0],Pos0[:,1],20,'r')
        plt.gca().set_aspect(1)
        plt.show()
    return Pos0

def InitPtclTypes(Pos0, initcond='one',ntype1=10,seed=0):
    '''initialize system in various configurations and return the indices of particles of type 1'''
    rng = np.random.default_rng(seed)
    Npts = len(Pos0)
    if initcond == 'rand':
        f1 = rng.integers(0,Npts,size=ntype1)
    else:
        # assume all particles are of the same 'type'
        initcond = 'one'
        f1 = np.arange(Npts)
    Type = np.zeros(Npts,dtype=np.int8) 
    Type[f1] = 1
    return initcond, Type
 
def nnsearch(pts, dist):
    ''' Find nearest neighbors within a certain distance of a set of pts (implies open boundary conditions (OBC) if using to initialize system). Also removes "self neighbors" '''
    import scipy.spatial as spatial
    point_tree = spatial.KDTree(pts)
    nns = point_tree.query_ball_point(pts,dist)
    selfidxs = np.array([np.where(np.array(nn) == i)[0][0] for i,nn in enumerate(nns)])
    nns = np.array([np.delete(nn,i) for i,nn in zip(selfidxs, nns)],dtype=object)
    #plt.plot(pts[:,0],pts[:,1],'k.')
    #plt.plot(pts[10,0],pts[10,1],'ro')
    #plt.plot(pts[nns[10],0],pts[nns[10],1],'g.')
    #plt.show()
    return nns   

def InitPosPerturb(Pos0,amp,rng=np.random.default_rng(0),opt='rand'):
    if opt == 'rand':
        x0 = Pos0[:,0] + amp*(rng.random(Pos0[:,0].shape)-0.5)
        y0 = Pos0[:,1] + amp*(rng.random(Pos0[:,0].shape)-0.5)
    elif opt == 'control': # controlled initialization for comparison with MATLAB code
        x0 = Pos0[:,0] + 2
        y0 = Pos0[:,1] + 1
    Pos = np.array([x0,y0]).T
    return Pos 

def DefSpringMats(idx_n, Type, ke, keA, keB, ko, koA, koB,alpha=0.):
    ''' Define even and odd spring constant matrices describing the spring strengths corresponding to each particle's nearest neighbor interactions
    Inputs:
        idx_n: nearest neighbor indices
    '''
    k_even = np.array([[keA,ke],[ke,keB]])
    k_odd = np.array([[koA, ko+alpha],[ko-alpha, koB]])
    odd_spring = np.array([k_odd[Type[i],Type[idx_n[i]]] for i in range(len(Type))],dtype=object)
    even_spring = np.array([k_even[Type[i],Type[idx_n[i]]] for i in range(len(Type))],dtype=object)
    return even_spring,odd_spring

def OddNetwork(t,y,even_spring,odd_spring,idx_n,r0=1,n_pot=1,noise=0.,seed=0):
    N = len(idx_n)
    Curr_pos_x = y[:N]
    Curr_pos_y = y[N:]
    # Compute differences dx, dy in x and y positions between each particle and its nearest neighbors
    rij_x = np.array([ Curr_pos_x[i] - Curr_pos_x[nn] for i,nn in enumerate(idx_n)],dtype=object)
    rij_y = np.array([ Curr_pos_y[i] - Curr_pos_y[nn] for i,nn in enumerate(idx_n)],dtype=object)

    # Distance b/w each particle and its nearest neighbors
    rij = np.array([np.sqrt(dx**2+dy**2) for dx, dy in zip(rij_x, rij_y)],dtype=object)

    # Determine normalized interaction direction vectors \hat{x} and \hat{y}
    rij_x = np.array([rx/r for rx,r in zip(rij_x, rij)],dtype=object)
    rij_y = np.array([ry/r for ry,r in zip(rij_y, rij)],dtype=object)

    # Set up interaction forces
    # Linear (even) Spring force
    # sum across rows (nearest neighbors)
    # Fx and Fy are Npart length (row) vectors
    Fx = -np.array([np.sum(((normrij - r0)**n_pot)*rijx*evensp) for rijx, normrij, evensp in zip(rij_x, rij,even_spring)],dtype=object)
    Fy = -np.array([np.sum(((normrij - r0)**n_pot)*rijy*evensp) for rijy, normrij, evensp in zip(rij_y, rij,even_spring)],dtype=object)
    # Odd spring forces
    Fx = Fx - np.array([np.sum((normrij - r0)*rijy*oddsp) for rijy,normrij,oddsp in zip(rij_y, rij,odd_spring)],dtype=object)
    Fy = Fy - np.array([np.sum((normrij - r0)*(-rijx)*oddsp) for rijx,normrij,oddsp in zip(rij_x, rij,odd_spring)],dtype=object)
    # Build output vector
    dydt = np.concatenate((Fx,Fy))
    if noise > 0:
        rng = np.random.default_rng(seed)
        dydt = dydt + noise*rng.standard_normal(dydt.shape); # 2 x Npart array detailing the system dxdt and dydt for Npart variables
    if t % 10 < 0.2: print(t)
    return dydt

def runsim(ti,tf,dt,Pos0,even_spring,odd_spring,idx_n,r0=1,n_pot=1,noise=0,seed=0):
    yinit = np.hstack((Pos0[:,0],Pos0[:,1]))
    ts = np.linspace(ti,tf,int(tf/dt)+1,endpoint=True)
    sol = solve_ivp(OddNetwork, [ti,tf],yinit, args=(even_spring, odd_spring, idx_n, r0,n_pot,noise,seed), t_eval=ts, dense_output=False,vectorized=False,rtol=1E-7,atol=1E-7,method='RK45') 
    return sol

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--a', type=np.float64, default=1) #lattice const
    parser.add_argument('--Ngrid', type=int, default=50) #grid size, must be even
    parser.add_argument('--Nstep', type=int, default=300) #number of sim steps
    parser.add_argument('--seed',type=int,default=0) #random seed
    parser.add_argument('--amp',type=np.float64,default=0.2) #perturbation amplitude
    parser.add_argument('--perturb',type=str,default='rand') #type of perturbation
    parser.add_argument('--dt',type=np.float64,default=0.1) #timestep
    parser.add_argument('--noise',type=np.float64,default=0.) #noise level in sys
    parser.add_argument('--keven',type=list,default=[0,0,0]) #[ke, keA, keB]
    parser.add_argument('--kodd',type=list,default=[1,1,1]) #[ko, koA, koB]
    parser.add_argument('--alpha',type=np.float64,default=0.) #degree of nonreciprocity between A and B particles
    parser.add_argument('--init',type=str,default='one')
    parser.add_argument('--r0',type=np.float64,default=1.)
    parser.add_argument('--deg',type=int,default=1) #degree of nonlinearity of even spring
    parser.add_argument('--outdir',type=str,default='.')

    args = parser.parse_args()

    a = args.a  # inter-electron spacing, controls density
    Ngrid = args.Ngrid
    Nstep = args.Nstep
    seed = args.seed
    amp = args.amp
    perturb = args.perturb
    dt = args.dt
    noise = args.noise
    ke,keA,keB = args.keven
    ko,koA,koB = args.kodd
    alpha = args.alpha
    initcond = args.init
    r0 = args.r0
    deg = args.deg
    outdir = args.outdir
     
    Pos0=InitHexagon(Ngrid,a=a,plotting=0)
    N= len(Pos0)
    nns=nnsearch(Pos0,1.4*a)
    Pos0 = InitPosPerturb(Pos0,amp,opt=perturb)
    initcond,Type = InitPtclTypes(Pos0,initcond,seed)
    even_spring, odd_spring = DefSpringMats(nns, Type, ke, keA, keB, ko, koA, koB,alpha)

    ntype1=np.sum(Type)
    savedir = os.path.join(outdir,f'tri_Ngrid_{Ngrid}_init_{initcond}_perturb_{perturb}_keven{ke}{keA}{keB}_kodd{ko}{koA}{koB}_dt_{dt}_Nstep_{Nstep}_noise_{noise}')
    print(savedir)
    if os.path.isdir(savedir) == False:
        os.mkdir(savedir)

    meta = {'a':a,'r0':r0,'deg':deg,'dt':dt,'Ngrid':Ngrid,'Nstep':Nstep,'init': initcond, 'seed':seed,'amp':amp,'perturb':perturb,'ke':even_spring,'ko':odd_spring,'alpha':alpha,'noise':noise,'ntype1':ntype1}
    scipy.io.savemat(os.path.join(savedir,'meta.mat'),meta)

    sol = runsim(0,Nstep*dt,dt,Pos0,even_spring,odd_spring,nns,r0=r0,n_pot=deg,noise=noise,seed=seed)
    Xmat = sol.y[:N,:]
    Ymat = sol.y[N:,:]
    scipy.io.savemat(os.path.join(savedir,'simdata.mat'),{"Xmat":Xmat,"Ymat":Ymat,'Type':Type})
    
    
