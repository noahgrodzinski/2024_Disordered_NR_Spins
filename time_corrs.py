import random
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

class Chain_1D():
    def __init__(self, N:int, r:float, u:float, h:float):
        self.N = N
        self.theta = np.array([np.pi*2*random.random() for i in range(N)])
        self.J_L = np.add(np.random.choice([-1.0, 1.0], size=(N), p=(0.5-u/2, 0.5+u/2)), h*np.ones((N)))
        self.J_R = np.add(np.random.choice([-1.0, 1.0], size=(N), p=(0.5-u/2, 0.5+u/2)), h*np.ones((N)))
    
    def evolve_timestep(self, dt: float):
        theta_dot = np.array([(self.J_R[i]*np.sin(self.theta[(i+1)%self.N] - self.theta[i]) + self.J_L[(i-1)%self.N]*np.sin(self.theta[(i-1)%self.N] - self.theta[i])) for i in range(self.N)])
        self.theta = np.add(self.theta, dt*theta_dot)

    def evolve_and_record(self, t:float, dt:float):
        n_T = int(t//dt)
        theta_history =[]
        for timestep in range(n_T):
            if timestep%int(1/dt) == 0:
                theta_history.append(self.theta%(np.pi))
            self.evolve_timestep(dt)
        return np.array(theta_history)
    
    def evolve_and_record_all(self, t:float, dt:float):
        n_T = int(t//dt)
        theta_history =[]
        for timestep in range(n_T):
            theta_history.append(self.theta%(np.pi))
            self.evolve_timestep(dt)
        return np.array(theta_history)
    

class Chain_2D(Chain_1D):
    def __init__(self, N: int, r: float, kappa: float):
        self.N = N
        self.theta = np.array([[np.pi*2*random.random() for i in range(N)] for j in range(N)])
        J_tilde_R = np.array([np.random.choice([-1.0, 1.0], size=(N), p=(0.5, 0.5)) for j in range(N)])
        J_tilde_L = np.array([[np.random.choice([J_tilde_R[j][i], -J_tilde_R[i][j]], p=(0.5+r/2, 0.5-r/2)) for i in range(N)] for j in range(N)])
        J_tilde_U = np.array([np.random.choice([-1.0, 1.0], size=(N), p=(0.5, 0.5)) for j in range(N)])
        J_tilde_D = np.array([[np.random.choice([J_tilde_U[j][i], -J_tilde_U[i][j]], p=(0.5+r/2, 0.5-r/2)) for i in range(N)] for j in range(N)])

        self.couplings_R = np.add(kappa*np.copy(J_tilde_R), (1-kappa)*np.ones((N,N)))
        self.couplings_L = np.add(kappa*np.copy(J_tilde_L), (1-kappa)*np.ones((N,N)))
        self.couplings_U = np.add(kappa*np.copy(J_tilde_U), (1-kappa)*np.ones((N,N)))
        self.couplings_D = np.add(kappa*np.copy(J_tilde_D), (1-kappa)*np.ones((N,N)))
        self.r = r
        self.kappa = kappa
        

    def evolve_timestep(self, dt: float):
        for i in range(self.N):
            for j in range(self.N):
                theta_dot_ij = (self.couplings_L[(i-1)%self.N][j]*np.sin(self.theta[(i-1)%self.N][j] - self.theta[i][j])
                + self.couplings_R[i][j]*np.sin(self.theta[(i+1)%self.N][j] - self.theta[i][j])
                + self.couplings_U[i][(j-1)%self.N]*np.sin(self.theta[i][(j-1)%self.N] - self.theta[i][j])
                + self.couplings_D[i][j]*np.sin(self.theta[i][(j+1)%self.N] - self.theta[i][j]))
                self.theta[i][j] = np.add(self.theta[i][j],dt*theta_dot_ij)
    

    def evolve_and_record(self, t: float, dt: float):
        self.__init__(self.N, self.r, self.kappa)
        n_T = int(t//dt)
        theta_history =[]
        for timestep in range(n_T):
            if timestep%int(1/dt) == 0:
                theta_history.append(self.theta%(np.pi))
            self.evolve_timestep(dt)
        return np.array(theta_history)
    

def calc_Ct(history, t, t_w):
    row1 = np.array(history)[t_w, :, :].flatten()
    row2 = np.array(history)[t_w+t, :, :].flatten()
    return np.mean(np.exp(2*1j*np.subtract(row2, row1)))

tic = time.perf_counter()

T=300
dt=0.1
r=0.0
kappa=0.6
N=50
runs=200
chain = Chain_2D(N=N, r=r, kappa=kappa)
histories = np.array([chain.evolve_and_record(t=T, dt=dt) for i in range(runs)])

t_ws = [10, 20, 40, 80, 160]
t_w_correlations =[]
for t_w in t_ws:
    t_w_correlations.append(np.hstack(([np.abs(np.mean([calc_Ct(histories[run, :, :, :], t, t_w) for run in range(histories.shape[0])])) for t in range(T-t_w)],np.zeros(t_w))))



np.save('2d_temporal_corrs', np.array(t_w_correlations))
toc = time.perf_counter()
print(f"Run in {toc - tic:0.4f} seconds")