import numpy as np
import matplotlib.pyplot as plt


class EulerMaruyama:

    def __init__(self,num_sims,t_init,t_end,grid_size,y_init,mu,sigma,verbose=0):
        self.num_sims = num_sims  # Display five runs

        self.t_init = t_init
        self.t_end  = t_end
        self.N = grid_size  # Compute 1000 grid points
        self.dt = float(t_end - t_init) / grid_size
        self.y_init = y_init
        self.mu = mu
        self.sigma = sigma
        self.ts = np.array([np.arange(self.t_init, self.t_end, self.dt)]*len(y_init))
        #self.dW = dW
        self.solve()

    def dW(self,delta_t):
        """Sample a random number at each call."""
        draw = np.random.normal(loc=0.0, scale=np.sqrt(delta_t),size=(len(self.y_init)))
        return draw

    def solve(self,verbose=0):

        ys = np.zeros((self.N,len(self.y_init)))
        ys[0] = self.y_init
        self.solution = []
        for n in range(self.num_sims):
            if n%100==0 and verbose==1:
                print('Number of simulations completed:',n,'of',self.num_sims)
            for i in range(1, len(self.ts.T)):
                t = (i-1) * self.dt
                y = ys[i-1]
                #print(y)
                ys[i,:] = y + self.mu(y, t) * np.array([self.dt]*len(self.y_init)).T + self.sigma(y, t) * self.dW(self.dt)
                #print(ys)
            if verbose==2:
                print('Ending Value:',ys[-1],'for run',n)
            self.solution.append(ys.copy())
        return None

    def plot_solution(self,eq=0):
        #order = (int(np.log10(len(self.solution)))-1)
        for ys in self.solution:
            #print(len(ys),len(self.ts))
            #print(self.ts.T[:,0])
            plt.plot(self.ts.T[:,eq], ys[:,eq], color='blue', alpha=(3e-3))
        plt.plot(self.ts.T[:,eq], np.min(np.array(self.solution),axis=0)[:,eq],color='black')
        plt.plot(self.ts.T[:,eq], np.max(np.array(self.solution),axis=0)[:,eq],color='black')
        plt.show()

    def plot_dist(self,t,eq=0):
        temp = []
        idx = (np.abs(self.ts - t)).argmin()
        #print(self.solution)
        for ys in self.solution:
            temp.append(ys[idx,eq])
        plt.hist(temp,bins=50,density=True)
        plt.title("The mean is: {0}\nThe sd is : {1}\nAt time: {2}".format(np.mean(temp),np.std(temp),self.ts[eq,idx]))
        plt.show()

if __name__=='__main__':
    beta = 0.105
    gamma = 0.07
    alpha = .85
    delta = 0.07
    def mu(y,t):
        #print(y)
        return -beta*y[0]*y[1],beta*y[1]*y[0]-gamma*y[1],gamma*y[1]
    def sigma(y,t):
        return np.array([-alpha*y[1]*y[0],alpha*y[1]*y[0],0])

    em = EulerMaruyama(num_sims=10000,t_init=0,t_end=200,grid_size=1000,y_init=np.array([0.9999,0.0001,0.0]),mu=mu,sigma=sigma,verbose=0)
    #print(em.solution)
    em.plot_solution(eq=1)
    em.plot_dist(45,eq=1)
    