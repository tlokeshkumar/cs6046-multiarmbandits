import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, d):
        self.d = d
        self.t=0
        self.w = np.zeros(d)
        self.mistake=[]
    
    def predict(self, xt):
        pred = np.sign(self.w.dot(xt))
        # accounting for 0 in the sign
        if(pred<=0):
            pred=-1.0
        else:
            pred=1.0
        
        return pred

    def update(self,xt,yt):
        # convert to {-1,1}
        yt=2*yt-1.0

        self.t = self.t+1
        
        pred=self.predict(xt)

        # update equation
        
        if(yt!=pred):
            self.mistake.append(1)
            self.w = self.w + yt*xt
        else:
            self.mistake.append(0)


    def weight(self):
        return self.w
    
    def mistakeHistory(self):
        return self.mistake

class Willow:
    def __init__(self, d):
        self.d=d
        self.w=np.ones(d)
        self.t=0
        self.mistake=[]

    
    def predict(self, xt):
        pred = np.sign(self.w.dot(xt)-self.d)
        
        # accounting for 0 in the sign
        if(pred<0):
            pred=0.0
        else:
            pred=1.0
        
        return pred
    
    def update(self,xt,yt):
        self.t = self.t+1
        
        pred=self.predict(xt)

        # update equation
        if(yt!=pred):
            self.mistake.append(1)
            if(yt==1 and pred==0):
                # w_{t+1} = w_t if xt==1
                self.w[np.where(xt==1)[0]]*=2
            else:
                self.w[np.where(xt==1)[0]]*=0.5
        else:
            self.mistake.append(0)

    def weight(self):
        return self.w

    def mistakeHistory(self):
        return self.mistake

class WeightedMajority:
    def __init__(self,d,eps):
        self.eps=eps
        self.d=d
        self.t=0
        self.w=np.ones(d)
        self.mistake=[]
    
    def predict(self, xt):
        idx = np.where(xt==1)[0]
        sum1 = np.sum(self.w[idx])
        sum0 = np.sum(self.w) - sum1

        if(sum1>sum0):
            return 1
        else:
            return 0
    
    def update(self, xt, yt):
        self.t=self.t+1
        pred=self.predict(xt)

        if(pred!=yt):
            self.mistake.append(1)
            wrongIdx=np.where(xt!=yt)[0]
            self.w[wrongIdx]*=(1-self.eps)
        else:
            self.mistake.append(0)
    
    def weight(self):
        return self.w
    
    def mistakeHistory(self):
        return self.mistake
class RandomizedWeightedMajority:
    def __init__(self,d,eps):
        self.eps=eps
        self.d=d
        self.t=0
        self.w=np.ones(d)
        self.mistake=[]
    
    def predict(self, xt):
        idx = np.where(xt==1)[0]
        sum1 = np.sum(self.w[idx])
        sum0 = np.sum(self.w) - sum1
        
        prob_of_success=sum1/(sum1+sum0)

        return np.random.binomial(1,prob_of_success,None)
    
    def update(self, xt, yt):
        self.t=self.t+1
        pred=self.predict(xt)

        if(pred!=yt):
            self.mistake.append(1)
            wrongIdx=np.where(xt!=yt)[0]
            self.w[wrongIdx]*=(1-self.eps)
        else:
            self.mistake.append(0)
    
    def weight(self):
        return self.w
    
    def mistakeHistory(self):
        return self.mistake

class RunExperiment:
    def __init__(self, x, y, ch=0, eps=0):
        self.x=x
        self.y=y
        self.ch=ch
        self.eps=eps
        self.d = self.x.shape[1]

        if(ch==0):
            self.algo = Perceptron(self.d)
        elif(ch==1):
            self.algo = Willow(self.d)
        elif(ch==2):
            if(self.eps==0):
                print ("EPS=0 for Weighted Majority Algorithm")
            self.algo = WeightedMajority(self.d, self.eps)
        elif(ch==3):
            if(self.eps==0):
                print ("EPS=0 for Weighted Majority Algorithm")
            self.algo = RandomizedWeightedMajority(self.d, self.eps)
        else:
            print ("Invalid Choice")
            exit(0)
    
    def run(self):
        for xt,yt in zip(self.x,self.y):
            self.algo.update(xt,yt)
    
    def mistakeHistory(self):
        return self.algo.mistakeHistory()

if __name__=='__main__':
    x=np.loadtxt("./Datasets/Dataset1_X.txt",delimiter=',')
    y=np.loadtxt("./Datasets/Dataset1_Y.txt")
    
    for i in x.T:
        if(np.array_equal(i,y)):
            print("Realizable")
    
    

    p1 = RunExperiment(x,y)
    p1.run()
    P_mistakes = p1.mistakeHistory()
    
    w1 = RunExperiment(x,y,1)
    w1.run()
    W_mistakes = w1.mistakeHistory()
    
    wm1 = RunExperiment(x,y,2,0.1)
    wm1.run()
    WM_mistakes = wm1.mistakeHistory()

    rwm1 = RunExperiment(x,y,3,0.3)
    rwm1.run()
    RWM_mistakes = rwm1.mistakeHistory()
    
    plt.plot(np.cumsum(RWM_mistakes), label='RWM')
    plt.plot(np.cumsum(WM_mistakes), label='WM')
    plt.plot(np.cumsum(W_mistakes), label='Willow')
    plt.plot(np.cumsum(P_mistakes), label='Perceptron')
    plt.xlabel("Timesteps")
    plt.ylabel("Number of Mistakes")
    plt.title("#Mistakes vs Timesteps of Online Algorithms")
    plt.legend(loc='best')
    plt.show()