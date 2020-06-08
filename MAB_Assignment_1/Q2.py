import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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

class CONSIST:
	def __init__(self,d):
		self.h_t=np.arange(d) # all the experts form the hypothesis class
		self.mask=np.ones_like(self.h_t)
		self.current_expert=np.random.choice(self.h_t, p=self.mask/self.mask.sum())
		self.mistake=[]
		self.t=0
		self.p=True
	def predict(self, xt):
		return xt[self.current_expert]
	
	def update(self, xt, yt):
		# plt.plot(self.mask)
		# plt.show()
		self.t=self.t+1
		pred=self.predict(xt)
		if(np.count_nonzero(self.mask)==1):
			if(self.p):
				print("Chosen Expert in CONSIST ", self.current_expert,' found at ', self.t)
			self.p=False
			return
		self.current_expert=np.random.choice(self.h_t, p=self.mask/self.mask.sum())

		if(pred!=yt):
			self.mistake.append(1)
			wrongIdx=np.where(xt!=yt)[0]
			self.mask[wrongIdx]=0
			if(np.count_nonzero(self.mask)==0):
				print("Non Realizable Adversary!")
				exit(0)
			# print('mask ', np.where(self.mask==1)[0])
		else:
			self.mistake.append(0)
	def mistakeHistory(self):
		return self.mistake
	def return_mask(self):
		return self.mask

class Halving:
	def __init__(self,d):
		self.h_t=np.arange(d) # all the experts form the hypothesis class
		self.mask=np.ones_like(self.h_t)
		self.mistake=[]
		self.t=0
		self.p=True
	def predict(self, xt):
		if (np.mean(xt[self.h_t]) >= 0.5):
			return 1.0
		else:
			return 0.0
	
	def update(self, xt, yt):
		self.t=self.t+1
		pred=self.predict(xt)
		if(self.h_t.shape[0]==1):
			if self.p:
				print ("Chosen Expert by Halving ", self.h_t[0],' found at ', self.t)
			self.p=False
			return
		if(pred!=yt):
			self.mistake.append(1)
			if(self.h_t.shape[0]==0):
				print("Non Realizable Adversary!")
				exit(0)
			active_experts_prediction = xt[self.h_t]
			rightIdx=np.where(active_experts_prediction==yt)[0]
			self.h_t = self.h_t[rightIdx]
		else:
			self.mistake.append(0)
	def mistakeHistory(self):
		return self.mistake

class RunExperiment:
	def __init__(self, x, y, ch=0, eps=0, runIter=10000):
		self.x=x
		self.y=y
		self.ch=ch
		self.eps=eps
		self.runIter=runIter
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
		elif(ch==4):
			self.algo = CONSIST(self.d)
		elif(ch==5):
			self.algo = Halving(self.d)
		else:
			print ("Invalid Choice")
			exit(0)
	
	def run(self):
		idx=0
		for xt,yt in zip(self.x,self.y):
			idx+=1
			if(idx>self.runIter):
				return
			self.algo.update(xt,yt)
		# if(self.ch==4):
		# 	columns=self.x[:, self.algo.return_mask()]
		# 	columns_same=np.array(columns == columns[:, 0], dtype='bool').all()
		# 	print(columns_same)
		# 	print(self.algo.return_mask())
	
	def mistakeHistory(self):
		return self.algo.mistakeHistory()

if __name__=='__main__':
	# x=np.loadtxt("./Datasets/Dataset1_X.txt",delimiter=',')
	# y=np.loadtxt("./Datasets/Dataset1_Y.txt")
	x=np.random.randint(0,2,size=(10000,1000))
	predictor_chosen=np.random.choice(x.shape[1])
	print("True Expert ", predictor_chosen)
	y = x[:, predictor_chosen]
	for i in x.T:
		if(np.array_equal(i,y)):
			print("Realizable")
	
	cons = RunExperiment(x,y,4,runIter=10000)
	cons.run()
	cons_mistakes = cons.mistakeHistory()
	
	half = RunExperiment(x,y,5,runIter=10000)
	half.run()
	half_mistakes = half.mistakeHistory()
	plt.plot(np.cumsum(cons_mistakes), label='CONSIST', color='g')
	plt.plot(np.cumsum(cons_mistakes), 'o', color='g')
	plt.plot(np.cumsum(half_mistakes), label='Halving', color='r')
	plt.plot(np.cumsum(half_mistakes), 'o', color='r')
	plt.legend(loc='best')
	plt.xlabel("Timesteps")
	plt.ylabel("Number of Mistakes")
	plt.title("No. of Mistakes vs Timesteps of Online Algorithms")
	plt.savefig('consist_halving.png')
	plt.show()
	'''
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
	'''