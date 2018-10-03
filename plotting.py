from matplotlib import rc
import pickle
from matplotlib import pyplot as plt

plots_folder = './plots/'

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size': 20})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
rc('xtick', labelsize=12)
rc('ytick', labelsize=14)
rc('axes', labelsize=18)
rc('figure', titlesize=18)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
rc('legend', fontsize=12)

results_folder = './results/'
plots_folder = './plots/'

max_steps = 2500000
freq = 100000
success = []
x = list(range(freq,max_steps+freq,freq))
for i in range(1,26):
	t = freq*i
	results_file_path = './results_phase_2/performance_results_' + str(t) + '.pkl'
	with open(results_file_path,'rb') as f:
		A=pickle.load(f)
		success.append(100*sum(A[5])/sum(A[4]))

DQN = [0]*len(success)
plt.figure()
plt.plot(x,success,'b-',label='Our Unified Model-Free HRL Method')
plt.plot(x,DQN,'r-',label='DeepMind DQN Algorithm (Mnih et. al., 2015)')
plt.xlabel('Training steps')
plt.ylabel('Success in reaching subgoals $\%$')
plot_path = plots_folder + 'montezuma-success.eps'
plt.legend(loc=0)
plt.savefig(plot_path, format='eps', dpi=1000,bbox_inches='tight')

step = max_steps // len(A[2])
x_vec = []
for i in range(0,len(A[2])):
	x_vec.append(i*step)

sum_rewards = []
for i in range(len(A[2])):
	a = sum(A[2][0:i])/(i+1)
	sum_rewards.append(a*350)

DQN = [0]*len(sum_rewards)

plt.figure()
plt.plot(x_vec,sum_rewards,'b-',label='Our Unified Model-Free HRL Method')
plt.plot(x_vec,DQN,'r-',label='DeepMind DQN Algorithm (Mnih et. al., 2015)')
plt.xlabel('Training steps')
plt.ylabel('Average return over 10 episdes')
plot_path = plots_folder + 'montezuma-returns.eps'
plt.legend(loc=0)
plt.savefig(plot_path, format='eps', dpi=1000,bbox_inches='tight')

from image_processing import *
import pickle
rec = Recognizer()
from matplotlib import pyplot as plt

C = [(79, 101),
 (83, 120),
 (128, 159),
 (110, 118),
 (136, 127),
 (78, 81),
 (120, 81),
 (33, 82),
 (35.0, 171.0),
 (25,129)]

O = [(12,120),(29,84),(127,83)]
G = C + O

img = rec.base_img
color = (0,0,255)
for g in C:
	g = (int(g[0]),int(g[1]))
	img = draw_circle(img, g, 4, color)

plt.figure()
color = (255,100,0)
for g in O:
	g = (int(g[0]),int(g[1]))
	img = draw_circle(img, g, 4, color)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()


