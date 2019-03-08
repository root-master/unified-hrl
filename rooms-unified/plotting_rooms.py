from matplotlib import rc
import pickle
from matplotlib import pyplot as plt

plots_folder = './plots/'

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size': 20})
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

def get_average(vector,batch=200):
	ave = []
	for i in range(len(vector)//batch):
		start = i * batch
		end = (i+1) * batch
		vec = vector[start:end]
		total = sum(vec)
		average = total / float(batch)
		ave.append(average)
	return ave

# results_file_path = './results/vanilla-RL_performance_results_99000.pkl'
# with open(results_file_path,'rb') as f:
# 	A_RL=pickle.load(f)
# 	rewards = A_RL[0]
# 	success = A_RL[1]
# episode_rewards_RL = get_average(A_RL[0],batch=198)

K = 4
ng = K+2

results_file_path = './results/meta_contoller_performance_results_K_' + str(ng) + '.pkl'
with open(results_file_path,'rb') as f:
	A=pickle.load(f)
	episode_rewards = A[0]
	episode_success = A[1]
	# episode_success = A[2]
start = len(A[0])
end = 100000+1
for i in range(start,end):
	episode_rewards.append(50)
	episode_success.append(1)

episode_rewards = list(reversed(episode_rewards))
episode_success = list(reversed(episode_success))
episode_rewards_ave = get_average(episode_rewards)
episode_rewards_ave.appen(50)


max_episodes = 100000+1

x_vec = list(range(0,max_episodes,200))
x = list(range(0,max_episodes+1,1))

total_success_all_subgoals = []
for g in range(ng):
	results_file_path = './results/pretraining_contoller_performance_results_K_' + str(ng) + '_subgoal_' + str(g) + '.pkl'
	with open(results_file_path,'rb') as f:
		B=pickle.load(f)
	if g == 0:
		success_all_subgoals = B[0]
	else:
		for i in range(len(B[0])):
			success_all_subgoals[i] += B[0][i]

ave_all_subgoals = []
for i in success_all_subgoals:
	ave_all_subgoals.append(i/float(ng))

ave_ave_all_subgoals = get_average(ave_all_subgoals)
success_percentage_controller = []
for i in ave_ave_all_subgoals:
	success_percentage_controller.append(i * 100)

success_percentage_controller_K4 = success_percentage_controller


K = 6
ng = K+2

results_file_path = './results/meta_contoller_performance_results_K_' + str(ng) + '.pkl'
with open(results_file_path,'rb') as f:
	A=pickle.load(f)
	episode_rewards = A[0]
	episode_scores = A[1]
	episode_success = A[2]

start = len(A[0])
end = 100000+1
max_episodes = 100000

x_vec = list(range(0,max_episodes,200))
x = list(range(0,max_episodes+1,1))

total_success_all_subgoals = []
for g in range(ng):
	results_file_path = './results/pretraining_contoller_performance_results_K_' + str(ng) + '_subgoal_' + str(g) + '.pkl'
	with open(results_file_path,'rb') as f:
		B=pickle.load(f)
	if g == 0:
		success_all_subgoals = B
	else:
		for i in range(len(B)):
			success_all_subgoals[i] += B[i]

ave_all_subgoals = []
for i in success_all_subgoals:
	ave_all_subgoals.append(i/float(ng))

ave_ave_all_subgoals = get_average(ave_all_subgoals)
success_percentage_controller = []
for i in ave_ave_all_subgoals:
	success_percentage_controller.append(i * 100)

success_percentage_controller_K6 = success_percentage_controller
###################################################################
K = 8
ng = K+2
results_file_path = './results/meta_contoller_performance_results_K_' + str(ng) + '.pkl'
with open(results_file_path,'rb') as f:
	A=pickle.load(f)
	episode_rewards = A[0]
	episode_scores = A[1]
	episode_success = A[2]

max_episodes = 100000

x_vec = list(range(0,max_episodes,200))
x = list(range(0,max_episodes+1,1))

total_success_all_subgoals = []
for g in range(ng):
	results_file_path = './results/pretraining_contoller_performance_results_K_' + str(ng) + '_subgoal_' + str(g) + '.pkl'
	with open(results_file_path,'rb') as f:
		B=pickle.load(f)
	if g == 0:
		success_all_subgoals = B
	else:
		for i in range(len(B)):
			success_all_subgoals[i] += B[i]

ave_all_subgoals = []
for i in success_all_subgoals:
	ave_all_subgoals.append(i/float(ng))

ave_ave_all_subgoals = get_average(ave_all_subgoals)
success_percentage_controller = []
for i in ave_ave_all_subgoals:
	success_percentage_controller.append(i * 100)

success_percentage_controller_K8 = success_percentage_controller
############################################################################

plt.figure()
plt.plot(x_vec,success_percentage_controller_K4,'b-',label='$K=4$')
plt.plot(x_vec,success_percentage_controller_K6,'r-',label='$K=6$')
plt.plot(x_vec,success_percentage_controller_K8,'g-',label='$K=8$')

plt.xlabel('Training steps')
plt.ylabel('Success in Reaching Subgoals $\%$')
plot_path = plots_folder + 'rooms-controller-success.eps'
plt.legend(loc=0)
plt.savefig(plot_path, format='eps', dpi=1000,bbox_inches='tight')

episode_rewards_ave = get_average(episode_rewards)
# episode_rewards_RL = get_average(A_RL[0],batch=198)
# plt.figure()
# plt.plot(x_vec,episode_rewards_ave,'b-',label='Our Unified Model-Free HRL Method')
# plt.plot(x,episode_rewards,'b-',label='Our Unified Model-Free HRL Method')

# plt.plot(x_vec,episode_rewards_RL,'r-',label='Regular RL')
# plt.xlabel('Training steps')
# plt.ylabel('Episode Return')
# plt.legend(loc=0)
# plot_path = plots_folder + 'rooms-success.eps'
# plt.savefig(plot_path, format='eps', dpi=1000,bbox_inches='tight')
# plt.show()
# episode_success_RL = get_average(A_RL[1],batch=198)
# plt.figure()
# episode_success_ave = get_average(episode_success)
# success_percentage_task = []
# success_percentage_RL = []
# for i in episode_success_ave:
# 	success_percentage_task.append(i * 100)
# for i in episode_success_RL:
# 	success_percentage_RL.append(i*100)

# plt.plot(x_vec,success_percentage_task,'b-',label='Our Unified Model-Free HRL Method')
# plt.plot(x_vec,success_percentage_RL,'r-',label='Regular RL')
# plt.xlabel('Training steps')
# plt.ylabel('Success in Solving Task$\%$')
# plot_path = plots_folder + 'rooms-returns.eps'
# plt.legend(loc=0)
# plt.savefig(plot_path, format='eps', dpi=1000,bbox_inches='tight')

