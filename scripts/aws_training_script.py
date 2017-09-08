# LIBRARIES
from collections import deque
import skimage.measure
import numpy as np
import gym
from gym import wrappers
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
import pickle


# USE GPUs?
gpu = 1


# GAME SETUP        
num_games = 2500  ## number of games to play
time_steps = 10000  ## max number of time steps per game
record = 1  ## record training or game play
render = 0  ## show game in real-time


# WHERE TO SAVE FILES
path_record = '/home/ubuntu/scp/breakout-training'
path_save_cnn = '/home/ubuntu/scp/DL_RL_Atari_breakout_5e_10000t'
path_save_score = '/home/ubuntu/scp/score.pkl'


# FUNCTIONS
def preprocess(rgb_tensor):
    '''
    Transforms 3D RGB numpy tensor: crop, convert to 2D grayscale, downsample.
    '''
    crop = rgb_tensor[30:194,:,:]
    grayscale = np.dot(crop[...,:3], [0.2989, 0.5870, 0.1140])  ## using Matlab's formula
    downsample = skimage.measure.block_reduce(grayscale, (2,2), np.max)
    standardize = (downsample - downsample.mean()) / np.sqrt(downsample.var() + 1e-5)
    return standardize
    
    
# CLASSES    
class Action:
    '''Returns an action value which is an int in range [0,3]'''
    
    def __init__(self):
        self.time_ = 0
        
    def update_time(self):
        self.time_ += 1
        
    def action(self):
        if self.time_ == 0:
            self.action_ = 1  ## start game by firing ball
        else:
            # take agent-based action every 4 time steps; else push action forward w/out agent computing
            if self.time_%4 == 0:
                if np.random.binomial(n=1, p=eg.epsilon_, size=1):
                    self.action_ = env.action_space.sample()  ## take random action
                else:
                    self.action_ = cnn(Variable(torch.Tensor(initial_seq).unsqueeze(0).unsqueeze(0).cuda())).data.max(1)[1][0]  ## take optimal action according to NN
        return self.action_
        
class ExperienceReplay:
    '''Long-term memory of experience tuples to break sequential correlations'''
    dq_ = deque(maxlen=int(1e6))

    def __init__(self, C):
        self.capacity_ = C  ## may not need this since not used anywhere?
        
    def add_experience(self, experience_tuple):
        '''add new experience to experience replay'''
        self.dq_.append(experience_tuple)
        
    def sample(self, capacity=32):
        '''sample from experience replay'''
        nb_items = len(self.dq_)
        if nb_items > capacity:
            idx = np.random.choice( nb_items, size=capacity, replace=False)
        else:
            idx = np.random.choice( nb_items, size=nb_items, replace=False)
        return [self.dq_[i] for i in idx]
        
class EpsilonGenerator():
    '''Linear annealing'''
    def __init__(self, start, stop, steps):
        self.epsilon_ = start
        self.stop_ = stop
        self.steps_ = steps
        self.step_size_ = (self.epsilon_ - stop) / (self.steps_)
        self.count_ = 1
        
    def epsilon_update(self):
        '''generate next epsilon value'''
        if (self.epsilon_ >= self.stop_ and self.count_ < self.steps_):
            self.count_ += 1
            self.epsilon_ -= self.step_size_
        else:
            self.epsilon_ = self.stop_
            self.count_ += 1
            
class CNN(nn.Module):
    '''Convolutional Neural Network'''
    def __init__(self,):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 4)  ## Conv2d(nChannels, filters, kernel, stride)
        self.conv2 = nn.Conv2d(16, 32, 4, 4)
        self.fc1 = nn.Linear(32 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 32 * 4 * 4)  ## reshape 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
class Dataset:
    '''Creates dataset and targets'''    
    def __init__(self):
        self.replay_size_ = None
        self.data_ = None
        self.target_ = None
        
    def get_data(self):
        '''get minibatch of dataset and targets'''
        self.replay_size_ = len(minibatch)
        # create tensor of initial observations
        self.data_ = Variable(torch.Tensor([minibatch[i][0] for i in range(self.replay_size_)]).unsqueeze(1).cuda())
        # create tensor of corresponding target variable values
        target_list = []
        for i in range(self.replay_size_):
            observed = Variable(torch.Tensor(minibatch[i][3]).cuda())
            if minibatch[i][4] == 'terminal':
                target_list.append(minibatch[i][2])
            else:
                target_list.append(minibatch[i][2] + discount * 
                                   cnn(observed.unsqueeze(0).unsqueeze(0)).data.max(1)[1][0])
        self.target_ = Variable(torch.Tensor(target_list).cuda())
        
class Score:
    '''Tracks score for each game and adds to list'''    
    def __init__(self):
        self.all_scores_ = []
        self.game_score_ = 0
        
    def reset(self):
        '''reset score to zero upon game termination'''
        self.game_score_ = 0
        
    def increment(self, reward):
        '''increment score per iteration'''
        self.game_score_ += reward
        
    def update(self):
    	'''append game score to running list'''
    	self.all_scores_.append(self.game_score_)


# Atari emulator
env = gym.make('Breakout-v0')
# whether to record training
if record:
    env = wrappers.Monitor(env, 
                           directory=path_record, 
                           video_callable=None, ## takes video when episode number is perfect cube
                           force=True)


# INSTANTIATE KEY CLASSES
cnn = torch.nn.DataParallel(CNN()).cuda() if gpu else CNN()
er = ExperienceReplay(C=1e6)
eg = EpsilonGenerator(start=1, stop=0.1, steps=1e6)
agent = Action()
dataset = Dataset()
score = Score()


# SETUP VARIABLES
discount = 0.9  
learning_rate = 0.01


# CNN SETUP
criterion = nn.MSELoss()
optimizer = optim.RMSprop(cnn.parameters(), 
                          lr=learning_rate, 
                          alpha=0.99, 
                          eps=1e-08, 
                          weight_decay=0, 
                          momentum=0, 
                          centered=False)


# PLAY GAME
for episode in range(num_games):
    
    ## start/reset environment + store observation
    initial_seq = preprocess(env.reset())
    
    ## reset score
    score.reset()
    
    for t in range(time_steps):
        
        ## show game in real-time
        if render:
            env.render()
        
        # take action (0=do nothing; 1=fire ball; 2=move right; 3=move left)
        action = agent.action()
        agent.update_time()
        
        # update epsilon for epsilon-greedy implementation
        eg.epsilon_update()
        
        # get feedback from emulator
        observation, reward, done, info = env.step(action)
        
        # update score
        score.increment(reward)
        
        # preprocess new observation post action    
        final_seq = preprocess(observation)
        
        # stop if no more moves, else continue and update
        if done:
            er.add_experience((initial_seq, action, reward, final_seq, 'terminal'))
            score.update()
            break
        else:
            er.add_experience((initial_seq, action, reward, final_seq, 'nonterminal'))
            
        # get mini-batch sample from experience replay (fyi - randomizes index)
        minibatch = er.sample()
        
        # get data for updating policy network
        dataset.get_data()
        
        # Update CNN
        optimizer.zero_grad()  ## zero the parameter gradients
        outputs = cnn(dataset.data_).max(1)[0]  ## feedforward pass
        loss = criterion(outputs, dataset.target_)  ## calculate loss
        loss.backward()  ## backpropagation
        optimizer.step()  ## update network weights
            
        # set new observation as initial observation
        initial_seq = final_seq
        
    # progress
    print("finished episode %d of %d" % (episode, num_games-1)) 
        
env.close()

# Save CNN
torch.save(cnn.state_dict(), path_save_cnn) 
# Save List of Scores
pickle.dump(score.all_scores_, open( path_save_score, "wb" ))

