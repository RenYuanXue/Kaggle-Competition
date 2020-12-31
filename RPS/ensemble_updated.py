import pandas as pd
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import json
from operator import itemgetter

#########################################################################################################
###############################   All used Methods for ensemble method   ################################
#########################################################################################################

# Classes used in multi_armed_bandit_agent.
class agent():
    def initial_step(self):
        return np.random.randint(3)
    
    def history_step(self, history):
        return np.random.randint(3)
    
    def step(self, history):
        if len(history) == 0:
            return int(self.initial_step())
        else:
            return int(self.history_step(history))

class rps(agent):
    def __init__(self, shift=0):
        self.shift = shift
    
    def rps(self, history):
        return self.shift % 3

# Class used in IOU2.
class Strategy:
  def __init__(self):
    # 2 different self.lengths of history, 3 kinds of history, both, mine, yours
    # 3 different self.limit self.length of reverse learning
    # 6 kinds of strategy based on Iocaine Powder
    self.num_predictor = 27


    self.len_rfind = [20]
    self.limit = [10,20,60]
    self.beat = { "R":"P" , "P":"S", "S":"R"}
    self.not_lose = { "R":"PPR" , "P":"SSP" , "S":"RRS" } #50-50 chance
    self.my_his   =""
    self.your_his =""
    self.both_his =""
    self.list_predictor = [""]*self.num_predictor
    self.length = 0
    self.temp1 = { "PP":"1" , "PR":"2" , "PS":"3",
              "RP":"4" , "RR":"5", "RS":"6",
              "SP":"7" , "SR":"8", "SS":"9"}
    self.temp2 = { "1":"PP","2":"PR","3":"PS",
                "4":"RP","5":"RR","6":"RS",
                "7":"SP","8":"SR","9":"SS"} 
    self.who_win = { "PP": 0, "PR":1 , "PS":-1,
                "RP": -1,"RR":0, "RS":1,
                "SP": 1, "SR":-1, "SS":0}
    self.score_predictor = [0]*self.num_predictor
    self.output = random.choice("RPS")
    self.predictors = [self.output]*self.num_predictor


  def prepare_next_move(self, prev_input):
    input = prev_input

    #update self.predictors
    #"""
    if len(self.list_predictor[0])<5:
        front =0
    else:
        front =1
    for i in range (self.num_predictor):
        if self.predictors[i]==input:
            result ="1"
        else:
            result ="0"
        self.list_predictor[i] = self.list_predictor[i][front:5]+result #only 5 rounds before
    #history matching 1-6
    self.my_his += self.output
    self.your_his += input
    self.both_his += self.temp1[input+self.output]
    self.length +=1
    for i in range(1):
        len_size = min(self.length,self.len_rfind[i])
        j=len_size
        #self.both_his
        while j>=1 and not self.both_his[self.length-j:self.length] in self.both_his[0:self.length-1]:
            j-=1
        if j>=1:
            k = self.both_his.rfind(self.both_his[self.length-j:self.length],0,self.length-1)
            self.predictors[0+6*i] = self.your_his[j+k]
            self.predictors[1+6*i] = self.beat[self.my_his[j+k]]
        else:
            self.predictors[0+6*i] = random.choice("RPS")
            self.predictors[1+6*i] = random.choice("RPS")
        j=len_size
        #self.your_his
        while j>=1 and not self.your_his[self.length-j:self.length] in self.your_his[0:self.length-1]:
            j-=1
        if j>=1:
            k = self.your_his.rfind(self.your_his[self.length-j:self.length],0,self.length-1)
            self.predictors[2+6*i] = self.your_his[j+k]
            self.predictors[3+6*i] = self.beat[self.my_his[j+k]]
        else:
            self.predictors[2+6*i] = random.choice("RPS")
            self.predictors[3+6*i] = random.choice("RPS")
        j=len_size
        #self.my_his
        while j>=1 and not self.my_his[self.length-j:self.length] in self.my_his[0:self.length-1]:
            j-=1
        if j>=1:
            k = self.my_his.rfind(self.my_his[self.length-j:self.length],0,self.length-1)
            self.predictors[4+6*i] = self.your_his[j+k]
            self.predictors[5+6*i] = self.beat[self.my_his[j+k]]
        else:
            self.predictors[4+6*i] = random.choice("RPS")
            self.predictors[5+6*i] = random.choice("RPS")

    for i in range(3):
        temp =""
        search = self.temp1[(self.output+input)] #last round
        for start in range(2, min(self.limit[i],self.length) ):
            if search == self.both_his[self.length-start]:
                temp+=self.both_his[self.length-start+1]
        if(temp==""):
            self.predictors[6+i] = random.choice("RPS")
        else:
            collectR = {"P":0,"R":0,"S":0} #take win/lose from opponent into account
            for sdf in temp:
                next_move = self.temp2[sdf]
                if(self.who_win[next_move]==-1):
                    collectR[self.temp2[sdf][1]]+=3
                elif(self.who_win[next_move]==0):
                    collectR[self.temp2[sdf][1]]+=1
                elif(self.who_win[next_move]==1):
                    collectR[self.beat[self.temp2[sdf][0]]]+=1
            max1 = -1
            p1 =""
            for key in collectR:
                if(collectR[key]>max1):
                    max1 = collectR[key]
                    p1 += key
            self.predictors[6+i] = random.choice(p1)
    
    #rotate 9-27:
    for i in range(9,27):
        self.predictors[i] = self.beat[self.beat[self.predictors[i-9]]]
        
    #choose a predictor
    len_his = len(self.list_predictor[0])
    for i in range(self.num_predictor):
        sum = 0
        for j in range(len_his):
            if self.list_predictor[i][j]=="1":
                sum+=(j+1)*(j+1)
            else:
                sum-=(j+1)*(j+1)
        self.score_predictor[i] = sum
    max_score = max(self.score_predictor)
    if max_score>0:
        predict = self.predictors[self.score_predictor.index(max_score)]
    else:
        predict = random.choice(self.your_his)
    self.output = random.choice(self.not_lose[predict])
    return self.output

# Global variables for random_forest method.
actions =  np.empty((0,0), dtype = int)
observations =  np.empty((0,0), dtype = int)
total_reward = 0

# Global variables for beta distribution method.
agents = {    
    'rps_0': rps(0),
    'rps_1': rps(1),
    'rps_2': rps(2),
}
history = []
bandit_state = {k:[1,1] for k in agents.keys()}

# Global variables for transition matrix.
T = np.zeros((3, 3))
P = np.zeros((3, 3))
a1, a2 = None, None

# Global variables for dllu1.
gg = {}

# Global variables for greenberg.
opponent_hist, my_hist = [], []
act = None

# Global variables for IOU2.
global GLOBAL_STRATEGY
GLOBAL_STRATEGY = Strategy()

# Global variables for memory pattern.
STEPS_MAX = 5
STEPS_MIN = 3
EFFICIENCY_THRESHOLD = -3
FORCED_RANDOM_ACTION_INTERVAL = random.randint(STEPS_MIN, STEPS_MAX)
current_memory = []
previous_action = {
    "action": None,
    # action was taken from pattern
    "action_from_pattern": False,
    "pattern_group_index": None,
    "pattern_index": None
}
steps_to_random = FORCED_RANDOM_ACTION_INTERVAL
current_memory_max_length = STEPS_MAX * 2
reward = 0
group_memory_length = current_memory_max_length
groups_of_memory_patterns = []
for i in range(STEPS_MAX, STEPS_MIN - 1, -1):
    groups_of_memory_patterns.append({
        "memory_length": group_memory_length,
        "memory_patterns": []
    })
    group_memory_length -= 2

def random_forest_random(observation, configuration):
    global actions, observations, total_reward
    
    if observation.step == 0:
        action = random.randint(0,2)
        actions = np.append(actions , [action])
        return action
    
    if observation.step == 1:
        action = random.randint(0,2)
        actions = np.append(actions , [action])
        observations = np.append(observations , [observation.lastOpponentAction])
        # Keep track of score
        winner = int((3 + actions[-1] - observation.lastOpponentAction) % 3)
        if winner == 1:
            total_reward = total_reward + 1
        elif winner == 2:
            total_reward = total_reward - 1        
        return action

    # Get Observation to make the tables (actions & obervations) even.
    observations = np.append(observations , [observation.lastOpponentAction])
    
    # Prepare Data for training
    # :-1 as we dont have feedback yet.
    X_train = np.vstack((actions[:-1], observations[:-1])).T
    
    # Create Y by rolling observations to bring future a step earlier 
    shifted_observations = np.roll(observations, -1)
    
    # trim rolled & last element from rolled observations
    y_train = shifted_observations[:-1].T
    
    # Set the history period. Long chains here will need a lot of time
    if len(X_train) > 25:
        random_window_size = 10 + random.randint(0,10)
        X_train = X_train[-random_window_size:]
        y_train = y_train[-random_window_size:]
   
    # Train a classifier model
    model = RandomForestClassifier(n_estimators=25)
    model.fit(X_train, y_train)

    # Predict
    X_test = np.empty((0,0), dtype = int)
    X_test = np.append(X_test, [int(actions[-1]), observation.lastOpponentAction])
    prediction = model.predict(X_test.reshape(1, -1))

    # Keep track of score
    winner = int((3 + actions[-1] - observation.lastOpponentAction) % 3)
    if winner == 1:
        total_reward = total_reward + 1
    elif winner == 2:
        total_reward = total_reward - 1
   
    # Prepare action
    action = int((prediction + 1) % 3)
    
    # If losing a bit then change strategy and break the patterns by playing a bit random
    if total_reward < -2:
        win_tie = random.randint(0,1)
        action = int((prediction + win_tie) % 3)

    # Update actions
    actions = np.append(actions , [action])

    # Action 
    return action 


def multi_armed_bandit_agent (observation, configuration):
    
    step_size = 3 
    decay_rate = 1.1
    
    global history, bandit_state
    
    def log_step(step = None, history = None, agent = None, competitorStep = None, file = 'history.csv'):
        if step is None:
            step = np.random.randint(3)
        if history is None:
            history = []
        history.append({'step': step, 'competitorStep': competitorStep, 'agent': agent})
        if file is not None:
            pd.DataFrame(history).to_csv(file, index = False)
        return step
    
    def update_competitor_step(history, competitorStep):
        history[-1]['competitorStep'] = int(competitorStep)
        return history
    
    if observation.step == 0:
        pass
    else:
        history = update_competitor_step(history, observation.lastOpponentAction)
        
        for name, agent in agents.items():
            agent_step = agent.step(history[:-1])
            bandit_state[name][1] = (bandit_state[name][1] - 1) / decay_rate + 1
            bandit_state[name][0] = (bandit_state[name][0] - 1) / decay_rate + 1
            
            if (history[-1]['competitorStep'] - agent_step) % 3 == 1:
                bandit_state[name][1] += step_size
            elif (history[-1]['competitorStep'] - agent_step) % 3 == 2:
                bandit_state[name][0] += step_size
            else:
                bandit_state[name][0] += step_size/2
                bandit_state[name][1] += step_size/2
            
    with open('bandit.json', 'w') as outfile:
        json.dump(bandit_state, outfile)
    
    
    # generate random number from Beta distribution for each agent and select the most lucky one
    best_proba = -1
    best_agent = None
    for k in bandit_state.keys():
        
        proba = np.random.beta(bandit_state[k][0],bandit_state[k][1])
        #proba = bandit_state[k][0]/(bandit_state[k][0]+bandit_state[k][1])        
        
        if proba > best_proba:
            best_proba = proba
            best_agent = k
        
    step = agents[best_agent].step(history)
    
    return log_step(step, history, best_agent)


def transition_agent(observation, configuration):
    global T, P, a1, a2
    if observation.step > 1:
        a1 = observation.lastOpponentAction
        T[a2, a1] += 1
        P = np.divide(T, np.maximum(1, T.sum(axis=1)).reshape(-1, 1))
        a2 = a1
        if np.sum(P[a1, :]) == 1:
            return int((np.random.choice(
                [0, 1, 2],
                p=P[a1, :]
            ) + 1) % 3)
        else:
            return int(np.random.randint(3))
    else:
        if observation.step == 1:
            a2 = observation.lastOpponentAction
        return int(np.random.randint(3))

code = compile(
    """
numPre = 30
numMeta = 6
if not input:
    limit = 8
    beat={'R':'P','P':'S','S':'R'}
    moves=['','','','']
    pScore=[[5]*numPre,[5]*numPre,[5]*numPre,[5]*numPre,[5]*numPre,[5]*numPre]
    centrifuge={'RP':0,'PS':1,'SR':2,'PR':3,'SP':4,'RS':5,'RR':6,'PP':7,'SS':8}
    centripete={'R':0,'P':1,'S':2}
    soma = [0,0,0,0,0,0,0,0,0];
    rps = [1,1,1];
    a="RPS"
    best = [0,0,0];
    length=0
    p=[random.choice("RPS")]*numPre
    m=[random.choice("RPS")]*numMeta
    mScore=[5,2,5,2,4,2]
else:
    for i in range(numPre):
        pp = p[i]
        bpp = beat[pp]
        bbpp = beat[bpp]
        pScore[0][i]=0.9*pScore[0][i]+((input==pp)-(input==bbpp))*3
        pScore[1][i]=0.9*pScore[1][i]+((output==pp)-(output==bbpp))*3
        pScore[2][i]=0.87*pScore[2][i]+(input==pp)*3.3-(input==bpp)*1.2-(input==bbpp)*2.3
        pScore[3][i]=0.87*pScore[3][i]+(output==pp)*3.3-(output==bpp)*1.2-(output==bbpp)*2.3
        pScore[4][i]=(pScore[4][i]+(input==pp)*3)*(1-(input==bbpp))
        pScore[5][i]=(pScore[5][i]+(output==pp)*3)*(1-(output==bbpp))
    for i in range(numMeta):
        mScore[i]=0.96*(mScore[i]+(input==m[i])-(input==beat[beat[m[i]]]))
    soma[centrifuge[input+output]] +=1;
    rps[centripete[input]] +=1;
    moves[0]+=str(centrifuge[input+output])
    moves[1]+=input
    moves[2]+=output
    length+=1
    for y in range(3):
        j=min([length,limit])
        while j>=1 and not moves[y][length-j:length] in moves[y][0:length-1]:
            j-=1
        i = moves[y].rfind(moves[y][length-j:length],0,length-1)
        p[0+2*y] = moves[1][j+i] 
        p[1+2*y] = beat[moves[2][j+i]]
    j=min([length,limit])
    while j>=2 and not moves[0][length-j:length-1] in moves[0][0:length-2]:
        j-=1
    i = moves[0].rfind(moves[0][length-j:length-1],0,length-2)
    if j+i>=length:
        p[6] = p[7] = random.choice("RPS")
    else:
        p[6] = moves[1][j+i] 
        p[7] = beat[moves[2][j+i]]
        
    best[0] = soma[centrifuge[output+'R']]*rps[0]/rps[centripete[output]]
    best[1] = soma[centrifuge[output+'P']]*rps[1]/rps[centripete[output]]
    best[2] = soma[centrifuge[output+'S']]*rps[2]/rps[centripete[output]]
    p[8] = p[9] = a[best.index(max(best))]
    
    for i in range(10,numPre):
        p[i]=beat[beat[p[i-10]]]
        
    for i in range(0,numMeta,2):
        m[i]=       p[pScore[i  ].index(max(pScore[i  ]))]
        m[i+1]=beat[p[pScore[i+1].index(max(pScore[i+1]))]]
output = beat[m[mScore.index(max(mScore))]]
if max(mScore)<0.07 or random.randint(3,40)>length:
    output=beat[random.choice("RPS")]
""", '<string>', 'exec')


def run(observation, configuration):
    global gg
    global code
    inp = ''
    try:
        inp = 'RPS'[observation.lastOpponentAction]
    except:
        pass
    gg['input'] = inp
    exec(code, gg)
    return {'R': 0, 'P': 1, 'S': 2}[gg['output']]


# GreenBerg.
def player(my_moves, opp_moves):
    rps_to_text = ('rock','paper','scissors')
    rps_to_num  = {'rock':0, 'paper':1, 'scissors':2}
    wins_with = (1,2,0)      #superior
    best_without = (2,0,1)   #inferior

    lengths = (10, 20, 30, 40, 49, 0)
    p_random = random.choice([0,1,2])  #called 'guess' in iocaine

    TRIALS = 1000
    score_table =((0,-1,1),(1,0,-1),(-1,1,0))
    T = len(opp_moves)  #so T is number of trials completed

    def min_index(values):
        return min(enumerate(values), key=itemgetter(1))[0]

    def max_index(values):
        return max(enumerate(values), key=itemgetter(1))[0]

    def find_best_prediction(l):  # l = len
        bs = -TRIALS
        bp = 0
        if player.p_random_score > bs:
            bs = player.p_random_score
            bp = p_random
        for i in range(3):
            for j in range(24):
                for k in range(4):
                    new_bs = player.p_full_score[T%50][j][k][i] - (player.p_full_score[(50+T-l)%50][j][k][i] if l else 0)
                    if new_bs > bs:
                        bs = new_bs
                        bp = (player.p_full[j][k] + i) % 3
                for k in range(2):
                    new_bs = player.r_full_score[T%50][j][k][i] - (player.r_full_score[(50+T-l)%50][j][k][i] if l else 0)
                    if new_bs > bs:
                        bs = new_bs
                        bp = (player.r_full[j][k] + i) % 3
            for j in range(2):
                for k in range(2):
                    new_bs = player.p_freq_score[T%50][j][k][i] - (player.p_freq_score[(50+T-l)%50][j][k][i] if l else 0)
                    if new_bs > bs:
                        bs = new_bs
                        bp = (player.p_freq[j][k] + i) % 3
                    new_bs = player.r_freq_score[T%50][j][k][i] - (player.r_freq_score[(50+T-l)%50][j][k][i] if l else 0)
                    if new_bs > bs:
                        bs = new_bs
                        bp = (player.r_freq[j][k] + i) % 3
        return bp


    if not my_moves:
        player.opp_history = [0]  #pad to match up with 1-based move indexing in original
        player.my_history = [0]
        player.gear = [[0] for _ in range(24)]
        # init()
        player.p_random_score = 0
        player.p_full_score = [[[[0 for i in range(3)] for k in range(4)] for j in range(24)] for l in range(50)]
        player.r_full_score = [[[[0 for i in range(3)] for k in range(2)] for j in range(24)] for l in range(50)]
        player.p_freq_score = [[[[0 for i in range(3)] for k in range(2)] for j in range(2)] for l in range(50)]
        player.r_freq_score = [[[[0 for i in range(3)] for k in range(2)] for j in range(2)] for l in range(50)]
        player.s_len = [0] * 6

        player.p_full = [[0,0,0,0] for _ in range(24)]
        player.r_full = [[0,0] for _ in range(24)]
    else:
        player.my_history.append(rps_to_num[my_moves[-1]])
        player.opp_history.append(rps_to_num[opp_moves[-1]])
        # update_scores()
        player.p_random_score += score_table[p_random][player.opp_history[-1]]
        player.p_full_score[T%50] = [[[player.p_full_score[(T+49)%50][j][k][i] + score_table[(player.p_full[j][k] + i) % 3][player.opp_history[-1]] for i in range(3)] for k in range(4)] for j in range(24)]
        player.r_full_score[T%50] = [[[player.r_full_score[(T+49)%50][j][k][i] + score_table[(player.r_full[j][k] + i) % 3][player.opp_history[-1]] for i in range(3)] for k in range(2)] for j in range(24)]
        player.p_freq_score[T%50] = [[[player.p_freq_score[(T+49)%50][j][k][i] + score_table[(player.p_freq[j][k] + i) % 3][player.opp_history[-1]] for i in range(3)] for k in range(2)] for j in range(2)]
        player.r_freq_score[T%50] = [[[player.r_freq_score[(T+49)%50][j][k][i] + score_table[(player.r_freq[j][k] + i) % 3][player.opp_history[-1]] for i in range(3)] for k in range(2)] for j in range(2)]
        player.s_len = [s + score_table[p][player.opp_history[-1]] for s,p in zip(player.s_len,player.p_len)]


    # update_history_hash()
    if not my_moves:
        player.my_history_hash = [[0],[0],[0],[0]]
        player.opp_history_hash = [[0],[0],[0],[0]]
    else:
        player.my_history_hash[0].append(player.my_history[-1])
        player.opp_history_hash[0].append(player.opp_history[-1])
        for i in range(1,4):
            player.my_history_hash[i].append(player.my_history_hash[i-1][-1] * 3 + player.my_history[-1])
            player.opp_history_hash[i].append(player.opp_history_hash[i-1][-1] * 3 + player.opp_history[-1])


    #make_predictions()

    for i in range(24):
        player.gear[i].append((3 + player.opp_history[-1] - player.p_full[i][2]) % 3)
        if T > 1:
            player.gear[i][T] += 3 * player.gear[i][T-1]
        player.gear[i][T] %= 9 # clearly there are 9 different gears, but original code only allocated 3 gear_freq's
                               # code apparently worked, but got lucky with undefined behavior
                               # I fixed by allocating gear_freq with length = 9
    if not my_moves:
        player.freq = [[0,0,0],[0,0,0]]
        value = [[0,0,0],[0,0,0]]
    else:
        player.freq[0][player.my_history[-1]] += 1
        player.freq[1][player.opp_history[-1]] += 1
        value = [[(1000 * (player.freq[i][2] - player.freq[i][1])) / float(T),
                  (1000 * (player.freq[i][0] - player.freq[i][2])) / float(T),
                  (1000 * (player.freq[i][1] - player.freq[i][0])) / float(T)] for i in range(2)]
    player.p_freq = [[wins_with[max_index(player.freq[i])], wins_with[max_index(value[i])]] for i in range(2)]
    player.r_freq = [[best_without[min_index(player.freq[i])], best_without[min_index(value[i])]] for i in range(2)]

    f = [[[[0,0,0] for k in range(4)] for j in range(2)] for i in range(3)]
    t = [[[0,0,0,0] for j in range(2)] for i in range(3)]

    m_len = [[0 for _ in range(T)] for i in range(3)]

    for i in range(T-1,0,-1):
        m_len[0][i] = 4
        for j in range(4):
            if player.my_history_hash[j][i] != player.my_history_hash[j][T]:
                m_len[0][i] = j
                break
        for j in range(4):
            if player.opp_history_hash[j][i] != player.opp_history_hash[j][T]:
                m_len[1][i] = j
                break
        for j in range(4):
            if player.my_history_hash[j][i] != player.my_history_hash[j][T] or player.opp_history_hash[j][i] != player.opp_history_hash[j][T]:
                m_len[2][i] = j
                break

    for i in range(T-1,0,-1):
        for j in range(3):
            for k in range(m_len[j][i]):
                f[j][0][k][player.my_history[i+1]] += 1
                f[j][1][k][player.opp_history[i+1]] += 1
                t[j][0][k] += 1
                t[j][1][k] += 1

                if t[j][0][k] == 1:
                    player.p_full[j*8 + 0*4 + k][0] = wins_with[player.my_history[i+1]]
                if t[j][1][k] == 1:
                    player.p_full[j*8 + 1*4 + k][0] = wins_with[player.opp_history[i+1]]
                if t[j][0][k] == 3:
                    player.p_full[j*8 + 0*4 + k][1] = wins_with[max_index(f[j][0][k])]
                    player.r_full[j*8 + 0*4 + k][0] = best_without[min_index(f[j][0][k])]
                if t[j][1][k] == 3:
                    player.p_full[j*8 + 1*4 + k][1] = wins_with[max_index(f[j][1][k])]
                    player.r_full[j*8 + 1*4 + k][0] = best_without[min_index(f[j][1][k])]

    for j in range(3):
        for k in range(4):
            player.p_full[j*8 + 0*4 + k][2] = wins_with[max_index(f[j][0][k])]
            player.r_full[j*8 + 0*4 + k][1] = best_without[min_index(f[j][0][k])]

            player.p_full[j*8 + 1*4 + k][2] = wins_with[max_index(f[j][1][k])]
            player.r_full[j*8 + 1*4 + k][1] = best_without[min_index(f[j][1][k])]

    for j in range(24):
        gear_freq = [0] * 9 # was [0,0,0] because original code incorrectly only allocated array length 3

        for i in range(T-1,0,-1):
            if player.gear[j][i] == player.gear[j][T]:
                gear_freq[player.gear[j][i+1]] += 1

        #original source allocated to 9 positions of gear_freq array, but only allocated first three
        #also, only looked at first 3 to find the max_index
        #unclear whether to seek max index over all 9 gear_freq's or just first 3 (as original code)
        player.p_full[j][3] = (player.p_full[j][1] + max_index(gear_freq)) % 3

    # end make_predictions()

    player.p_len = [find_best_prediction(l) for l in lengths]

    return rps_to_num[rps_to_text[player.p_len[max_index(player.s_len)]]]


def greenberg_agent(observation, configuration):
    global opponent_hist, my_hist, act
    
    rps_to_text = ('rock','paper','scissors')
    if observation.step > 0:
        my_hist.append(rps_to_text[act])
        opponent_hist.append(rps_to_text[observation.lastOpponentAction])
        
    act = player(my_hist, opponent_hist)
    return act


# IOU2.
def agent_IOU(observation, configuration):
  global GLOBAL_STRATEGY

  # Action mapping
  to_char = ["R", "P", "S"]
  from_char = {"R": 0, "P": 1, "S": 2}

  if observation.step > 0:
    GLOBAL_STRATEGY.prepare_next_move(to_char[observation.lastOpponentAction])
  action = from_char[GLOBAL_STRATEGY.output]
  return action


# Memory pattern.
def evaluate_pattern_efficiency(previous_step_result):
    """ 
        evaluate efficiency of the pattern and, if pattern is inefficient,
        remove it from agent's memory
    """
    pattern_group_index = previous_action["pattern_group_index"]
    pattern_index = previous_action["pattern_index"]
    pattern = groups_of_memory_patterns[pattern_group_index]["memory_patterns"][pattern_index]
    pattern["reward"] += previous_step_result
    # if pattern is inefficient
    if pattern["reward"] <= EFFICIENCY_THRESHOLD:
        # remove pattern from agent's memory
        del groups_of_memory_patterns[pattern_group_index]["memory_patterns"][pattern_index]
    
def find_action(group, group_index):
    """ if possible, find my_action in this group of memory patterns """
    if len(current_memory) > group["memory_length"]:
        this_step_memory = current_memory[-group["memory_length"]:]
        memory_pattern, pattern_index = find_pattern(group["memory_patterns"], this_step_memory, group["memory_length"])
        if memory_pattern != None:
            my_action_amount = 0
            for action in memory_pattern["opp_next_actions"]:
                # if this opponent's action occurred more times than currently chosen action
                # or, if it occured the same amount of times and this one is choosen randomly among them
                if (action["amount"] > my_action_amount or
                        (action["amount"] == my_action_amount and random.random() > 0.5)):
                    my_action_amount = action["amount"]
                    my_action = action["response"]
            return my_action, pattern_index
    return None, None

def find_pattern(memory_patterns, memory, memory_length):
    """ find appropriate pattern and its index in memory """
    for i in range(len(memory_patterns)):
        actions_matched = 0
        for j in range(memory_length):
            if memory_patterns[i]["actions"][j] == memory[j]:
                actions_matched += 1
            else:
                break
        # if memory fits this pattern
        if actions_matched == memory_length:
            return memory_patterns[i], i
    # appropriate pattern not found
    return None, None

def get_step_result_for_my_agent(my_agent_action, opp_action):
    """ 
        get result of the step for my_agent
        1, 0 and -1 representing win, tie and lost results of the game respectively
        reward will be taken from observation in the next release of kaggle environments
    """
    if my_agent_action == opp_action:
        return 0
    elif (my_agent_action == (opp_action + 1)) or (my_agent_action == 0 and opp_action == 2):
        return 1
    else:
        return -1
    
def update_current_memory(obs, my_action):
    """ add my_agent's current step to current_memory """
    # if there's too many actions in the current_memory
    if len(current_memory) > current_memory_max_length:
        # delete first two elements in current memory
        # (actions of the oldest step in current memory)
        del current_memory[:2]
    # add agent's last action to agent's current memory
    current_memory.append(my_action)
    
def update_memory_pattern(obs, group):
    """ if possible, update or add some memory pattern in this group """
    # if length of current memory is suitable for this group of memory patterns
    if len(current_memory) > group["memory_length"]:
        # get memory of the previous step
        # considering that last step actions of both agents are already present in current_memory
        previous_step_memory = current_memory[-group["memory_length"] - 2 : -2]
        previous_pattern, pattern_index = find_pattern(group["memory_patterns"], previous_step_memory, group["memory_length"])
        if previous_pattern == None:
            previous_pattern = {
                # list of actions of both players
                "actions": previous_step_memory.copy(),
                # total reward earned by using this pattern
                "reward": 0,
                # list of observed opponent's actions after each occurrence of this pattern
                "opp_next_actions": [
                    # action that was made by opponent,
                    # amount of times that action occurred,
                    # what should be the response of my_agent
                    {"action": 0, "amount": 0, "response": 1},
                    {"action": 1, "amount": 0, "response": 2},
                    {"action": 2, "amount": 0, "response": 0}
                ]
            }
            group["memory_patterns"].append(previous_pattern)
        # update previous_pattern
        for action in previous_pattern["opp_next_actions"]:
            if action["action"] == obs["lastOpponentAction"]:
                action["amount"] += 1


def my_agent(obs, conf):
    """ your ad here """
    # action of my_agent
    my_action = None
    
    # forced random action
    global steps_to_random
    steps_to_random -= 1
    if steps_to_random <= 0:
        steps_to_random = FORCED_RANDOM_ACTION_INTERVAL
        # choose action randomly
        my_action = random.randint(0, 2)
        # save action's data
        previous_action["action"] = my_action
        previous_action["action_from_pattern"] = False
        previous_action["pattern_group_index"] = None
        previous_action["pattern_index"] = None
    
    # if it's not first step
    if obs["step"] > 0:
        # add opponent's last step to current_memory
        current_memory.append(obs["lastOpponentAction"])
        # previous step won or lost
        previous_step_result = get_step_result_for_my_agent(current_memory[-2], current_memory[-1])
        global reward
        reward += previous_step_result
        # if previous action of my_agent was taken from pattern
        if previous_action["action_from_pattern"]:
            evaluate_pattern_efficiency(previous_step_result)
    
    for i in range(len(groups_of_memory_patterns)):
        # if possible, update or add some memory pattern in this group
        update_memory_pattern(obs, groups_of_memory_patterns[i])
        # if action was not yet found
        if my_action == None:
            my_action, pattern_index = find_action(groups_of_memory_patterns[i], i)
            if my_action != None:
                # save action's data
                previous_action["action"] = my_action
                previous_action["action_from_pattern"] = True
                previous_action["pattern_group_index"] = i
                previous_action["pattern_index"] = pattern_index
    
    # if no action was found
    if my_action == None:
        # choose action randomly
        my_action = random.randint(0, 2)
        # save action's data
        previous_action["action"] = my_action
        previous_action["action_from_pattern"] = False
        previous_action["pattern_group_index"] = None
        previous_action["pattern_index"] = None
    
    # add my_agent's current step to current_memory
    update_current_memory(obs, my_action)
    return my_action

#########################################################################################################
#########################################################################################################
#########################################################################################################

# Use Proportional Representation to find most favorable move.
def proportionalRepresentation(array_of_moves, array_of_won_rounds, step):
    array_of_win_rates = array_of_won_rounds / step
    move_dict = {0: np.array([]), 1: np.array([]), 2:np.array([])}
    for index, element in enumerate(array_of_moves):
        move_dict[element] = np.append(move_dict[element], array_of_win_rates[index])
    for move in move_dict:
        if len(move_dict[move]) == 0:
            move_dict[move] = 0
        else:
            move_dict[move] = sum(move_dict[move]) / len(move_dict[move])
    best_move = 0
    best_rating = -1
    for move in move_dict:
        if move_dict[move] > best_rating:
            best_move = move
            best_rating = move_dict[move]
    return best_move

def updateWinRate(array_of_moves, opponent_move, array_won_rounds):
    current_round = np.array([0] * len(array_of_moves))
    for index, move in enumerate(array_of_moves):
        if move == (opponent_move + 1) % 3:
            current_round[index] = 1
    return array_won_rounds + current_round

main_step = 0
main_won_rounds = np.array([0, 0, 0, 0, 0, 0, 0])
main_method_steps = np.array([0, 0, 0, 0, 0, 0, 0])


def main(observation, configuration):
    global main_step
    global main_won_rounds
    global main_method_steps
    # If not first step, update # of won round first.
    if main_step > 0:
        main_won_rounds = updateWinRate(main_method_steps, 
                                       int(observation.lastOpponentAction), 
                                       main_won_rounds)

    # Compute moves generated by each method.
    step_random_forest = random_forest_random(observation, configuration)
    step_transition = transition_agent(observation, configuration)
    step_multi_armed = multi_armed_bandit_agent (observation, configuration)
    step_dllu1 = run(observation, configuration)
    step_greenberg = greenberg_agent(observation, configuration)
    step_IOU = agent_IOU(observation, configuration)
    step_memory = my_agent(observation, configuration)
    # Have all the moves in a np array.
    main_method_steps = np.array([step_random_forest, step_transition, step_multi_armed, step_dllu1, 
                                  step_greenberg, step_IOU, step_memory])

    # If first step, return random number.
    if main_step == 0:
        main_step += 1
        return np.random.randint(3)
    # Otherwise, find current optimized step.
    else:
        selected_step = proportionalRepresentation(main_method_steps, main_won_rounds, main_step)
        main_step += 1
        return selected_step