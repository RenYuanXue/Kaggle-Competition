
import pandas as pd
import numpy as np
import json
from random import randrange
    

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
    

agents = {    
    'rps_0': rps(0),
    'rps_1': rps(1),
    'rps_2': rps(2),
}

history = []
bandit_state = {k:[1,1] for k in agents.keys()}


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
