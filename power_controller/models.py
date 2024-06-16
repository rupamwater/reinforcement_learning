
import numpy as np
from math import floor , isnan

import gym
from gym.spaces import *

from functools import *


class Home:

    def __init__(self, device_consumption):
        self.time_period_unit = 1440 # 24 * 60
        self.dev_cons = device_consumption
        self.dev_status =  [ (0, 0, 0) for i in range(len(device_consumption))]


    def power_demand(self, T):
        random = np.random.random(len(self.dev_cons))
        status = self.dev_status 
        consumption = self.dev_cons 

        #If device switched on switch it off is time past the end time
        switch_off  = [  (status[i][0] and status[i][2] > T) and (0,0,0) or status[i]   for i in range(len(self.dev_cons))]
        
        # If device switched off then swicth on if random number less than probability of the device(to simulate percentage of time it is switched on)
        switch_on   = [ (not status[i][0] and random[i] < consumption[i][2] ) and (1, T,T + consumption[i][1]) or status[i]   for i in range(len(self.dev_cons))]
        
        switch      = list(zip(switch_off, switch_on))

        self.dev_status   = [ ( switch[i][0][0] > switch[i][1][0] ) and switch[i][0]  or switch[i][1]      for i in range(len(switch))]

        power_drawn = [ (self.dev_status[i][0] and  self.dev_cons[i][1] or 0.0 )  for i in range(len(self.dev_status))]

        return reduce(lambda a, b: a + b , power_drawn)

    def max_demand(self):
        device_power = [ d[0] for d in self.dev_cons]
        return reduce(lambda a , b: a + b, device_power )
 
 
class Panel:

    def __init__(self, generator ):
        self.periods_in_day = 24 * 60
        m = [1.0] * 60
        self.gen_capacity_minutes = [ m[i]* s for s in generator for i in range(len(m))]
        self.max_panel_supply = reduce(lambda a, b: (a > b) and a or b , self.gen_capacity_minutes)
        self.power_unit_multiplier = 1000

    def power_supply(self, T):  
        t = T % self.periods_in_day  
        return self.gen_capacity_minutes[t] * self.power_unit_multiplier


    def max_supply(self):
        return self.max_panel_supply * self.power_unit_multiplier


class Battery:

    def __init__(self, max_charge , charge=0.0):
        self.periods_in_day = 24 * 60
        self.charge = charge
        self.maxCapacity = max_charge
        self.minCharge = 0.2 * max_charge

    def recharge(self, charge_power , charge_period=1):
        if self.charge + charge_power * charge_period/ self.periods_in_day < self.maxCapacity:   
            self.charge += charge_power * charge_period/ self.periods_in_day

    def discharge(self, charge_power, discharge_period=1):
        if self.charge - charge_power * discharge_period/self.periods_in_day > 0:   
            self.charge -= charge_power * discharge_period/self.periods_in_day

    def current_charge(self):
        return self.charge

    def max_charge(self):
        return self.maxCapacity

class SmartWatt(gym.Env):

    def __init__(self, cons_param , gen_param , sim_period, max_charge):
        super().__init__()

        self.periods_in_day = 24 * 60
        self.simulation_period = sim_period

        rng     = np.random
        
        self.consumption_param = cons_param
        self.generation_param = gen_param
 
        self.consumer = Home(self.consumption_param)
        self.storage = Battery(max_charge )
        self.supply = Panel(self.generation_param)

        self.max_demand = self.consumer.max_demand()
        self.max_charge = self.storage.max_charge()
        self.max_panel_supply = self.supply.max_supply()
        self.max_grid_supply = 8000
        self.max_battery_supply = 2000
        self.max_charge_consumption = 2000

        charge  = floor(rng.uniform() * self.max_charge)

        self.storage = Battery(max_charge , charge )

        self.consumer = Home(self.consumption_param)
        self.storage = Battery(max_charge , charge)
        self.supply = Panel(self.generation_param)


        #Rewards
        self.panel_supply_reward = 1.0
        self.battery_supply_reward_matrix = np.array([[0.0 , 0.3 , -1.25],[0.3 , 0.5, 0.0],[0.5 , 1.0, 0.5]])
        self.grid_supply_reward = -1.0 

        self.obs_dim =  4
        low_obs = np.zeros((self.obs_dim,))
        high_obs = np.array([self.max_demand , self.max_panel_supply, self.max_charge , self.periods_in_day  ])
        
        self.action_space = Discrete(5)
        self.observation_space = Box(low=low_obs , high=high_obs )

        self.actions = dict([
                             (0 , np.array([[1,0],[1,0],[0,0]]) ),
                             (1 , np.array([[1,0],[1,0],[1,0]]) ),
                             (2 , np.array([[0,0],[1,1],[1,0]]) ),
                             (3 , np.array([[1,0],[1,0],[0,0]]) ),
                             (4 , np.array([[0,0],[1,0],[1,1]]) )
                           ])


        self.current_obs =  None
        self.terminal_period = None

        

    def reset(self):
        """
           Samples parameter values from parameter space 
        """
        rng     = np.random

        charge  = floor(rng.uniform() * self.max_charge)
        demand  = floor(rng.uniform() * self.max_demand) 
 
        period  = floor(rng.uniform() * self.periods_in_day) 

        panel_supply = self.supply.power_supply(period)

        debug(("Reset : Max Charge ", self.max_charge , ",  Max Demand", self.max_demand  , ", Max Panel Supply ", self.max_panel_supply ))
        self.battery = Battery( self.max_charge , charge)


        self.current_obs = np.array([ demand, panel_supply , charge , period  ])
        self.terminal_period = period + self.simulation_period
        self.current_period = period

        return self.current_obs        


    def step(self, action):
        """ 
            Returns : Given the current obs and action, returns the
            next observation, the reward terminal state and  optionally additional info 
        """

        curr_action_mtx     = self.actions[action]


        x_dim = curr_action_mtx.shape[0]
        y_dim = curr_action_mtx.shape[1]
        
        panel_charge_switch =  curr_action_mtx[2][1]
        grid_charge_switch  = curr_action_mtx[1][1]

        battery_supply_switch = curr_action_mtx[0][0]
        grid_supply_switch    = curr_action_mtx[1][0]
        panel_supply_switch   = curr_action_mtx[2][0]


        curr_demand = self.consumer.power_demand(self.current_period + 1)
        curr_charge = self.storage.current_charge()
        curr_panel_supply = self.supply.power_supply(self.current_period + 1) 

        debug(("Action ", action, ", Demand ", curr_demand, ", Charge ", curr_charge , ", Panel Supply ", curr_panel_supply ))
        debug(("Battery Supply Switch ", battery_supply_switch, ", Grid Supply Switch ", grid_supply_switch, ", Panel Supply Switch ", panel_supply_switch ))
        debug(("Grid Charge Switch ", grid_charge_switch, ", Panel Charge Switch ", panel_charge_switch ))
        
        panel_consumer_supply  = panel_supply_switch * min(curr_panel_supply, curr_demand)

        excess_panel_supply    = max( (curr_panel_supply - panel_consumer_supply) if panel_supply_switch else curr_panel_supply , 0 )

        panel_charge_supply    = panel_charge_switch * excess_panel_supply
        
        battery_supply         = battery_supply_switch * min(curr_demand - panel_consumer_supply, self.max_battery_supply)

        panel_supply           = panel_charge_supply + panel_consumer_supply

        grid_consumer_supply   = grid_supply_switch * (curr_demand - panel_supply - battery_supply) 

        excess_grid_supply     = self.max_grid_supply - grid_consumer_supply

        grid_charge_supply     = grid_charge_switch * min( excess_grid_supply , self.max_charge_consumption)

        charge_indicator       = grid_charge_switch or panel_charge_switch

        discharge_indicator    = battery_supply_switch
        
        battery_charge_supply  = grid_charge_supply + panel_charge_supply

        debug((" Panel Supply Switch ", panel_supply_switch ,  "Panel Consumer Supply : ", panel_consumer_supply,", Panel Charge Switch ", panel_charge_switch ,  " , Panel Charge Supply ", panel_charge_supply))
        debug( ("Excess Panel Supply ", excess_panel_supply, ", Panel SUpply Switch  ", panel_supply_switch , ", Current Panel Supply " ,  curr_panel_supply , ", Panel Consumer Supply " , panel_consumer_supply))
        
        debug((" Grid Supply Switch ", grid_supply_switch , "Grid Consumer Supply : ", grid_consumer_supply, ", Grid Charge Switch ", grid_charge_switch,  " , Grid Charge Supply ", grid_charge_supply))
        debug((" Charge Indicator ", charge_indicator, ", Battery Chareg Supply " , battery_charge_supply))
        debug((" Battery Supply Switch ", battery_supply_switch , ", Battery Supply ",  battery_supply ))

        if charge_indicator:
            battery_charge_delta = battery_charge_supply / self.periods_in_day
            charge_power         = curr_charge + battery_charge_delta

            self.storage.recharge( charge_power  )

        elif discharge_indicator:
            battery_supply_delta = battery_supply / self.periods_in_day
            discharge_power      = curr_charge - battery_supply_delta
            self.storage.discharge(discharge_power   )
        
        battery_supply_reward = reduce(lambda  x, y : x + y ,  [ r[2] if ((curr_charge <= (r[1] * self.max_charge) ) and (curr_charge > (r[0] * self.max_charge))) else 0.0    for r in self.battery_supply_reward_matrix ] ) #0.5 # use the logic based on curr_charge

        debug((" battery_supply_reward ", battery_supply_reward))
        #battery_supply_reward =  reduce(lambda  x, y : x + y ,  [  r[2] if (curr_charge <= (r[1] * self.max_charge) and curr_charge > (r[0] * self.max_charge)) else 0.0   for r in self.battery_supply_reward_matrix ])  #0.5 # use the logic based on curr_charge

        panel_supply = panel_charge_supply + panel_consumer_supply

        grid_supply = grid_consumer_supply + grid_charge_supply

        debug(("Battery Supply Reward", battery_supply_reward , " Panel Supply reward" , self.panel_supply_reward , " Grid Supply Reward ", self.grid_supply_reward ))

        reward = 100 * ((battery_supply_reward  * battery_supply)/curr_demand + (self.panel_supply_reward * panel_supply) / curr_demand + (self.grid_supply_reward * grid_supply ) / curr_demand )

        debug(("Battery Supply ", battery_supply , " Panel Supply " , panel_supply , " Grid Supply ", grid_supply ))

        debug(("Reward " , reward))
        debug("    " )
        debug("==================================================" )
        debug("     " )
        debug("==================================================" )

        next_charge = self.storage.current_charge()
        next_period = (self.current_period + 1) % self.periods_in_day
        next_panel_supply = panel_charge_supply + panel_consumer_supply
        next_demand = curr_demand

        next_obs = np.array([ next_demand, next_panel_supply , next_charge , next_period  ])

        self.current_obs = next_obs
        self.current_period += 1


        done = self.current_period > self.terminal_period
        
        debug(("Terminal ", self.terminal_period , ", Current Period : ", self.current_period , ", Done ", done))


        return self.current_obs , reward , done , {}



def debug(message):
    log = False
    if(log):
        print(message)


if __name__ == '__main__':
    # device power conmsumption as the power, time [period of operation , frequency of opearation as power
    device_power = [(200, 60 , 0.3),(1500, 4, 0.02 ),(1400, 10, 0.1),(250, 1440, 1.0),(2000, 45, 0.02),(400, 60, 0.01), (200, 180, 0.01) , (50, 1440 , 1.00)]
    consumption = np.array(device_power)
    home = Home(consumption)

    # Generation capacity of Panels for each hour of the day, starting from 0100 hours to 2400 hours in the array
    generation_schedule=[0.0 ,0.0 ,0.0 ,0.0 ,0.0 , 0.3, 0.5 , 0.7, 1.0 , 1.3 , 2.0, 2.0, 2.0, 2.0, 1.7, 1.2 , 0.7, 0.2, 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0]
    
    generator = np.array(generation_schedule)
    panel = Panel(generator)
    T = floor(np.random.rand() * 1440)
    debug(("Time : ", T))
    p = panel.power_supply(T)
    debug(("Supply", p))

    periods = 1440 * 10

    max_charge = 5000

    smartEnv = SmartWatt(consumption , generator , periods  , max_charge  ) 


    debug(smartEnv.observation_space.high)
    win_os_size = [250] * len(smartEnv.observation_space.high)
    discrete_os_size = np.ceil((smartEnv.observation_space.high-smartEnv.observation_space.low)/win_os_size)
    discrete_size_multiplier = np.array([1,1,1,4])
    discrete_os_size_table = discrete_os_size * discrete_size_multiplier


    discrete_os_win_size = ((smartEnv.observation_space.high-smartEnv.observation_space.low)/discrete_os_size_table)
    discrete_os_table_size = ((smartEnv.observation_space.high + discrete_os_win_size -smartEnv.observation_space.low)/discrete_os_win_size).astype(np.int_)
    debug(np.concatenate([discrete_os_table_size , [smartEnv.action_space.n]], 0))

    LEARNING_RATE = 0.15
    DISCOUNT = 0.90
    EPISODES = 4000

    EPSILON = 0.3
    EPSILON_DECAY =  0.01 
    SHOW_EVERY = 50

    def get_discrete_state(state):
        discrete_state = ((state-smartEnv.observation_space.low) / discrete_os_win_size)
        return tuple(discrete_state.astype(np.int_))

    debug(("discrete_os_win_size",np.ceil(discrete_os_win_size)))

    q_table = np.random.uniform(low=0, high=100, size=(np.concatenate([discrete_os_table_size , [smartEnv.action_space.n]], 0)))

    debug((" Q Table shape : " , q_table.shape))
    ep_reward = []
    aggr_ep_rewards = {'ep': [], 'avg' : [] , 'min' : [] , 'max' : [] }
    supply_metrics = {'grid': [], 'panel':[] , 'battery' : [] }


    for i in range(EPISODES):
        done = False
        episode_reward = 0.0

        state_reset = smartEnv.reset()
        debug(("Reset" , state_reset))
        discrete_state = get_discrete_state(state_reset)
        debug(discrete_state )

        while not done  :
            debug(("Discrete ", discrete_state ))
            action = np.argmax(q_table[discrete_state])
            debug(("action" , action))
            
            new_state , reward, done, _ = smartEnv.step(action) 
            episode_reward +=  reward/100      
            debug((new_state, reward))

            new_discrete_state = get_discrete_state(new_state)
            #print(("new_discrete_state", new_discrete_state))
            debug("=============================================")
            debug("   ")
            debug("=============================================")

            if not done:
                max_future_q = np.max(q_table[new_discrete_state])
                current_q = q_table[new_discrete_state][action]
                new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
                #print(" LEARNING RATE ",LEARNING_RATE , " , DISCOUNT ", DISCOUNT)
                #print(" current_q ", current_q ,", reward ", reward , ", max_future_q ", max_future_q)
                #print("New Q value ",new_q, " = (1-LEARNING_RATE) * current_q ", (1-LEARNING_RATE) * current_q , " + LEARNING_RATE * (reward * DISCOUNT * max_future_q) ", LEARNING_RATE * (reward * DISCOUNT * max_future_q) )
                if isnan(new_q):
                    break
                q_table[new_discrete_state][action] = new_q
            discrete_state = new_discrete_state
        #print("Episode ", i , ", Episode Reward ", episode_reward )
        ep_reward.append(episode_reward)    

        if not i % SHOW_EVERY :
            average_reward = sum(ep_reward[-SHOW_EVERY:])/len(ep_reward[-SHOW_EVERY:])
            aggr_ep_rewards['ep'].append(i)
            aggr_ep_rewards['avg'].append(average_reward)
            aggr_ep_rewards['min'].append(min(ep_reward[-SHOW_EVERY:]))
            aggr_ep_rewards['max'].append(max(ep_reward[-SHOW_EVERY:]))
            print("Expisode ", i , " Average ", average_reward , " Minimum ", min(ep_reward[-SHOW_EVERY:]), " " , max(ep_reward[-SHOW_EVERY:]))



    smartEnv.close()






             

    


    