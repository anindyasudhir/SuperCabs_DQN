# Import routines
 
import numpy as np
import math
import random
 
# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger
 
# Lambda for Poisson Distribution
lambda_loc_0 = 2
lambda_loc_1 = 12
lambda_loc_2 = 4
lambda_loc_3 = 7
lambda_loc_4 = 8
 
class CabDriver():
 
    def __init__(self):
        """initialise your state and define your action space and state space"""
        # Action Space: (p,q) where p is start location and q is destination (and p != q)
        #                [0,0] is for 'no-ride' action
        self.action_space = [(i,j) for i in range(1, m + 1) for j in range(1, m + 1) if i != j] + [(0,0)]
       
        # The state space is defined by the driver’s current location along with the time components
        self.state_space = [(i,j,k) for i in range(1, m + 1) for j in range(t) for k in range(d)]
       
        # State Size for architecture-1
        self.state_size = m + t + d
        
        # State Size for architecture-2. This is not used as we chose to use architecure-1 after trying both
        self.state_size_arch2 = m + t + d + m + m
        
        # Action Size
        self.action_size = len(self.action_space)
       
        # Init with a random state
        self.state_init = random.sample(self.state_space, 1)[0]
       
        # Start the first round
        self.reset()
 
    # Encoding state for NN input as per Architecture 1
    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
       
        # Create one hot encoding arrays for location, hour and day
        location_onehot = np.zeros(m)
        hour_onehot = np.zeros(t)
        day_onehot = np.zeros(d)
       
        # Set the position in the array corresponding to location, hour and day mentioned the state to 1
        location_onehot[state[0] - 1] = 1
        hour_onehot[state[1]] = 1
        day_onehot[state[2]] = 1
 
        # Architecture 1 means, the state as one hot encoded array is the input to the NN
        state_encod = np.concatenate((location_onehot, hour_onehot, day_onehot), axis = None)
       
        # The first index of the input to the NN model is the batch size, so reshape state to (1, state_size)
        state_encod = state_encod.reshape(1, self.state_size)
 
        return state_encod
 
    
    # Use this function if you are using architecture-2
    # This was written as we experimented with both architecture-1 and architecture-2. We decided to go with
    # architecture-1. So below function never gets used
    def state_encod_arch2(self, state, action):
        """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""
       
        # Create one hot encoding arrays for location, hour and day
        location_onehot = np.zeros(m)
        hour_onehot = np.zeros(t)
        day_onehot = np.zeros(d)
        #action encoding
        pickup_loc_onehot = np.zeros(m)
        drop_loc_onehot = np.zeros(m)
       
        # Set the position in the array corresponding to location, hour and day mentioned the state to 1
        location_onehot[state[0] - 1] = 1
        hour_onehot[state[1]] = 1
        day_onehot[state[2]] = 1
        if (action[0] != 0) and (action[1] != 0):
            pickup_loc_onehot[action[0] - 1] = 1
            drop_loc_onehot[action[1] - 1] = 1
            
 
        # Architecture 2 means, both state and action are the input to the NN
        state_encod = np.concatenate((location_onehot, hour_onehot, day_onehot, pickup_loc_onehot, drop_loc_onehot), axis = None)
       
        # The first index of the input to the NN model is the batch size, so reshape state to (1, state_size)
        state_encod = state_encod.reshape(1, self.state_size_arch2) 

        return state_encod
 
 
    # Getting number of requests
    def requests(self, state):
        """Determining the number of requests basis the location.
        Use the table specified in the MDP and complete for rest of the locations"""
       
        location = state[0]
       
        requests = 0
       
        # Wait for a valid request to come
        while(requests == 0):
            # Get requests based on the location's Poisson Distribution mean
            if location == 1:
                requests = np.random.poisson(lambda_loc_0)
            elif location == 2:
                requests = np.random.poisson(lambda_loc_1)
            elif location == 3:
                requests = np.random.poisson(lambda_loc_2)
            elif location == 4:
                requests = np.random.poisson(lambda_loc_3)
            elif location == 5:
                requests = np.random.poisson(lambda_loc_4)
 
        # Maximum number of customer requests is 15 as per problem statement
        if requests > 15:
            requests = 15
 
        # Since range is 0 to (m-1)*m i.e. 0 to 20, it means indices between 0 to 19 are considered for random sampling
        # This means index 20 which is for no-ride option i.e. (0,0) does not get selected. This is intentional
        possible_actions_index = random.sample(range(0, (m-1)*m), requests)
        
        # Get the corresponding actions from the action space
        actions = [self.action_space[i] for i in possible_actions_index]
 
        # Now we add the no-ride option to the random samples as the driver will always have this option available to him
        actions.append((0,0))
        random.shuffle(actions)
 
        return actions  
 
 
 
    def step(self, state, action, time_matrix):       
        """Takes in state, action and Time-matrix and returns the reward, next state and terminal state status"""
       
        start_loc, hour, day = state  # Current State     
        pickup_loc, drop_loc = action # Action taken by the driver in the Current State
       
        # When the driver chooses no-ride
        if action == (0,0):
            # reward = −𝐶𝑓 if 𝑎 = (0,0) i.e. driver chooses no-ride
            reward = -1 * C
            
            # Time taken by the driver to move to the pick up location from his current location is zero
            pickup_time = 0
            # Time taken to complete the ride is zero
            ride_time = 0
 
            # Details for the next state
            # Time is incremented by 1 hour
            next_hour = hour + 1
            next_day = day
            # Start location still remains the same
            next_loc = start_loc
           
        # When the driver accepts and executes the ride
        else:     
            # In Time_matrix, the location indices are 0-4 but in our logic the location indices are 1-5
            # Hence we need to subtract 1 from the indices
           
            # Time taken by the driver to move to the pick up location from his current location
            pickup_time = int(time_matrix[start_loc - 1][pickup_loc - 1][hour][day])
               
            # Calculation of Time taken to complete the ride
            # Ride starts when the driver arrives at the pick up location
            start_hour = hour + pickup_time
            start_day = day
            # Check for the day roll over
            if start_hour >= 24:
                start_hour = start_hour - 24
                start_day = start_day + 1
                # Check for the week roll over
                if start_day >= 7:
                    start_day = start_day - 7
                   
            # Time taken to complete the ride
            ride_time = int(time_matrix[pickup_loc - 1][drop_loc - 1][start_hour][start_day])
         
            # reward = 𝑅𝑘 ∗ (𝑇𝑖𝑚𝑒(𝑝,𝑞)) − 𝐶𝑓 ∗ (𝑇𝑖𝑚𝑒(𝑝,𝑞) + 𝑇𝑖𝑚𝑒(𝑖,𝑝)) if 𝑎 = (𝑝,𝑞) i.e. when driver accepts the ride
            reward = (R * ride_time) - (C * (ride_time + pickup_time))
           
            # Details for the next state
            # Next start location is the last drop location
            next_loc = drop_loc
            # Update the start hour i.e. next hour = hour at the start of ride + time for pick-up + time for ride
            next_hour = start_hour + ride_time
            next_day = day
                     
        # Check for the day roll over
        if next_hour >= 24:
            next_hour = next_hour - 24
            next_day = next_day + 1
            # Check for the week roll over
            if next_day >= 7:
                next_day = next_day - 7
               
        # Next state
        next_state = (next_loc, next_hour, next_day)
       
        # Accumulate the total ride hours
        if (pickup_time + ride_time) > 0:
            self.total_ride_time = self.total_ride_time + pickup_time + ride_time
        else:
            self.total_ride_time = self.total_ride_time + 1
           
        # Determine if the terminal state is reached i.e. when 30 days are reached i.e. when 24*30 hours are completed
        thirty_days = t * 30 
        if self.total_ride_time >= thirty_days:
            terminal_state = True
        else:
            terminal_state = False
       
        return reward, next_state, terminal_state
 
 
    def reset(self):
        
        self.total_ride_time = 0
     
        # Choose a random state to begin with
        self.location_random = np.random.choice(np.arange(1,m+1)) 
        self.hour_random = np.random.choice(np.arange(0,t)) 
        self.day_random = np.random.choice(np.arange(0,d)) 
      
        self.state_init = (self.location_random, self.hour_random, self.day_random)

        return self.state_init
