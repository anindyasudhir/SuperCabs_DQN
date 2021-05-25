# Problem Statement

You are hired as a Sr. Machine Learning Er. at SuperCabs, a leading app-based cab provider in a large Indian metro city. In this highly competitive industry, retention of good cab drivers is a crucial business driver, and you believe that a sound RL-based system for assisting cab drivers can potentially retain and attract new cab drivers. 
Cab drivers, like most people, are incentivised by a healthy growth in income. The goal of your project is to build an RL-based algorithm which can help cab drivers maximise their profits by improving their decision-making process on the field.

# The Need for Choosing the 'Right' Requests
Most drivers get a healthy number of ride requests from customers throughout the day. But with the recent hikes in electricity prices (all cabs are electric), many drivers complain that although their revenues are gradually increasing, their profits are almost flat. Thus, it is important that drivers choose the 'right' rides, i.e. choose the rides which are likely to maximise the total profit earned by the driver that day. 
For example, say a driver gets three ride requests at 5 PM. The first one is a long-distance ride guaranteeing high fare, but it will take him to a location which is unlikely to get him another ride for the next few hours. The second one ends in a better location, but it requires him to take a slight detour to pick the customer up, adding to fuel costs. Perhaps the best choice is to choose the third one, which although is medium-distance, it will likely get him another ride subsequently and avoid most of the traffic. 
There are some basic rules governing the ride-allocation system. If the cab is already in use, then the driver won’t get any requests. Otherwise, he may get multiple request(s). He can either decide to take any one of these requests or can go ‘offline’, i.e., not accept any request at all. 

In this project, you need to create the environment and an RL agent that learns to choose the best request. You need to train your agent using vanilla Deep Q-learning (DQN) only and NOT a double DQN. You have learnt about the two architectures of DQN (shown below) - you are free to choose any of these.

![image](https://user-images.githubusercontent.com/8203312/119506133-3f989480-bd8b-11eb-886a-083518a9d23d.png)

![image](https://user-images.githubusercontent.com/8203312/119506163-48896600-bd8b-11eb-9e5e-7282a146bd59.png)

We have implemented both of these architectures.


# Why did we choose Architecture 1 (State input neural network) over Architecture 2 (State-action input neural network)?

  - Architecture 2 works but is way slower than Architecture 1.
  - Architecture 2 took Total execution time of 4016.45 seconds to train 1000 episodes while Architecture 1 took just ~450 seconds to train on 1000 episodes making Architecture 2 (state-action input) infeasible with our system configuration.
