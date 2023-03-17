# RLDyna
 Reinforcement learning from scratch (Frozen Lake).
 
 ![FrozenLake](https://user-images.githubusercontent.com/17081096/225944512-9061e8d4-694f-4d0e-b8b2-5b8d314e538c.jpg)

 A little project for pyhton 3+ that shows an essence of reinforcement learning.
 
 The project uses 'graphics' module that can be installed automatically from PyCharm or by using - *pip install graphics*, and gym - *pip install gym*.
 
 The robot walks on the lake trying to find the exit.
 
 The robot uses **temporal difference off policy control tabular Q-learning** method.
 
 The entry point (main) *FrozenLakeMain.py*.
 
 The size and structure of the lake can be easely adjusted right from *FrozenLakeMain.py*.
 
 Different TD algorithms for testing (SARSA, MonteCarlo, DQ, TreeBackup, QSigma etc) can be added by extending from *StepControl.py* abstraction.
 
 NN models can be added by extending from *RModel.py*.
 
 Planning can be added by extending from *RPlanning.py*.
 
 Policy can be added by extending from *RPolicy.py*.
 
 *StateType* enum contains {blind, around, all_map} where:
 - blind - the robot sees only cell it stands on (good for tabular methods).
 - around - the robot sees cells around (good for different NN).
 - all_map - the robot sees all map (good for CNN).
 
 # How it works?
 
 Imagine you have some task / goal that you need to accomplish / achieve.
 Let's say in the most simple form our task is to play tic-tac-toe and our goal is to win.
 When we play the tic-tac-toe we focus on its grid, that is our **Environment** for this particular task.

 The **Environment** at each particular moment (its frame) can be described as **State**.

 We can represent the **State** like a bunch of numbers. For example, the empty tic-tac-toe grid, at start, can look like this {0, 0, 0, 0, 0, 0, 0, 0, 0} just one    dimensional array.

 Technically we can assign any number to any part of the **State** that has meaning for us, let us say 0 is an empty cell, 1 - O, 2 - X, then if someone put the X in the center of the empty grid its state becomes {0, 0, 0, 0, 2, 0, 0, 0, 0}

 The same way we can describe the **State** of the chess board, where 0 is an empty field, 11 - pawn, 12 - bishop, 13 - knight, 14 - rook, 15 - queen, 16 - king for black,  for white - 21, 22, 23, 24, 25, 26 accordingly. 
 If you have some robot that has a camera the **State** can be an image where each particular pixel's color of the image is a number.

 Having the **State** we can define the number of **Actions** that we can perform to modify **State** and hence an **Environment**, to run through our task to achieve the goal. For tic-tac-toe our set of **Actions** can be something like this {0, 1, 2, 3, 4, 5, 6, 7, 8 } where each number represents placing the O or X in the appropriate cell, the number of **Actions** can be restricted depending on the **State**.

 Performing **Actions** we change the **Environment** that provides us a new **State**.
 
![RL](https://user-images.githubusercontent.com/17081096/225954537-34483668-aaa3-4c82-820e-78ae9733e785.jpg)

## Reward

The question is what action we need to choose, being in a particular state, to complete the task and achieve our goal.
Choosing an action we can define how good it is, to go to another state by using the **Reward**. For example if the robot sees the road and decides to go forward, it can get the 0 reward, but if it sees the wall right in front and decides to go forward, it can get the -1 reward, which signals that it is not a good decision to do so.

Task can be **Continuous** when we just do something infinitely many times without the end adapting to the conditions over and over, and **Episodic** when it has some started conditions and it ends after several steps. For example Black Jack has a strict and small amount of states with reward at the end of an episode. Reward can be assigned after every step or only when an episode ends. For tasks with a small number of states having the episodic nature and getting a reward only when an episode ends, Monte Carlo algorithms often suit very well.

By using **Reward** an **Agent** can decide what **Action** is more suitable to choose being in a particular **State**.

Having, **State**, **Action**, **Reward** we now can see how we move through the states.

![States](https://user-images.githubusercontent.com/17081096/225958307-91f8c87e-36af-4870-9191-a300d9e09375.jpg)

## The Bellman equation

This picture shows us that being in a particular state and choosing a particular action we end up in the next state. Moreover we can see that even if we are in the same state and do the same action we can end up in different states. It is because we are not the ones who can influence the environment. Imagine you sell the cups, and you have 10 cups at the start (S10) and at the end of the day you have only 5 cups that moves you to (S5) and gives R = 5 (1 for each sold cup), but if someone broke 3 cups and you sold 2 cups that means you are still in (S5) but the R = 2.

We now can create **transition probability** model - **$\ p(S', r | S, a)$**. This function answers the question, what is the probability that being in **S**, choosing specific **a** you end up in **S'** with particular **r**? But as we know, from example above, that being in some **S** and choosing an **a** we can get any other **r** depending on the situation, then we get this - **$\sum_{r} p(S', r | S, a)$** - the sum must be 1 in case we have only one state to go to.

For example, let’s say we have only one state and the probability that you sold 5 cups ending up in S5 with R = 5 is 0.95. It could mean that in 95 days from 100 no exceptions happen but in 5 cases someone breaks 1 - 5 cups down evenly 1% on each case. We now have a transition model with probabilities - $\ 0.95 + 0.01 + 0.01 + 0.01 + 0.01 + 0.01 = 1$. We also need to specify reward - **$\sum_{r} p(S', r | S, a) * r$**. Now we can calculate the value - $\ 0.95 * 5 + 0.01 * 4 + 0.01 * 3 + 0.01 * 2 + 0.01 + 1 + 0.01 * 0 = 4.85$.

From the picture above we also see that being in **S** and choosing an **a** we can end up in different states that leads us to - **$\sum_{S'}\sum_{r} p(S', r | S, a) * r$**.

But we also can choose different **a** that leads us to - **$\sum_{a} \pi(a | S)\sum_{S'}\sum_{r} p(S', r | S, a) * r$**, where $\pi(a | S)$ is our **Policy**.

### Policy

Policy gives us the answer to the question, what is the best action to choose to complete a task / achieve a goal with maximum efficiency. Policy is the probability of choosing an action being in a particular state. From the example above we can say that we have two actions - “go to sell cups”, “stay home” = {1, 0}, if we work 5 days per week our policy is $\pi(1 | S) = 5 / 7$ and $\pi(0 | S) = 2 / 7$, but if our goal is to make maximum amount of money, then our policy is $\pi(1 | S) = 1$ and $\pi(0 | S) = 0$.

If our goal is to find the optimal policy, how do we choose the policy at the start? We can specify the policy with uniform probability. For example, imagine you have a robot that can go {west, east, north, south}. Its goal is to find the exit from the room, but it does not know where to go. Then for each action we can specify the policy as {0.25, 0.25, 0.25, 0.25}.

Our final equation is - **$U(S) = \sum_{a} \pi(a | S)\sum_{S'}\sum_{r} p(S', r | S, a) * (r + \gamma U(S'))$**.

Now the questions are: What the **$U(S)$** value actually is? What is **$\gamma U(S')$**? What if we need to somehow evaluate the transition probabilities? How to evaluate policy?

