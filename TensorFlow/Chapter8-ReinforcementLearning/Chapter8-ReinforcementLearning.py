''' Reinforcement learning is all about designing a system that is driven by
criticisms and rewards. Sometimes a system might recieve this feedback immediately,
making it easy to associate recent actions with that feedback (e.g. learning how
to perform an action). Other times, feedback might be very delayed, making it harder
to work out the consequences of previous actions (e.g. investing in a start-up
company). Reinforcement learning tries to make the right actions given any state.
It also has to find appropriate solutions to the exploration vs. exploitation
problem. A system may have been performing the same kinds of actions, and recieving
the same kinds of rewards for a long time. However, might there be a better series
of actions? Should the system experiment with new options, potentially risking
guaranteed rewards?

    Whereas supervised and unsupervised learning appear at opposite ends of the
spectrum, reinforcement learning (RL) exists somewhere in the middle. It is not
supervised learning, because the training data comes from the algorithm deciding
between exploration and exploitation. And it is not unsupervised because the algorithm
receives feedback from the environment. As long as you are in a situation where
performing an action in a state produces a reward, you can use reinforcement learning
to discover a good sequence of actions to take that maximize expected rewards.
'''

######################################################
# POLICY
######################################################
''' The system which tells an agent how to decide which actions to take is called
a POLICY. The goal of reinforcement learning is to discover a good policy. This
policy has to weigh the different advantages of short- vs. long-term consequences.
The best possible policy is called the OPTIMAL POLICY, and it is often the holy
grail of reinforcement learning. Learning the optimal policy, tells you the optimal
action given any state. There are many different types of policy. For example,
an agent could always chooses the action with the greatest immediate reward (called
a 'greedy' policy). Another option might be to arbitrarily choose an action (called
a 'random' policy). If you come up with a policy to solve a reinforcement learning
problem, it is often a good idea to double-check that your learned policy performs
better than both the random and greedy policies.
'''

######################################################
# UTILITY
######################################################
''' The long-term reward is called a UTILITY. It turns out that, if we know the utility
of performing an action at a given state, then it is easy to solve using reinforcement
learning. For example, to decide which action to take, we simply select the action that
produces the highest utility. The hard part is uncovering these utility values. The
utility of performing an action 'a' at a state 's' is written as a UTILITY FUNCTION
[ Q(s, a) ].

    An elegant way to calculate the utility of a particular state-action pair (s, a) is
by recursively considering the utilities of future actions. The utility of your current
action is not just influenced by the immediate reward, but also the next best action.
This concept is expressed in the formula below, where s' denotes the next state, and a'
denotes the next action. The reward of taking action a in state s is denoted by r(s, a):

        Q(s, a) = r(s, a) + gamma * max(Q(s', a'))

    Here, gamma is a hyper-parameter that you get to choose, called the DISCOUNT FACTOR.
If gamma is 0, then the agent chooses the action that maximizes the immediate reward.
Higher values of gamma will make the agent put more importance in considering long-term
consequences. We can read the formula as "the value of this action is the immediate reward
provided by taking this action, added to the discount factor times the best thing that can
happen after that".

    The extent to which one focuses on short- vs. long-term rewards is one way to play
with a system. In other applications of reinforcement learning though, newly available
information might be more important than historical records, or vice versa. For example,
if a robot is expected to learn to solve tasks quickly but not necessarily optimally, we
might want to set a faster learning rate. Or if a robot is allowed more time to explore and
exploit, we might tune down the learning rate. Let's call the learning rate 'alpha', and
change our utility function as follows (notice when alpha = 1, this equation is the same
as the one above):

    Q(s, a) = Q(s,a) + alpha( r(s,a) + gamma * max(Q(s',a') - Q(s,a)) )

    Reinforcement learning can be solved if we know this Q(s, a) function (called Q-function)
Conveniently for us, neural networks are a way to approximate any function given enough
training data.

Note that most reinforcement learning algorithms boil down to just three main steps:

    - Infer(s) => a (Select the best action (a) given a state (s) using the knowledge it has so far)

    - Do(s, a) => r, s' (Perform action to find out the reward (r) as well as the next state (s'))

    - Learn(s, r, a, s') (Improve understanding of the world using the newly acquired knowledge (s, r, a, s').)
'''

######################################################
# APPLYING REINFORCEMENT LEARNING
######################################################
''' Application of reinforcement learning requires defining a way to retrieve rewards once
an action is taken from a state. In the current example, we will focus on stock trading, as
because buying and selling a stock changes the state of the trader (cash on hand), and each
action generates a reward (or loss). In this situation, the states are representated as a
multidimensional vector containing information about the current budget, current number of
stocks, and a recent history of stock prices (the last 200 stock prices). Therefore each
state is a 202-dimensional vector.

    For simplicity, there are only three actions: buy, sell, and hold:

    1. Buying a stock at the current stock price decreases the budget while incrementing the
    current stock count.

    2. Selling a stock trades it in for money at the current share price.

    3. Holding does neither, and performing that action simply waits a single time-period,
    and yields no reward.

    Ideally our algorithm should buy low and sell high. We may increase our profits by buying
low and selling high as frequently as possible within a given period of time (i.e. high-frequency
trading). Our goal is to learn a policy that gains the maximum net-worth from trading in a stock
market.
'''

# Import dependencies
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import random
from pandas_datareader import data as pdr
import fix_yahoo_finance

######################################################
# Functions to download and display stock price data
######################################################

# Get stock prices
def get_prices(cache_filename='stock_prices.npy'):
    try:
        stock_prices = np.load(cache_filename)
    except:
        data = pdr.get_data_yahoo('MSFT', '1992-07-22', '2016-07-22') # Microsoft shares
        cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        data = data.reindex(columns=cols)
        data.reset_index(inplace=True,drop=False)
        stock_prices = data['Open'].values
        np.save(cache_filename, stock_prices)
    return stock_prices

# Visualise stock price data, and save figure to .png
def plot_prices(prices):
    plt.title('Opening stock prices')
    plt.xlabel('day')
    plt.ylabel('price ($)')
    plt.plot(prices)
    plt.savefig('prices.png')


######################################################
# Define a superclass for all decision policies
######################################################
''' Most reinforcement learning algorithms follow similar implementation patterns. As a
result, it is a good idea to create a class with the relevant methods to reference later.
Basically, reinforcement learning needs two operations well defined:
    (1) how to select an action,
    (2) how to improve the utility Q-function'''
class DecisionPolicy:
    def select_action(self, current_state, step): # Given a state, the decision policy will calculate the next action to take
        pass # The pass statement is a null operation; nothing happens when it executes. The pass is also useful in places where your code will eventually go, but has not been written yet

    def update_q(self, state, action, reward, next_state): # Improve the Q-function from a new experience of taking an action
        pass

######################################################
# Implement a random decision policy (using the previously defined DecisionPolicy superclass)
######################################################
''' Here, w will only need to define the select_action() method, which will randomly pick an
action without even looking at the state'''
class RandomDecisionPolicy(DecisionPolicy): # Inherit from DecisionPolicy to implement its functions
    def __init__(self, actions):
        self.actions = actions

    def select_action(self, current_state, step):
        action = self.actions[random.randint(0, len(self.actions) - 1)] # Randomly pick action
        return action

######################################################
# Implement a more intelligent decision policy
######################################################
''' With our random decision policy (defined above) working as our baseline, we can now
implement a more intelligent decision policy where we learn the utility, or 'Q' function.
This Q-function gives the utility of a given action in a given state by analysing the
current state and the best next state, and scaling this by the discount factor.

    This policy uses a deep neural network to approximate the Q-function. The input layers
is a state-space vector, which contains all of the state inforamtion (i.e.g current budget,
number of shares, and price history as many nodes). These inputs are then fed into a number
of deep, hidden layers, which eventually give a three-dimensional output of either: buy,
sell, or hold (see 'Design of reinforcement learning network.png')

    This policy introduces a new hyper-parameters: epsilon. This parameter keeps the
solution from getting "stuck" when applying the same action over and over. The lesser its
value, the more often it will randomly explore new actions'''
class QLearningDecisionPolicy(DecisionPolicy):
    # Constructor function
    def __init__(self, actions, input_dim):

        # Set hyper-parameters for the Q-function
        self.epsilon = 0.9 # Exploration vs. exploitation preference
        self.gamma = 0.001 # Discount factor (i.e. short- vs. long-term reward)
        self.actions = actions # Associate actions input with self object
        output_dim = len(actions) # State that the number of outputs is equivalent to the number of possible actions
        h1_dim = 200 # State the number of hidden neurons

        # Define the input and output tensors
        self.x = tf.placeholder(tf.float32, [None, input_dim])
        self.y = tf.placeholder(tf.float32, [output_dim])

        # Design the neural network architecture (i.e. weights and biases between input=>hidden, and hidden=>output)
        W1 = tf.Variable(tf.random_normal([input_dim, h1_dim]))
        b1 = tf.Variable(tf.constant(0.1, shape=[h1_dim]))
        h1 = tf.nn.relu(tf.matmul(self.x, W1) + b1)
        W2 = tf.Variable(tf.random_normal([h1_dim, output_dim]))
        b2 = tf.Variable(tf.constant(0.1, shape=[output_dim]))

        # Define the op to compute the utility (essentially the output of the network)
        self.q = tf.nn.relu(tf.matmul(h1, W2) + b2)

        # Set the loss as the square error
        loss = tf.square(self.y - self.q)

        # Use an optimizer to update model parameters to minimize the loss
        self.train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)

        # Set up the session and initialize variables
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    # Define algorithm for selecting an action
    def select_action(self, current_state, step):
        threshold = min(self.epsilon, step / 1000.)
        if random.random() < threshold: # Exploit best option with probability epsilon
            # Get suggested action from neural network
            action_q_vals = self.sess.run(self.q, feed_dict={self.x: current_state})
            # Find the index of action with the greatest activation
            action_idx = np.argmax(action_q_vals)  # TODO: replace w/ tensorflow's argmax
            # Get action from index
            action = self.actions[action_idx]
        else: # Explore random option with probability 1 - epsilon
            action = self.actions[random.randint(0, len(self.actions) - 1)] # Choose random action
        return action

    # Update the Q-function by updating its model parameters
    def update_q(self, state, action, reward, next_state):
        # Get current model suggestions for actions
        action_q_vals = self.sess.run(self.q, feed_dict={self.x: state})

        # Get next model suggestions for actions
        next_action_q_vals = self.sess.run(self.q, feed_dict={self.x: next_state})
        next_action_idx = np.argmax(next_action_q_vals)

        # Update action strengths based on current reward and the best 
        current_action_idx = self.actions.index(action)
        action_q_vals[0, next_action_idx] = reward + self.gamma * next_action_q_vals[0, next_action_idx] # r(s,a) + gamma * max(Q(s',a')
        action_q_vals = np.squeeze(np.asarray(action_q_vals))

        # Update weights based on previously calculated utility
        self.sess.run(self.train_op, feed_dict={self.x: state, self.y: action_q_vals})


######################################################
# Use a given policy to make decisions and return the performance
######################################################
def run_simulation(policy, initial_budget, initial_num_stocks, prices, hist, debug=False):
    
    # Initialize values that track key trading values
    budget = initial_budget
    num_stocks = initial_num_stocks
    share_value = 0
    transitions = list() # keeps tracking of all states
    
    # Loop over prices in current history window (defined by hist-1)
    for i in range(len(prices) - hist-1):

        # Print progress on every hundreth iteration
        if i % 100 == 0:
            print('progress {:.2f}%'.format(float(100*i) / (len(prices) - hist - 1)))

        # Get a single measure of the current state (by forcing the recent stock prices, current budget, and number of stocks into a single numpy matrix)
        current_state = np.asmatrix(np.hstack((prices[i:i+hist], budget, num_stocks)))

        # Calculate metric for current value of the stock portfolio
        current_portfolio = budget + num_stocks * share_value
        
        # Select an action from the current policy (using the select.action() method)
        action = policy.select_action(current_state, i)

        # Update portfolio based on action
        share_value = float(prices[i + hist + 1])
        if action == 'Buy' and budget >= share_value:
            budget -= share_value
            num_stocks += 1
        elif action == 'Sell' and num_stocks > 0:
            budget += share_value
            num_stocks -= 1
        else:
            action = 'Hold'

        # Compute new portfolio value after taking action
        new_portfolio = budget + num_stocks * share_value

        # Compute the reward from taking an action at a state
        reward = new_portfolio - current_portfolio

        # Get a single measure of the next state
        next_state = np.asmatrix(np.hstack((prices[i+1:i+hist+1], budget, num_stocks)))
        
        # Log the current state
        transitions.append((current_state, action, reward, next_state))

        # Update the policy after experiencing a new action (using the update_q() method)
        policy.update_q(current_state, action, reward, next_state)

    # Once loop has completed, compute the final portfolio worth
    portfolio = budget + num_stocks * share_value
    if debug:
        print('${}\t{} shares'.format(budget, num_stocks))
    return portfolio

######################################################
# Run multiple simulations to calculate an average performance
######################################################
'''To obtain a more robust measurement of success, let's run the simulation a
couple times and average the results. Doing so may take a while to complete
(perhaps 5 minutes), but your results will be more reliable'''
def run_simulations(policy, budget, num_stocks, prices, hist):
    num_tries = 10 # Decide number of times to re-run the simulations
    final_portfolios = list() # Store portfolio worth of each run in this array
    for i in range(num_tries): # Loop through number of tries
        final_portfolio = run_simulation(policy, budget, num_stocks, prices, hist)
        final_portfolios.append(final_portfolio) # Store portfolio worth
    avg, std = np.mean(final_portfolios), np.std(final_portfolios) # Average the values from all the runs
    return avg, std

''' RECORD OF FINAL RESULTS
RandomDecisionPolicy:       (Average:  1922.387, STD: 665.354)
QLearningDecisionPolicy:    (Average: 12941.613, STD: 458.116)
'''

######################################################
# Applying reinforcement learning to stock price data
######################################################
if __name__ == '__main__':
    prices = get_prices()
    plot_prices(prices)
    actions = ['Buy', 'Sell', 'Hold']
    hist = 200
    # policy = RandomDecisionPolicy(actions)
    policy = QLearningDecisionPolicy(actions, hist + 2)
    budget = 1000.0
    num_stocks = 0
    avg, std = run_simulations(policy, budget, num_stocks, prices, hist)
    print(avg, std)

