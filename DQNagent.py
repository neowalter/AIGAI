import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from collections import deque
import time
import matplotlib.pyplot as plt
import random

# Load data from CSV file
data = pd.read_csv('msft_days_data.csv')
dates = data['Date'].apply(lambda x: pd.to_datetime(x).timestamp()).values
prices = data['Close'].values
volumes = data['Volume'].values

# Set game parameters
initial_cash = 10000
transaction_fee = 1
n_shares = 0
cash = initial_cash

# Set DRQN parameters
state_size = 3 # date, close price and volume
action_size = 3 # buy, sell, hold
batch_size = 32
memory_size = 1000

# Initialize DRQN model
model = Sequential()
model.add(LSTM(16, input_shape=(1, state_size)))
model.add(Dense(action_size, activation='linear'))
model.compile(loss='mse', optimizer='adam')

# Initialize memory buffer
memory = deque(maxlen=memory_size)

# Define function to choose action based on epsilon-greedy policy
def choose_action(state, epsilon):
    if np.random.rand() <= epsilon:
        return np.random.randint(0, action_size)
    else:
        q_values = model.predict(state,verbose=None)
        return np.argmax(q_values[0])

# Define function to train DRQN model on a batch of experiences
def train_model():
    if len(memory) < batch_size:
        return

    batch = np.random.choice(len(memory), batch_size)
    for i in batch:
        state, action, reward, next_state, done = memory[i]
        target = reward
        if not done:
            target += 0.95 * np.amax(model.predict(next_state,verbose=None)[0])
        target_f = model.predict(state,verbose=None)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=0)
df = pd.DataFrame(columns=['epos','score'])

scores = []
for run in range(1000):
    start_time = time.time()
    # Play the game for one episode
    for t in range(len(prices)-1):
        # Observe current state
        state = np.array([dates[t], prices[t], volumes[t]]).reshape(1, 1, state_size)

        # Choose action based on epsilon-greedy policy
        action = choose_action(state, 0.1)

        # Take action and observe next state and reward
        next_state = np.array([dates[t+1], prices[t+1], volumes[t+1]]).reshape(1, 1, state_size)
        reward = 0
        if action == 0: # buy
            n_buy = cash // (prices[t] + transaction_fee)
            if n_buy > 0:
                cash -= n_buy * (prices[t] + transaction_fee)
                buy_price = prices[t]
                n_shares += n_buy
                # reward -= transaction_fee * n_buy
                reward = n_buy * (prices[t-1] - prices[t] - transaction_fee)
        elif action == 2: # hold
            reward = n_shares * (prices[t] - prices[t-1] + transaction_fee)
        elif action == 1: # sell
            if n_shares > 0:
                n_sell = random.randint(1, n_shares)
                cash += n_sell * (prices[t] - transaction_fee)
                n_shares -= n_sell
                # reward += (prices[t]- buy_price) * n_shares - transaction_fee
                # n_shares = 0
                reward = n_sell * (prices[t] - buy_price - transaction_fee)
        # else:
             

        # Compute game score and update reward based on score increase
        score = cash + n_shares * prices[t]
        reward += score - initial_cash

        # Store experience in memory buffer
        memory.append((state, action, reward, next_state, False))

        # Train DRQN model on a batch of experiences
        train_model()

        # if time.time() - start_time >= 2: # training time limit of 5 minutes (300 seconds)
        #     break

    # Print final game score and asset values after each run 
    score = cash + n_shares * prices[-1]
    scores.append(score)
    print('Run:', run+1)
    print('Final game score:', score)
    # print('Final cash value:', cash)
    # print('Final share value:', n_shares * prices[-1])

# Plot the scores over multiple runs 
plt.plot(scores)
plt.xlabel('Run')
plt.ylabel('Score')
plt.show()
