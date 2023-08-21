import random
import numpy as np
import pandas as pd
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('msft_days_data.csv')
data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')

# Define state space size and action space size
state_size = 4 # cash, shares, price, volume
action_size = 3 # buy, sell, skip

# Define DQN agent class
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95 # discount factor
        self.epsilon = 1.0 # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
    
    def _build_model(self):
        # Neural network for approximating Q-function
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        # Store experience in memory
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        # Choose action based on epsilon-greedy policy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        # Train model on a batch of experiences from memory
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Initialize game variables and DQN agent
cash = 10000
shares = 0
score = cash
transaction_fee = 1
current_date_index = 0

agent = DQNAgent(state_size, action_size)
batch_size = 32

# Run game for multiple episodes (i.e., until final date is reached)
for e in range(3):
    # Reset game variables for new episode
    cash = 10000
    shares = 0
    score = cash
    current_date_index = 0
    
    # Get initial state from data and reshape to (1,state_size)
    current_price = data.iloc[current_date_index]['Close']
    current_volume = data.iloc[current_date_index]['Volume']
    state = np.array([cash, shares, current_price, current_volume]).reshape((1,state_size))
    
    while current_date_index < len(data) - 1 and score > 0:
        # Choose action based on current state and agent's policy
        action = agent.act(state)
        
        # Update cash and shares based on chosen action (buy/sell/skip)
        if action == 0: # buy one share if possible (subject to cash availability)
            if cash >= current_price + transaction_fee:
                cash -= current_price + transaction_fee
                shares += 1
                current_date_index +=1
                # print(f"Episode {e}: Buy at {current_price:.2f}")
        
        elif action == 1: # sell one share if possible (subject to share ownership)
            if shares > 0:
                cash += current_price - transaction_fee
                shares -= 1
                current_date_index +=1
                # print(f"Episode {e}: Sell at {current_price:.2f}")
        
        else: # skip (do nothing)
            current_date_index +=1
            # print(f"Episode {e}: Skip at step {current_date_index}")
        
        # Update score and increment current date index to move to next row in data
        score = cash + shares * current_price
    
    # print(f"Episode {e}: score is:  {score}")
    df = pd.DataFrame()
    new_row = {"epo": round(e+1), "score": score}
    df = df._append(new_row, ignore_index=True)
    df = df.reset_index()
    print(df)


# fig = plt.figure(figsize=(8, 6))
# plt = fig.add_subplot(111)

sns.lineplot(x='epo', y='score', data=df)

# Add title and labels
plt.title('learning result')
plt.xlabel('epo')
plt.ylabel('score')
plt.show()
