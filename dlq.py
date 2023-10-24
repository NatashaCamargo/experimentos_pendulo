import numpy as np
import random
import tensorflow as tf
from collections import deque
from pendulum_env_teste import PendulumEnvironment

class DQLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.train_on_batch(state, target_f)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        self.screen.fill((255, 255, 255))
        x = int(self.screen_width / 2 + self.length * np.sin(self.theta))
        y = int(self.screen_height / 2 - self.length * np.cos(self.theta))

        pygame.draw.line(self.screen, (0, 0, 0), (self.screen_width // 2, self.screen_height // 2), (x, y), 3)
        pygame.draw.circle(self.screen, (0, 0, 255), (x, y), 10)

        pygame.display.flip()
        self.clock.tick(60)


if __name__ == "__main__":
    episodes = 1000
    batch_size = 32

    env = PendulumEnvironment()
    state_size = 2
    action_size = 3
    agent = DQLAgent(state_size, action_size)

    done = False
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        for time in range(500):
            action = agent.act(state)
            torque = action - 1  # torque pode ser -1, 0, 1
            next_state, reward, done, _ = env.step([torque])
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                print(f"Episode: {e}/{episodes}, Score: {time}, Epsilon: {agent.epsilon:.2}")
                break

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        
        agent.render()
