import pygame
import numpy as np

class PendulumEnvironment:
    def __init__(self, length=200, mass=10, dt=0.05, g=9.81):
        self.length = length
        self.mass = mass
        self.dt = dt
        self.g = g

        self.theta = np.pi / 4
        self.theta_dot = 0

        pygame.init()
        self.screen_width, self.screen_height = 800, 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('Pendulum Simulation')

        self.clock = pygame.time.Clock()

    def reset(self):
        self.theta = np.random.uniform(-np.pi, np.pi)
        self.theta_dot = np.random.uniform(-1, 1)
        return [self.theta, self.theta_dot]

    def step(self, action):
        torque = action[0]
        theta_double_dot = (self.g / self.length) * np.sin(self.theta) + (1.0 / (self.mass * self.length**2)) * torque
        self.theta_dot += theta_double_dot * self.dt
        self.theta += self.theta_dot * self.dt
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi

        reward = -(self.theta**2 + 0.1 * self.theta_dot**2 + 0.001 * action[0]**2)
        done = False
        return [self.theta, self.theta_dot], reward, done, {}

    def render(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.screen.fill((255, 255, 255))
            x = int(self.screen_width / 2 + self.length * np.sin(self.theta))
            y = int(self.screen_height / 2 - self.length * np.cos(self.theta))

            pygame.draw.line(self.screen, (0, 0, 0), (self.screen_width // 2, self.screen_height // 2), (x, y), 3)
            pygame.draw.circle(self.screen, (0, 0, 255), (x, y), 10)

            pygame.display.flip()
            self.clock.tick(60)
            action = [0.0]
            self.step(action)

        pygame.quit()

    def close(self):
        pygame.quit()

# Testando a animação
env = PendulumEnvironment()
env.reset()
env.render()
