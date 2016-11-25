import gym
import cv2
import random

class GymEnv:
    def __init__(self, config):
        self.env = gym.make(config.env_name)
        self._screen = None
        self.reward = 0
        self.term = True
        self.display = config.display
        self.dims = (config.screen_h, config.screen_w)
        self.action_repeat = config.action_repeat
        self.random_start = config.random_start

    def newRandomGame(self):
        self.newGame()
        for _ in xrange(random.randint(0, self.random_start - 1)):
            self.step(0)
        if self.display:
            self.render()
        return self.screen, 0, 0, self.term

    def newGame(self):
        if self.env.ale.lives() == 0:
            self._screen = self.env.reset()
        self._screen, self.reward, self.term, _ = self.env.step(0)
        if self.display:
            self.render()
        return self.screen, 0, 0, self.term

    def step(self, action):
        self._screen, self.reward, self.term, _ = self.env.step(action)
        return self.screen, self.reward, self.term

    def render(self):
        self.env.render()

    def act(self, action, is_training=True):
        cumulated_reward = 0
        start_lives = self.env.ale.lives()
        for _ in xrange(self.action_repeat):
            self.step(action)
            cumulated_reward += self.reward
            if is_training and start_lives > self.env.ale.lives():
                cumulated_reward -= 1
                self.term = True
            if self.term:
                break
        self.reward = cumulated_reward
        if self.display:
            self.render()
        return self.screen, self.reward, self.term

    @property
    def action_size(self):
        return self.env.action_space.n

    @property
    def screen(self):
        return cv2.resize(cv2.cvtColor(self._screen, cv2.COLOR_RGB2GRAY)/255.0, self.dims)

