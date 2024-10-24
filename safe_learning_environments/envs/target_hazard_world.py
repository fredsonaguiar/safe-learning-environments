import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces


class TargetHazardWorld(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 24}

    def __init__(self,render_mode=None,
                 window_size=500,
                 location_precision=0.01,
                 max_absolute_location=1,
                 max_absolute_velocity=1,
                 max_absolute_acceleration=1,
                 show_observation_traces=False):
        
        self.window_size = window_size # The size of the PyGame window
        self.location_precision = location_precision
        self.max_absolute_location = max_absolute_location
        self.max_absolute_velocity = max_absolute_velocity
        self.max_absolute_acceleration = max_absolute_acceleration

        # define observation_space
        self.observation_space = spaces.Tuple((
            # agent position and velocity
            spaces.Box(-max_absolute_location, max_absolute_location, shape=(2,), dtype=float),
            spaces.Box(-max_absolute_velocity, max_absolute_velocity, shape=(2,), dtype=float),
            
            # target position and velocity
            spaces.Box(-max_absolute_location, max_absolute_location, shape=(2,), dtype=float),
            spaces.Box(-max_absolute_velocity, max_absolute_velocity, shape=(2,), dtype=float),
            
            # hazard position and velocity
            spaces.Box(-max_absolute_location, max_absolute_location, shape=(2,), dtype=float),
            spaces.Box(-max_absolute_velocity, max_absolute_velocity, shape=(2,), dtype=float),
        ))

        # define action_space        
        self.action_space = spaces.Tuple((
            spaces.Box(-max_absolute_acceleration, max_absolute_acceleration, shape=(2,), dtype=float),
            spaces.Box(-max_absolute_acceleration, max_absolute_acceleration, shape=(2,), dtype=float),
            spaces.Box(-max_absolute_acceleration, max_absolute_acceleration, shape=(2,), dtype=float)
        ))

        # defines suitable render_mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # window and clock in case of human-rendering is used
        self.window = None
        self.clock = None

        # if set and human-rendering is used show the traces
        self.show_observation_traces = show_observation_traces
    
    def _get_obs(self):
        return (
            self._agent_location, self._agent_velocity,
            self._target_location, self._target_velocity,
            self._hazard_location, self._hazard_velocity
        )
    

    def _get_info(self):
        return {
            "target_distance": np.linalg.norm(self._agent_location - self._target_location),
            "hazard_distance": np.linalg.norm(self._agent_location - self._hazard_location),
        }
    
    
    def reset(self, seed=None, options=None):
        # we need the following line to seed self.np_random
        super().reset(seed=seed)

        # choose the new locations uniformly at random or check if given
        self._agent_velocity = np.zeros(shape=(2,))
        self._agent_location = self.np_random.uniform(-self.max_absolute_location, self.max_absolute_location, size=2)
        if options is not None:
            self._agent_location = options.get("agent_location", self._agent_location)

        # choose the new locations uniformly at random or check if given
        self._target_velocity = np.zeros(shape=(2,))
        self._target_location = self.np_random.uniform(-self.max_absolute_location, self.max_absolute_location, size=2)
        if options is not None:
            self._target_location = options.get("target_location", self._target_location)

        # choose the new locations uniformly at random or check if given
        self._hazard_velocity = np.zeros(shape=(2,))
        self._hazard_location = self.np_random.uniform(-self.max_absolute_location, self.max_absolute_location, size=2)
        if options is not None:
            self._hazard_location = options.get("hazard_location", self._hazard_location)

        # reinitializes observations
        if self.show_observation_traces:
            self.observations = []

        # renders frame if in human render mode
        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()
    

    def _step(self, location, velocity, accelaretion, dt):
        # force limited
        accelaretion = np.clip(accelaretion, -self.max_absolute_acceleration, self.max_absolute_acceleration)

        # velocity limited
        velocity = velocity + accelaretion * dt
        velocity = np.clip(velocity, -self.max_absolute_velocity, self.max_absolute_velocity)

        # location limited to window
        location = location + velocity * dt
        location = np.clip(location, -self.max_absolute_location, self.max_absolute_location)

        return location, velocity


    def step(self, action):
        # updated agents position and velocity
        dt = 1/self.metadata["render_fps"]
        self._agent_location, self._agent_velocity = self._step(self._agent_location, self._agent_velocity, action[0], dt)
        self._target_location, self._target_velocity = self._step(self._target_location, self._target_velocity, action[1], dt)
        self._hazard_location, self._hazard_velocity = self._step(self._hazard_location, self._hazard_velocity, action[2], dt)
        
        # check if terminated
        terminated = np.linalg.norm(self._agent_location - self._target_location) < self.location_precision

        # calculates reward
        reward = - dt

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, False, self._get_info()
    

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()


    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        
        # defines isometrics
        window_isometry = lambda location: (1 + location/self.max_absolute_location)*self.window_size/2

        # saves observations and draws lines
        if self.show_observation_traces:
            # saves observations    
            self.observations.append(self._get_obs())
            # draw lines when possible
            if len(self.observations) >= 2:
                _agent_locations, _, _target_locations, _, _hazard_locations, _ = zip(*self.observations)

                # draw the target traces
                _target_locations =  [window_isometry(position) for position in _target_locations]
                pygame.draw.aalines(canvas, (50, 150, 50), False, _target_locations, 2)
                # draw the hazard traces
                _hazard_locations =  [window_isometry(position) for position in _hazard_locations]
                pygame.draw.aalines(canvas, (150, 50, 50), False, _hazard_locations, 2)
                # draw the agent traces
                _agent_locations =  [window_isometry(position) for position in _agent_locations]
                pygame.draw.aalines(canvas, (0, 0, 0), False, _agent_locations, 2)

        # draw the target
        pygame.draw.circle(canvas, (50, 150, 50), window_isometry(self._target_location) , 10)
        # draw the hazard
        pygame.draw.circle(canvas, (150, 50, 50), window_isometry(self._hazard_location), 10)
        # draw the agent
        pygame.draw.circle(canvas, (0, 0, 0), window_isometry(self._agent_location), 8)


        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
        

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()