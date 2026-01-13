from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3

class LockedGemEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 480  # The size of the PyGame window


        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.window_size, self.window_size, 3), dtype=np.uint8
        )

        """
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "pressure_plate": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )
        """

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            Actions.right.value: np.array([1, 0]),
            Actions.up.value: np.array([0, 1]),
            Actions.left.value: np.array([-1, 0]),
            Actions.down.value: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return self._render_frame()
        #return {"agent": self._agent_location, "pressure_plate": self._pressure_plate_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }
    
    @staticmethod
    def is_point_inside_square(point, square_top_left_corner, square_side_length):
        return point[0] >= square_top_left_corner[0] and \
            point[0] <= square_top_left_corner[0] + square_side_length and \
            point[1] >= square_top_left_corner[1] and \
            point[1] <= square_top_left_corner[1] + square_side_length    
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Generate the locked room
        top_left_corner = self.np_random.integers(1, self.size - 3, size=2, dtype=int)

        # Generate the door
        rel_door_locations = [(0, 1), (1, 0), (1, 2), (2, 1)]
        rel_door_location_idx = self.np_random.choice(len(rel_door_locations))
        self._door_locked = True
        self._door_location = top_left_corner + rel_door_locations[rel_door_location_idx]
        
        self._walls = []
        for i in range(top_left_corner[0], top_left_corner[0] + 3):
            for j in range(top_left_corner[1], top_left_corner[1] + 3):
                if i == top_left_corner[0] + 1 and j == top_left_corner[1] + 1: continue
                if i == self._door_location[0] and j == self._door_location[1]: continue
                self._walls.append(np.array([i, j]))          

        # The gem will be inside of the room
        self._target_location = np.array([top_left_corner[0] + 1, top_left_corner[1] + 1])

        # Sample the agent's location randomly until it does not coincide with the room
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        while True:
            bad = False
            for wall in self._walls:
                bad = bad or np.array_equal(self._agent_location, wall)

            bad = bad or np.array_equal(self._agent_location, self._door_location)

            if bad:
                self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
            else:
                break

        # And we do the same for the pressure plate location
        self._pressure_plate_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        while True:
            bad = False
            for wall in self._walls:
                bad = bad or np.array_equal(self._pressure_plate_location, wall)

            bad = bad or np.array_equal(self._pressure_plate_location, self._door_location)
            bad = bad or np.array_equal(self._pressure_plate_location, self._agent_location)

            if bad:
                self._pressure_plate_location = self.np_random.integers(0, self.size, size=2, dtype=int)
            else:
                break
        
        observation = self._render_frame()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]

        moving_into_walls = False
        for wall in self._walls:
            moving_into_walls = np.array_equal(self._agent_location + direction, wall)
            if moving_into_walls: break

        moving_into_door = np.array_equal(self._agent_location + direction, self._door_location) and self._door_locked

        if not moving_into_walls and not moving_into_door:
            # We use `np.clip` to make sure we don't leave the grid
            self._agent_location = np.clip(
                self._agent_location + direction, 0, self.size - 1
            )

        if np.array_equal(self._agent_location, self._pressure_plate_location):
            self._door_locked = not self._door_locked

        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._render_frame()
        info = self._get_info()

        return observation, reward, terminated, False, info

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
        canvas.fill((112, 130, 56))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the walls and the door
        for wall in self._walls:
            pygame.draw.rect(
                canvas,
                (97, 101, 105),
                pygame.Rect(
                    pix_square_size * wall[::-1],
                    (pix_square_size, pix_square_size),
                ),
            )

        if self._door_locked:
            pygame.draw.rect(
                canvas,
                (64, 35, 32),
                pygame.Rect(
                    pix_square_size * self._door_location[::-1],
                    (pix_square_size, pix_square_size),
                ),
            )

        # Then we draw the pressure plate
        pygame.draw.rect(
            canvas,
            (64, 35, 32),
            pygame.Rect(
                (self._pressure_plate_location[::-1] + 0.25) * pix_square_size ,
                (pix_square_size // 2, pix_square_size // 2),
            ),
        )

        # Then we draw the agent
        pygame.draw.circle(
            canvas,
            (24, 41, 88),
            (self._agent_location[::-1] + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Now we draw the target
        pygame.draw.circle(
            canvas,
            (128, 0, 128),
            (self._target_location[::-1] + 0.5) * pix_square_size,
            pix_square_size / 6,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])

        #else:  # rgb_array
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
