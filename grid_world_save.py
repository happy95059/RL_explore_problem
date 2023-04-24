import gym
from gym import spaces
import pygame
import numpy as np
import math
import copy
class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5, agent_see=3, wall=20):
        self.size = size  # The size of the square grid
        self.agent_see=agent_see
        self.window_size = 512  # The size of the PyGame window
        self.full_map=np.zeros([size,size])
        self.wall_size=wall
        env_state=self.build()
        self._target_location=env_state[wall]
        self.wall=env_state[0:wall]
        print(self.wall)
        print(self._target_location)
        #self.wall=[[2,3],[2,2],[2,1]]
        #self._target_location=[1,2]
        
        self.filter=int((agent_see-1)/2)
        self.full_map[self._target_location[0],self._target_location[1]]=2
        for i in self.wall:
            self.full_map[i[0],i[1]]=1
        print(self.full_map)
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).

        self.observation_space = spaces.Discrete(int(math.pow(3,(agent_see*agent_see))))
        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
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
     
        
     
    def change_render(self,render_mode):
        self.render_mode=render_mode
    
    def build(self):
        env_state=[]
        for i in range(self.wall_size+1):
            flag=True
            while flag:
                temp=self.np_random.integers(0, self.size, size=(2), dtype=int)
                flag=False
                for j in range(len(env_state)):
                    if np.equal(env_state[j],list(temp)).all():
                        flag=True
            env_state.append(list(temp))
        return env_state
    def _get_obs(self):
        pad_map=np.pad(self.full_map,(self.filter),'constant',constant_values=1)
        '''
        print('======================================')
        print('real state = '+str(self._agent_location))
        print('target = '+str(self._target_location))
        print('======================================')
        '''
        return pad_map[self._agent_location[0]:self._agent_location[0]+2*self.filter+1,self._agent_location[1]:self._agent_location[1]+2*self.filter+1]

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        flag=True
        walls=copy.deepcopy(self.wall)
        walls.append(self._target_location)

        walls=np.array(walls)
        while flag:
            self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
            flag=False
            for i in walls:
                if np.array_equal(i ,self._agent_location):
                    flag=True
        #use agent_state get map
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        next_location=self._agent_location+direction
        pad_map=np.pad(self.full_map,(self.filter),'constant',constant_values=1)
        die=False
        if pad_map[next_location[0]+self.filter,next_location[1]+self.filter]!=1:
            self._agent_location=next_location
        else:
            die=True
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 200 if terminated else -3  # Binary sparse rewards
        reward = -10  if die else reward
        observation = self._get_obs()
        info = self._get_info()
        info={
            "distance": np.array(
                self._agent_location
            )
        }
        if self.render_mode == "human":
            self._render_frame()
        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size*2, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size*2, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * np.array(self._target_location).astype('int'),
                (pix_square_size, pix_square_size),
            ),
        )
        # drow wall
        for wall in self.wall:
            pygame.draw.rect(
                canvas,
                (0, 255, 0),
                pygame.Rect(
                    pix_square_size * np.array(wall).astype('int'),
                    (pix_square_size, pix_square_size),
                ),
            )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
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
        
        
        
        
        pix_square_size = (
            self.window_size / self.agent_see
        )  # The size of a single grid square in pixels

        
        state=self._get_obs()
        walls,target=self.state_to_chanel(state)
        agent=[self.filter,self.filter]
        # First we draw the target
        '''
        if target!=[]:
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    pix_square_size * (np.array(target).astype('int')+[self.agent_see,0]),
                    (pix_square_size, pix_square_size),
                ),
            )
        '''
        # drow wall
        for wall in walls:
            pygame.draw.rect(
                canvas,
                (0, 255, 0),
                pygame.Rect(
                    pix_square_size * (np.array(wall).astype('int')+[self.agent_see,0]),
                    (pix_square_size, pix_square_size),
                ),
            )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            ((np.array(agent).astype('int') + 0.5)+[self.agent_see,0]) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.agent_see + 1):
            pygame.draw.line(
                canvas,
                0,
                (self.window_size, pix_square_size * x),
                (self.window_size*2, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * (x+self.agent_see), 0),
                (pix_square_size * (x+self.agent_see), self.window_size),
                width=3,
            ) 
        
        
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    

    
    def state_to_chanel(self,state):
        size=state[0].size
        matrix_1=np.ones([size,size])
        matrix_2=np.ones([size,size])*2
        
        chanel_1 = np.equal(matrix_1,state).astype('int')
        chanel_2 = np.equal(matrix_2,state).astype('int')
        wall=[]
        target=[]
        for i in range(size):
            for j in range(size):
                if chanel_1[i,j]==1:
                    wall.append([i,j])
                if chanel_2[i,j]==1:
                    target.append([i,j])
        return wall,target
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()



