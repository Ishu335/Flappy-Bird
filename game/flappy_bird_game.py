import gymnasium as gym
import flappy_bird_gymnasium
import pygame

env=gym.make("FlappyBird-v0",render_mode='human')
state,info=env.reset()
done=False

pygame.init() 
screen=pygame.display.get_surface()


while not done: 
    action =0
    for envent in pygame.event.get():
        if envent.type==pygame.QUIT: 
            done=True
        elif envent.type==pygame.KEYDOWN:
            if envent.key==pygame.K_SPACE:
                action=1
    state,reward,done,truncated,info=env.step(action)
    env.render()