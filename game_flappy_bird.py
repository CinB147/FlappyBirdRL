import gymnasium as gym
import flappy_bird_gymnasium  # noqa: F401 - registers "FlappyBird-v0"
import pygame

# use_lidar=False -> state is 12 simple numbers (pipe positions, bird y, velocity...)
env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)
state, info = env.reset()
done = False

while not done:
    action = 0  # 0 = do nothing, 1 = flap

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            action = 1

    # step() also draws the frame and keeps it at 30 FPS, so no env.render() needed
    state, reward, terminated, truncated, info = env.step(action)
    done = done or terminated or truncated

print("Score:", info["score"])
env.close()
