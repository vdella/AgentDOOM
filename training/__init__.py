import gymnasium
import ale_py


env = gymnasium.make("ALE/Tetris-v5", render_mode="human")
epochs = 10000