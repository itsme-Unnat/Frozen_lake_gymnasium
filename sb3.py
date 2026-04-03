import gymnasium as gym
from stable_baselines3 import SAC, TD3, A2C
import os
import argparse

# 1. Setup Directories
MODEL_DIR = "models"
LOG_DIR = "logs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def train(env, sb3_algo):
    print(f"--- Starting Training Loop: {sb3_algo} ---")
    
    # Initialize the specific model based on user input
    match sb3_algo:
        case 'SAC':
            model = SAC('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=LOG_DIR)
        case 'TD3':
            model = TD3('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=LOG_DIR)
        case 'A2C':
            model = A2C('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=LOG_DIR)
        case _:
            print(f'Error: Algorithm "{sb3_algo}" not supported by this script.')
            return

    TIMESTEPS = 25000
    iters = 0
    
    try:
        while True:
            iters += 1
            # We use reset_num_timesteps=False to keep the x-axis in Tensorboard continuous
            model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=sb3_algo)
            
            # Save periodic checkpoints
            save_path = os.path.join(MODEL_DIR, f"{sb3_algo}_{TIMESTEPS * iters}")
            model.save(save_path)
            print(f"Checkpoint saved: {save_path}")
            
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving final model...")
        model.save(os.path.join(MODEL_DIR, f"{sb3_algo}_latest"))

def test(env, sb3_algo, path_to_model):
    print(f"--- Testing Model: {path_to_model} ---")
    
    # Load the specific model class
    match sb3_algo:
        case 'SAC':
            model = SAC.load(path_to_model, env=env)
        case 'TD3':
            model = TD3.load(path_to_model, env=env)
        case 'A2C':
            model = A2C.load(path_to_model, env=env)
        case _:
            print(f'Error: Algorithm "{sb3_algo}" not recognized.')
            return

    # Evaluation Loop
    obs = env.reset()[0]
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            print("Episode Finished. Resetting...")
            obs = env.reset()[0]

if __name__ == '__main__':
    # 2. Command Line Arguments
    parser = argparse.ArgumentParser(description='Train or test specific SB3 models.')
    parser.add_argument('gymenv', help='Gymnasium environment (e.g., Humanoid-v4, Pendulum-v1)')
    parser.add_argument('sb3_algo', help='RL algorithm: SAC, TD3, or A2C')
    parser.add_argument('-t', '--train', action='store_true', help='Train the model')
    parser.add_argument('-s', '--test', metavar='path_to_model', help='Test the model from a specific path')
    args = parser.parse_args()

    # 3. Execution Logic
    if args.train:
        # Training usually doesn't need a UI (render_mode=None is faster)
        gymenv = gym.make(args.gymenv, render_mode=None)
        train(gymenv, args.sb3_algo)

    if args.test:
        # Check if file exists (with or without .zip extension)
        actual_path = args.test if os.path.isfile(args.test) else args.test + ".zip"
        
        if os.path.isfile(actual_path):
            gymenv = gym.make(args.gymenv, render_mode='human')
            test(gymenv, args.sb3_algo, path_to_model=args.test)
        else:
            print(f'Error: File "{args.test}" not found.')