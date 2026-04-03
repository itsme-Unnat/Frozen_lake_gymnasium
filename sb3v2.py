import gymnasium as gym
import stable_baselines3
import os
import argparse

# 1. Setup Directories
MODEL_DIR = "models"
LOG_DIR = "logs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def train(env, sb3_algo, sb3_class):
    print(f"--- Starting Training Loop for {sb3_algo} ---")
    
    # Initialize the model
    # device='cuda' uses GPU. If you don't have one, SB3 will fallback to CPU automatically.
    model = sb3_class('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=LOG_DIR)

    TIMESTEPS = 25000
    iters = 0
    
    try:
        while True:
            iters += 1
            # reset_num_timesteps=False ensures the total count in logs continues upward
            model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=sb3_algo)
            
            # Save the model with the current total timestep count
            save_path = os.path.join(MODEL_DIR, f"{sb3_algo}_{TIMESTEPS * iters}")
            model.save(save_path)
            print(f"Model saved to: {save_path}")
            
    except KeyboardInterrupt:
        print("\nTraining paused by user. Final model saved.")
        model.save(os.path.join(MODEL_DIR, f"{sb3_algo}_final"))

def test(env, path_to_model, sb3_class):
    print(f"--- Testing Model: {path_to_model} ---")
    model = sb3_class.load(path_to_model, env=env)

    while True: # Loop episodes for continuous viewing
        obs, info = env.reset()
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
        print("Episode finished. Restarting...")

if __name__ == '__main__':
    # 2. Command Line Arguments
    parser = argparse.ArgumentParser(description='Managed SB3 Training/Testing')
    parser.add_argument('gymenv', help='Gymnasium environment (e.g., Humanoid-v4, CartPole-v1)')
    parser.add_argument('sb3_algo', help='SB3 algorithm (e.g., PPO, SAC, DQN)')
    parser.add_argument('-t', '--train', action='store_true', help='Start training mode')
    parser.add_argument('-s', '--test', metavar='path_to_model', help='Path to a .zip model file to test')
    args = parser.parse_args()

    # 3. Dynamic Algorithm Import
    try:
        sb3_class = getattr(stable_baselines3, args.sb3_algo)
    except AttributeError:
        print(f"Error: {args.sb3_algo} is not a valid algorithm in Stable Baselines3.")
        exit()

    # 4. Execution Logic
    if args.train:
        env = gym.make(args.gymenv, render_mode=None)
        train(env, args.sb3_algo, sb3_class)

    elif args.test:
        if os.path.isfile(args.test) or os.path.isfile(args.test + ".zip"):
            env = gym.make(args.gymenv, render_mode='human')
            test(env, path_to_model=args.test, sb3_class=sb3_class)
        else:
            print(f'Error: Model file "{args.test}" not found.')