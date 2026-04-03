import gymnasium as gym
import stable_baselines3
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, StopTrainingOnRewardThreshold, EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
import argparse

# 1. Setup Directories
MODEL_DIR = "models"
LOG_DIR = "logs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def train(env, sb3_class, args):
    print(f"--- Starting Training: {args.sb3_algo} on {args.gymenv} ---")
    
    # Initialize the model with Tensorboard logging
    model = sb3_class(
        'MlpPolicy', 
        env, 
        verbose=1, 
        device='cuda', # Uses GPU if available
        tensorboard_log=LOG_DIR
    )

    # Callback A: Stop if we reach a specific performance goal
    # Note: Adjust reward_threshold based on the environment (e.g., 200 for CartPole, 300 for BipedalWalker)
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=300, verbose=1)

    # Callback B: Stop if the model stops getting better
    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=5, 
        min_evals=10, # Wait for at least 10 evaluations before checking for stagnation
        verbose=1
    )

    # Main Callback: Handles evaluation and saving the best model
    eval_callback = EvalCallback(
        env, 
        eval_freq=10000, 
        callback_on_new_best=callback_on_best, 
        callback_after_eval=stop_train_callback, 
        verbose=1, 
        best_model_save_path=os.path.join(MODEL_DIR, f"{args.gymenv}_{args.sb3_algo}"),
        log_path=LOG_DIR,
    )
    
    try:
        # Train for a very large number of timesteps; callbacks will stop it early
        model.learn(
            total_timesteps=int(1e10), 
            tb_log_name=f"{args.gymenv}_{args.sb3_algo}", 
            callback=eval_callback
        )
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving current state...")
        model.save(os.path.join(MODEL_DIR, f"{args.gymenv}_{args.sb3_algo}_interrupted"))

def test(env, sb3_class, args): 
    print(f"--- Testing Model: {args.gymenv}_{args.sb3_algo} ---")
    model_path = os.path.join(MODEL_DIR, f"{args.gymenv}_{args.sb3_algo}", "best_model")
    
    if not os.path.exists(model_path + ".zip"):
        print(f"Error: No model found at {model_path}. Train it first!")
        return

    model = sb3_class.load(model_path, env=env)

    obs = env.reset()[0]   
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            obs = env.reset()[0] # Restart for continuous viewing

if __name__ == '__main__':
    # 2. Command Line Arguments
    parser = argparse.ArgumentParser(description='Train or test SB3 models.')
    parser.add_argument('gymenv', help='Gymnasium environment (e.g., BipedalWalker-v3)')
    parser.add_argument('sb3_algo', help='SB3 algorithm (e.g., PPO, SAC, DQN, TD3)')    
    parser.add_argument('--test', help='Run in test mode with rendering', action='store_true')
    args = parser.parse_args()

    # 3. Dynamic Algorithm Import
    try:
        sb3_class = getattr(stable_baselines3, args.sb3_algo)
    except AttributeError:
        print(f"Error: {args.sb3_algo} is not a valid Stable Baselines3 algorithm.")
        exit()

    # 4. Environment Selection
    if args.test:
        env = gym.make(args.gymenv, render_mode='human')
        test(env, sb3_class, args)
    else:
        env = gym.make(args.gymenv)
        env = Monitor(env) # Tracks stats for logs
        train(env, sb3_class, args)
        

