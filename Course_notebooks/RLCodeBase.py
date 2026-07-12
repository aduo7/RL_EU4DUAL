import imageio
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.stats import  gaussian_kde
from scipy import stats

class ReinforcementLearningBase:
  def __init__(self, env, gamma, epsilon, min_epsilon, epsilon_decay):
    self.env = env
    self.gamma = gamma
    self.epsilon = epsilon
    self.min_epsilon = min_epsilon
    self.epsilon_decay = epsilon_decay
    self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
    self.episode_returns = []  # To store the return of each episode
    self.episode_epsilon = []

  def epsilon_greedy_policy(self, state):
    if np.random.rand() < self.epsilon:
        return self.env.action_space.sample()  # Random action (exploration)
    else:
        return np.argmax(self.Q[state])  # Greedy action (exploitation)

  def epsilon_soft_policy(self, state):
    probabilities = np.ones(self.env.action_space.n) * (self.epsilon / self.env.action_space.n)  # Minimum probability for all actions
    best_action = np.argmax(self.Q[state])
    probabilities[best_action] += 1 - self.epsilon  # Distribute remaining probability to the greedy action
    return np.random.choice(self.env.action_space.n, p=probabilities)


  # Plot the learning progress
  def plot_learning_progress(self, window_size=100):
      """
      Plot the returns over episodes and a moving average to smooth the curve.
      """
      moving_avg = np.convolve(self.episode_returns, np.ones(window_size) / window_size, mode='valid')
      plt.figure(figsize=(10, 6))
      plt.plot(self.episode_returns, label='Episode Return', alpha=0.3)
      plt.plot(moving_avg, label=f'Moving Average (last {window_size} episodes)', color='red')
      plt.xlabel('Episode')
      plt.ylabel('Return')
      plt.title('Learning Progress: Returns Over Episodes')
      plt.legend()
      plt.grid()
      plt.show()

  # Evaluate the trained policy
  def evaluate_policy(self, num_episodes=5, save_video=False, video_dir=""):
      total_rewards = 0
      evaluation_returns=[]
      frames=[]
      for _ in range(int(num_episodes)):
          state, _ = self.env.reset()
          frames.append(self.env.render())
          done = False
          while not done:
              action = np.argmax(self.Q[state])  # Greedy policy
              state, reward, terminated, truncated, _ = self.env.step(action)
              done = terminated or truncated
              total_rewards += reward
              frames.append(self.env.render())
          evaluation_returns.append(total_rewards)
      if save_video:
        self.save_video(frames, video_dir)

      return total_rewards / num_episodes, evaluation_returns

  def save_video(self, frames, dir):
    imageio.mimsave(dir, frames, fps=3)


def moving_average(data, window_size=100):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_learning_curves(returns_dict, window_size=100):
    plt.figure(figsize=(10, 6))
    for algo, returns in returns_dict.items():
        all_returns = np.array(returns)
        mean_returns = np.mean(all_returns, axis=0)
        std_returns = np.std(all_returns, axis=0)

        smoothed_mean = moving_average(mean_returns, window_size)
        smoothed_std = moving_average(std_returns, window_size)

        x_values = np.arange(len(smoothed_mean))
        plt.plot(x_values, smoothed_mean, label=algo)
        plt.fill_between(x_values, smoothed_mean - smoothed_std, smoothed_mean + smoothed_std, alpha=0.2)

    plt.xlabel("Episodes")
    plt.ylabel("Mean Return")
    plt.title("Learning Curves")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_avg_reward_dist(mean_rewards):
  # Analyze and visualize the distribution of mean rewards
  mean_reward_distribution = gaussian_kde(mean_rewards)
  x_values = np.linspace(min(mean_rewards), max(mean_rewards), 1000)

  plt.plot(x_values, mean_reward_distribution(x_values), label="Mean Reward Distribution")
  plt.axvline(np.mean(mean_rewards), color='r', linestyle='--', label=f"Overall Mean: {np.mean(mean_rewards):.2f}")
  plt.fill_between(x_values, mean_reward_distribution(x_values), alpha=0.3)
  plt.xlabel("Mean Reward")
  plt.ylabel("Density")
  plt.title("Distribution of Mean Rewards Across Trained Agents")
  plt.legend()
  plt.show()

def compute_confidence_intervals(data, confidence_level=0.95):
    mean = np.mean(data, axis=0)
    std_err = stats.sem(data, axis=0)
    n = data.shape[0]
    h = std_err * stats.t.ppf((1 + confidence_level) / 2, n - 1)
    return mean, mean - h, mean + h

def plot_learning_curves_with_confidence_intervals(returns_dict, window_size=100, confidence_level=0.95):
    plt.figure(figsize=(10, 6))
    for algo, returns in returns_dict.items():
        all_returns = np.array(returns)

        # Compute mean and confidence intervals
        mean_returns, lower_bound, upper_bound = compute_confidence_intervals(all_returns, confidence_level)

        # Smooth the mean and bounds
        smoothed_mean = moving_average(mean_returns, window_size)
        smoothed_lower_bound = moving_average(lower_bound, window_size)
        smoothed_upper_bound = moving_average(upper_bound, window_size)

        # Adjust x-values for the moving average
        x_values = np.arange(len(smoothed_mean)) + window_size - 1

        # Plot the mean with confidence intervals
        plt.plot(x_values, smoothed_mean, label=algo)
        plt.fill_between(x_values, smoothed_lower_bound, smoothed_upper_bound, alpha=0.2)

    plt.xlabel("Episodes")
    plt.ylabel("Mean Return")
    plt.title("Learning Curves with Confidence Intervals")
    plt.legend()
    plt.grid(True)
    plt.show()

def compute_tolerance_intervals(data, alpha=0.05, beta=0.9):
    """
    Compute tolerance intervals for the given data.

    Parameters:
        data: Array of shape (num_runs, num_episodes).
        alpha: Confidence level (1 - alpha).
        beta: Proportion of population to capture.

    Returns:
        lower_bound, upper_bound: Tolerance interval bounds.
    """
    num_runs, num_episodes = data.shape
    data_sorted = np.sort(data, axis=0)

    # Compute empirical percentiles
    lower_percentile = (1 - beta) / 2
    upper_percentile = 1 - (1 - beta) / 2
    lower_index = int(np.floor(lower_percentile * num_runs))
    upper_index = int(np.ceil(upper_percentile * num_runs))

    lower_bound = data_sorted[lower_index]
    upper_bound = data_sorted[upper_index-1]

    # Adjust for sample size (optional, for more conservative intervals)
    k = stats.norm.ppf(1 - alpha / 2) * np.sqrt((num_runs + 1) / (num_runs * beta))
    std_dev = np.std(data, axis=0)

    # Ensure data types are compatible for broadcasting
    lower_bound = lower_bound.astype(np.float64)
    upper_bound = upper_bound.astype(np.float64)

    lower_bound -= k * std_dev
    upper_bound += k * std_dev

    return lower_bound, upper_bound

def plot_learning_curves_with_tolerance_intervals(returns_dict, window_size=100, alpha=0.05, beta=0.9):
    plt.figure(figsize=(10, 6))
    for algo, returns in returns_dict.items():
        all_returns = np.array(returns)

        # Compute mean and tolerance intervals
        mean_returns = np.mean(all_returns, axis=0)
        lower_bound, upper_bound = compute_tolerance_intervals(all_returns, alpha, beta)

        # Smooth the mean and bounds
        smoothed_mean = moving_average(mean_returns, window_size)
        smoothed_lower_bound = moving_average(lower_bound, window_size)
        smoothed_upper_bound = moving_average(upper_bound, window_size)

        # Adjust x-values for the moving average
        x_values = np.arange(len(smoothed_mean)) + window_size - 1

        # Plot the mean with tolerance intervals
        plt.plot(x_values, smoothed_mean, label=algo)
        plt.fill_between(x_values, smoothed_lower_bound, smoothed_upper_bound, alpha=0.2)

    plt.xlabel("Episodes")
    plt.ylabel("Mean Return")
    plt.title("Learning Curves with Tolerance Intervals")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def perform_welch_ttest(td_returns, mc_returns):
    """
    Perform Welch's t-test to compare two sets of returns

    Args:
    td_returns (np.array): Returns from Temporal Difference algorithm
    mc_returns (np.array): Returns from Monte Carlo algorithm

    Returns:
    dict: Test statistics and p-value
    """
    # Aggregate results (e.g., take the mean of each run)
    td_returns = [np.mean(run) for run in td_returns]  # Mean return per run for TD
    mc_returns = [np.mean(run) for run in mc_returns]  # Mean return per run for MC

    # Perform Welch's t-test
    t_statistic, p_value = stats.ttest_ind(td_returns, mc_returns, equal_var=False)

    # Calculate summary statistics
    td_mean = np.mean(td_returns)
    mc_mean = np.mean(mc_returns)
    td_std = np.std(td_returns, ddof=1)  # Sample standard deviation
    mc_std = np.std(mc_returns, ddof=1)

    return {
        't_statistic': t_statistic,
        'p_value': p_value,
        'td_mean': td_mean,
        'mc_mean': mc_mean,
        'td_std': td_std,
        'mc_std': mc_std
    }