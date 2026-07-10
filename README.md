# Reinforcement Learning (RL) Course Repository

This repository houses the code, practical exercises, and theoretical foundations developed for the **Reinforcement Learning (RL)** module.
---

## Course Objective

The main objective of this course is to build a theoretical and practical foundation for training autonomous agents to solve sequential optimization tasks via trial-and-error environment interaction over discrete time steps. Unlike supervised machine learning, which depends on independent, identically distributed static datasets, reinforcement learning works within environments characterized by sequential dependence and delayed feedback. The agent relies entirely on scalar reward signals to learn optimal long-term behaviors.

### Core Competencies Developed
* **Sequential Modeling**: Formulating complex continuous or discrete control problems through the mathematical lens of a Markov Decision Process (MDP).
* **Tabular Methods**: Implementing exact value-based algorithms for low-dimensional tracking tasks.
* **Deep Architectures**: Scaling architectures to high-dimensional state-action spaces using Deep Q-Networks (DQN) and policy gradient models (A2C/PPO).
* **Validation**: Deploying automated tuning pipelines and comparative statistical significance metrics to navigate the stochastic nature of RL environments.

---

## Course Outline

The repository is organized into five structured core modules:


### Module 1: Introduction
* **General Concepts**: Explains basic terminology, mapping agent interactions (actions, states, rewards) to real-world applications. It highlights critical design obstacles like exploration vs. exploitation, massive sample inefficiencies, and the sim-to-real gap.
* **Gymnasium**: Hands-on onboarding with standard OpenAI-derived benchmark environments using the `gymnasium` library.
  
[01_Introduction](https://github.com/aduo7/RL_EU4DUAL/blob/main/01_EU4DUAL_Reinforcement_Learning_Intro_MGEP.pdf)
### Module 2: RL Basics
* **Markov Decision Processes (MDPs)**: Formulates the foundational mathematical framework of reinforcement learning. Topics cover state transition probabilities, return definitions for episodic and continuing tasks, state/action value functions, and solving systems using the Bellman optimality equations.

### Module 3: Classic RL
* **Temporal Difference (TD) Learning**: Model-free, data-driven evaluation mechanisms using the Generalized Policy Iteration (GPI) paradigm.
* **Sarsa**: Implementations of on-policy single-step state-action value updates.
* **Q-Learning**: Implementations of off-policy action-value exploration, tested across classic tabular setups like Cliff Walking.

### Module 4: Deep RL
* **Deep Q-Networks (DQN)**: Implementing neural network function approximation to resolve continuous or high-dimensional state profiles. Key patterns include experience replay memory buffers to break temporal data correlation and stabilize gradient updates.
* **Policy Gradient Frameworks (A2C & PPO)**: Advanced actor-critic methods optimizing policies directly. Covers Advantage Actor-Critic (A2C) base estimators alongside Proximal Policy Optimization (PPO) using clipped surrogate objective functions, value losses, and exploration-driving entropy coefficients to secure stable policy shifts.

### Module 5: Evaluation
* **Hyperparameter Optimization**: Automated hyperparameter searching (learning rates, discount factors, batch sizes) using Optuna's Tree-structured Parzen Estimator (TPE) samplers and Median Pruners.
* **Evaluation Metrics**: Standardized run logging metrics capturing episodic mean returns, step tracking, loss curves, and throughput frames per second (FPS).
* **Statistical Significance**: Robust verification utilizing Welch's t-test, confidence intervals, and tolerance intervals to confidently establish algorithmic performance improvements across highly stochastic environments.
