import random
import numpy as np


# Compute a single rollout.
#
# env: Environment
# policy: Policy
# render: bool
# return: [(np.array([state_dim]), np.array([action_dim]), float, np.array([state_dim]))]
#          ((state, action, reward, next_state) tuples)
def get_rollout(env, policy, render):
    # Step 1: Initialization
    state = env.reset()
    done = False

    # Step 2: Compute rollout
    sarss = []
    while not done:
        # Step 2a: Render environment
        if render:
            env.render()

        # Step 2b: Action
        action = policy.get_action(state)

        # Step 2c: Transition environment
        next_state, reward, done, _ = env.step(action)

        # Step 2d: Rollout (s, a, r)
        sarss.append((state, action, reward, next_state))

        # Step 2e: Update state
        state = next_state

    # Step 3: Render final state
    if render:
        env.render()

    return sarss


# Estimate the cumulative reward of the policy.
#
# env: Environment
# policy: Policy
# n_rollouts: int
# return: float
def test_policy(env, policy, n_rollouts):
    cum_reward = 0.0
    succ_rate = 0.0
    for i in range(n_rollouts):
        sarss = get_rollout(env, policy, False)
        tmp_rew = env.cum_reward(np.array([state for state, _, _, _ in sarss]))
        cum_reward += tmp_rew
        if tmp_rew > 0:
            succ_rate += 1.0
    return cum_reward / n_rollouts, (succ_rate * 100.0 / n_rollouts)
