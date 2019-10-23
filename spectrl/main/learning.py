from spectrl.main.monitor import Resource_Model
from spectrl.main.monitor import Compiled_Spec
from spectrl.ars.ars_discrete import *

import numpy as np


class ProductMDP:

    # system : System MDP (no need for reward function)
    # action_dim: action space dimension for the system
    # res_model : Resource_Model (optional)
    # spec : TaskSpec
    # min_reward (C_l) = Min possible unshaped reward
    # local_reward_bound (C_u) = Max possible absolute value of local reward (quant. sem. value)
    def __init__(self, system, action_dim, spec, min_reward, local_reward_bound,
                 res_model=None, use_shaped_rewards=True):
        self.system = system
        init_system_state = self.system.reset()
        system_state_dim = len(init_system_state)
        if res_model is None:
            def delta(sys_state, res_state, sys_action):
                return np.array([])
            res_model = Resource_Model(system_state_dim, action_dim, 0, np.array([]), delta)
        monitor = spec.get_monitor(system_state_dim, res_model.res_dim, local_reward_bound)
        self.spec = Compiled_Spec(res_model, monitor, min_reward, local_reward_bound)
        self.state = (np.append(init_system_state, self.spec.init_extra_state()), 0)
        self.is_shaped = use_shaped_rewards

    def reset(self):
        self.state = (np.append(self.system.reset(), self.spec.init_extra_state()), 0)
        return self.state

    def step(self, action):
        next_state, rew, done, render = self.system.step(action[:self.spec.action_dim])
        res_reg_state, monitor_state = self.spec.extra_step(self.state, action)
        self.state = (np.append(next_state, res_reg_state), monitor_state)
        return self.state, rew, done, render

    def cum_reward(self, rollout):
        if self.is_shaped:
            return self.spec.cum_reward_shaped(rollout)
        else:
            return self.spec.cum_reward_unshaped(rollout)

    def render(self):
        self.system.render()

# ==================================================================================================


class HyperParams:

    # hidden_layer_dim = Number of neurons in the hidden layers
    # actions confined to the range [-action_bound, action_bound]
    # n_iters: int (ending condition)
    # n_samples: int (N)
    # n_top_samples: int (b)
    # delta_std (nu)
    # lr: float (alpha)
    # min_lr: float (minimum alpha)
    def __init__(self, hidden_layer_dim, action_bound, n_iters, n_samples, n_top_samples, delta_std,
                 lr, min_lr):
        self.hidden_dim = hidden_layer_dim
        self.action_bound = action_bound
        self.ars_params = ARSParams(n_iters, n_samples, n_top_samples, delta_std, lr, min_lr)

# ==================================================================================================


# env : ProductMDP
# params : HyperParams
# variant : {'normal', 'unshaped', 'state_based', 'deterministic'}
#     'normal' - SPECTRL
#     'unshaped' - Use unshaped rewards
#     'state-based' - State based policy that does not depend on monitor state
#     'deterministic' - Use greedy deterministic monitor
def learn_policy(env, params, variant='normal'):

    # dimension of input vector to nn
    nn_input_dim = env.spec.total_state_dim
    if variant == 'state-based':
        nn_input_dim = env.spec.register_split

    # dimension of output vector of nn
    nn_output_dim = env.spec.total_action_dim
    if variant == 'deterministic':
        nn_output_dim = env.spec.action_dim

    policy_params = NNParams(nn_input_dim, nn_output_dim, params.action_bound,
                             params.hidden_dim, env.spec.monitor.n_states)

    policy = NNPolicy(policy_params)
    if variant == 'state-based':
        policy = NNPolicySimple(policy_params)

    log_info = ars(env, policy, params.ars_params)

    return policy, log_info

# ==================================================================================================


# Print formatted rollout
def print_rollout(env, policy, render=False):
    test_rollout = get_rollout(env, policy, render)
    state_list = [(state[0][:env.spec.state_dim].tolist(),
                   state[0][env.spec.state_dim:env.spec.register_split].tolist(),
                   state[1],
                   state[0][env.spec.register_split:].tolist())
                  for state, _, _, _ in test_rollout]
    print('------------------------------ Rollout ---------------------------------')
    for state in state_list:
        print(str(["{0:0.2f}".format(i) for i in state[0]]).replace("'", "") + ", "
              + str(["{0:0.2f}".format(i) for i in state[1]]).replace("'", "") + ", "
              + str(state[2]) + ", "
              + str(["{0:0.2f}".format(i) for i in state[3]]).replace("'", ""))
