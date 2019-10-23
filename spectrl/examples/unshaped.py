from spectrl.main.learning import *
from spectrl.main.spec_compiler import *
from spectrl.examples.util import *
from scipy.stats import truncnorm
from numpy import linalg as LA

import pickle


# Define model of the system
# System of car in 2d with controllable velocity
class VC_Env:
    def __init__(self, time_limit, std=0.5):
        self.state = np.array([5.0, 0.]) + truncnorm.rvs(-1, 1, 0, 0.3, 2)
        self.time_limit = time_limit
        self.time = 0
        self.std = std

    def reset(self):
        self.state = np.array([5.0, 0.]) + truncnorm.rvs(-1, 1, 0, 0.3, 2)
        self.time = 0
        return self.state

    def step(self, action):
        next_state = self.state + action + truncnorm.rvs(-1, 1, 0, self.std, 2)
        self.state = next_state
        self.time = self.time + 1
        return next_state, 0, self.time > self.time_limit, None

    def render(self):
        pass


# Define the resource model
# Fuel consumption proportional to distance from x-axis and the velocity
# sys_state: np.array(2)
# res_state: np.array(1)
# sys_action: np.array(2)
def resource_delta(sys_state, res_state, sys_action):
    return np.array([res_state[0] - 0.1 * abs(sys_state[0]) * LA.norm(sys_action)])

# ==================================================================================================


# Define the specification
# 1. Relevant atomic predicates:
# a. Reach predicate
#    goal: np.array(2), err: float
def reach(goal, err):
    def predicate(sys_state, res_state):
        return (min([sys_state[0] - goal[0],
                     goal[0] - sys_state[0],
                     sys_state[1] - goal[1],
                     goal[1] - sys_state[1]]) + err)
    return predicate


# b. Avoid predicate
#    obstacle: np.array(4): [x_min, y_min, x_max, y_max]
def avoid(obstacle):
    def predicate(sys_state, res_state):
        return max([obstacle[0] - sys_state[0],
                    obstacle[1] - sys_state[1],
                    sys_state[0] - obstacle[2],
                    sys_state[1] - obstacle[3]])
    return predicate


def have_fuel(sys_state, res_state):
    return res_state[0]


# Goals and obstacles
gtop = np.array([5.0, 10.0])
gbot = np.array([5.0, 0.0])
gright = np.array([10.0, 0.0])
gcorner = np.array([10.0, 10.0])
gcorner2 = np.array([0.0, 10.0])
origin = np.array([0.0, 0.0])
obs = np.array([4.0, 4.0, 6.0, 6.0])

# Specifications
spec1 = alw(avoid(obs), ev(reach(gtop, 1.0)))
spec2 = alw(avoid(obs), alw(have_fuel, ev(reach(gtop, 1.0))))
spec3 = seq(alw(avoid(obs), ev(reach(gtop, 1.0))), alw(avoid(obs), ev(reach(gbot, 1.0))))
spec4 = seq(choose(alw(avoid(obs), ev(reach(gtop, 1.0))), alw(avoid(obs), ev(reach(gright, 1.0)))),
            alw(avoid(obs), ev(reach(gcorner, 1.0))))
spec5 = seq(spec3, alw(avoid(obs), ev(reach(gright, 1.0))))
spec6 = seq(spec5, alw(avoid(obs), ev(reach(gcorner, 1.0))))
spec7 = seq(spec6, alw(avoid(obs), ev(reach(origin, 1.0))))

specs = [spec1, spec2, spec3, spec4, spec5, spec6, spec7]

params_list = [HyperParams(30, 1, 1500, 20, 8, 0.1, 1, 0.2),
               HyperParams(30, 1, 1500, 20, 8, 0.05, 0.6, 0.1),
               HyperParams(30, 1, 1500, 20, 8, 0.05, 0.5, 0.1),
               HyperParams(30, 1, 1500, 20, 8, 0.05, 0.6, 0.1),
               HyperParams(30, 1, 2000, 30, 10, 0.03, 0.8, 0.1),
               HyperParams(30, 1, 10000, 30, 8, 0.02, 0.2, 0.05),
               HyperParams(30, 1, 15000, 30, 5, 0.01, 0.2, 0.05)]

lb = [10., 20., 10., 10., 10., 9., 9.]

f = [False, True, False, False, False, False, False]

# ==================================================================================================

# Construct Product MDP and learn policy
if __name__ == '__main__':
    itno, folder = parse_command_line_options()
    for i in range(len(specs)):
        print('====================== Learning Policy for Spec {} =========================='.
              format(i))

        # Step 1: initialize system environment
        time_limit = 40
        if i >= 5:
            time_limit = 80
        system = VC_Env(time_limit, std=0.05)

        # Step 2 (optional): construct resource model
        resource = Resource_Model(2, 2, 1, np.array([7.0]), resource_delta)

        # Step 3: construct product MDP
        env = ProductMDP(system, 2, specs[i], 0.0, lb[i],
                         res_model=resource,
                         use_shaped_rewards=False) if f[i] else ProductMDP(system, 2, specs[i],
                                                                           0.0, lb[i],
                                                                           use_shaped_rewards=False)

        # Step 4: Set hyper parameters
        params = params_list[i]

        # Step 5: Learn policy
        policy, log_info = learn_policy(env, params)

        # Save policy and log information
        np.save(folder + '/log_info{}{}'.format(i, itno), log_info)
        policy_file = open(folder + '/policy{}{}.pkl'.format(i, itno), 'wb')
        pickle.dump(policy, policy_file)
        policy_file.close()

        # Print rollout and performance
        print_rollout(env, policy)
        _, succ_rate = test_policy(env, policy, 100)
        print('Estimated Satisfaction Rate: {}%'.format(succ_rate))
        rollout = get_rollout(env, policy, False)
        np.save(folder + '/rollout{}{}'.format(i, itno), rollout)
