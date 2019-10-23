from spectrl.main.spec_compiler import *
from spectrl.ars.ars import *
from spectrl.examples.util import *
from scipy.stats import truncnorm
from numpy import linalg as LA

import pickle


# Define model of the system
# System of car in 2d with controllable velocity
class VC_Env:
    def __init__(self, time_limit, spec, fuel, std=0.05):
        self.state = np.array([5.0, 0., 7.0]) + truncnorm.rvs(-1, 1, 0, 0.3, 3)
        self.state[2] = 7.0
        self.time_limit = time_limit
        self.time = 0
        self.std = std
        self.fuel = fuel
        self.spec = spec

    def reset(self):
        self.state = np.array([5.0, 0., 7.0]) + truncnorm.rvs(-1, 1, 0, 0.3, 3)
        self.state[2] = 7.0
        self.time = 0
        if self.fuel:
            return self.state
        else:
            return self.state[:2]

    def step(self, action):
        next_state = self.state[:2] + action + truncnorm.rvs(-1, 1, 0, self.std, 2)
        next_state = np.concatenate([next_state, np.array([self.state[2] - 0.1 * abs(self.state[0])
                                                           * LA.norm(action)])])
        self.state = next_state
        self.time = self.time + 1
        if self.fuel:
            return next_state, 0, self.time > self.time_limit, None
        else:
            return next_state[:2], 0, self.time > self.time_limit, None

    def cum_reward(self, rollout):
        return self.spec.quantitative_semantics(rollout, 2)

    def render(self):
        pass

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
spec3 = alw(avoid(obs), seq(ev(reach(gtop, 1.0)), ev(reach(gbot, 1.0))))
spec4 = alw(avoid(obs), seq(ev(lor(reach(gtop, 1.0), reach(gright, 1.0))), ev(reach(gcorner, 1.0))))
spec5 = alw(avoid(obs), seq(ev(reach(gtop, 1.0)), seq(ev(reach(gbot, 1.0)),
                                                      ev(reach(gright, 1.0)))))
spec6 = alw(avoid(obs), seq(ev(reach(gtop, 1.0)), seq(ev(reach(gbot, 1.0)),
                                                      seq(ev(reach(gright, 1.0)),
                                                          ev(reach(gcorner, 1.0))))))
spec7 = alw(avoid(obs), seq(ev(reach(gtop, 1.0)), seq(ev(reach(gbot, 1.0)),
                                                      seq(ev(reach(gright, 1.0)),
                                                          seq(ev(reach(gcorner, 1.0)),
                                                              ev(reach(origin, 1.0)))))))


specs = [spec1, spec2, spec3, spec4, spec5, spec6, spec7]

fuel = [False, True, False, False, False, False, False]

params_list = [ARSParams(1500, 20, 8, 0.04, 0.4),
               ARSParams(1500, 20, 8, 0.04, 0.4),
               ARSParams(1500, 20, 8, 0.04, 0.4),
               ARSParams(1500, 20, 8, 0.04, 0.5),
               ARSParams(2000, 30, 10, 0.04, 0.6),
               ARSParams(10000, 30, 10, 0.04, 0.3),
               ARSParams(15000, 30, 10, 0.04, 0.3)]

# ==================================================================================================

# Construct Product MDP and learn policy
if __name__ == '__main__':
    itno, folder = parse_command_line_options()
    for i in range(len(specs)):
        print('====================== Learning Policy for Spec {} =========================='.
              format(i))
        time_limit = 40
        if i >= 5:
            time_limit = 80
        env = VC_Env(time_limit, specs[i], fuel[i], std=0.05)
        nnparams = NNParams(2, 2, 1, 50)
        if (fuel[i]):
            nnparams = NNParams(3, 2, 1, 50)
        params = params_list[i]
        policy = NNPolicy(nnparams)

        log_info = ars(env, policy, params)
        np.save(folder + '/log_info{}{}'.format(i, itno), log_info)
        policy_file = open(folder + '/policy{}{}.pkl'.format(i, itno), 'wb')
        pickle.dump(policy, policy_file)
        policy_file.close()

        _, succ_rate = test_policy(env, policy, 100)
        sarss = get_rollout(env, policy, False)
        print([state for state, _, _, _ in sarss])
        print('Estimated Success Rate: {}%'.format(succ_rate))
        np.save(folder + '/rollout{}{}'.format(i, itno), sarss)
