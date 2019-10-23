from spectrl.main.monitor import Monitor_Automaton
from spectrl.main.compiler_utility import *

import enum

MAX_PRED_VAL = 1000000.


# Constructors for TaskSpec
class Cons(enum.Enum):
    ev = 1
    alw = 2
    seq = 3
    choose = 4
    ite = 5

# ==================================================================================================


# Specification AST
class TaskSpec:

    # cons: int (Has to refer to a valid constructor in Cons)
    # predicate: np.array(state_dim), np.array(resource_dim) -> float
    # subtasks = [TaskSpec]
    # Some functions in this class assume presence of subtasks as needed,
    # according to syntax of language in the paper
    def __init__(self, cons, predicate, subtasks):
        self.cons = cons
        self.predicate = predicate
        self.subtasks = subtasks

    def quantitative_semantics(self, traj, sys_dim, use_prefix=False):
        sys_traj = [state[:sys_dim] for state in traj]
        res_traj = [state[sys_dim:] for state in traj]
        if use_prefix:
            return self.quantitative_semantics_dp(sys_traj, res_traj)[1][len(traj)-1]
        else:
            return self.quantitative_semantics_dp(sys_traj, res_traj)[0][0]

    # Does not support conditional statements
    # Returns quantitative semnatics for all suffixes (retval[0]) and prefixes (retval[1])
    # suffixes not available once a sequence operator is encountered with first task != ev
    # prefixes not available once a sequence operator is encountered with second task != ev
    def quantitative_semantics_dp(self, sys_traj, res_traj):
        n = len(sys_traj)
        retval = np.zeros((2, n))

        # atomic task
        if self.cons == Cons.ev:
            for i in range(n):
                if i == 0:
                    retval[0][n-1] = self.predicate(sys_traj[n-1], res_traj[n-1])
                    retval[1][0] = self.predicate(sys_traj[0], res_traj[0])
                else:
                    retval[0][n-i-1] = max(retval[0][n-i], self.predicate(sys_traj[n-i-1],
                                                                          res_traj[n-i-1]))
                    retval[1][i] = max(retval[1][i-1], self.predicate(sys_traj[i],
                                                                      res_traj[i]))
            return retval

        # always constraint
        # this function also supports always formala without a subtask
        if self.cons == Cons.alw:
            subval = np.array([[MAX_PRED_VAL]*n]*2)
            if self.subtasks[0] is not None:
                subval = self.subtasks[0].quantitative_semantics_dp(sys_traj, res_traj)
            for i in range(n):
                if i == 0:
                    retval[0][n-1] = self.predicate(sys_traj[n-1], res_traj[n-1])
                    retval[1][0] = self.predicate(sys_traj[0], res_traj[0])
                else:
                    retval[0][n-i-1] = min(retval[0][n-i], self.predicate(sys_traj[n-i-1],
                                                                          res_traj[n-i-1]))
                    retval[1][i] = min(retval[1][i-1], self.predicate(sys_traj[i],
                                                                      res_traj[i]))
            for i in range(n):
                retval[0][i] = min(retval[0][i], subval[0][i])
                retval[1][i] = min(retval[1][i], subval[1][i])
            return retval

        # sequence
        if self.cons == Cons.seq:
            if self.subtasks[0].cons == Cons.ev:
                subval = self.subtasks[1].quantitative_semantics_dp(sys_traj, res_traj)
                for i in range(n):
                    if i == 0:
                        retval[0][n-1] = -MAX_PRED_VAL
                    else:
                        retval[0][n-i-1] = max(retval[0][n-i],
                                               min(self.subtasks[0].predicate(sys_traj[n-i-1],
                                                                              res_traj[n-i-1]),
                                                   subval[0][n-i]))
                        subval[0][n-i-1] = max(subval[0][n-i], subval[0][n-i-1])
            if self.subtasks[1].cons == Cons.ev:
                subval = self.subtasks[0].quantitative_semantics_dp(sys_traj, res_traj)
                for i in range(n):
                    if i == 0:
                        retval[1][0] = -MAX_PRED_VAL
                    else:
                        retval[1][i] = max(retval[1][i-1],
                                           min(self.subtasks[1].predicate(sys_traj[i],
                                                                          res_traj[i]),
                                               subval[1][i-1]))
                        subval[1][i] = max(subval[1][i-1], subval[1][i])
            return retval

        # choice
        if self.cons == Cons.choose:
            subval1 = self.subtasks[0].quantitative_semantics_dp(sys_traj, res_traj)
            subval2 = self.subtasks[1].quantitative_semantics_dp(sys_traj, res_traj)
            for i in range(n):
                retval[0][i] = max(subval1[0][i], subval2[0][i])
                retval[1][i] = max(subval1[1][i], subval2[1][i])
            return retval

    # returns a monitor: Monitor_Automaton
    def get_monitor(self, state_dim, res_dim, local_reward_bound):

        # atomic evetually task
        if self.cons == Cons.ev:

            mpred = monitor_predicate(self.predicate, state_dim)

            def pred_update(state, reg):
                return np.array([mpred(state, reg)[1]])

            def frew(state, reg):
                return reg[0]

            e00 = (0, true_predicate(local_reward_bound), id_update)
            e01 = (1, mpred, pred_update)
            e11 = (1, true_predicate(local_reward_bound), id_update)
            transitions = [[e00, e01], [e11]]
            rewards = [None, frew]

            return Monitor_Automaton(2, 1, state_dim + res_dim, np.array([0.0]), transitions,
                                     rewards)

        # adding safety constraints
        elif self.cons == Cons.alw:

            # construct monitor for sub-formula
            mon = self.subtasks[0].get_monitor(state_dim, res_dim, local_reward_bound)

            # add saftey constraint
            mpred = monitor_predicate(self.predicate, state_dim)

            n_states = mon.n_states
            n_registers = mon.n_registers + 1
            input_dim = mon.input_dim
            init_reg = np.concatenate([mon.init_registers, np.array([local_reward_bound])])

            transitions = []
            for ve in mon.transitions:
                ve_new = []
                for (q, p, u) in ve:
                    ve_new.append((q,
                                   project_predicate(p, 0, mon.n_registers),
                                   alw_update(u, mpred)))
                transitions.append(ve_new)

            rewards = []
            for rew in mon.rewards:
                if rew is None:
                    rewards.append(None)
                else:
                    rewards.append(alw_reward(rew))

            return Monitor_Automaton(n_states, n_registers, input_dim, init_reg, transitions,
                                     rewards)

        # sequence
        elif self.cons == Cons.seq:

            # construct monitors for subformulas
            mon1 = self.subtasks[0].get_monitor(state_dim, res_dim, local_reward_bound)
            mon2 = self.subtasks[1].get_monitor(state_dim, res_dim, local_reward_bound)

            # construct monitor for sequence
            n_states = mon1.n_states + mon2.n_states
            n_registers = max(mon1.n_registers, mon2.n_registers + 1)
            input_dim = mon1.input_dim
            init_reg = np.zeros(n_registers)
            init_reg[:mon1.n_registers] = mon1.init_registers

            transitions = []
            for qu in range(mon1.n_states):
                ve = mon1.transitions[qu]
                ve_new = []

                # Delta1
                for (qv, p, u) in ve:
                    ve_new.append((qv,
                                   project_predicate(p, 0, mon1.n_registers),
                                   project_update(u, 0, mon1.n_registers)))

                # Delta1->2
                if mon1.rewards[qu] is not None:
                    for (qv, p, u) in mon2.transitions[0]:
                        q2 = qv + mon1.n_states
                        p2 = rew_pred(p, mon1.rewards[qu], mon2.init_registers, 0, mon1.n_registers)
                        u2 = seq_update(n_registers, mon1.n_registers, mon2.n_registers,
                                        mon2.init_registers, mon1.rewards[qu], u)
                        ve_new.append((q2, p2, u2))

                transitions.append(ve_new)

            for ve in mon2.transitions:
                ve_new = []
                for (q, p, u) in ve:
                    ve_new.append((q + mon1.n_states,
                                   project_predicate(p, 0, mon2.n_registers),
                                   project_update(u, 0, mon2.n_registers)))
                transitions.append(ve_new)

            rewards = [None]*mon1.n_states
            for rew in mon2.rewards:
                if rew is not None:
                    rewards.append(seq_reward(rew, mon2.n_registers))
                else:
                    rewards.append(None)

            return Monitor_Automaton(n_states, n_registers, input_dim, init_reg, transitions,
                                     rewards)

        # choice
        elif self.cons == Cons.choose:

            # Construct monitors for subformulas
            mon1 = self.subtasks[0].get_monitor(state_dim, res_dim, local_reward_bound)
            mon2 = self.subtasks[1].get_monitor(state_dim, res_dim, local_reward_bound)

            # combine
            # initial state is merged, state numbers of first monitor do not change
            n_states = mon1.n_states + mon2.n_states - 1
            n_registers = mon1.n_registers + mon2.n_registers
            input_dim = mon1.input_dim
            init_reg = np.concatenate([mon1.init_registers, mon2.init_registers])

            transitions = []

            # Delta0
            # us[0] stores loop update for the first monitor, us[1] for second monitor
            us = []
            # Set of transitions from initial state: None is used as a placeholder for self loop
            trans_init = [None]
            for (q1, p1, u1) in mon1.transitions[0]:
                if q1 == 0:
                    us.append(u1)
                else:
                    trans_init.append((q1,
                                       project_predicate(p1, 0, mon1.n_registers),
                                       project_update(u1, 0, mon1.n_registers, clean=True)))

            for (q2, p2, u2) in mon2.transitions[0]:
                if q2 == 0:
                    us.append(u2)
                else:
                    trans_init.append((q2 + mon1.n_states - 1,
                                       project_predicate(p2, mon1.n_registers, n_registers),
                                       project_update(u2, mon1.n_registers, n_registers,
                                                      clean=True)))

            def loop_update(state, reg):
                return np.concatenate([us[0](state, reg[0:mon1.n_registers]),
                                       us[1](state, reg[mon1.n_registers:n_registers])])
            trans_init[0] = (0, true_predicate(local_reward_bound), loop_update)
            transitions.append(trans_init)

            # Delta1: Add all transitions in monitor 1
            for ve in mon1.transitions[1:]:
                ve_new = []
                for (q, p, u) in ve:
                    ve_new.append((q,
                                   project_predicate(p, 0, mon1.n_registers),
                                   project_update(u, 0, mon1.n_registers)))
                transitions.append(ve_new)

            # Delta2: Add all transitions in monitor 2
            for ve in mon2.transitions[1:]:
                ve_new = []
                for (q, p, u) in ve:
                    ve_new.append((q + mon1.n_states - 1,
                                   project_predicate(p, mon1.n_registers, n_registers),
                                   project_update(u, mon1.n_registers, n_registers)))
                transitions.append(ve_new)

            rewards = [None]
            for rew in mon1.rewards[1:]:
                if rew is not None:
                    rewards.append(project_reward(rew, 0, mon1.n_registers))
                else:
                    rewards.append(None)

            for rew in mon2.rewards[1:]:
                if rew is not None:
                    rewards.append(project_reward(rew, mon1.n_registers, n_registers))
                else:
                    rewards.append(None)

            return Monitor_Automaton(n_states, n_registers, input_dim, init_reg, transitions,
                                     rewards)

        # conditional
        else:

            # Construct monitors for sub formulas
            mon1 = self.subtasks[0].get_monitor(state_dim, res_dim, local_reward_bound)
            mon2 = self.subtasks[1].get_monitor(state_dim, res_dim, local_reward_bound)

            b = monitor_predicate(self.predicate, state_dim)
            notb = neg_pred(b)

            # Combine monitors
            n_states = mon1.n_states + mon2.n_states + 1
            n_registers = max(mon1.n_registers, mon2.n_registers)
            input_dim = mon1.input_dim
            init_reg = np.zeros(n_registers)

            transitions = []

            # Delta0
            trans_init = []

            for (q, p, u) in mon1.transitions[0]:
                trans_init.append((q+1,
                                   conj_pred(p, b, mon1.init_registers),
                                   init_update(u, mon1.init_registers)))
            for (q, p, u) in mon2.transitions[0]:
                trans_init.append((q + mon1.n_states + 1,
                                   conj_pred(p, notb, mon2.init_registers),
                                   init_update(u, mon2.init_registers)))
            transitions.append(trans_init)

            # Delta1

            for ve in mon1.transitions:
                ve_new = []
                for (q, p, u) in ve:
                    ve_new.append((q+1,
                                   project_predicate(p, 0, mon1.n_registers),
                                   project_update(u, 0, mon1.n_registers)))
                transitions.append(ve_new)

            # Delta2

            for ve in mon2.transitions:
                ve_new = []
                for (q, p, u) in ve:
                    ve_new.append((q + mon1.n_states + 1,
                                   project_predicate(p, 0, mon2.n_registers),
                                   project_update(u, 0, mon2.n_registers)))
                transitions.append(ve_new)

            rewards = [None]

            for rew in mon1.rewards:
                if rew is None:
                    rewards.append(None)
                else:
                    rewards.append(project_reward(rew, 0, mon1.n_registers))

            for rew in mon2.rewards:
                if rew is None:
                    rewards.append(None)
                else:
                    rewards.append(project_reward(rew, 0, mon2.n_registers))

            return Monitor_Automaton(n_states, n_registers, input_dim, init_reg, transitions,
                                     rewards)

# ==================================================================================================
# API for building Task Specifications


# Logical OR
# p1, p2 : np.array([state_dim]) * np.array([res_dim]) -> Float
def lor(p1, p2):
    def p(sys_state, res_state):
        return max(p1(sys_state, res_state), p2(sys_state, res_state))
    return p


# Logical AND
# p1, p2 : np.array([state_dim]) * np.array([res_dim]) -> Float
def land(p1, p2):
    def p(sys_state, res_state):
        return min(p1(sys_state, res_state), p2(sys_state, res_state))
    return p


# Atomic eventually task
# p : np.array([state_dim]) * np.array([res_dim]) -> Float
def ev(p):
    return TaskSpec(Cons.ev, p, [])


# Safety constraints
# p : np.array([state_dim]) * np.array([res_dim]) -> Float
# phi : TaskSpec
def alw(p, phi):
    return TaskSpec(Cons.alw, p, [phi])


# Sequence of tasks
# phi1, phi2 : TaskSpec
def seq(phi1, phi2):
    return TaskSpec(Cons.seq, None, [phi1, phi2])


# Choice of tasks
# phi1, phi2 : TaskSpec
def choose(phi1, phi2):
    return TaskSpec(Cons.choose, None, [phi1, phi2])


# Conditional
# p : np.array([state_dim]) * np.array([res_dim]) -> Float
# phi1, phi2 : TaskSpec
def ite(p, phi1, phi2):
    return TaskSpec(Cons.ite, p, [phi1, phi2])
