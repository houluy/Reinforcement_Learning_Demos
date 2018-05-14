from Q.Q import Q

def gen_st(s, k, K, i):
    new = []
    if i == K:
        for j in range(k + 1):
            s[i] = j
            new.append(s[:])
        return new
    else:
        for j in range(k + 1):
            s[i] = j
            new += gen_st(s, k, K, i + 1)
        return new

class ProactiveCache:
    def __init__(self):
        self.k = 3
        self.K = 3
        self.V = 10
        self.C = 3
        self.L = 300
        self.u = self.L/self.C
        self.nu = 10
        self.mu = 100
        self.rho = 0.9
        self.zeta = 0.5
        self.req_air = 10
        self.req_line = 1000
        self.req_time = self.req_air + self.req_line
        self.generate_states()

    def generate_states(self):
        state = [0 for x in range(self.K + 1)]
        self.space = gen_st(state, self.k, self.K, 0)
        self.state_count = len(self.space)
        self.init_state = self.space[0]

    def available_actions(self, state):
        actions = [(0, 0)]
        for rsu, s in enumerate(state):
            if s < self.K:
                for a in range(1, self.K - s + 1):
                    actions.append((rsu, a))
        return actions

    def reward(self, state, action):
        self.zeta*self.time_ell(action) + (1 - self.zeta)*self.u*self.occupation(state=state, action=action)

    def transit(self, state, action):
        if action[1] == 0:
            return state
        else:
            state[action[0]] += action[1]
            return state[:]

    def occupation(self, state, action):
        return sum(state) + action[1]

    def time_ell(self, action):
        if action[1] == 0:
            return self.t_m
        else:
            return self.rho*self.t_h + (1 - self.rho)*self.t_m

    @property
    def t_m(self):
        return 2*self.req_time + self.L/self.nu

    def t_h(self, action):
        return 2*self.req_air + self.req_line + (action[1]*self.u)/self.mu + (self.L - self.u*action[1])/self.nu

    @property
    def action_set(self):
        return self.available_action(self.init_state)

class Adaptor(ProactiveCache):
    def __init__(self):
        super().__init__()
        self.Q = Q(
            state_set=self.space,
            action_set=self.action_set,
            available_actions=self.available_actions,
            reward_func=self.reward,
            transition_func=self.transit
        )