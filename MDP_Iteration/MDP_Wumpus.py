import numpy as np
import argparse


class WumpusWorld:
    def __init__(self):
        self.grid_size = 4
        self.num_states = self.grid_size * self.grid_size
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.rewards = self._initialize_rewards()
        # print('rewards is:', self.rewards)
        self.transition_probabilities = self._initialize_transition_probabilities()
        # print('transition_probilities is:', self.transition_probabilities)

    def _initialize_rewards(self):
        rewards = np.full((self.grid_size, self.grid_size), -0.4)
        rewards[0, 3] = 10  # Gold position
        rewards[2, 2] = -5   # Pit position
        rewards[3, 2] = -5   # Pit position
        rewards[3, 1] = -10  # Wumpus position
        return rewards

    def _initialize_transition_probabilities(self):
        # Transition probabilities for each action
        transition_probs = {}
        for action in self.actions:
            transition_probs[action] = np.zeros((self.num_states, self.num_states))
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                state = i * self.grid_size + j
                for action in self.actions:
                    for next_action in self.actions:
                        if action == next_action:
                            prob = 0.8
                        elif (self._orthogonal(action, next_action)):
                            prob = 0.1
                        else:
                            continue
                        
                        next_state = self._move(i, j, next_action)
                        if next_state is not None:
                            transition_probs[action][state, next_state] = prob
        
        return transition_probs

    def _orthogonal(self, action1, action2):
        return (action1 in ['UP', 'DOWN'] and action2 in ['LEFT', 'RIGHT']) or \
               (action1 in ['LEFT', 'RIGHT'] and action2 in ['UP', 'DOWN'])

    def _move(self, i, j, action):
        if action == 'UP' and i > 0:
            return (i - 1) * self.grid_size + j
        elif action == 'DOWN' and i < self.grid_size - 1:
            return (i + 1) * self.grid_size + j
        elif action == 'LEFT' and j > 0:
            return i * self.grid_size + (j - 1)
        elif action == 'RIGHT' and j < self.grid_size - 1:
            return i * self.grid_size + (j + 1)
        return None

    def get_reward(self, state, action, next_state):
        reward = self.rewards[next_state // self.grid_size, next_state % self.grid_size]
        return reward
    
    def get_transition_prob(self, state, action, next_state):
        return self.transition_probabilities[action][state, next_state]

    def MDP_value_iteration(self, gamma, eta, max_iter):
        V = np.zeros(self.num_states)
        for i in range (max_iter): #Bellman 1957 value iteration - lecture notes
            V_previous = np.copy (V) #t-1
            for state in range (self.num_states):
                action_values = []
                for action in self.actions:
                    action_value = sum(
                        self.get_transition_prob(state, action, next_state) * 
                        (self.get_reward(state, action, next_state) + gamma * V_previous[next_state])
                        for next_state in range(self.num_states)
                    )
                    action_values.append(action_value)
                chosen_V = max(action_values)
                V[state] = chosen_V
            if np.max(np.abs(V - V_previous)) < eta:
                break

        return V

    def MDP_policy_iteration(self, gamma, eta, max_iter):
        policy = np.random.choice(self.actions, size=self.num_states)
        V = np.zeros(self.num_states)
        for _ in range(max_iter):
            V = np.zeros(self.num_states)
            # Policy Evaluation - Iterate until value converges
            while True:
                V_prev = np.copy(V)
                for state in range(self.num_states):
                    action = policy[state]
                    V[state] = sum(
                        self.get_transition_prob(state, action, next_state) * 
                        (self.get_reward(state, action, next_state) + gamma * V_prev[next_state])
                        for next_state in range(self.num_states)
                    )
                if np.max(np.abs(V - V_prev)) < eta:
                    break
            # Policy Improvement
            policy_converges = True
            for state in range (self.num_states):
                previous_action = policy[state]
                action_values = []
                for action in self.actions:
                    action_value = sum(
                        self.get_transition_prob(state, action, next_state) * 
                        (self.get_reward(state, action, next_state) + gamma * V[next_state])
                        for next_state in range(self.num_states)
                    )
                    action_values.append(action_value)
                    
                new_updated_policy = self.actions[np.argmax(action_values)]
                policy [state] = new_updated_policy
                if previous_action != policy[state]:
                    policy_converges = False

            if policy_converges:
                    break

        return V, policy
    
    def MDP_policy(self, V, gamma):
        # policy[s] is the best action to take in state s, firstly set it to 0 for all states
        policy = np.random.choice(self.actions, size=self.num_states)
        for state in range(self.num_states):
            action_values = []
            for action in self.actions:
                action_value = sum(
                    self.get_transition_prob(state, action, next_state) * 
                    (self.get_reward(state, action, next_state) + gamma * V[next_state])
                    for next_state in range(self.num_states)
                )
                action_values.append(action_value)
            policy[state] = self.actions[np.argmax(action_values)]
        return policy


def main(gamma, eta, max_iter):
    wumpus_world = WumpusWorld()
    
    # Value Iteration
    print(">>>>>>Running Value Iteration...")
    V_value = wumpus_world.MDP_value_iteration(gamma, eta, max_iter)
    print("Value Function (Value Iteration):\n", V_value.reshape(wumpus_world.grid_size, -1))
    print('Policy is:\n', wumpus_world.MDP_policy(V_value, gamma=gamma).reshape(wumpus_world.grid_size, -1))
    print(">>>>>-----------------------------")

    # Policy Iteration
    print(">>>>>>Running Policy Iteration...")
    V_policy, policy = wumpus_world.MDP_policy_iteration(gamma, eta, max_iter)
    print("Value Function (Policy Iteration):\n", V_policy.reshape(wumpus_world.grid_size, -1))
    print("Policy is:\n", policy.reshape(wumpus_world.grid_size, -1))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, required=True, help='Discount factor')
    parser.add_argument('--eta', type=float, required=True, help='Convergence threshold')
    parser.add_argument('--e', type=int, required=True, help='Maximum iterations')
    args = parser.parse_args()
    
    main(args.gamma, args.eta, args.e)