import numpy as np
import graphics
import rover

def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states
    all_possible_observed_states: a list of possible observed states
    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    rover.py), and the i-th Distribution should correspond to time
    step i
    """

    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    forward_messages[0] = prior_distribution
    backward_messages = [None] * num_time_steps
    marginals = [None] * num_time_steps 
    
    # TODO: Compute the forward messages
    forward_messages[0] = rover.Distribution()
    
    # Initialization Step: alpha_0 = p(z_0) * p(x_0 | z_0)
    for state in all_possible_hidden_states:
        prob = prior_distribution[state]
        if observations[0] is not None:
            prob *= observation_model(state)[observations[0]]
        forward_messages[0][state] = prob
        
    forward_messages[0].renormalize()

    # Recursive Step: alpha_i = p(x_i | z_i) * sum(alpha_{i-1} * p(z_i | z_{i-1}))
    for i in range(1, num_time_steps):
        forward_messages[i] = rover.Distribution()
        
        for current_state in all_possible_hidden_states:
            sum_prob = 0.0
            
            # Sum over all possible previous states
            for prev_state in all_possible_hidden_states:
                trans_prob = transition_model(prev_state)[current_state]
                if trans_prob > 0:
                    sum_prob += forward_messages[i-1][prev_state] * trans_prob
            
            # Multiply by observation probability (if observation exists)
            obs_prob = 1.0
            if observations[i] is not None:
                obs_prob = observation_model(current_state)[observations[i]]
                
            forward_messages[i][current_state] = sum_prob * obs_prob
            
        forward_messages[i].renormalize()
                   
    # TODO: Compute the backward messages
    backward_messages[-1] = rover.Distribution()
    
    # Initialization Step: beta_{N-1} = 1
    for state in all_possible_hidden_states:
        backward_messages[-1][state] = 1.0
        
    # Recursive Step: beta_i = sum(beta_{i+1} * p(z_{i+1} | z_i) * p(x_{i+1} | z_{i+1}))
    for i in range(num_time_steps - 2, -1, -1):
        backward_messages[i] = rover.Distribution()
        
        for current_state in all_possible_hidden_states:
            sum_prob = 0.0
            trans_dist = transition_model(current_state)
            
            # Sum over all possible next states
            for next_state in all_possible_hidden_states:
                trans_prob = trans_dist[next_state]
                if trans_prob > 0:
                    obs_prob = 1.0
                    if observations[i+1] is not None:
                        obs_prob = observation_model(next_state)[observations[i+1]]
                    sum_prob += backward_messages[i+1][next_state] * trans_prob * obs_prob
                    
            backward_messages[i][current_state] = sum_prob
            
        backward_messages[i].renormalize()
    
    # TODO: Compute the marginals 
    # Marginal distribution: gamma_i = alpha_i * beta_i
    for i in range(num_time_steps):
        marginals[i] = rover.Distribution()
        for state in all_possible_hidden_states:
            marginals[i][state] = forward_messages[i][state] * backward_messages[i][state]
        marginals[i].renormalize()
            
    return marginals

def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esitmated hidden states, each state is encoded as a tuple
    (<x>, <y>, <action>)
    """

    # TODO: Write your code here
    num_time_steps = len(observations)
    max_probs = [rover.Distribution() for _ in range(num_time_steps)]
    backpointers = [{} for _ in range(num_time_steps)]
    
    # Initialization Step: omega_0 = p(z_0) * p(x_0 | z_0)
    for state in all_possible_hidden_states:
        prob = prior_distribution[state]
        if observations[0] is not None:
            prob *= observation_model(state)[observations[0]]
        max_probs[0][state] = prob
        
    max_probs[0].renormalize()

    # Recursive Step: omega_i = p(x_i | z_i) * max(omega_{i-1} * p(z_i | z_{i-1}))
    for i in range(1, num_time_steps):
        for current_state in all_possible_hidden_states:
            best_prob = -1.0
            best_prev_state = None
            
            obs_prob = 1.0
            if observations[i] is not None:
                obs_prob = observation_model(current_state)[observations[i]]
                
            # Find the previous state that yields the maximum probability
            for prev_state in all_possible_hidden_states:
                trans_prob = transition_model(prev_state)[current_state]
                if trans_prob > 0:
                    prob = max_probs[i-1][prev_state] * trans_prob * obs_prob
                    if prob > best_prob:
                        best_prob = prob
                        best_prev_state = prev_state
                        
            max_probs[i][current_state] = best_prob
            backpointers[i][current_state] = best_prev_state
            
        max_probs[i].renormalize()

    # Backtracking Step: Follow the backpointers from the best final state
    estimated_states = [None] * num_time_steps
    best_final_state = max_probs[-1].get_mode()
    estimated_states[-1] = best_final_state
    
    for i in range(num_time_steps - 1, 0, -1):
        estimated_states[i-1] = backpointers[i][estimated_states[i]]

    return estimated_states


if __name__ == '__main__':
   
    enable_graphics = True 
    missing_observations = True 
    
    if missing_observations:
        filename = 'test_missing.txt'
    else:
        filename = 'test.txt'
            
    # load data    
    hidden_states, observations = rover.load_data(filename)
    num_time_steps = len(hidden_states)

    all_possible_hidden_states   = rover.get_all_hidden_states()
    all_possible_observed_states = rover.get_all_observed_states()
    prior_distribution           = rover.initial_distribution()
    
    print('Running forward-backward...')
    marginals = forward_backward(all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 rover.transition_model,
                                 rover.observation_model,
                                 observations)
    print('\n')

    timestep = num_time_steps - 1
    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')

    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               rover.transition_model,
                               rover.observation_model,
                               observations)
    print('\n')
    
    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        print(estimated_states[time_step])
  
    if enable_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()