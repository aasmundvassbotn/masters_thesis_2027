import numpy as np
from .ddpg_torch import Agent, MPC_Agent
from .environments import ClosedField_v22, ClosedField_v23_dyna, MPC_environment_v40
from .utils import plotLearning

def v22_training(episodes=5000, sim_dt=0.1,decision_dt=0.1, save_folder="DDPG/checkpoints/v22_5", loadfolder="DDPG/checkpoints/v22", plot_file = 'DDPG/plots/openfield_v22.png', environment="four_walls", add_noise=True,
                  store_plot_data=None, flip_noise_off=False, flip_episode=None, add_disturbance=None, disturbance_episode=None):
    """
    Here, we added knowledge of previous action, and punish for jerk.
    """
    actual_collisions_during_training = 0
    env = ClosedField_v22(sim_dt=sim_dt, decision_dt=decision_dt, render=False, v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5, horizon=200, episode_s=100, environment_selection=environment)
    best_score = -10000 # Impossibly bad
    collision_history = np.zeros((episodes,))
    agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[42], tau=0.1, env=env, #alpha=0.000025, beta=0.00025, tau=0.001
                batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2, chkpt_dir=save_folder)
    np.random.seed(42)
    if loadfolder:
        print("Loading model from:'" + loadfolder+"'")
        agent.load_models(load_directory=loadfolder)
    actual_collisions_during_training
    score_history = []
    for i in range(episodes):
        if flip_noise_off and i == flip_episode:
            add_noise = False
        obs = env.reset()
        done = False
        score = 0
        episode_lenght = 0
        while not done:
            act = agent.choose_action(obs, add_noise=add_noise)
            if add_disturbance and i > disturbance_episode:
                new_state, reward, done, info = env.step(act, add_disturbance=add_disturbance)
            else:
                new_state, reward, done, info = env.step(act)
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            score += reward
            obs = new_state
            #env.render()
            episode_lenght += 1

        score_history.append(score)
        
        # Store best average models
        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
        
        if info == "'Collided'":
            actual_collisions_during_training +=1
            collision_history[i] = 1

        print('episode ', i, 'score %.2f' % score,
            'trailing 100 games avg %.3f' % np.mean(score_history[-100:]), "ep_lenght:", episode_lenght, "info:", info)
        print("Total collisions;", actual_collisions_during_training)
        if i % 100 == 0:
            plotLearning(score_history, plot_file, window=100)
            if store_plot_data:
                np.savetxt('DDPG/plots/'+store_plot_data+'_ch.txt', collision_history, fmt='%d')
                np.savetxt('DDPG/plots/'+store_plot_data+'_sh.txt', np.asarray(score_history))

    plotLearning(score_history, plot_file, window=100)
    if store_plot_data:
        np.savetxt('DDPG/plots/'+store_plot_data+'_ch.txt', collision_history, fmt='%d')
        np.savetxt('DDPG/plots/'+store_plot_data+'_sh.txt', np.asarray(score_history))
    
def v23_training(episodes=5000, episode_s=20, sim_dt=0.1, decision_dt=0.5, save_folder="DDPG/checkpoints/v23",
                  loadfolder="DDPG/checkpoints/v22", plot_folder = 'DDPG/plots/openfield_v23.png', add_noise=False, reverse_ok=False, total_goals = 5):
    """
    Here, we added knowledge of previous action, and punish for jerk.
    """
    
    env = ClosedField_v23_dyna(sim_dt=sim_dt, decision_dt=decision_dt, render=False, v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5, horizon=200, episode_s=episode_s,
                               reverse_ok=reverse_ok, total_goals=total_goals)
    best_score = -10000 # Impossibly bad

    agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[42], tau=0.1, env=env, #alpha=0.000025, beta=0.00025, tau=0.001
                batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2, chkpt_dir=save_folder)
    np.random.seed(42)
    if loadfolder:
        print("Loading model from:'" + loadfolder+"'")
        agent.load_models(load_directory=loadfolder)

    score_history = []
    for i in range(episodes):
        obs = env.reset()
        done = False
        score = 0
        episode_lenght = 0
        while not done:
            act = agent.choose_action(obs, add_noise=add_noise)
            new_state, reward, done, info = env.step(act)
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            score += reward
            obs = new_state
            #env.render()
            episode_lenght += 1

        score_history.append(score)
        
        # Store best average models
        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode ', i, 'score %.2f' % score,
            'trailing 100 games avg %.3f' % np.mean(score_history[-100:]), "ep_lenght:", episode_lenght, "info:", info)

    plotLearning(score_history, plot_folder, window=100)

def v40_MPC_IRL_training(episodes=5000, sim_dt=0.05, decision_dt=0.5, plotting = 'DDPG/plots/mpc_v40.png', save_folder="DDPG/checkpoints/v40", loadfolder="DDPG/checkpoints/v22_5",
                        environment_selection="four_walls", add_noise=True, collision_rejection=False, include_collision_state = False, reset_after_collision_avoidance = True,
                        store_plot_data=None, flip_noise_off=False, flip_episode=None, add_disturbance=None):
    """ 
    As opposed to _1, this training does -no- training of the hallucination, but allows collision courses to pass!
    Could also be called the "IRL" trainer!
    It does all its predictions / moves in the hallucinations, and then - after executing them IRL it gets a reward and learns!
    """

    ###############################################################################################################
    # Parameters
    #times = np.int32(decision_dt/sim_dt)
    v_max=20
    v_min=-4
    alpha_max=0.5
    tau_steering=0.5
    tau_throttle=0.5
    best_score = -20000 # Impossibly bad
    # Initialization
    env = MPC_environment_v40(sim_dt=sim_dt, decision_dt=decision_dt, render=False, v_max=v_max, v_min=v_min,
	       alpha_max=alpha_max, tau_steering=tau_steering, tau_throttle=tau_throttle, horizon=200, edge=150,
		   episode_s=60, mpc=True, environment_selection=environment_selection)
    agent = MPC_Agent(alpha=0.000025, beta=0.00025, input_dims=[42], tau=0.1, env=env, 
            batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2, chkpt_dir=save_folder)
    
    if loadfolder:
        print("Loading model from:'" + loadfolder+"'")
        agent.load_models(load_directory=loadfolder)
    ###############################################################################################################
    # Sets certain parameters
    trajectory_length = np.int32(3.0/decision_dt) # to get 3 second trajectories
    # Book-keeping during training
    score_history = []
    collision_history = np.zeros((episodes,))
    rejection_history = np.zeros((episodes,))
    ###############################################################################################################
    """ One episode"""
    for e in range(episodes):
        # Should noice be turned off?
        if flip_noise_off and e == flip_episode:
            add_noise = False
        ##############################
        # Readying for a new episode #
        ##############################
        current_IRL_state = env.reset()
        done = False
        score = 0
        episode_lenght = 0
        update_vision = True # need to make initial update
        reset_episode_flag = False
        rejected_trajectories = 0
        #####################
        # Start the episode #
        #####################
        while not done:
            #################################
            # If vision box needs an update #
            #################################
            while update_vision and not reset_episode_flag: 
                # Do not allow colliding trajectories
                # 'Collided' deciedes if the trajectory is going to collide.
                # Should "include collision state = TRUE" be included? to learn from collisions?"
                action_queue, _, _, halu_d2s, states, collided = \
                    env.hallucinate(trajectory_length, sim_dt, decision_dt, agent, add_noise=add_noise, goal_stop=False,
                                     collision_stop=collision_rejection, include_collision_state=include_collision_state)
                #
                update_vision = False
                # There was a collision in the plan
                if include_collision_state and collided:
                    rejected_trajectories += 1
                    # 1
                    reward = -500 # Collision
                    s_next = states.pop()
                    action = action_queue.pop()
                    agent.remember(state=states[-1], action=action, reward=reward,
                                new_state=s_next, done=int(True))
                    agent.learn()
                    # 2
                    discount = agent.gamma
                    while len(states)>1:
                        reward *= discount
                        s_next = states.pop()
                        action = action_queue.pop()
                        agent.remember(state=states[-1], action=action, reward=reward,
                                new_state=s_next, done=int(True))
                        agent.learn()

                    
                    # Reset the episode
                    if reset_after_collision_avoidance:
                        update_vision = False
                        reset_episode_flag = True
                    # Try again
                    else:
                        update_vision = True
            
            if reset_episode_flag:
                done = True
                reward = -500
                info = "'Avoid and reset episode'"
            else:
                ####################################
                # Do we need to update trajectory? #
                ####################################
                real_circogram = env.SC
                _, d2, _, _, _ = real_circogram
                d2_halu = halu_d2s.popleft() # Retrieve believed/halucinated SC from trajectory
                update_vision = env.update_vision(len(action_queue), d2, d2_halu)

                #############################
                # Select and execute action #
                #############################
                act = action_queue.popleft()

                ###################
                # Normal training #
                ###################
                next_IRL_state, reward, done, info = env.step(act, add_disturbance=add_disturbance)
                halu_state = states.popleft()

                # Remember the transition
                if len(states)>0:
                    agent.remember(state=halu_state, action=act, reward=reward,
                                    new_state=halu_state[0], done=int(done))
                # Learn from replay buffer, given batch size
                agent.learn()

            ################
            # Book-keeping #
            ################
            score += reward
            current_IRL_state = next_IRL_state
            #env.render()
            episode_lenght += 1

        ####################################
        ## BOOK-Keeping from the training ##
        ####################################
        if info == "'Collided'":
            collision_history[e] = 1
        score_history.append(score)
        rejection_history[e] = rejected_trajectories
        
        # Store best average models
        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
        # Print some data
        print('episode ', e, 'score %.2f' % score,
            'trailing 100 games avg %.3f' % np.mean(score_history[-100:]), "ep_lenght:", episode_lenght, "info:", info)
        print("Rejected trajectories:", rejected_trajectories, "/", int(np.sum(rejection_history)))
        ####################################
        if e % 100 == 0:
            plotLearning(score_history, plotting, window=100)
            if store_plot_data:
                np.savetxt('DDPG/plots/'+store_plot_data+'_ch.txt', collision_history, fmt='%d')
                np.savetxt('DDPG/plots/'+store_plot_data+'_rh.txt', np.asarray(rejection_history))
                np.savetxt('DDPG/plots/'+store_plot_data+'_sh.txt', np.asarray(score_history))
    
    plotLearning(score_history, plotting, window=100)
    np.savetxt('DDPG/plots/'+store_plot_data+'_ch.txt', collision_history, fmt='%d')
    np.savetxt('DDPG/plots/'+store_plot_data+'_rh.txt', np.asarray(rejection_history))
    np.savetxt('DDPG/plots/'+store_plot_data+'_sh.txt', np.asarray(score_history))



if __name__ == "__main__":
    # Usage (from the repo root):
    #   python -m DDPG.main_torch ddpg           # plain DDPG, static walls          (ClosedField_v22)
    #   python -m DDPG.main_torch ddpg_dynamic   # plain DDPG, other moving vehicles (ClosedField_v23_dyna)
    #   python -m DDPG.main_torch pdrl           # PDRL (DDPG + MPC-style planning)  (MPC_environment_v40)
    # Optional second argument: number of episodes (default 5000).
    # New runs are saved under DDPG/checkpoints/my_*, so the shipped checkpoints are never overwritten.
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "pdrl"
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 5000

    if mode == "ddpg":
        v22_training(episodes=episodes, sim_dt=0.05, decision_dt=0.5,
                     save_folder="DDPG/checkpoints/my_ddpg", loadfolder=None,
                     plot_file="DDPG/plots/my_ddpg.png", environment="four_walls")
    elif mode == "ddpg_dynamic":
        v23_training(episodes=episodes, episode_s=20, sim_dt=0.05, decision_dt=0.5,
                     save_folder="DDPG/checkpoints/my_ddpg_dynamic", loadfolder="DDPG/checkpoints/v22",
                     plot_folder="DDPG/plots/my_ddpg_dynamic.png")
    elif mode == "pdrl":
        # Starts from the pre-trained DDPG weights (v22), as in the thesis.
        v40_MPC_IRL_training(episodes=episodes, sim_dt=0.05, decision_dt=0.5,
                             plotting="DDPG/plots/my_pdrl.png", save_folder="DDPG/checkpoints/my_pdrl",
                             loadfolder="DDPG/checkpoints/v22", environment_selection="four_walls",
                             add_noise=True, collision_rejection=True)
    else:
        sys.exit("Unknown mode '%s'. Use: ddpg | ddpg_dynamic | pdrl" % mode)
