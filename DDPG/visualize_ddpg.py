from .ddpg_torch import MPC_Agent, Agent
import numpy as np
from .environments import ClosedField_v22, ClosedField_v23_dyna, MPC_environment_v40, MPC_environment_v41
import pygame
import sys


        #env.close() # Automatically when env is garbage-collected.

        #env.close() # Automatically when env is garbage-collected.

        #env.close() # Automatically when env is garbage-collected.

        #env.close() # Automatically when env is garbage-collected.

        #env.close() # Automatically when env is garbage-collected.

        #env.close() # Automatically when env is garbage-collected.

        #env.close() # Automatically when env is garbage-collected.

        #env.close() # Automatically when env is garbage-collected.

def visualize_v22(sim_dt=0.05, decision_dt=0.5, folder="...",
                  v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5, environment_selection="four_walls",
                  ):

    env = ClosedField_v22(sim_dt=sim_dt, decision_dt=decision_dt, render=True,
                                v_max=v_max, v_min=v_min, alpha_max=alpha_max, tau_steering=tau_steering, tau_throttle=tau_throttle,
                                    horizon=200, environment_selection=environment_selection)
    # Start a new agent
    agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[42], tau=0.1, env=env, 
                    batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2, chkpt_dir=folder)
    
    # Load a pre-trained model
    agent.load_models(Verbose=True)
    render_all_frames=True
    while True:
        obs = env.reset() # restart env
        done = False
        score = 0
        # One episode step should be 0.5 seconds = decision_dt
        # So to get realtime, we need to limit ourselves to 2 fps...
        clock = pygame.time.Clock()
        while not done:
            act = agent.choose_action(obs, add_noise=False)
            new_state, reward, done, info = env.step(act)
            obs = new_state
            score += reward
            if render_all_frames:
                env.render(render_all_frames=True)
                clock.tick(1/sim_dt)
            else:
                env.render()
                clock.tick(1/decision_dt)
        print("Total score was:", score, "info:", info)
        #env.close() # Automatically when env is garbage-collected.

def visualize_v23(sim_dt=0.05, decision_dt=0.5, folder="...",
                  v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5, user_controlled=False):

    env = ClosedField_v23_dyna(sim_dt=sim_dt, decision_dt=decision_dt, render=True,
                                v_max=v_max, v_min=v_min, alpha_max=alpha_max, tau_steering=tau_steering, tau_throttle=tau_throttle,
                                    horizon=200, episode_s=200)
    # Start a new agent
    agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[42], tau=0.1, env=env, 
                    batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2, chkpt_dir=folder)
    
    # Load a pre-trained model
    agent.load_models(Verbose=True)
    while True:
        obs = env.reset() # restart env
        done = False
        score = 0
        # One episode step should be 0.5 seconds = decision_dt
        # So to get realtime, we need to limit ourselves to 2 fps...
        clock = pygame.time.Clock()
        while not done:
            if user_controlled:
                act = np.array([0, 0])
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: # Press x button
                        #exit()
                        pygame.quit()
                        sys.exit()
                keys = pygame.key.get_pressed()
                if keys[pygame.K_UP]:
                    #print("UP")
                    act[0] = 1
                if keys[pygame.K_DOWN]:
                    #print("DOWN")
                    act[0] = -1
                if keys[pygame.K_LEFT]:
                    #print("LEFT")
                    act[1] = -1
                if keys[pygame.K_RIGHT]:
                    #print("RIGHT")
                    act[1] = 1
            else:
                act = agent.choose_action(obs, add_noise=False)

            new_state, reward, done, info = env.step(act)
            obs = new_state
            score += reward

            

            env.render(render_all_frames=True)
            clock.tick(1/sim_dt)

        print("Total score was:", score, "info:", info)
        #env.close() # Automatically when env is garbage-collected.


############################################################## NEW technology! ##############################################################

def visualize_v40(sim_dt=0.05, decision_dt=0.5, save_folder="DDPG/checkpoints/v40", loadfolder="DDPG/checkpoints/v22",
                  v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5, environment_selection="four_walls",
                  trajectory_time=3.0, user_controlled=False, add_noise=False, collision_stop=True, include_collision_state=False,
                  goal_stop=True, add_disturbance=None):
    """
        - loadfolder="DDPG/checkpoints/v22"; holds a good starting point, based on a basic DDPG algorithm
        - loadfolder="DDPG/checkpoints/v40"; is a MPC-specifically trained agent to be visualized
    """
    
    # Initialization
    env = MPC_environment_v40(sim_dt=sim_dt, decision_dt=decision_dt, render=True, v_max=v_max, v_min=v_min,
	       alpha_max=alpha_max, tau_steering=tau_steering, tau_throttle=tau_throttle, horizon=200, edge=150,
		   episode_s=100, mpc=True, boost_N=False, environment_selection=environment_selection)
    agent = MPC_Agent(alpha=0.000025, beta=0.00025, input_dims=[42], tau=0.1, env=env, 
            batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2, chkpt_dir=save_folder)
    
    if loadfolder:
        print("Loading model from:'" + loadfolder+"'")
        agent.load_models(load_directory=loadfolder)
    #np.random.seed(0)
    ###############################################################################################################
    # Sets certain parameters
    trajectory_length = np.int32(trajectory_time/decision_dt) # to get 3 second trajectories
    ###############################################################################################################
    """ One episode"""
    while True: # Keeps repeating until exited
        current_state = env.reset()
        done = False
        update_vision=True # need to make initial update
        """ One decision_dt """
        while not done:
            ##############
            # Get new SC #
            ##############
            real_circogram = env.SC
            d1, d2, _, P2, _ = real_circogram
            #
            if update_vision:
                action_queue, decision_trajectory, sim_trajectory, halu_d2s, halu_states, collided = \
                    env.hallucinate(trajectory_length, sim_dt, decision_dt, agent, add_noise=add_noise, collision_stop=collision_stop, include_collision_state=include_collision_state,
                                    goal_stop=goal_stop)
                # TODO: add rejection to collision trajectories, even in display

 
            ####################################
            # Do we need to update trajectory? #
            ####################################
            d2_halu = halu_d2s.popleft() # Retrieve believed/halucinated SC from trajectory
            update_vision = env.update_vision(len(action_queue), d2, d2_halu)


            #############################
            # Select and execute action #
            #############################
            if len(action_queue) == 0:
                # This happens when trajectory is only 1 step, and it lead to collision... :(
                print("No actions")
                act = [0, 0]
            else:
                act = action_queue.popleft()
            """ NOTE during env.step, the SC for the new step is calculated! """

            if user_controlled:
                act = np.array([0, 0])
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: # Press x button
                        #exit()
                        pygame.quit()
                        sys.exit()
                keys = pygame.key.get_pressed()
                if keys[pygame.K_UP]:
                    #print("UP")
                    act[0] = 1
                if keys[pygame.K_DOWN]:
                    #print("DOWN")
                    act[0] = -1
                if keys[pygame.K_LEFT]:
                    #print("LEFT")
                    act[1] = -1
                if keys[pygame.K_RIGHT]:
                    #print("RIGHT")
                    act[1] = 1

            next_state, _, done, info = env.step(act, add_disturbance=add_disturbance)

        
            ########################
            # End of state actions #
            ########################
            current_state = next_state
            env.render(decision_trajectory, sim_trajectory, display_vision_box=True)
            # TODO: remove visited points of the trajectory, as we go...
        print("Episode finished with code;", info)

def visualize_v41(sim_dt=0.05, decision_dt=0.5, save_folder="DDPG/checkpoints/v40", loadfolder="DDPG/checkpoints/v22",
                    v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5, trajectory_time=3.0,
                    user_controlled=False, add_noise=False, collision_stop=True, include_collision_state=False, repeat_forever=False):
    
    """
        - loadfolder="DDPG/checkpoints/v22"; holds a good starting point, based on a basic DDPG algorithm
        - loadfolder="DDPG/checkpoints/v40"; is a MPC-specifically trained agent to be visualized
    """
    
    # Initialization
    env = MPC_environment_v41(sim_dt=sim_dt, decision_dt=decision_dt, render=True, v_max=v_max, v_min=v_min,
	       alpha_max=alpha_max, tau_steering=tau_steering, tau_throttle=tau_throttle, horizon=200, edge=150,
		   episode_s=100, mpc=True, boost_N=False)
    agent = MPC_Agent(alpha=0.000025, beta=0.00025, input_dims=[42], tau=0.1, env=env, 
            batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2, chkpt_dir=save_folder)
    
    if loadfolder:
        print("Loading model from:'" + loadfolder+"'")
        agent.load_models(load_directory=loadfolder)
    #np.random.seed(0)
    ###############################################################################################################
    # Sets certain parameters
    trajectory_length = np.int32(trajectory_time/decision_dt) # to get 3 second trajectories
    ###############################################################################################################
    """ One episode"""
    while True: # Keeps repeating until exited
        current_state = env.reset()
        done = False
        update_vision=True # need to make initial update
        """ One decision_dt """
        while not done:
            ##############
            # Get new SC #
            ##############
            real_circogram = env.SC
            d1, d2, _, P2, _ = real_circogram
            #
            if update_vision:
                action_queue, decision_trajectory, sim_trajectory, halu_d2s, _, collided = \
                    env.hallucinate(trajectory_length, sim_dt, decision_dt, agent, add_noise=add_noise, collision_stop=collision_stop, include_collision_state=include_collision_state)
                # TODO: add rejection to collision trajectories, even in display
                
 
            ####################################
            # Do we need to update trajectory? #
            ####################################
            d2_halu = halu_d2s.popleft() # Retrieve believed/halucinated SC from trajectory
            update_vision = env.update_vision(len(action_queue), d2, d2_halu, out_=0.9, in_=0.2) # Update very time!


            #############################
            # Select and execute action #
            #############################
            if len(action_queue) == 0:
                # This happens when trajectory is only 1 step, and it lead to collision... :(
                print("No actions")
                act = [0, 0]
            else:
                act = action_queue.popleft()
            # NOTE: often the last action in trajectory is right in front of collision, so it eventually leads to collision
            #if len(action_queue) == 0:
            #    print("Last action: stop!")
            #    act = [0, 0]

            if user_controlled:
                act = np.array([0, 0])
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: # Press x button
                        #exit()
                        pygame.quit()
                        sys.exit()
                keys = pygame.key.get_pressed()
                if keys[pygame.K_UP]:
                    #print("UP")
                    act[0] = 1
                if keys[pygame.K_DOWN]:
                    #print("DOWN")
                    act[0] = -1
                if keys[pygame.K_LEFT]:
                    #print("LEFT")
                    act[1] = -1
                if keys[pygame.K_RIGHT]:
                    #print("RIGHT")
                    act[1] = 1

            next_state, _, done, _ = env.step(act)
            if repeat_forever and len(env.goal_stack)==1:
                env.generate_new_goal_stack()
        
            ########################
            # End of state actions #
            ########################
            current_state = next_state
            env.render(decision_trajectory, sim_trajectory, display_vision_box=True)
            





if __name__ == "__main__":
    # Usage (from the repo root):
    #   python -m DDPG.visualize_ddpg pdrl           # PDRL planning in a street map, shows planned trajectories
    #   python -m DDPG.visualize_ddpg pdrl_dynamic   # PDRL among other moving vehicles
    #   python -m DDPG.visualize_ddpg ddpg           # plain DDPG agent, static walls
    #   python -m DDPG.visualize_ddpg ddpg_dynamic   # plain DDPG agent among other moving vehicles
    # The weights in DDPG/checkpoints are the ones Henrik trained for the car.
    mode = sys.argv[1] if len(sys.argv) > 1 else "pdrl"
    common = dict(v_max=20, v_min=-4, alpha_max=0.5, tau_steering=0.5, tau_throttle=0.5)

    if mode == "pdrl":
        visualize_v40(sim_dt=0.02, decision_dt=0.25, save_folder="None", loadfolder="DDPG/checkpoints/v22_naples_nn",
                      environment_selection="naples_street", trajectory_time=2.0,
                      add_noise=False, collision_stop=True, include_collision_state=False, **common)
    elif mode == "pdrl_dynamic":
        visualize_v41(sim_dt=0.05, decision_dt=0.5, save_folder="None", loadfolder="DDPG/checkpoints/v23",
                      trajectory_time=3.0, add_noise=False, collision_stop=False, include_collision_state=True,
                      repeat_forever=True, **common)
    elif mode == "ddpg":
        visualize_v22(sim_dt=0.05, decision_dt=0.5, folder="DDPG/checkpoints/v22",
                      environment_selection="four_walls", **common)
    elif mode == "ddpg_dynamic":
        visualize_v23(sim_dt=0.05, decision_dt=0.5, folder="DDPG/checkpoints/v23", user_controlled=False, **common)
    else:
        sys.exit("Unknown mode '%s'. Use: pdrl | pdrl_dynamic | ddpg | ddpg_dynamic" % mode)
