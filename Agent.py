import numpy as np

class Agent:

    def __init__(self, v_max : float = 10, v_min : float = 0, alpha_max : float = 1.2, var_vel : float = 0.5, var_alpha : float = 0.2) -> None:
        self.v_max = v_max
        self.v_min = v_min
        self.alpha_max = alpha_max
        self.var_vel = var_vel
        self.var_alpha = var_alpha
        self.state = 0

        # needed for brownian_DC_action
        self.risky_distance = None
        self.risky_index = None
        # Debugging
        self.prev_dist = None
        self.new_dists = []
        self.states = []
        self.v_refs = []
        self.wait_counter = 0

        # NEW DC tech
        self.prev_DC_risks = None

    def determined_dc_avoidance_action(self, angles, risks, v_ref, alpha_ref, stop_threshold):
        """
        Take a single action to avoid colliding, after detecting too high risks.
        - Prefers right hand turns
        - Smoother turns
        """
        max_index = np.argmax(risks)
        max_risk = risks[max_index] # Risks are primarily in the region 0->1 (sometimes (above!))
        max_angle = angles[max_index]

        # 1 The turning angle:
        if v_ref>= 0: #Accelerating forward
            # NOTE prefers the right hand turn.
            if np.pi >= max_angle >= np.pi/2+0.1: # left turn needed
                #print("TURN LEFT!")
                # The closer the risk is to straight in front, the sharper the turn
                alpha_ref = -self.alpha_max
                #alpha_ref = -self.alpha_max * (1-(max_angle-np.pi/2)/(np.pi))
                #alpha_ref *= 1.1


            elif np.pi/2+0.1 > max_angle >= 0: # Right turn needed (max_angle < pi/2)
                #print("TURN RIGHT!")
                # The closer the risk is to straight in front, the sharper the turn
                alpha_ref = self.alpha_max
                #alpha_ref = self.alpha_max * (0.5+(max_angle)/(np.pi))
                #alpha_ref *= 1.1

            else:
                # That is weird... Max angle should be somewhere in driving direction.
                pass # Weird behaviour... # see TODO

        else: # Accelerating backwards!
            if 3*np.pi/2 >= max_angle >= np.pi: # left turn needed
                # NOTE turns towards the obstacle.
                alpha_ref = -self.alpha_max

            elif 2*np.pi >= max_angle >= 3*np.pi/2: # Right turn needed
                # NOTE turns towards the obstacle.
                alpha_ref = self.alpha_max

            else:
                # That is weird... Max angle should be somewhere in driving direction.
                pass # Weird behaviour... # see TODO


        # 2 v_ref: The higher the risk, the **slower** the driving.
        if v_ref >=0: # Forward
            #v_ref = self.v_max*(1/(1+max_risk))
            v_ref = self.v_max*(1-max_risk/(stop_threshold))
        else:
            v_ref = self.v_min*(1/(1+max_risk)) # Keep backing
            #v_ref = self.v_min*(1-max_risk/stop_threshold)
        return v_ref, alpha_ref

    def determined_driver(self, dynamic_circogram, static_circogram, v_ref, alpha_ref, risk_threshold=0.8, stop_threshold = 2, dist_wait=10, verbose=False):
        """ Collision avoidance, and keeping realaively high speeds
        """
        # State machine:
        # {"Determined": 0, "dc_avoid" : 1, "stop, reverse" : 2}
        # 
        d1, d2, _, _, _ = static_circogram
        angles = dynamic_circogram[:,0]
        risks = dynamic_circogram[:,2]
        max_index = np.argmax(risks)
        ####################################################
        # Stop and reverse
        # Reverse until: path is cleared OR backing risk is too high again
        if (self.state == 2):
            # NOTE we stop to make sure we are stationary
            # TODO: add a 50% of waiting a bit before proceeding?
            #if self.stop_counter<20:
            #    v_ref = 0
            #    alpha_ref = 0
            #    self.stop_counter += 1
            #    return v_ref, alpha_ref

            # TODO: problem when car is actually stuck - will move forward, as risk == 0, then see danger, stop - but slide into the obstacle... :(

            # State remains
            v_ref = self.v_min
            if risks[max_index] >= risk_threshold:
                # Drive towards angle
                v_ref, alpha_ref = self.determined_dc_avoidance_action(angles, risks, self.v_min, alpha_ref, stop_threshold)
            else:
                alpha_ref = self.backing_angle

            # State change?
            new_distance = np.abs(d2[self.risky_index] - d1[self.risky_index])
            if new_distance > self.risky_distance*2: # One way out of state 2
                self.dist_counter +=1
                if self.dist_counter > dist_wait: # To be sure
                    v_ref = 0
                    self.state = 1
                    if verbose:
                        print("Risky dist was cleared!")    
            else:
                self.dist_counter = 0

            if risks[max_index] > stop_threshold: # Reverse risk is greater again.
                v_ref = 0 # Stop again
                self.state = 1
                if verbose:
                    print("Reversing was more dangerous!")

        ####################################################
        # STATE DC avoidance
        elif (self.state == 1):
            # State change?
            if (risks[max_index] >= stop_threshold):
                # STOP
                v_ref = 0
                self.state = 2
                self.risky_distance = np.abs(d2[max_index] - d1[max_index]) # distance along risky ray
                self.risky_index = max_index
                self.dist_counter = 0
                ## 50% / 50% of what turn to stop and wait or not
                #num = np.random.choice([0, 1])
                #if num==0:
                #    self.stop_counter = 30 # Done waiting
                #elif num==1:
                #    self.stop_counter = 0 # Wait
                
                # 50% / 50% of what turn to take
                num = np.random.choice([0, 1])
                if num==0:
                    self.backing_angle = self.alpha_max/1.5 
                elif num==1:
                    self.backing_angle = -self.alpha_max/1.5
                else:
                    self.backing_angle = 0 #Straight
                
                if verbose:
                    print("CAR IS STOPPING...")
            elif risks[max_index] < risk_threshold:
                # Return to determined
                self.state = 0
                alpha_ref = 0
                v_ref = self.v_max
            else:
                # Default state 1 behaviour: prefers rht
                v_ref, alpha_ref = self.determined_dc_avoidance_action(angles, risks, v_ref, alpha_ref, stop_threshold)
        ####################################################
        # STATE Determined
        elif (self.state==0):
            # State change?
            if risks[max_index] >= risk_threshold:
                # Avoid collision!!
                self.state = 1
            else:
                # Default state 0 behaviour
                v_ref = self.v_max
                alpha_ref = 0
        else:
            raise Exception("Not a valid state: "+str(self.state))
        ####################################################
        return v_ref, alpha_ref

