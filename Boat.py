"""
Time-discrete 2D motion model of a boat with a single rotatable stern thruster.

Follows the course model "Time-Discrete Motion Model of a Simple Boat with Rotatable Thruster":
  - reference point: centre of mass (CoM)
  - body-frame (surge u, sway v) linear drag, with c_sway >> c_surge   (appendix, "Refined Dynamics")
      M du/dt = F_x - c_surge * u
      M dv/dt = F_y - c_sway  * v
  - yaw:  I dw/dt = tau - c_yaw * w - c_cross * v
  - each first-order equation is discretised with its exact solution over one step T,
    with thrust F and thruster angle constant during the step
  - kinematics: p_{k+1} = p_k + T R(psi_k) v_b[k],   psi_{k+1} = psi_k + T w_k
  - frame rotation: v_b is stored in the body frame, which turns with the hull. After each step
    v_b is re-expressed in the new body frame, v_b <- R(-(psi_{k+1} - psi_k)) v_b (the omega x v
    term that the "refined dynamics" equation in the notes leaves out). Without it the velocity
    would turn together with the hull, as if the boat were on rails.
With c_sway = c_surge (and tau_thrust, tau_steering ~ 0) this reproduces the notes' Task 3
world-frame model exactly, with the stern-thruster sign convention below.

Sign convention (differs from the course notes):
  The thruster sits at the stern, a distance d behind the CoM. Its physical angle delta is measured
  from the forward axis. A thrust pointing to port at the stern pushes the stern to port and turns
  the bow to starboard, so  tau = -d * F * sin(delta). The notes write tau = +d F sin(delta), which
  corresponds to a thruster ahead of the CoM. To keep the same interface as Henrik's car,
  `alpha` is the *steering* angle: alpha > 0 turns the heading psi in the positive direction,
  and the physical thruster angle is delta = -alpha. So tau = d F sin(alpha), as in the notes.

Interface: a Boat is a drop-in replacement for Vehicle. Drivers, environments and PDRL call
one_step_algorithm(_2)(alpha_ref, v_ref); v_ref is mapped to the thrust that gives that steady-state
surge speed (F = c_surge * v_ref). For direct control use step(F_cmd, alpha_cmd, dt).
"""
import numpy as np
from Vehicle import Vehicle


def _first_order_step(x, force, c, m, T):
    """Exact solution of m dx/dt = force - c x over T, force constant."""
    if c <= 1e-12:
        return x + T * force / m
    a = np.exp(-c * T / m)
    return a * x + (1 - a) * force / c


class Boat(Vehicle):

    def __init__(self, center: np.array, length: float = 8, width: float = 3, heading: float = 0,
                 dt: float = 0.05,
                 mass: float = 1000.0,           # M  [kg]
                 inertia: float = None,          # I  [kg m^2], default: solid rectangle
                 c_surge: float = 400.0,         # surge drag [N s/m]  -> top speed F_max / c_surge = 7 m/s
                 c_sway: float = 4000.0,         # sway drag  [N s/m]  (>> c_surge: the hull resists sliding)
                 c_yaw: float = 15000.0,         # yaw drag   [N m s/rad]
                 c_cross: float = 0.0,           # sway -> yaw coupling [N s] (optional refinement)
                 F_max: float = 2800.0,          # max forward thrust [N]
                 F_min: float = -1200.0,         # max reverse thrust [N]
                 alpha_max: float = 0.6,         # max thruster angle [rad]
                 tau_thrust: float = 0.8,        # thrust actuator time constant [s]
                 tau_steering: float = 0.4,      # thruster-angle actuator time constant [s]
                 **_ignored):                    # accepts Vehicle-only kwargs (tau_throttle, ...) for drop-in use
        super().__init__(center, length=length, width=width, heading=heading, dt=dt,
                         alpha_max=alpha_max, v_max=F_max / c_surge,
                         tau_throttle=tau_thrust, tau_steering=tau_steering)
        self.M = mass
        self.I = inertia if inertia is not None else mass * (length**2 + width**2) / 12
        self.c_surge, self.c_sway, self.c_yaw, self.c_cross = c_surge, c_sway, c_yaw, c_cross
        self.F_max, self.F_min = F_max, F_min
        self.tau_thrust = tau_thrust
        self.d_thruster = length / 2      # CoM -> thruster distance

        # Boat state (on top of Vehicle's heading / position_center)
        self.p = np.array(center, dtype=float)    # CoM position, world frame
        self.v_b = np.zeros(2)                    # body-frame velocity [u (surge, forward), v (sway, port)]
        self.omega = 0.0                          # yaw rate
        self.F = 0.0                              # current thrust (after actuator lag)
        self.alpha = 0.0                          # current steering angle (after actuator lag)
        self._sync_geometry()

    # ------------------------------------------------------------------ dynamics
    def step(self, F_cmd: float, alpha_cmd: float, dt: float = None):
        """Advance one time step with thrust command F_cmd [N] and steering command alpha_cmd [rad]."""
        T = self.dt if dt is None else dt
        F_cmd = float(np.clip(F_cmd, self.F_min, self.F_max))
        alpha_cmd = float(np.clip(alpha_cmd, -self.alpha_max, self.alpha_max))

        # Actuators: first-order lag towards the commands
        self.F += (1 - np.exp(-T / self.tau_thrust)) * (F_cmd - self.F)
        self.alpha += (1 - np.exp(-T / self.tau_steering)) * (alpha_cmd - self.alpha)

        # Thruster force in the body frame (x forward, y to port) and yaw moment about the CoM
        delta = -self.alpha                       # physical thruster angle
        Fx = self.F * np.cos(delta)
        Fy = self.F * np.sin(delta)
        tau = -self.d_thruster * Fy               # = d F sin(alpha)

        # Kinematics with the old velocities (as in the notes)
        R = self._R(self.heading)
        self.p = self.p + T * (R @ self.v_b)
        psi_new = self.heading + T * self.omega

        # Exact first-order updates
        u, v = self.v_b
        u_new = _first_order_step(u, Fx, self.c_surge, self.M, T)
        v_new = _first_order_step(v, Fy, self.c_sway, self.M, T)
        omega_new = _first_order_step(self.omega, tau - self.c_cross * v, self.c_yaw, self.I, T)

        # The body frame turned by (psi_new - psi_k): express the velocity in the new body frame
        dpsi = psi_new - self.heading
        c, s = np.cos(dpsi), np.sin(dpsi)
        self.v_b = np.array([[c, s], [-s, c]]) @ np.array([u_new, v_new])

        self.heading = psi_new
        self.omega = omega_new
        self._sync_geometry()

    # Henrik's interface: (steering ref, speed ref) -> boat
    def one_step_algorithm(self, alpha_ref, v_ref):
        self.one_step_algorithm_2(alpha_ref, v_ref, self.dt)

    def one_step_algorithm_2(self, alpha_ref, v_ref, dt):
        self.focus_direction = int(np.sign(v_ref))
        self.step(F_cmd=self.c_surge * v_ref, alpha_cmd=alpha_ref, dt=dt)

    def one_step_algorithm_3(self, action, dt, params=None):
        raise NotImplementedError("Disturbance testing (add_disturbance) is car-specific; "
                                  "vary the Boat parameters instead.")

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _R(psi):
        return np.array([[np.cos(psi), -np.sin(psi)],
                         [np.sin(psi),  np.cos(psi)]])

    @property
    def velocity_world(self):
        """Velocity in the world frame."""
        return self._R(self.heading) @ self.v_b

    def _sync_geometry(self):
        """Keep the Vehicle fields that the rest of the code reads in sync with the boat state."""
        self.position_center = self.p.copy()
        self.originVCF = self.CCFtoWCF(np.array([0.0, -self.length / 2]))
        vw = self.velocity_world
        self.X = np.array([[self.originVCF[0]], [self.originVCF[1]], [vw[0]], [vw[1]]])
        u = self.v_b[0]
        self.actual_direction = 1 if u > 1e-6 else (-1 if u < -1e-6 else 0)
        self.vertices = self.vertices_VCTtoWCF(self.verticesVCF)
        self.sides = [[self.vertices[0], self.vertices[1]],
                      [self.vertices[1], self.vertices[2]],
                      [self.vertices[2], self.vertices[3]],
                      [self.vertices[3], self.vertices[0]]]
        self.lines = self.eval_lines(self.sides)
