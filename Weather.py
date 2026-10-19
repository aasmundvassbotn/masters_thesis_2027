"""
Weather for the boat simulation: water current and wind.

Change the settings below and save. They are used by:
    python -m DDPG.visualize_ddpg keyboard_boat

Directions are (x, y) in m/s. On screen x points right and y points down:
    (1, 0) right, (-1, 0) left, (0, -1) up, (0, 1) down.

The weather only gives the current and wind vectors. How they act on a boat is in Boat.py.
"""
import numpy as np

# ---------------------------------------------------------------- settings
CURRENT = (0.0, 0.0)            # water current, m/s; (0, 0) = no current
CURRENT_CHANGE = 0.0            # chance (0-1) per second that the current changes direction

WIND = (0.0, -5.0)              # wind direction and starting speed, m/s; (0, 0) = no wind
WIND_CHANGE = 0.1               # chance (0-1) per second that the wind changes direction
WIND_GUST = (2.0, 20.0)         # wind speed varies between (min, max) m/s; None = constant speed
GUST_TIME = 2.0                 # typical gust duration, s

MAX_DIRECTION_CHANGE = 180.0    # largest change in direction, degrees (180 = any direction)
SEED = None                     # a number gives the same random sequence every run
# -------------------------------------------------------------------------


class ChangingVector:
    """A 2D vector (m/s) whose direction and speed can change at random.

    change_chance:  chance that the direction changes within one second (0 = never).
                    Converted to a chance per step, 1 - (1 - change_chance)**dt, so it does not
                    depend on the time step.
    max_change_deg: a change turns the direction by a random angle in [-max, +max] degrees.
    speed_range:    (min, max) m/s, or None for a constant speed. The speed follows a random walk
                    that is pulled towards the middle of the range (Ornstein-Uhlenbeck process),
                    with a spread of a quarter of the range, and is clipped to the range.
    gust_time:      time constant of that random walk, s.
    """

    def __init__(self, vector=(0.0, 0.0), change_chance=0.0, max_change_deg=180.0, rng=None,
                 speed_range=None, gust_time=2.0):
        self.vector = np.array(vector, dtype=float)
        self.change_chance = float(np.clip(change_chance, 0.0, 1.0))
        self.max_change = np.radians(max_change_deg)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.speed_range = None
        self.gust_time = gust_time
        self.direction_changed = False   # True right after an update that changed the direction
        if speed_range is not None:
            lo, hi = sorted(float(x) for x in speed_range)
            if lo < 0:
                raise ValueError("speed range must be non-negative")
            if not self.vector.any():
                raise ValueError("a speed range needs a direction, give a non-zero vector")
            self.speed_range = (lo, hi)
            start = np.clip(np.linalg.norm(self.vector), lo, hi)
            self.vector = self.vector / np.linalg.norm(self.vector) * start

    @property
    def speed(self):
        return float(np.linalg.norm(self.vector))

    @property
    def direction_deg(self):
        """Direction on screen: 0 = right, 90 = up."""
        return np.degrees(np.arctan2(-self.vector[1], self.vector[0])) % 360

    def update(self, dt):
        """Move dt seconds forward. Returns True if the vector changed."""
        self.direction_changed = False
        if not self.vector.any():
            return False
        changed = False
        # Direction: random jump
        if self.change_chance > 0 and self.rng.random() < 1 - (1 - self.change_chance) ** dt:
            a = self.rng.uniform(-self.max_change, self.max_change)
            c, s = np.cos(a), np.sin(a)
            self.vector = np.array([[c, -s], [s, c]]) @ self.vector
            self.direction_changed = changed = True
        # Speed: Ornstein-Uhlenbeck random walk, clipped to the range
        if self.speed_range is not None:
            lo, hi = self.speed_range
            mean, spread = (lo + hi) / 2, (hi - lo) / 4
            a = np.exp(-dt / self.gust_time)
            speed = mean + (self.speed - mean) * a + spread * np.sqrt(1 - a * a) * self.rng.standard_normal()
            speed = float(np.clip(speed, max(lo, 1e-6), hi))
            self.vector = self.vector / np.linalg.norm(self.vector) * speed
            changed = True
        return changed


# Default for Weather arguments: use the value from the settings above
SETTING = object()


class Weather:
    """Current and wind for the whole map. Arguments that are not given use the settings above."""

    def __init__(self, current=SETTING, current_change=SETTING, wind=SETTING, wind_change=SETTING,
                 wind_gust=SETTING, gust_time=SETTING, max_direction_change=SETTING, seed=SETTING):
        def pick(value, setting):
            return setting if value is SETTING else value
        max_change = pick(max_direction_change, MAX_DIRECTION_CHANGE)
        rng = np.random.default_rng(pick(seed, SEED))
        self.wind_gust = pick(wind_gust, WIND_GUST)
        self.current_field = ChangingVector(pick(current, CURRENT), pick(current_change, CURRENT_CHANGE),
                                            max_change, rng)
        self.wind_field = ChangingVector(pick(wind, WIND), pick(wind_change, WIND_CHANGE), max_change, rng,
                                         speed_range=self.wind_gust, gust_time=pick(gust_time, GUST_TIME))

    @property
    def current(self):
        return self.current_field.vector.copy()

    @property
    def wind(self):
        return self.wind_field.vector.copy()

    def update(self, dt):
        """Move the weather dt seconds forward."""
        self.current_field.update(dt)
        if self.current_field.direction_changed:
            print("Current changed direction: %.0f deg" % self.current_field.direction_deg)
        self.wind_field.update(dt)
        if self.wind_field.direction_changed:
            print("Wind changed direction: %.0f deg" % self.wind_field.direction_deg)

    def apply(self, boats):
        """Give the current weather to each boat."""
        for boat in boats:
            boat.current = self.current
            boat.wind = self.wind
