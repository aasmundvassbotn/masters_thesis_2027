from Vehicle import Vehicle
from Agent import Agent

class Limo(Vehicle):

    def __init__(self, vehicle : Vehicle, driver : Agent ):
        self.vehicle = vehicle
        self.driver = driver
