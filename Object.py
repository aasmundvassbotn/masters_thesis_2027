from typing import List
import numpy as np
from math import sqrt
from numba import jit

class Object:
    def __init__(self, center: np.array, vertices: List[np.array]):
        #self.heading = heading
        self.position_center = center
        self.vertices = vertices
        # TODO: an object should be able to hold more than 4 sides
        #self.sides = [[vertices[0], vertices[1]],
        #              [vertices[1], vertices[2]],
        #              [vertices[2], vertices[3]],
        #              [vertices[3], vertices[0]]]
        self.sides = self.vertices_to_sides(vertices)
        self.lines = self.eval_lines(self.sides)
        self.radius = self.eval_radius()

    def vertices_to_sides(self, vertices):
        sides = [[vertices[-1], vertices[0]]]
        for i in range(len(vertices)-1):
            sides.append([vertices[i], vertices[i+1]])
        return sides
   
    """Evaluate radius of circle surrounding vehicle"""
    def eval_radius(self) -> float:
        # given the initial shape of the vehicle, that for now we consider as rectangular, we evaluate the radius.
        radius = self.dist_point_point(self.position_center, self.vertices[0])
        return radius

    """Evaluate lines passing throught vertices"""
    def eval_lines(self, sides) -> List[np.array]:
        # evaluate the lines passing through the vertices and insert them in a list. every line is expressed as an
        # array with 3 values [a,b,c]
        lines = []
        for side in self.sides:
            lines.append(self.eval_line_point_point(side[0], side[1]))
        return lines

    # Evaluate line passing throught two points"""
    def eval_line_point_point(self, point1: np.array, point2: np.array) -> np.array:
        # evaluate the line passing throught two points, expressing it as a vector with 3 values [a,b,c]
        delta_x = point1[0] - point2[0]
        if delta_x != 0: # Check if the line is vertical, otherwise there is a division by 0
            delta_y = point1[1] - point2[1]
            m = delta_y / delta_x
            c = (delta_x * point1[1] - delta_y * point1[0]) / delta_x
            line = np.array([m, -1, c])
        else:
            line = np.array([1, 0, - point1[0]]) # The equation of a vertical line
        return line

    """ Methods useful for evaluation distances"""
    # METHOD 1: The distance between two points
    @staticmethod
    @jit(nopython=True)
    def dist_point_point(point1: np.array, point2: np.array) -> float:
        distance = sqrt((point1[0]-point2[0])*(point1[0]-point2[0]) + (point1[1]-point2[1])*(point1[1]-point2[1]))
        return distance

    # METHOD 2: The distance between a line (defined in the normal form $a x+b y + c = 0$) and a point
    # METHOD 3: find the closest point P on a given line $a x+b y + c = 0$ (which is the projection)
    # to a given point A ($x_0,y_0$)
    # METHOD 4: find if a point P($x_0,y_0$) on a line stands between two other points A($x_1,y_1$)
    # and B($x_2,y_2$) on  same line
    # METHOD 5: find if a point H has its projection between two other points A and B
    # METHOD 6: distance of a point to a segment
    # METHOD 7: distance of a point to an Obj
    # This is the far-sighted distance, which is approximation of the distance. Very fast to evaluate
    # This is the near-sighted distance, which is the precise distance.
