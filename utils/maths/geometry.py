import numpy as np

def diff_angles(angle1, angle2):
    a = angle1 - angle2
    return (a + np.pi) % (2 * np.pi) - np.pi

def vector_angle(v, w):
    return np.arctan2(w[1] - v[1], w[0] - v[0])

def vectors_angle(v, w):
    angle = np.arctan2(v[1], v[0]) - np.arctan2(w[1], w[0])

    if angle > np.pi:
        angle -= 2 * np.pi
    elif angle <= -np.pi:
        angle += 2 * np.pi

    return angle

def angle_between_vectors(v, w):
    return np.arccos(np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w)))

def get_min_distance_from_point_to_line(point, line_start, line_end):
    return np.linalg.norm(
        np.cross(line_end - line_start, line_start - point)
    ) / np.linalg.norm(line_end - line_start)

def get_intersection_of_two_lines(p1, v1, p2, v2):
    """
    Get the intersection of two lines
    the first line starts at p1 and has the direction v1
    the second line starts at p2 and has the direction v2
    """
    # check if the lines are parallel
    if np.cross(v1, v2) == 0:
        return None
    else:
        t = np.cross(p2 - p1, v2) / np.cross(v1, v2)
        return p1 + t * v1

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y