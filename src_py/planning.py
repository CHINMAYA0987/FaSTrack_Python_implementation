import numpy as np
import math
import random
from typing import Tuple, Optional, List
from dataclasses import dataclass
from models import Point, Obstacle, TrackingModelState, PlanningModelState


@dataclass
class Environment:
    start: Point
    target: Point
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    obstacles: Optional[List[Obstacle]] = None


class RRT:
    def __init__(self, start: Point, step_size=2.0, max_iter=2000):
        self.start = start
        self.step_size = step_size
        self.max_iter = max_iter
        self.nodes = [self.start]
        self.parent = {0: None}
        self.path = []
        self.goal_radius = 1.0

    def is_in_obstacle(self, p: Point, env: Environment) -> bool:
        for obs in env.obstacles:
            if (p.x - obs.x)**2 + (p.y - obs.y)**2 <= obs.radius**2:
                return True
        return False

    def is_edge_valid(self, p1: Point, p2: Point, env: Environment) -> bool:
        for obs in env.obstacles:
            cx, cy, r = obs.x, obs.y, obs.radius
            dx, dy = p2.x - p1.x, p2.y - p1.y
            
            a = dx*dx + dy*dy
            
            if a < 1e-6:
                return True 
                
            fx, fy = p1.x - cx, p1.y - cy
            
            b = 2 * (fx*dx + fy*dy)
            c = (fx*fx + fy*fy) - r*r
            
            discriminant = b*b - 4*a*c
            if discriminant >= 0:
                sqrt_disc = math.sqrt(discriminant)
                t1 = (-b - sqrt_disc) / (2*a)
                t2 = (-b + sqrt_disc) / (2*a)
                
                if (0 <= t1 <= 1) or (0 <= t2 <= 1):
                    return False
        return True

    def get_nearest_node_idx(self, p_rand: Point) -> int:
        min_dist = float('inf')
        nearest_idx = -1
        for i, node in enumerate(self.nodes):
            dist = math.hypot(p_rand.x - node.x, p_rand.y - node.y)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        return nearest_idx

    def extend(self, p_nearest: Point, p_rand: Point) -> Point:
        dist = math.hypot(p_rand.x - p_nearest.x, p_rand.y - p_nearest.y)
        if dist < self.step_size:
            return p_rand
        
        theta = math.atan2(p_rand.y - p_nearest.y, p_rand.x - p_nearest.x)
        x = p_nearest.x + self.step_size * math.cos(theta)
        y = p_nearest.y + self.step_size * math.sin(theta)
        return Point(x, y)

    def is_goal_reached(self, p: Point, target: Point) -> bool:
        return math.hypot(p.x - target.x, p.y - target.y) < self.goal_radius

    def build_rrt(self, env: Environment, ax=None) -> bool:
        self.nodes = [self.start]
        self.parent = {0: None}
        self.path = []
        
        x_min, x_max = env.x_range
        y_min, y_max = env.y_range

        for _ in range(self.max_iter):
            if random.random() < 0.1:
                p_rand = env.target
            else:
                p_rand = Point(random.uniform(x_min, x_max), random.uniform(y_min, y_max))

            if self.is_in_obstacle(p_rand, env):
                continue

            nearest_idx = self.get_nearest_node_idx(p_rand)
            nearest = self.nodes[nearest_idx]
            
            new_node = self.extend(nearest, p_rand)

            if self.is_in_obstacle(new_node, env) or not self.is_edge_valid(nearest, new_node, env):
                continue

            self.nodes.append(new_node)
            new_node_idx = len(self.nodes) - 1
            self.parent[new_node_idx] = nearest_idx
            
            if ax is not None:
                ax.plot([nearest.x, new_node.x], [nearest.y, new_node.y], c="gray", linewidth=0.8, alpha=0.6, zorder=1)

            if self.is_goal_reached(new_node, env.target):
                self.nodes.append(env.target)
                self.parent[len(self.nodes)-1] = new_node_idx
                self.extract_path()
                return True

        return False

    def extract_path(self):
        path = []
        idx = len(self.nodes)-1
        while idx is not None:
            path.append(self.nodes[idx])
            idx = self.parent.get(idx)
        self.path = list(reversed(path))


class SimplePlanner:
    def __init__(self, planning_model):
        self.planning_model = planning_model
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.kp_omega = 1.0
        self.lookahead_dist = 2.0

    def set_waypoints(self, waypoints: List[Point]):
        self.waypoints = waypoints
        self.current_waypoint_idx = 0
        
    def get_control(self, state: PlanningModelState) -> float:
        if not self.waypoints or self.current_waypoint_idx >= len(self.waypoints):
            return 0.0

        current_pos = np.array([state.x, state.y])
        
        base_idx = self.current_waypoint_idx
        min_dist_to_wp = float('inf')
        lookahead_point = None
        
        for i in range(self.current_waypoint_idx, len(self.waypoints)):
            waypoint_pos = np.array([self.waypoints[i].x, self.waypoints[i].y])
            dist = np.linalg.norm(waypoint_pos - current_pos)
            
            if dist < min_dist_to_wp:
                min_dist_to_wp = dist
                base_idx = i
                
            if dist > self.lookahead_dist or i == len(self.waypoints) - 1:
                lookahead_point = self.waypoints[i]
                self.current_waypoint_idx = base_idx 
                break
        
        if lookahead_point is None:
            return 0.0

        desired_heading = math.atan2(lookahead_point.y - state.y, lookahead_point.x - state.x)
        heading_error = (desired_heading - state.theta + math.pi) % (2*math.pi) - math.pi
        omega = self.kp_omega * heading_error
        
        if np.linalg.norm(current_pos - np.array([self.waypoints[-1].x, self.waypoints[-1].y])) < 0.5:
            self.current_waypoint_idx = len(self.waypoints)
            return 0.0

        return np.clip(omega, self.planning_model.omega_min, self.planning_model.omega_max)


def inflate_obstacles(obstacles, inflation_radius):
    inflated = []
    for obs in obstacles:
        inflated.append(Obstacle(obs.x, obs.y, obs.radius + inflation_radius))
    return inflated