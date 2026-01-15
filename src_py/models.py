import numpy as np
import math
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class Point:
    x: float
    y: float


@dataclass
class Obstacle:
    x: float
    y: float
    radius: float
    color: str = 'red'
    alpha: float = 0.5


@dataclass
class TrackingModelState:
    x: float
    y: float
    theta: float
    v: float
    omega: float
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta, self.v, self.omega])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'TrackingModelState':
        return cls(arr[0], arr[1], arr[2], arr[3], arr[4])


@dataclass
class PlanningModelState:
    x: float
    y: float
    theta: float
    v: float = 0.0
    omega: float = 0.0
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'PlanningModelState':
        return cls(arr[0], arr[1], arr[2])


@dataclass
class RelativeState:
    x_r: float
    y_r: float
    theta_r: float
    v: float
    omega: float
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x_r, self.y_r, self.theta_r, self.v, self.omega])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'RelativeState':
        return cls(arr[0], arr[1], arr[2], arr[3], arr[4])


class TrackingModel:
    def __init__(self, a_bounds: Tuple[float, float], 
                 alpha_bounds: Tuple[float, float],
                 d_bounds: dict):
        self.a_min, self.a_max = a_bounds
        self.alpha_min, self.alpha_max = alpha_bounds
        self.d_bounds = d_bounds
        
    def dynamics(self, state: TrackingModelState, 
                 control: Tuple[float, float],
                 disturbance: Optional[Tuple[float, float, float, float]] = None
                ) -> np.ndarray:
        a, alpha = control
        
        if disturbance is None:
            d_x, d_y, d_a, d_alpha = 0.0, 0.0, 0.0, 0.0
        else:
            d_x, d_y, d_a, d_alpha = disturbance
        
        x_dot = state.v * np.cos(state.theta) + d_x
        y_dot = state.v * np.sin(state.theta) + d_y
        theta_dot = state.omega
        v_dot = a + d_a
        omega_dot = alpha + d_alpha
        
        return np.array([x_dot, y_dot, theta_dot, v_dot, omega_dot])


class PlanningModel:
    def __init__(self, v_const: float, omega_bounds: Tuple[float, float]):
        self.v = v_const
        self.omega_min, self.omega_max = omega_bounds
    
    def dynamics(self, state: PlanningModelState, 
                 control: float) -> np.ndarray:
        omega = control
        
        x_dot = self.v * np.cos(state.theta)
        y_dot = self.v * np.sin(state.theta)
        theta_dot = omega
        
        return np.array([x_dot, y_dot, theta_dot])


def wrap_angle(a):
    return (a + math.pi) % (2*math.pi) - math.pi


class RelativeDynamics:
    def __init__(self, tracking_model: TrackingModel, 
                 planning_model: PlanningModel):
        self.tracking = tracking_model
        self.planning = planning_model
    
    def compute_relative_state(self, tracker: TrackingModelState,
                               planner: PlanningModelState) -> RelativeState:
        dx = tracker.x - planner.x
        dy = tracker.y - planner.y
        
        cos_theta_p = np.cos(planner.theta)
        sin_theta_p = np.sin(planner.theta)
        
        x_r = cos_theta_p * dx + sin_theta_p * dy
        y_r = -sin_theta_p * dx + cos_theta_p * dy
        
        theta_r = tracker.theta - planner.theta
        theta_r = wrap_angle(theta_r)
        
        return RelativeState(x_r, y_r, theta_r, tracker.v, tracker.omega)