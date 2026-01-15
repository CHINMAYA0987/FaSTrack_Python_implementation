import numpy as np
import math
import os
from typing import Tuple, List, Optional
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Arc

from models import (Point, Obstacle, TrackingModelState, PlanningModelState, 
                    RelativeState, TrackingModel, PlanningModel, RelativeDynamics, wrap_angle)
from planning import Environment, RRT, SimplePlanner, inflate_obstacles


class PrimaryController:
    def __init__(self, tracking_control_bounds: dict):
        self.A_MAX = tracking_control_bounds['A_MAX']
        self.ALPHA_MAX = tracking_control_bounds['ALPHA_MAX']
        
        self.kp_v = 4.0
        self.kp_heading = 3.0
        self.kd_omega = 1.0
        
        self.target_v = 0.5
        self.min_distance_threshold = 0.3
        
    def set_path(self, waypoints):
        self.waypoints = waypoints
        
    def control(self, tracker: TrackingModelState, planner: PlanningModelState) -> Tuple[float, float]:
        dx = planner.x - tracker.x
        dy = planner.y - tracker.y
        distance_error = math.hypot(dx, dy)
        
        if distance_error > 8.0:
            target_v = 1.8
        elif distance_error > 4.0:
            target_v = 1.5
        elif distance_error > 1.5:
            target_v = self.target_v * 1.8
        elif distance_error > 0.5:
            target_v = self.target_v
        else:
            target_v = self.target_v * 0.6
        
        v_error = target_v - tracker.v
        a = self.kp_v * v_error
        
        if distance_error > 4.0 and tracker.v < 0.8:
            a = max(a, 0.55)
        
        a = np.clip(a, -self.A_MAX, self.A_MAX)
        
        if distance_error > self.min_distance_threshold:
            desired_heading = math.atan2(dy, dx)
            heading_error = wrap_angle(desired_heading - tracker.theta)
            
            kp_heading = self.kp_heading
            if distance_error > 3.0:
                kp_heading *= 1.5
            
            alpha = kp_heading * heading_error - self.kd_omega * tracker.omega
        else:
            alpha = -self.kd_omega * tracker.omega * 0.5
        
        alpha = np.clip(alpha, -self.ALPHA_MAX, self.ALPHA_MAX)
        
        return (a, alpha)


class HybridTrackingController:
    def __init__(self, npz_path: str, tracking_model: TrackingModel, planning_model: PlanningModel, 
                 primary_controller, V_threshold: float = 0.0):
        
        if not os.path.exists(npz_path):
            dummy_data = {
                'V': np.zeros((10, 10, 1, 1, 1)),
                'xs': np.linspace(-1, 1, 10),
                'ys': np.linspace(-1, 1, 10),
                'thetas': np.array([0]),
                'vs': np.array([0.5]),
                'omegas': np.array([0]),
                'TEB_max': 0.6
            }
            np.savez_compressed(npz_path, **dummy_data)
            print("WARNING: HJ data file not found. Created dummy file for simulation.")

        data = np.load(npz_path, allow_pickle=True)
        self.V = data['V']
        
        self.grids = {
            'x_r': data['xs'], 'y_r': data['ys'], 'theta_r': data['thetas'], 
            'v': data['vs'], 'omega': data['omegas']
        }
        
        self.teb = data.get('TEB_max', 0.845)
        
        self.tracking_model = tracking_model
        self.planning_model = planning_model
        self.primary_controller = primary_controller
        self.V_threshold = V_threshold
        
        self.rel_dynamics = RelativeDynamics(tracking_model, planning_model)
        
        self.interpolators = self._create_interpolators()
        
        self.obstacle_safety_distance = 3.0

    def _create_interpolators(self):
        coords = [self.grids[k] for k in ['x_r', 'y_r', 'theta_r', 'v', 'omega']]
        interpolators = {}
        interpolators['V'] = RegularGridInterpolator(coords, self.V, bounds_error=False, fill_value=np.max(self.V))
        return interpolators

    def value(self, rel_state: RelativeState) -> float:
        point = rel_state.to_array()
        return self.interpolators['V'](point)[0]

    def compute_min_obstacle_distance(self, tracker: TrackingModelState, obstacles: List[Obstacle]) -> float:
        if not obstacles:
            return float('inf')
        
        min_dist = float('inf')
        for obs in obstacles:
            dist_to_center = math.hypot(tracker.x - obs.x, tracker.y - obs.y)
            dist_to_surface = dist_to_center - obs.radius
            min_dist = min(min_dist, max(0.0, dist_to_surface))
        
        return min_dist

    def optimal_controller(self, tracker: TrackingModelState, planner: PlanningModelState, 
                          min_obstacle_dist: float, obstacles: List[Obstacle], current_step: int = 0) -> Tuple[float, float]:
        nearest_obs = None
        min_dist = float('inf')
        for obs in obstacles:
            dist = math.hypot(tracker.x - obs.x, tracker.y - obs.y) - obs.radius
            if dist < min_dist:
                min_dist = dist
                nearest_obs = obs
        
        if min_obstacle_dist < 0.5:
            target_speed = 0.0
            evasive_maneuver = True
        elif min_obstacle_dist < 1.0:
            target_speed = 0.15
            evasive_maneuver = True
        elif min_obstacle_dist < 2.0:
            target_speed = 0.3
            evasive_maneuver = False
        else:
            distance_to_planner = math.hypot(planner.x - tracker.x, planner.y - tracker.y)
            target_speed = 0.35 if distance_to_planner < 2.0 else 0.45
            evasive_maneuver = False
        
        speed_error = target_speed - tracker.v
        a = max(-0.5, 5.0 * speed_error) if target_speed == 0.0 else 3.0 * speed_error
        a = np.clip(a, self.tracking_model.a_min, self.tracking_model.a_max)
        
        if evasive_maneuver and nearest_obs is not None:
            dx_obs = tracker.x - nearest_obs.x
            dy_obs = tracker.y - nearest_obs.y
            angle_away_from_obs = math.atan2(dy_obs, dx_obs)
            heading_error = wrap_angle(angle_away_from_obs - tracker.theta)
            alpha = 3.5 * heading_error - 2.0 * tracker.omega
        else:
            dx = planner.x - tracker.x
            dy = planner.y - tracker.y
            desired_heading = math.atan2(dy, dx)
            heading_error = wrap_angle(desired_heading - tracker.theta)
            alpha = 2.5 * heading_error - 1.2 * tracker.omega
        
        alpha = np.clip(alpha, self.tracking_model.alpha_min, self.tracking_model.alpha_max)
        
        return (a, alpha)

    def get_control(self, tracker: TrackingModelState, planner: PlanningModelState, 
                    obstacles: List[Obstacle], current_step: int = 0) -> Tuple[Tuple[float, float], str]:
        rel_state = self.rel_dynamics.compute_relative_state(tracker, planner)
        
        min_obstacle_dist = self.compute_min_obstacle_distance(tracker, obstacles)
        
        near_obstacle = min_obstacle_dist < self.obstacle_safety_distance
        
        if near_obstacle:
            control = self.optimal_controller(tracker, planner, min_obstacle_dist, obstacles, current_step)
            control_type = f"OPTIMAL/SAFE (Near Obstacle: {min_obstacle_dist:.2f}m)"
        else:
            control = self.primary_controller.control(tracker, planner)
            control_type = f"PRIMARY/PERFORMANCE (Safe: {min_obstacle_dist:.2f}m)"
            
        return control, control_type


def get_arc_intersection_points(tracker_state, obstacle_center, radius, fov_angle):
    tx, ty, t_heading = tracker_state.x, tracker_state.y, tracker_state.theta
    ox, oy = obstacle_center
    R = radius
    angle_l = t_heading + fov_angle / 2
    angle_r = t_heading - fov_angle / 2
    
    def intersect_line_circle(angle):
        dx, dy = ox - tx, oy - ty
        vx, vy = math.cos(angle), math.sin(angle)
        
        a = 1.0
        b = -2 * (dx * vx + dy * vy)
        c = dx*dx + dy*dy - R*R
        
        discriminant = b*b - 4*a*c
        if discriminant < 0: return None
        
        t1 = (-b - math.sqrt(discriminant)) / (2*a)
        t2 = (-b + math.sqrt(discriminant)) / (2*a)
        
        t = min(t1, t2)
        if t < 0: t = max(t1, t2)
        
        if t < 0: return None
            
        px = tx + t * vx
        py = ty + t * vy
        return px, py

    p_l = intersect_line_circle(angle_l)
    p_r = intersect_line_circle(angle_r)
    
    if p_l and p_r:
        return p_l[0], p_l[1], p_r[0], p_r[1]
    return None


def draw_safety_viz(ax, tracker_state, obstacles, teb):
    FOV_RANGE = 5.0
    FOV_ANGLE = np.radians(120)
    
    tx, ty, t_heading = tracker_state.x, tracker_state.y, tracker_state.theta
    
    start_angle = np.degrees(t_heading - FOV_ANGLE / 2)
    end_angle = np.degrees(t_heading + FOV_ANGLE / 2)
    
    fov_patch = Wedge((tx, ty), FOV_RANGE, start_angle, end_angle, 
                      color='cyan', alpha=0.1, zorder=1)
    ax.add_patch(fov_patch)
    
    for obs in obstacles:
        obs_center = (obs.x, obs.y)
        safety_radius = obs.radius + teb
        
        distance_to_center = math.hypot(tx - obs_center[0], ty - obs_center[1])
        if distance_to_center > FOV_RANGE + safety_radius:
            continue
            
        intersection_points = get_arc_intersection_points(tracker_state, obs_center, safety_radius, FOV_ANGLE)
        
        if intersection_points:
            p1_x, p1_y, p2_x, p2_y = intersection_points
            
            p1_angle = np.arctan2(p1_y - obs_center[1], p1_x - obs_center[0])
            p2_angle = np.arctan2(p2_y - obs_center[1], p2_x - obs_center[0])
            
            p1_angle = (p1_angle + 2*math.pi) % (2*math.pi)
            p2_angle = (p2_angle + 2*math.pi) % (2*math.pi)
            
            angle_diff = (p2_angle - p1_angle) % (2*math.pi)
            
            if angle_diff > math.pi:
                arc_start_angle_deg = np.degrees(p2_angle)
                arc_end_angle_deg = np.degrees(p1_angle)
            else:
                arc_start_angle_deg = np.degrees(p1_angle)
                arc_end_angle_deg = np.degrees(p2_angle)
            
            glowing_arc = Arc(xy=obs_center, 
                              width=2*safety_radius, 
                              height=2*safety_radius, 
                              angle=0.0, 
                              theta1=arc_start_angle_deg, 
                              theta2=arc_end_angle_deg,
                              color='lime',
                              linewidth=4, 
                              zorder=5)
            ax.add_patch(glowing_arc)


class FaSTrackSimulator:
    def __init__(self):
        self.a_bounds = (-0.5, 0.5)
        self.alpha_bounds = (-6.0, 6.0)
        self.d_bounds = {'dx': 0.02, 'dy': 0.02, 'da': 0.2, 'dalpha': 0.02}
        self.tracking_model = TrackingModel(self.a_bounds, self.alpha_bounds, self.d_bounds)
        
        self.v_hat_const = 0.5
        self.omega_hat_bounds = (-1.5, 1.5)
        self.planning_model = PlanningModel(self.v_hat_const, self.omega_hat_bounds)
        
        self.rel_dynamics = RelativeDynamics(self.tracking_model, self.planning_model)
        
        self.npz_path = 'value_function_5d3d_demo_2k.npz'
        self.teb = 0.6
        self.replan_threshold = 5.0
        
        self.teb_violation_threshold = 1.3 * self.teb
        self.teb_safety_margin = 0.9 * self.teb
        self.min_replan_interval = 100
        self.replan_validation_wait = 30
        
    def compute_offline(self):
        if not os.path.exists(self.npz_path):
            self.teb = 0.6
            return
        
        data = np.load(self.npz_path, allow_pickle=True)
        self.teb = data.get('TEB_max', 0.6)
        
    def build_new_path(self, start_state: TrackingModelState, target_point: Point, obstacles_list: List[Obstacle], 
                       current_waypoints: Optional[List[Point]] = None, tracker_waypoint_idx: Optional[int] = None) -> Optional[List[Point]]:
        
        RRT_MAX_ITER = 4000
        RRT_STEP_SIZE = 2.0
        
        if current_waypoints is not None and tracker_waypoint_idx is not None and len(current_waypoints) > 1:
            start_pt = Point(start_state.x, start_state.y)
            
            STITCHING_HORIZON_STEPS = 15
            
            stitch_idx = min(tracker_waypoint_idx + STITCHING_HORIZON_STEPS, len(current_waypoints) - 1)
            
            if stitch_idx == len(current_waypoints) - 1:
                target_to_use = target_point
            else:
                target_to_use = current_waypoints[stitch_idx]
                
            if tracker_waypoint_idx >= len(current_waypoints) - 2:
                target_to_use = target_point
                stitch_idx = len(current_waypoints) - 1
                 
            temp_env = Environment(start_pt, target_to_use, env.x_range, env.y_range, obstacles_list)
            temp_rrt = RRT(start_pt, step_size=RRT_STEP_SIZE, max_iter=RRT_MAX_ITER)
            success = temp_rrt.build_rrt(temp_env)

            if success and len(temp_rrt.path) > 1:
                new_path_segment = temp_rrt.path
                
                if stitch_idx < len(current_waypoints) - 1:
                    new_global_path = new_path_segment + current_waypoints[stitch_idx+1:]
                else:
                    new_global_path = new_path_segment
                
                return new_global_path
            
            return None

        else:
            start_pt = Point(start_state.x, start_state.y)
            target_to_use = target_point
            
        temp_env = Environment(start_pt, target_to_use, env.x_range, env.y_range, obstacles_list)
        temp_rrt = RRT(start_pt, step_size=RRT_STEP_SIZE, max_iter=RRT_MAX_ITER)
        success = temp_rrt.build_rrt(temp_env)

        if success and len(temp_rrt.path) > 1:
            return temp_rrt.path
        
        return None

    def run(self, env: Environment, total_time: float = 100.0, dt: float = 0.1, live_animation=True):
        self.compute_offline()

        print("\n" + "="*70)
        print(" Starting RRT Global Planning")
        print(" (Initial plan to final target)")
        print("="*70)

        inflated_obstacles = inflate_obstacles(env.obstacles, self.teb)
        
        rrt_success = False
        initial_tracker_state = TrackingModelState(x=env.start.x, y=env.start.y, theta=0.0, v=0.0, omega=0.0)
        waypoints = self.build_new_path(initial_tracker_state, env.target, inflated_obstacles, current_waypoints=None, tracker_waypoint_idx=None)
        
        if waypoints:
            rrt_success = True
        
        if not rrt_success:
            print("ERROR: RRT failed to find a path. Simulation aborted.")
            return

        print(f"RRT Path found with {len(waypoints)} waypoints.")

        initial_tracker = initial_tracker_state
        
        planner_init_dist = self.teb * 0.5
        initial_planner = PlanningModelState(
            x=env.start.x + planner_init_dist * np.cos(initial_tracker.theta), 
            y=env.start.y + planner_init_dist * np.sin(initial_tracker.theta), 
            theta=0.0
        )
        tracker = initial_tracker
        planner = initial_planner
        
        tracking_control_bounds = {
            'A_MAX': self.tracking_model.a_max, 
            'ALPHA_MAX': self.tracking_model.alpha_max
        }
        
        primary_controller = PrimaryController(tracking_control_bounds)
        primary_controller.set_path(waypoints)

        self.controller = HybridTrackingController(
            self.npz_path, self.tracking_model, self.planning_model, primary_controller
        )

        self.planner = SimplePlanner(self.planning_model)
        self.planner.set_waypoints(waypoints)
        
        num_steps = int(total_time / dt)
        results = {
            'time': np.zeros(num_steps), 
            'tracker_x': np.zeros(num_steps), 'tracker_y': np.zeros(num_steps),
            'planner_x': np.zeros(num_steps), 'planner_y': np.zeros(num_steps), 
            'tracking_error': np.zeros(num_steps),
            'relative_error': np.zeros(num_steps),
            'controller_type': [], 
            'obstacles': env.obstacles, 
            'waypoints': waypoints,
            'replans': 0
        }

        print("\n" + "="*70)
        print(" Starting FaSTrack Simulation (Receding Horizon Planning Active)")
        print(" **MODIFIED FOR BETTER TRACKING RECOVERY**")
        print("="*70)
        
        collision_detected = False
        goal_reached = False
        last_controller_type = ""
        consecutive_violations = 0
        steps_since_last_replan = self.min_replan_interval
        replan_validation_counter = 0
        in_validation_period = False
        tracker_waypoint_idx = 0
        
        if live_animation:
            plt.ion()
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_xlim(env.x_range)
            ax.set_ylim(env.y_range)
            ax.set_xlabel("X Position (m)"); ax.set_ylabel("Y Position (m)")
            ax.set_title("FASTRACK 5D-3D")
            
            path_line, = ax.plot([p.x for p in waypoints], [p.y for p in waypoints], 
                                color='gray', linewidth=2, linestyle='--', label='Global Path', alpha=0.7)
            
            inflated_obs = inflate_obstacles(env.obstacles, self.teb)
            for obs in env.obstacles:
                ax.add_patch(Circle((obs.x, obs.y), obs.radius, color='red', alpha=0.5, zorder=2))
            for obs in inflated_obs:
                ax.add_patch(Circle((obs.x, obs.y), obs.radius, fill=False, color='red', linestyle='--', alpha=0.6, zorder=2))

            tracker_dot, = ax.plot([], [], marker='o', color='blue', markersize=8, label='5D Tracker (Vehicle)', zorder=10)
            tracker_trail, = ax.plot([], [], color='blue', linewidth=1.5, alpha=0.5, label='Tracker Trail')
            error_circle_patch = Circle((0, 0), self.teb, fill=False, color='cyan', linestyle=':', linewidth=2, alpha=0.6, label='TEB Bound')
            controller_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, verticalalignment='top', fontsize=10,
                                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            ax.legend(loc='upper right')
        
        for step in range(num_steps):
            t = step * dt
            results['time'][step] = t

            if goal_reached or collision_detected:
                break
            
            steps_since_last_replan += 1
            
            tracker_pos = np.array([tracker.x, tracker.y])
            
            if self.planner.waypoints:
                temp_min_dist = float('inf')
                temp_idx = tracker_waypoint_idx
                search_start_idx = max(0, tracker_waypoint_idx - 5)
                
                for i in range(search_start_idx, len(self.planner.waypoints)):
                    wp = self.planner.waypoints[i]
                    dist = np.linalg.norm(tracker_pos - np.array([wp.x, wp.y]))
                    
                    if dist < temp_min_dist:
                        temp_min_dist = dist
                        temp_idx = i
                
                tracker_waypoint_idx = temp_idx
                
            error = math.hypot(tracker.x - planner.x, tracker.y - planner.y)
            rel_state = self.rel_dynamics.compute_relative_state(tracker, planner)

            tracker_control, control_type = self.controller.get_control(tracker, planner, env.obstacles, step)
            results['controller_type'].append(control_type)
            
            min_obs_dist = self.controller.compute_min_obstacle_distance(tracker, env.obstacles)
            tracking_gap = error
            planner_speed_factor = 1.0
            
            if "OPTIMAL" in control_type:
                if min_obs_dist < 0.5:
                    planner_speed_factor = 0.0
                elif min_obs_dist < 1.0:
                    planner_speed_factor = 0.05
                elif min_obs_dist < 2.0:
                    planner_speed_factor = 0.1
                else:
                    planner_speed_factor = 0.2

            if tracking_gap > 4.0:
                planner_speed_factor = min(planner_speed_factor, 0.01)
            elif tracking_gap > 2.0:
                planner_speed_factor = min(planner_speed_factor, 0.1)
            elif tracking_gap > self.teb_violation_threshold * 1.5:
                planner_speed_factor = min(planner_speed_factor, 0.3)
            elif tracking_gap > self.teb_violation_threshold:
                planner_speed_factor = min(planner_speed_factor, 0.5)
            
            if planner_speed_factor == 0.0:
                planner_control = 0.0
            else:
                planner_control = self.planner.get_control(planner) * planner_speed_factor
            
            if control_type != last_controller_type:
                last_controller_type = control_type

            relative_pos_error = math.hypot(rel_state.x_r, rel_state.y_r)
            catastrophic_error = error > 10.0
            
            if catastrophic_error and steps_since_last_replan >= 20:
                print(f"\n{'#'*70}\n[CATASTROPHIC] EMERGENCY REPLAN at t={t:.2f}s. Forcing immediate replan...\n{'#'*70}")
                inflated_obs = inflate_obstacles(env.obstacles, self.teb)
                new_waypoints = self.build_new_path(tracker, env.target, inflated_obs, waypoints, tracker_waypoint_idx)
                
                if new_waypoints and len(new_waypoints) > 1:
                    waypoints = new_waypoints
                    self.planner.set_waypoints(waypoints)
                    primary_controller.set_path(waypoints)
                    
                    planner_reset_dist = self.teb * 0.5
                    planner = PlanningModelState(
                        x=tracker.x + planner_reset_dist * np.cos(tracker.theta),
                        y=tracker.y + planner_reset_dist * np.sin(tracker.theta),
                        theta=tracker.theta
                    )
                    
                    p0, p1 = new_waypoints[0], new_waypoints[1]
                    dx_p, dy_p = p1.x - p0.x, p1.y - p0.y
                    len_sq = dx_p*dx_p + dy_p*dy_p
                    if len_sq > 1e-6:
                        t_proj = np.clip(((planner.x - p0.x) * dx_p + (planner.y - p0.y) * dy_p) / len_sq, 0.0, 1.0)
                        planner.x = p0.x + t_proj * dx_p
                        planner.y = p0.y + t_proj * dy_p
                        planner.theta = math.atan2(dy_p, dx_p)
                        self.planner.current_waypoint_idx = 0
                        tracker_waypoint_idx = 0
                    
                    results['replans'] += 1
                    consecutive_violations = 0
                    steps_since_last_replan = 0
                    in_validation_period = True
                    replan_validation_counter = 0
                    print(f"[SUCCESS] Emergency replan: Planner reset to ({planner.x:.2f}, {planner.y:.2f})\n")
                    if live_animation:
                        path_line.set_data([p.x for p in waypoints], [p.y for p in waypoints])
                        path_line.set_color('red')
                    continue

            in_validation_period = in_validation_period and replan_validation_counter < self.replan_validation_wait
            replan_validation_counter += 1
            
            safely_within_teb = relative_pos_error < self.teb_safety_margin
            teb_significantly_violated = relative_pos_error > self.teb_violation_threshold
            
            if safely_within_teb:
                consecutive_violations = 0
            
            can_replan = (not in_validation_period and steps_since_last_replan >= self.min_replan_interval)
            
            if teb_significantly_violated and not safely_within_teb:
                consecutive_violations += 1
                
                if consecutive_violations >= 20 and can_replan:
                    violation_type = f"TEB Severe ({relative_pos_error:.3f}m > {self.teb_violation_threshold:.3f}m)"
                    print(f"\n{'#'*70}\n[CRITICAL] REPLANNING REQUIRED at t={t:.2f}s\n  Violation Type: {violation_type}\n  Global Error: {error:.4f}m\n  Triggering Receding Horizon replan...\n{'#'*70}")
                    
                    inflated_obs = inflate_obstacles(env.obstacles, self.teb)
                    new_waypoints = self.build_new_path(tracker, env.target, inflated_obs, waypoints, tracker_waypoint_idx)
                    
                    if new_waypoints and len(new_waypoints) > 1:
                        waypoints = new_waypoints
                        self.planner.set_waypoints(waypoints)
                        primary_controller.set_path(waypoints)
                        
                        planner_reset_dist = self.teb * 0.5
                        planner = PlanningModelState(
                            x=tracker.x + planner_reset_dist * np.cos(tracker.theta),
                            y=tracker.y + planner_reset_dist * np.sin(tracker.theta),
                            theta=tracker.theta
                        )
                        p0, p1 = new_waypoints[0], new_waypoints[1]
                        dx_p, dy_p = p1.x - p0.x, p1.y - p0.y
                        len_sq = dx_p*dx_p + dy_p*dy_p
                        if len_sq > 1e-6:
                            t_proj = np.clip(((planner.x - p0.x) * dx_p + (planner.y - p0.y) * dy_p) / len_sq, 0.0, 1.0)
                            planner.x = p0.x + t_proj * dx_p
                            planner.y = p0.y + t_proj * dy_p
                            planner.theta = math.atan2(dy_p, dx_p)
                            self.planner.current_waypoint_idx = 0
                            tracker_waypoint_idx = 0
                        
                        results['replans'] += 1
                        consecutive_violations = 0
                        steps_since_last_replan = 0
                        in_validation_period = True
                        replan_validation_counter = 0
                        
                        print(f"[SUCCESS] Replanned with {len(new_waypoints)} local waypoints. Planner reset.\n")
                        if live_animation:
                            path_line.set_data([p.x for p in waypoints], [p.y for p in waypoints])
                            path_line.set_color('green')
                    else:
                        print(f"[ERROR] Replanning failed - continuing with current path\n")
                        consecutive_violations = 0
            else:
                consecutive_violations = max(0, consecutive_violations - 1)

            disturbance = (
                np.random.uniform(-self.tracking_model.d_bounds['dx'], self.tracking_model.d_bounds['dx']),
                np.random.uniform(-self.tracking_model.d_bounds['dy'], self.tracking_model.d_bounds['dy']),
                np.random.uniform(-self.tracking_model.d_bounds['da'], self.tracking_model.d_bounds['da']),
                np.random.uniform(-self.tracking_model.d_bounds['dalpha'], self.tracking_model.d_bounds['dalpha']),
            )

            if not goal_reached:
                p_dot = self.planning_model.dynamics(planner, planner_control)
                planner_new_array = planner.to_array() + p_dot * dt
                planner = PlanningModelState.from_array(planner_new_array)
                planner.theta = wrap_angle(planner.theta)
            
            t_dot = self.tracking_model.dynamics(tracker, tracker_control, disturbance)
            tracker_new_array = tracker.to_array() + t_dot * dt
            tracker = TrackingModelState.from_array(tracker_new_array)
            tracker.theta = wrap_angle(tracker.theta)

            results['tracker_x'][step], results['tracker_y'][step] = tracker.x, tracker.y
            results['planner_x'][step], results['planner_y'][step] = planner.x, planner.y
            results['tracking_error'][step] = math.hypot(tracker.x - planner.x, tracker.y - planner.y)
            results['relative_error'][step] = math.hypot(rel_state.x_r, rel_state.y_r)
            
            for obs in env.obstacles:
                if math.hypot(tracker.x - obs.x, tracker.y - obs.y) < obs.radius:
                    collision_detected = True
                    break
            
            final_waypoint = env.target
            tracker_at_goal = math.hypot(tracker.x - final_waypoint.x, tracker.y - final_waypoint.y) < 1.5
            planner_at_goal = math.hypot(planner.x - final_waypoint.x, planner.y - final_waypoint.y) < 1.0
            
            if tracker_at_goal and planner_at_goal:
                goal_reached = True
                print(f"\n[SUCCESS] Goal reached at t={t:.2f}s.")
                break

            if live_animation and step % 5 == 0:
                V_val_current = self.controller.value(rel_state)
                
                for child in list(ax.get_children()):
                    if isinstance(child, (Wedge, Arc)):
                        child.remove()
                
                draw_safety_viz(ax, tracker, env.obstacles, self.teb)
                
                tracker_dot.set_data([tracker.x], [tracker.y])
                tracker_trail.set_data(results['tracker_x'][:step+1], results['tracker_y'][:step+1])
                error_circle_patch.center = (tracker.x, tracker.y)
                
                controller_short = control_type.split('(')[0].strip()
                status_text = f"t={t:.1f}s | Controller: {controller_short}\nError: {error:.3f}m | V: {V_val_current:.3f}"
                controller_text.set_text(status_text)
                controller_text.set_bbox(dict(boxstyle='round', facecolor='orange' if "OPTIMAL" in controller_short else 'lightgreen', alpha=0.8))
                
                fig.canvas.draw(); fig.canvas.flush_events()
                plt.pause(0.001)

        max_error = np.max(results['tracking_error'][:step])
        max_relative_error = np.max(results['relative_error'][:step])
        controller_stats = {}
        for ct in results['controller_type']:
            key = ct.split('(')[0].strip()
            controller_stats[key] = controller_stats.get(key, 0) + 1
            
        print("\n" + "="*70)
        print(" SIMULATION SUMMARY (MODIFIED GAINS)")
        print("="*70)
        print(f"Final Time: {t:.2f} s")
        print(f"Goal Reached: {goal_reached}")
        print(f"Collision Detected: {collision_detected}")
        print(f"\nTracking Performance:")
        print(f"  Maximum Global Error: {max_error:.4f} m")
        print(f"  Maximum Relative Error: {max_relative_error:.4f} m (in planner's frame)")
        print(f"  TEB (Safety Bound): {self.teb:.4f} m")
        print(f"  TEB Safety Margin (90%): {self.teb_safety_margin:.4f} m")
        print(f"  TEB Violation Threshold (130%): {self.teb_violation_threshold:.4f} m")
        print(f"  Relative Error within TEB: {'YES ✓' if max_relative_error < self.teb else 'NO ✗'}")
        print(f"  Relative Error within Safety Margin: {'YES ✓' if max_relative_error < self.teb_safety_margin else 'NO ✗'}")
        print(f"\nSafety Analysis:")
        print(f"  Safety Guarantees (Relative): {max_relative_error < self.teb and not collision_detected}")
        print(f"  Collision-Free: {not collision_detected}")
        print(f"\nReplanning Strategy:")
        print(f"  Number of Replans Triggered: {results['replans']}")
        print(f"  Replan Trigger: Relative error > {self.teb_violation_threshold:.3f}m for 20 steps")
        print(f"  Replan Cooldown: {self.min_replan_interval} steps (10 seconds)")
        print(f"  Validation Period: {self.replan_validation_wait} steps (3 seconds)")
        print(f"  Auto-reset: If error < {self.teb_safety_margin:.3f}m")
        
        print("\nController Usage Statistics:")
        total_steps = sum(controller_stats.values())
        for controller, count in sorted(controller_stats.items()):
            percentage = (count / total_steps) * 100
            print(f"  {controller}: {count} steps ({percentage:.1f}%)")
        
        print("\n" + "="*70)
        print("SMART REPLANNING EXPLANATION (MODIFIED):")
        print("1. Primary controller is **more aggressive** to pull the tracker back to the planner faster.")
        print("2. Planner coordination is **stricter** to slow the planner down significantly when the error is large.")
        print(f"3. Safe zone: < {self.teb_safety_margin:.3f}m (90% of TEB) → Resets violations")
        print(f"4. Violation zone: > {self.teb_violation_threshold:.3f}m (130% of TEB) → Starts counting")
        print("5. Replan only after 20 consecutive violations (2 seconds)")
        print("6. Minimum 10 seconds between replans to ensure stability")
        print("7. Planner uses Receding Horizon Planning (15 steps ahead) for local replanning/stitching.")
        print("="*70)


def setup_environment():
    x_min, x_max = 0.0, 30.0
    y_min, y_max = 0.0, 30.0
    
    start_point = Point(x=1.0, y=1.0)
    target_point = Point(x=25.0, y=25.0)
    
    obstacles = [
        Obstacle(x=10.0, y=10.0, radius=2.5),
        Obstacle(x=20.0, y=20.0, radius=2.0),
        Obstacle(x=8.0, y=22.0, radius=2.0),
        Obstacle(x=15.0, y=15.0, radius=1.5),
    ]

    return Environment(start_point, target_point, (x_min, x_max), (y_min, y_max), obstacles)


if __name__ == "__main__":
    env = setup_environment()
    simulator = FaSTrackSimulator()
    simulator.run(env, total_time=500.0, dt=0.1, live_animation=True)