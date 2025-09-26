import numpy as np
import math
import random
from guidance.VehicleParams import VehicleParams
from guidance.FinControlSystem import FinControlSystem
class GuidedMissile:
    def __init__(self, target, defense_systems, params: VehicleParams, launch_speed, launch_elev_deg, launch_azim_deg):
        self.target = target
        self.defense_systems = defense_systems
        self.params = params
        self.launch_speed = float(launch_speed)
        self.launch_elev_deg = float(launch_elev_deg)
        self.launch_azim_deg = float(launch_azim_deg)
        self.terminal_phase = False
        self.fin_control = FinControlSystem(params)
        self.seeker_lock = False
        self.lock_time = 0.0
        self.radar_contacts = []
        self.destroyed = False

    def is_target_in_radar_range(self, missile_pos, target_pos):
        """Check if target is within radar range"""
        dist_to_target = np.linalg.norm(target_pos - missile_pos)
        return dist_to_target <= self.params.radar_range

    def detect_defenses(self, missile_pos):
        """Detect defense systems within radar range"""
        detected_defenses = []
        for defense in self.defense_systems:
            if defense.active and np.linalg.norm(defense.position - missile_pos) <= self.params.radar_range:
                detected_defenses.append(defense)
        return detected_defenses

    def calculate_avoidance_vector(self, missile_pos, missile_vel, defenses, target_pos, dt):
        """Calculate balanced avoidance vector that doesn't compromise mission"""
        if not defenses:
            return np.zeros(3)

        avoidance_vector = np.zeros(3)
        missile_speed = np.linalg.norm(missile_vel)

        if missile_speed < 1e-9:
            return avoidance_vector

        missile_dir = missile_vel / missile_speed

        # Calculate direction to target
        to_target = target_pos - missile_pos
        to_target_dist = np.linalg.norm(to_target)
        if to_target_dist > 1e-9:
            target_dir = to_target / to_target_dist
        else:
            target_dir = missile_dir

        total_threat_level = 0.0

        for defense in defenses:
            if defense.is_missile_in_range(missile_pos):
                # Calculate threat direction and distance
                threat_dir = defense.get_threat_direction(missile_pos)
                threat_dist = np.linalg.norm(defense.position - missile_pos)

                # Calculate threat urgency (closer = more urgent)
                threat_urgency = 1.0 - (threat_dist / defense.range)
                threat_urgency = max(0.1, min(1.0, threat_urgency))  # Clamp between 0.1-1.0

                # We want to move away from the threat, but not straight away
                # Use perpendicular avoidance to maintain forward momentum
                avoidance_dir = -threat_dir

                # Blend with target direction to maintain course
                blend_factor = 0.3 + (0.7 * threat_urgency)  # More threat = more avoidance
                avoidance_dir = (blend_factor * avoidance_dir +
                                 (1 - blend_factor) * target_dir)
                avoidance_dir /= np.linalg.norm(avoidance_dir) + 1e-12

                # Weight by threat level and urgency
                weight = defense.threat_level * threat_urgency

                # Scale avoidance based on missile speed (faster = less drastic maneuvers)
                speed_factor = min(1.0, missile_speed / 200.0)  # Normalize to 200 m/s
                weight *= speed_factor

                avoidance_vector += avoidance_dir * weight
                total_threat_level += weight

        if total_threat_level > 1e-9:
            # Normalize the avoidance vector
            avoidance_vector /= total_threat_level

            # Apply obstacle avoidance weight, but scale it based on threat level
            # Don't let avoidance completely override target pursuit
            max_avoidance = min(self.params.obstacle_avoidance_weight,
                                2.0 * total_threat_level)  # Cap avoidance strength

            avoidance_vector *= max_avoidance

            # Ensure we don't lose too much forward momentum
            forward_component = np.dot(avoidance_vector, missile_dir)
            if forward_component < -0.5:  # If avoidance is pushing too much backward
                # Add some forward bias to maintain momentum
                avoidance_vector += missile_dir * 0.3 * abs(forward_component)

        return avoidance_vector

    def run_guided(self, dt=0.01, max_time=60.0, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        elev = math.radians(self.launch_elev_deg)
        azim = math.radians(self.launch_azim_deg)
        pos = np.array([0.0, 0.0, 0.0], dtype=float)
        vel = self.launch_speed * np.array([
            math.cos(elev) * math.cos(azim),
            math.cos(elev) * math.sin(azim),
            math.sin(elev)
        ], dtype=float)

        initial_forward = vel / (np.linalg.norm(vel) + 1e-12)
        t = 0.0
        traj = [pos.copy()]
        rocket_time_remain = self.params.thrust_duration
        prev_heading_error = 0.0
        integral_heading_error = 0.0
        prev_speed = self.launch_speed

        # For recording history
        time_history = [t]
        fin_history = [self.fin_control.deflections.copy()]
        guidance_history = [np.zeros(3)]
        radar_history = [False]
        target_history = [self.target.position.copy()]
        defense_history = []
        projectile_history = [[] for _ in self.defense_systems]
        velocity_history = [vel.copy()]

        # For predictive guidance
        target_velocity_estimate = np.zeros(3)
        last_target_pos = self.target.position.copy()
        prediction_horizon = 2.0  # seconds to predict ahead

        while t < max_time and not self.destroyed:
            # Update target position
            target_pos = self.target.update(dt, t)

            # Estimate target velocity for predictive guidance
            if t > 0:
                target_velocity_estimate = (target_pos - last_target_pos) / dt
                last_target_pos = target_pos.copy()

            # Update defense systems and their projectiles
            for i, defense in enumerate(self.defense_systems):
                defense.update_projectiles(dt)
                projectile_history[i].append(
                    defense.projectiles[-1].position.copy() if defense.projectiles else np.array(
                        [np.nan, np.nan, np.nan]))

                # Try to fire at missile
                defense.try_fire_at_target(pos, t)

                # Check for collisions with projectiles
                if defense.check_projectile_collision(pos):
                    print("💥 MISSILE DESTROYED BY DEFENSE!")
                    self.destroyed = True
                    break

            # Check if target is in radar range
            in_radar_range = self.is_target_in_radar_range(pos, target_pos)
            radar_history.append(in_radar_range)

            # Detect defense systems
            detected_defenses = self.detect_defenses(pos)
            defense_history.append([defense.position.copy() for defense in detected_defenses])

            # Calculate avoidance vector if defenses are detected
            avoidance_vector = self.calculate_avoidance_vector(pos, vel, detected_defenses, target_pos, dt)

            # For radar homing, only guide if in radar range
            dist_to_target = np.linalg.norm(target_pos - pos)
            if dist_to_target > self.params.radar_range or not in_radar_range:
                # No guidance - ballistic trajectory
                guidance_accel = np.zeros(3)
                self.seeker_lock = False
                self.lock_time = 0.0
            else:
                # Radar homing guidance with predictive targeting
                if not self.seeker_lock:
                    self.seeker_lock = True
                    print(f"🔒 RADAR LOCK at {dist_to_target:.0f}m")

                self.lock_time += dt

                # Predictive guidance: aim where the target will be, not where it is
                time_to_impact = dist_to_target / max(np.linalg.norm(vel), 1.0)
                prediction_time = min(time_to_impact, prediction_horizon)

                # Predict target position
                predicted_target_pos = target_pos + target_velocity_estimate * prediction_time

                to_target = predicted_target_pos - pos
                r = np.linalg.norm(to_target) + 1e-12
                los_unit = to_target / r

                # BALANCED guidance: Blend target direction with avoidance
                threat_level = min(1.0, len(detected_defenses) * 0.3)  # Scale with number of threats

                guidance_dir = (1.0 - threat_level) * los_unit + threat_level * avoidance_vector
                guidance_dir_norm = np.linalg.norm(guidance_dir)
                if guidance_dir_norm > 1e-9:
                    guidance_dir /= guidance_dir_norm

                # Calculate guidance acceleration
                missile_dir = vel / (np.linalg.norm(vel) + 1e-12)
                error_dir = guidance_dir - missile_dir

                # Adaptive guidance gain based on distance to target
                # Higher gain when closer to target for precision
                distance_factor = max(0.1, 1.0 - (dist_to_target / self.params.radar_range))
                guidance_gain = self.params.pn_gain * (1.0 + threat_level * 0.5 + distance_factor * 0.5)

                guidance_accel = guidance_gain * error_dir * max(np.linalg.norm(vel), 1.0)

                # Terminal phase guidance (higher precision)
                if dist_to_target < self.params.terminal_phase_start:
                    if not self.terminal_phase:
                        self.terminal_phase = True
                        print("🎯 TERMINAL GUIDANCE PHASE ACTIVATED")

                    # Increase guidance gain for terminal phase
                    guidance_accel *= (self.params.terminal_pn_gain / self.params.pn_gain)

                    # Limit acceleration to terminal phase maximum
                    guidance_accel_norm = np.linalg.norm(guidance_accel)
                    if guidance_accel_norm > self.params.terminal_control_max:
                        guidance_accel = guidance_accel * (self.params.terminal_control_max / guidance_accel_norm)

            # Update fin control system
            fin_deflections = self.fin_control.update(guidance_accel, vel, dt)

            # Thrust and propulsion
            thrust_accel = np.zeros(3)
            if rocket_time_remain > 0:
                thrust_accel = self.params.thrust_accel * initial_forward
                rocket_time_remain -= dt

            if self.params.edf_accel > 0 and np.linalg.norm(vel) > 1e-9:
                thrust_accel += self.params.edf_accel * (vel / np.linalg.norm(vel))

            # Drag and physics
            speed = np.linalg.norm(vel)
            if speed > 1e-9:
                drag_force = -0.5 * self.params.air_density * self.params.drag_coeff * self.params.area * speed * speed
                drag_accel = drag_force * (vel / speed) / self.params.mass
            else:
                drag_accel = np.zeros(3)

            a_grav = np.array([0.0, 0.0, -self.params.g])
            a_total = guidance_accel + thrust_accel + drag_accel + a_grav

            # Integration
            vel += a_total * dt
            pos += vel * dt
            t += dt
            traj.append(pos.copy())

            # Record history for animation
            time_history.append(t)
            fin_history.append(fin_deflections.copy())
            guidance_history.append(guidance_accel.copy())
            target_history.append(target_pos.copy())
            velocity_history.append(vel.copy())

            # Termination conditions
            dist_to_target = np.linalg.norm(target_pos - pos)
            if dist_to_target <= 5.0:
                return {
                    'hit': True,
                    'destroyed': self.destroyed,
                    'flight_time': t,
                    'trajectory': np.array(traj),
                    'miss_distance': dist_to_target,
                    'time_history': np.array(time_history),
                    'fin_history': np.array(fin_history),
                    'guidance_history': np.array(guidance_history),
                    'radar_history': np.array(radar_history),
                    'target_history': np.array(target_history),
                    'defense_history': defense_history,
                    'projectile_history': projectile_history,
                    'velocity_history': np.array(velocity_history)
                }
            if pos[2] < 0 and t > 0.05:
                break
            if t > self.params.thrust_duration and dist_to_target > 1000 and np.dot(vel, to_target) < 0:
                break

        impact = pos
        miss = np.linalg.norm(target_pos - pos)
        return {
            'hit': not self.destroyed and miss <= 5.0,
            'destroyed': self.destroyed,
            'flight_time': t,
            'trajectory': np.array(traj),
            'miss_distance': miss,
            'time_history': np.array(time_history),
            'fin_history': np.array(fin_history),
            'guidance_history': np.array(guidance_history),
            'radar_history': np.array(radar_history),
            'target_history': np.array(target_history),
            'defense_history': defense_history,
            'projectile_history': projectile_history,
            'velocity_history': np.array(velocity_history)
        }

    # Add this to your GuidedMissile class
    def get_missile_geometry(self):
        """Return the 3D geometry of the missile for visualization"""
        # Use the same parameters as your streamlined design
        length = 2.2
        diameter = 0.12
        fin_span = 0.22
        fin_length = 0.18

        # Pre-calculate geometry (similar to your design code)
        nose_z = np.linspace(0, length * 0.3, 20)
        nose_theta = np.linspace(0, 2 * np.pi, 20)
        nose_theta_grid, nose_z_grid = np.meshgrid(nose_theta, nose_z)

        # Pointed nose cone
        L_nose = length * 0.3
        R = diameter / 2
        z_norm = nose_z_grid / L_nose
        nose_radius = R * (1 - z_norm) ** 3

        nose_x_grid = nose_radius * np.cos(nose_theta_grid)
        nose_y_grid = nose_radius * np.sin(nose_theta_grid)
        nose_z_grid = L_nose - nose_z_grid  # Correct forward direction

        # Main body
        body_z = np.linspace(0, length * 0.7, 30)
        body_theta = np.linspace(0, 2 * np.pi, 20)
        body_theta_grid, body_z_grid = np.meshgrid(body_theta, body_z)

        body_taper = 0.92
        taper_factor = 1.0 - (1.0 - body_taper) * (body_z_grid) / (length * 0.7)
        body_x_grid = (diameter / 2) * np.cos(body_theta_grid) * taper_factor
        body_y_grid = (diameter / 2) * np.sin(body_theta_grid) * taper_factor
        body_z_grid += length * 0.3

        # Fins (highly swept back)
        fin_angles = [0, 90, 180, 270]
        fin_z_start = length * 0.85
        fin_z_end = fin_z_start + fin_length
        fin_sweep_angle = 55

        fin_vertices = []
        for angle in fin_angles:
            rad_angle = np.radians(angle)
            sweep_rad = np.radians(fin_sweep_angle)

            x = [
                0,
                np.cos(rad_angle) * fin_span / 4 * np.cos(sweep_rad),
                np.cos(rad_angle) * fin_span / 2,
                0
            ]

            y = [
                0,
                np.sin(rad_angle) * fin_span / 4 * np.cos(sweep_rad),
                np.sin(rad_angle) * fin_span / 2,
                0
            ]

            z = [
                fin_z_start,
                fin_z_start + fin_length * np.sin(sweep_rad) * 0.8,
                fin_z_end,
                fin_z_end
            ]

            fin_vertices.append((x, y, z))

        return {
            'nose': (nose_x_grid, nose_y_grid, nose_z_grid),
            'body': (body_x_grid, body_y_grid, body_z_grid),
            'fins': fin_vertices,
            'length': length,
            'diameter': diameter
        }