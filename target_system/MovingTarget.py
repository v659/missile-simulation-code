import random
import numpy as np

TARGET_TYPES = {
    "ground": {"color": "red", "marker": "X", "size": 200, "label": "Ground Target", "movement": "stationary",
               "speed": 0},
    "airborne": {"color": "blue", "marker": "^", "size": 150, "label": "Air Target", "movement": "smooth_random",
                 "speed": 50},
    "naval": {"color": "navy", "marker": "s", "size": 180, "label": "Naval Target", "movement": "smooth_evasive",
              "speed": 30},
    "bunker": {"color": "gray", "marker": "D", "size": 220, "label": "Bunker", "movement": "stationary", "speed": 0}
}


class MovingTarget:
    def __init__(self, initial_position, target_type, speed=None):
        self.position = np.array(initial_position, dtype=float)
        self.type = target_type
        self.speed = speed if speed is not None else TARGET_TYPES[target_type]["speed"]
        self.direction = np.array([1.0, 0.0, 0.0])  # Initial direction
        self.target_direction = self.direction.copy()  # Target direction for smooth transitions
        self.angle = 0.0
        self.radius = 200.0  # For circular movement
        self.center = initial_position.copy()
        self.last_direction_change = 0.0
        self.direction_change_interval = random.uniform(3.0, 6.0)  # Less frequent changes
        self.velocity = np.zeros(3)  # Track velocity for predictive guidance
        self.smoothness_factor = 0.1  # How quickly direction changes (lower = smoother)

        # For pattern-based movement
        self.movement_pattern = "linear"
        self.pattern_timer = 0.0
        self.pattern_duration = random.uniform(5.0, 10.0)
        self.waypoints = []
        self.current_waypoint = 0

        # Initialize based on target type
        if TARGET_TYPES[target_type]["movement"] in ["smooth_random", "smooth_evasive"]:
            self._generate_waypoints()

    def _generate_waypoints(self):
        """Generate a set of waypoints for smooth movement"""
        num_waypoints = random.randint(3, 6)
        self.waypoints = []

        for i in range(num_waypoints):
            # Create waypoints in a generally forward direction with some variation
            angle_variation = random.uniform(-30, 30)  # Degrees
            distance = random.uniform(200, 500)

            # Convert to radians and calculate offset
            angle_rad = np.radians(angle_variation)
            dx = distance * np.cos(angle_rad)
            dy = distance * np.sin(angle_rad)
            dz = random.uniform(-50, 100) if self.type == "airborne" else 0

            waypoint = self.position + np.array([dx, dy, dz])
            self.waypoints.append(waypoint)

        self.current_waypoint = 0

    def update(self, dt, current_time):
        """Update target position based on its movement type with smooth transitions"""
        old_position = self.position.copy()
        movement_type = TARGET_TYPES[self.type]["movement"]

        if movement_type == "stationary":
            # No movement
            pass

        elif movement_type == "smooth_random":
            # Smooth random movement with waypoint navigation
            self._update_smooth_random(dt, current_time)

        elif movement_type == "smooth_evasive":
            # Smooth evasive maneuvers with occasional sharp turns
            self._update_smooth_evasive(dt, current_time)

        else:
            # Fallback to original behavior for other types
            self._update_legacy(dt, current_time, movement_type)

        # Calculate velocity for predictive guidance
        self.velocity = (self.position - old_position) / dt if dt > 0 else np.zeros(3)

        return self.position

    def _update_smooth_random(self, dt, current_time):
        """Smooth random movement using waypoint navigation"""
        if not self.waypoints:
            self._generate_waypoints()

        # Check if we've reached the current waypoint
        current_target = self.waypoints[self.current_waypoint]
        distance_to_waypoint = np.linalg.norm(current_target - self.position)

        if distance_to_waypoint < 50:  # Close enough to waypoint
            self.current_waypoint = (self.current_waypoint + 1) % len(self.waypoints)
            current_target = self.waypoints[self.current_waypoint]

        # Calculate direction to waypoint
        to_waypoint = current_target - self.position
        if np.linalg.norm(to_waypoint) > 1e-9:
            target_direction = to_waypoint / np.linalg.norm(to_waypoint)
        else:
            target_direction = self.direction

        # Smoothly transition to the new direction
        self.direction = self._smooth_direction_transition(self.direction, target_direction)

        # Move in the current direction
        self.position += self.direction * self.speed * dt

    def _update_smooth_evasive(self, dt, current_time):
        """Smooth evasive maneuvers with occasional direction changes"""
        # Occasionally change movement pattern
        self.pattern_timer += dt
        if self.pattern_timer > self.pattern_duration:
            self.pattern_timer = 0
            self.pattern_duration = random.uniform(4.0, 8.0)
            self.movement_pattern = random.choice(["linear", "zigzag", "circle", "weave"])

            if self.movement_pattern in ["zigzag", "weave"]:
                self._generate_waypoints()

        # Execute current movement pattern
        if self.movement_pattern == "linear":
            # Continue in current direction with minor adjustments
            if current_time - self.last_direction_change > self.direction_change_interval:
                # Small random adjustment to direction
                adjustment = np.random.uniform(-0.2, 0.2, 3)
                adjustment[2] *= 0.5  # Less vertical adjustment
                self.target_direction = self.direction + adjustment
                self.target_direction /= np.linalg.norm(self.target_direction) + 1e-12
                self.last_direction_change = current_time
                self.direction_change_interval = random.uniform(2.0, 4.0)

            # Smooth transition to target direction
            self.direction = self._smooth_direction_transition(self.direction, self.target_direction)

        elif self.movement_pattern == "zigzag":
            # Zigzag between waypoints
            if not self.waypoints:
                self._generate_waypoints()

            current_target = self.waypoints[self.current_waypoint]
            distance_to_waypoint = np.linalg.norm(current_target - self.position)

            if distance_to_waypoint < 50:
                self.current_waypoint = (self.current_waypoint + 1) % len(self.waypoints)
                current_target = self.waypoints[self.current_waypoint]

            to_waypoint = current_target - self.position
            if np.linalg.norm(to_waypoint) > 1e-9:
                target_direction = to_waypoint / np.linalg.norm(to_waypoint)
            else:
                target_direction = self.direction

            self.direction = self._smooth_direction_transition(self.direction, target_direction)

        elif self.movement_pattern == "circle":
            # Circular movement
            self.angle += dt * 0.5  # Angular velocity
            self.direction = np.array([
                np.cos(self.angle),
                np.sin(self.angle),
                0  # Mostly horizontal circle
            ])

        elif self.movement_pattern == "weave":
            # Weaving pattern (sinusoidal)
            self.angle += dt * 0.8
            weave_strength = 0.3
            self.direction = np.array([
                np.cos(self.angle),
                np.sin(self.angle) * weave_strength,
                np.sin(self.angle * 1.5) * weave_strength * 0.5
            ])
            self.direction /= np.linalg.norm(self.direction) + 1e-12

        # Move in the current direction
        self.position += self.direction * self.speed * dt * random.uniform(0.9, 1.1)

    def _update_legacy(self, dt, current_time, movement_type):
        """Legacy movement behavior for compatibility"""
        if movement_type == "random":
            # Random movement with occasional direction changes
            if current_time - self.last_direction_change > self.direction_change_interval:
                self.direction = np.random.uniform(-1, 1, 3)
                self.direction[2] = max(0, self.direction[2])  # Keep it at or above ground level
                self.direction /= np.linalg.norm(self.direction) + 1e-12
                self.last_direction_change = current_time
                self.direction_change_interval = random.uniform(1.0, 4.0)

            self.position += self.direction * self.speed * dt * random.uniform(0.8, 1.2)

        elif movement_type == "evasive":
            # Evasive maneuvers - more aggressive direction changes
            if current_time - self.last_direction_change > self.direction_change_interval:
                self.direction = np.random.uniform(-1, 1, 3)
                self.direction[2] = max(0, self.direction[2])  # Keep it at or above ground level
                self.direction /= np.linalg.norm(self.direction) + 1e-12
                self.last_direction_change = current_time
                self.direction_change_interval = random.uniform(0.5, 2.0)

            # Add some random jitter to the movement
            jitter = np.random.uniform(-0.3, 0.3, 3)
            movement_dir = self.direction + jitter
            movement_dir /= np.linalg.norm(movement_dir) + 1e-12

            self.position += movement_dir * self.speed * dt * random.uniform(0.7, 1.3)

    def _smooth_direction_transition(self, current_dir, target_dir, factor=None):
        """Smoothly transition between directions"""
        if factor is None:
            factor = self.smoothness_factor

        # Use linear interpolation for smooth transition
        new_dir = current_dir * (1 - factor) + target_dir * factor
        new_dir_norm = np.linalg.norm(new_dir)

        if new_dir_norm > 1e-9:
            return new_dir / new_dir_norm
        else:
            return current_dir