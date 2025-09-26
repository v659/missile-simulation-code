from target_system.target_defense.DefenseProjectile import DefenseProjectile
import numpy as np
DEFENSE_TYPES = {
    "sam": {"color": "orange", "marker": "s", "size": 120, "label": "SAM Site", "range": 300, "threat_level": 0.8},
    "aaa": {"color": "purple", "marker": "D", "size": 100, "label": "AAA", "range": 200, "threat_level": 0.6},
    "radar": {"color": "cyan", "marker": "o", "size": 80, "label": "Radar", "range": 400, "threat_level": 0.4}
}


class DefenseSystem:
    def __init__(self, position, defense_type):
        self.position = np.array(position, dtype=float)
        self.type = defense_type
        self.range = DEFENSE_TYPES[defense_type]["range"]
        self.threat_level = DEFENSE_TYPES[defense_type]["threat_level"]
        self.active = True
        self.last_shot_time = 0
        self.fire_rate = 1.0  # Shots per second
        self.projectiles = []

    def is_missile_in_range(self, missile_pos):
        """Check if missile is within range of this defense system"""
        distance = np.linalg.norm(missile_pos - self.position)
        return distance <= self.range

    def get_threat_direction(self, missile_pos):
        """Get direction from missile to this defense system (normalized)"""
        direction = self.position - missile_pos
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            return np.zeros(3)
        return direction / norm

    def try_fire_at_target(self, target_pos, current_time):
        """Attempt to fire at target if ready"""
        if not self.active:
            return None

        # Check if ready to fire again
        if current_time - self.last_shot_time < 1.0 / self.fire_rate:
            return None

        # Check if target is in range
        if not self.is_missile_in_range(target_pos):
            return None

        # Calculate direction to target
        direction = target_pos - self.position
        distance = np.linalg.norm(direction)
        if distance < 1e-9:
            return None

        direction = direction / distance

        # Determine projectile speed based on defense type
        if self.type == "sam":
            speed = 600.0  # SAM missiles are fast
        elif self.type == "aaa":
            speed = 400.0  # AAA shells are slower
        else:  # radar
            speed = 300.0  # Radar-guided weapons

        # Create projectile
        velocity = direction * speed

        # Add some random dispersion based on defense type
        if self.type == "aaa":  # AAA has more dispersion
            dispersion = np.random.uniform(-0.1, 0.1, 3)
        else:
            dispersion = np.random.uniform(-0.05, 0.05, 3)

        velocity += dispersion * speed

        projectile = DefenseProjectile(self.position.copy(), velocity, self.type)
        self.projectiles.append(projectile)
        self.last_shot_time = current_time

        return projectile

    def update_projectiles(self, dt):
        """Update all projectiles and remove inactive ones"""
        active_projectiles = []
        for projectile in self.projectiles:
            projectile.update(dt)
            if projectile.active:
                active_projectiles.append(projectile)
        self.projectiles = active_projectiles

    def check_projectile_collision(self, missile_pos, missile_radius=2.0):
        """Check if any projectile hit the missile"""
        for projectile in self.projectiles:
            distance = np.linalg.norm(projectile.position - missile_pos)
            if distance < missile_radius:
                return True
        return False