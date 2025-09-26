import numpy as np
class DefenseProjectile:
    """Projectile fired by a defense system"""

    def __init__(self, position, velocity, defense_type, ttl=5.0):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.type = defense_type
        self.ttl = ttl  # Time to live in seconds
        self.active = True

    def update(self, dt):
        """Update projectile position"""
        self.position += self.velocity * dt
        self.ttl -= dt
        if self.ttl <= 0:
            self.active = False
        return self.position