from dataclasses import dataclass
@dataclass
class VehicleParams:
    # Mass and physical properties
    mass: float = 2.0  # kg
    length: float = 1.8  # meters
    diameter: float = 0.15  # meters

    # Aerodynamic properties
    drag_coeff: float = 0.25  # dimensionless
    area: float = 0.012  # m² (cross-sectional area)
    lift_coeff: float = 0.1  # dimensionless
    air_density: float = 1.225  # kg/m³ (sea level)
    g: float = 9.81  # m/s² (gravity)

    # Propulsion parameters
    initial_speed_guess: float = 170.0  # m/s
    thrust_accel: float = 200.0  # m/s²
    thrust_duration: float = 4.0  # seconds
    edf_accel: float = 15.0  # m/s² (electric ducted fan for terminal phase)
    specific_impulse: float = 250.0  # seconds
    propellant_mass: float = 0.8  # kg

    # Guidance & control parameters
    pn_gain: float = 35.0  # Proportional navigation gain
    kp_heading: float = 25.0  # Proportional heading gain
    kd_heading: float = 6.0  # Derivative heading gain
    ki_heading: float = 1.5  # Integral heading gain
    a_control_max: float = 500.0  # m/s² (maximum guidance acceleration)

    # Precision guidance enhancements
    terminal_guidance: bool = True  # Enable terminal phase guidance
    terminal_phase_start: float = 100.0  # meters (distance to target)
    terminal_pn_gain: float = 60.0  # Higher gain for terminal phase
    terminal_control_max: float = 600.0  # m/s² (higher acceleration in terminal phase)

    # Advanced guidance parameters
    adaptive_guidance: bool = True  # Enable adaptive guidance
    min_control_speed: float = 30.0  # m/s (minimum speed for effective control)
    impact_angle_constraint: float = -45.0  # degrees (desired impact angle)
    energy_management: bool = True  # Enable energy management

    # Energy management parameters
    thrust_tracking_blend: float = 0.8  # Blend factor for thrust direction tracking
    energy_gain: float = 0.8  # Energy management gain
    max_energy_rate: float = 50.0  # J/s (maximum energy change rate)

    # Fin control parameters - REALISTIC AERODYNAMIC CONTROL
    fin_max_deflection: float = 25.0  # degrees - realistic maximum
    fin_response_time: float = 0.08  # seconds - realistic response time
    fin_effectiveness: float = 15.0  # scaling factor for fin effectiveness
    fin_damping: float = 0.7  # damping ratio for fin dynamics
    fin_count: int = 4  # number of control fins
    fin_area: float = 0.002  # m² (area of each fin)

    # Seeker parameters
    seeker_fov: float = 45.0  # degrees field of view
    radar_range: float = 1500.0  # meters
    radar_update_rate: float = 20.0  # Hz
    seeker_noise: float = 0.1  # radians (angular measurement noise)

    # Obstacle avoidance parameters
    obstacle_avoidance_weight: float = 2.0  # weight for avoiding defenses
    min_avoidance_distance: float = 50.0  # meters
    avoidance_gain: float = 3.0  # avoidance maneuver gain

    # Navigation parameters
    imu_noise: float = 0.01  # m/s² (IMU measurement noise)
    gps_noise: float = 1.0  # meters (GPS position noise)
    nav_update_rate: float = 100.0  # Hz

    # Simulation parameters
    max_flight_time: float = 60.0  # seconds
    simulation_dt: float = 0.01  # seconds (simulation time step)
    guidance_dt: float = 0.02  # seconds (guidance update interval)

    # Performance limits
    max_speed: float = 800.0  # m/s
    max_acceleration: float = 800.0  # m/s²
    max_altitude: float = 10000.0  # meters
    min_altitude: float = -10.0  # meters (below ground for impact)

    # Environmental parameters
    wind_speed: float = 3.0  # m/s
    wind_direction: float = 30.0  # degrees
    turbulence_intensity: float = 0.1  # dimensionless

    # Diagnostic and logging
    enable_logging: bool = True
    log_interval: float = 0.1  # seconds
    visualization_update: float = 0.05  # seconds

    def validate(self):
        """Validate parameter constraints"""
        assert self.mass > 0, "Mass must be positive"
        assert self.thrust_duration > 0, "Thrust duration must be positive"
        assert self.fin_max_deflection > 0, "Fin deflection must be positive"
        assert self.radar_range > 0, "Radar range must be positive"
        return True

    def get_aerodynamic_properties(self):
        """Return aerodynamic properties as dict"""
        return {
            'drag_coeff': self.drag_coeff,
            'lift_coeff': self.lift_coeff,
            'area': self.area,
            'fin_area': self.fin_area,
            'fin_count': self.fin_count
        }

    def get_guidance_properties(self):
        """Return guidance properties as dict"""
        return {
            'pn_gain': self.pn_gain,
            'terminal_pn_gain': self.terminal_pn_gain,
            'kp_heading': self.kp_heading,
            'ki_heading': self.ki_heading,
            'kd_heading': self.kd_heading,
            'adaptive_guidance': self.adaptive_guidance
        }

    def get_propulsion_properties(self):
        """Return propulsion properties as dict"""
        return {
            'thrust_accel': self.thrust_accel,
            'thrust_duration': self.thrust_duration,
            'edf_accel': self.edf_accel,
            'specific_impulse': self.specific_impulse,
            'propellant_mass': self.propellant_mass
        }