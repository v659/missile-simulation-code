"""
🚀 Advanced Missile Guidance Simulation with Radar Homing and Defense Systems

Features:
- Real-time 3D animation with missile visualization
- Moving targets with randomized behaviors
- Defense systems that the missile must avoid
- Radar homing guidance with obstacle avoidance
- Field of View (FOV) visualization for the missile seeker
- Four-fin control visualization with random movements
- Multiple target scenarios

Author: Enhanced with radar homing, defense systems, and randomized target movements
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from guidance.GuidedMissile import GuidedMissile
from guidance.VehicleParams import VehicleParams
from target_system.MovingTarget import MovingTarget
from target_system.target_defense.DefenseSystem import DefenseSystem, DEFENSE_TYPES
from mpl_toolkits.mplot3d import art3d

# Remove this line: from modeler3D import StreamlinedMissileDesigner

def get_missile_designer():
    """Import and create missile designer only when needed"""
    import sys
    import io

    # Capture any output during import
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    try:
        from modeler3D import StreamlinedMissileDesigner
        designer = StreamlinedMissileDesigner(
            length=2.2,
            diameter=0.12,
            fin_span=0.22,
            fin_length=0.18
        )
        return designer
    finally:
        # Restore stdout
        sys.stdout = old_stdout
        # Discard any captured output
        buffer.close()
# Try to import py-ballisticcalc (optional high-fidelity solver)
PYBALL_AVAILABLE = False
try:
    import py_ballisticcalc as pbc

    PYBALL_AVAILABLE = True
except ImportError:
    try:
        import py_ballisticcalc as pbc

        PYBALL_AVAILABLE = True
    except ImportError:
        pbc = None
        PYBALL_AVAILABLE = False

BALLISTIC_LIB_AVAILABLE = PYBALL_AVAILABLE

# Optional: use SciPy differential_evolution if available
try:
    from scipy.optimize import differential_evolution

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Custom colormap for fin visualization
fin_cmap = LinearSegmentedColormap.from_list('fin_cmap', ['green', 'yellow', 'red'])





def simulate_open_loop(speed, elev_deg, azim_deg, target, params: VehicleParams, dt=0.02, max_time=60.0):
    """Simulate open-loop flight with realistic physics"""

    elev = math.radians(elev_deg)
    azim = math.radians(azim_deg)
    pos = np.array([0.0, 0.0, 0.0], dtype=float)
    vel = speed * np.array([
        math.cos(elev) * math.cos(azim),
        math.cos(elev) * math.sin(azim),
        math.sin(elev)
    ], dtype=float)

    initial_forward = vel / np.linalg.norm(vel)
    t = 0.0
    last_pos = pos.copy()

    while t < max_time:
        # Ground impact check
        if pos[2] < 0 and t > 0.05:
            p_last = last_pos
            z1, z2 = p_last[2], pos[2]
            if z1 - z2 == 0:
                return pos
            frac = z1 / (z1 - z2)
            return p_last + frac * (pos - p_last)

        last_pos = pos.copy()

        # Realistic atmospheric effects
        altitude = pos[2]
        air_density = params.air_density * math.exp(-altitude / 8500.0)

        # Wind effects
        wind_speed = 3.0 * math.sin(t * 0.3)
        wind_direction = math.radians(30.0)
        wind = np.array([
            wind_speed * math.cos(wind_direction),
            wind_speed * math.sin(wind_direction),
            0.0
        ])

        # Drag calculation
        v_rel = vel - wind
        speed_norm = np.linalg.norm(v_rel)
        if speed_norm > 1e-9:
            drag_force = -0.5 * air_density * params.drag_coeff * params.area * speed_norm * speed_norm
            drag_accel = drag_force * (v_rel / speed_norm) / params.mass
        else:
            drag_accel = np.zeros(3)

        # Thrust with target tracking
        thrust_accel = np.zeros(3)
        if t < params.thrust_duration:
            to_target_vec = target - pos
            to_target_norm = np.linalg.norm(to_target_vec)
            to_target_dir = to_target_vec / to_target_norm if to_target_norm > 1e-9 else initial_forward

            blend = params.thrust_tracking_blend
            thrust_dir = blend * to_target_dir + (1.0 - blend) * initial_forward
            thrust_dir /= np.linalg.norm(thrust_dir) + 1e-12
            thrust_accel = params.thrust_accel * thrust_dir

        # EDF assist
        if params.edf_accel > 0 and speed_norm > 1e-9:
            thrust_accel += params.edf_accel * (v_rel / speed_norm)

        # Gravity and integration
        a_grav = np.array([0.0, 0.0, -params.g])
        a_total = drag_accel + thrust_accel + a_grav

        vel += a_total * dt
        pos += vel * dt
        t += dt

    return pos


def optimize_launch_to_target(target, params: VehicleParams, azim_search_range=(-10, 10),
                              speed_bounds=(80, 300), elev_bounds=(5, 60), dt=0.02,
                              use_scipy=False, grid_resolution=(20, 18), prefer_high_angle=True):
    dx, dy = target[0], target[1]
    az_center = math.degrees(math.atan2(dy, dx))
    az_min = az_center + azim_search_range[0]
    az_max = az_center + azim_search_range[1]
    best = {'miss': float('inf'), 'elev': 0}

    def eval_for_az(az_deg):
        if use_scipy and SCIPY_AVAILABLE:
            def obj(x):
                s, elev = float(x[0]), float(x[1])
                impact = simulate_open_loop(s, elev, az_deg, target, params, dt=dt)
                miss = np.linalg.norm(impact - target)
                if prefer_high_angle:
                    angle_penalty = 2.0 * (elev_bounds[1] - elev)
                    return miss + angle_penalty
                return miss

            bounds = [(speed_bounds[0], speed_bounds[1]), (elev_bounds[0], elev_bounds[1])]
            res = differential_evolution(obj, bounds, maxiter=15, popsize=8, tol=1e-2)
            actual_impact = simulate_open_loop(res.x[0], res.x[1], az_deg, target, params, dt=dt)
            return float(res.x[0]), float(res.x[1]), float(np.linalg.norm(actual_impact - target))
        else:
            s_vals = np.linspace(speed_bounds[0], speed_bounds[1], grid_resolution[0])
            elev_vals = np.linspace(elev_bounds[0], elev_bounds[1], grid_resolution[1])
            local_best = (None, None, float('inf'))

            for s in s_vals:
                for elev in elev_vals:
                    impact = simulate_open_loop(s, elev, az_deg, target, params, dt=dt)
                    miss = np.linalg.norm(impact - target)
                    adjusted_miss = miss
                    if prefer_high_angle:
                        angle_penalty = 2.0 * (elev_bounds[1] - elev)
                        adjusted_miss = miss + angle_penalty
                    if adjusted_miss < local_best[2]:
                        local_best = (s, elev, adjusted_miss)
            return local_best[0], local_best[1], miss

    # Optimization process
    az_vals = np.linspace(az_min, az_max, 9)
    for az in az_vals:
        s, elev, miss = eval_for_az(az)
        if (miss < best['miss'] - 1.0 or
                (abs(miss - best['miss']) < 5.0 and elev > best['elev'] + 5.0 and prefer_high_angle)):
            best = {'speed': s, 'elev': elev, 'azim': az, 'miss': miss}

    # Refinement
    az_ref = np.linspace(max(az_min, best['azim'] - 1.5), min(az_max, best['azim'] + 1.5), 7)
    for az in az_ref:
        s, elev, miss = eval_for_az(az)
        if (miss < best['miss'] - 0.5 or
                (abs(miss - best['miss']) < 3.0 and elev > best['elev'] + 3.0 and prefer_high_angle)):
            best = {'speed': s, 'elev': elev, 'azim': az, 'miss': miss}

    return best['speed'], best['elev'], best['azim'], best['miss']



def choose_target_scenario():
    print("\n🎯 CHOOSE TARGET SCENARIO:")
    scenarios = {
        1: {"pos": [60.0, 800.0, 0.0], "type": "ground", "name": "Standard Ground Target",
            "defenses": [([400, 600, 0], "sam"), ([700, 400, 0], "aaa")]},
        2: {"pos": [200.0, 1500.0, 500.0], "type": "airborne", "name": "Airborne Target (Smooth)",
            "defenses": [([800, 1200, 0], "sam"), ([1200, 1000, 0], "radar")]},
        3: {"pos": [1200.0, 800.0, 0.0], "type": "naval", "name": "Naval Target (Evasive)",
            "defenses": [([900, 600, 0], "aaa"), ([1100, 1000, 0], "sam")]},
        4: {"pos": [800.0, 600.0, 50.0], "type": "bunker", "name": "Mountain Bunker",
            "defenses": [([600, 400, 0], "sam"), ([1000, 400, 0], "aaa"), ([800, 800, 0], "radar")]},
        5: {"pos": [5.0, 5.0, 150.0], "type": "ground", "name": "Close-range Test",
            "defenses": []},
        6: {"pos": [300.0, 1200.0, 300.0], "type": "airborne", "name": "Advanced Air Target",
            "defenses": [([600, 800, 0], "sam"), ([900, 1100, 0], "radar"), ([1200, 900, 0], "aaa")],
            "speed": 70}  # Faster air target
    }

    for i, scenario in scenarios.items():
        print(f"{i}. {scenario['name']} - {scenario['pos']}")

    choice = int(input("Select target (1-6): ") or "1")
    selected = scenarios.get(choice, scenarios[1])

    # Create target
    target = MovingTarget(selected["pos"], selected["type"], selected.get("speed"))

    # Create defense systems
    defense_systems = []
    for pos, defense_type in selected["defenses"]:
        defense_systems.append(DefenseSystem(pos, defense_type))

    return target, defense_systems, selected["type"]

    for i, scenario in scenarios.items():
        print(f"{i}. {scenario['name']} - {scenario['pos']}")

    choice = int(input("Select target (1-5): ") or "1")
    selected = scenarios.get(choice, scenarios[1])

    # Create target
    target = MovingTarget(selected["pos"], selected["type"])

    # Create defense systems
    defense_systems = []
    for pos, defense_type in selected["defenses"]:
        defense_systems.append(DefenseSystem(pos, defense_type))

    return target, defense_systems, selected["type"]


def display_performance_metrics(result, target, params, best_speed, best_elev, best_az):
    traj = result['trajectory']
    max_altitude = np.max(traj[:, 2])
    speeds = [np.linalg.norm(traj[i] - traj[i - 1]) / 0.01 if i > 0 else best_speed for i in range(len(traj))]
    max_speed = np.max(speeds)
    avg_speed = np.mean(speeds)

    # Calculate max fin deflections
    max_fin_deflections = np.max(np.abs(result['fin_history']), axis=0) if 'fin_history' in result else np.zeros(4)

    # Calculate radar statistics
    radar_percentage = np.mean(result['radar_history']) * 100 if 'radar_history' in result else 0

    print("\n" + "=" * 60)
    print("🎯 MISSILE PERFORMANCE DASHBOARD")
    print("=" * 60)
    print(f"🚀 Launch Parameters:")
    print(f"   Speed: {best_speed:.0f} m/s | Elevation: {best_elev:.1f}° | Azimuth: {best_az:.1f}°")
    print(f"📊 Flight Metrics:")
    print(f"   Max Altitude: {max_altitude:.0f} m")
    print(f"   Max Speed: {max_speed:.0f} m/s ({max_speed * 3.6:.0f} km/h)")
    print(f"   Avg Speed: {avg_speed:.0f} m/s")
    print(f"   Flight Time: {result['flight_time']:.2f} s")
    print(f"🎯 Target Engagement:")
    print(f"   Final Distance: {result['miss_distance']:.3f} m")
    print(f"   Status: {'DIRECT HIT! 💥' if result['hit'] else 'MISS ❌'}")
    if result['hit']:
        accuracy = 100 * (1 - result['miss_distance'] / 5)
        print(f"   Accuracy: {accuracy:.1f}%")
    print(f"🔧 Fin Performance:")
    print(f"   Max Up Fin: {max_fin_deflections[0]:.1f}°")
    print(f"   Max Down Fin: {max_fin_deflections[1]:.1f}°")
    print(f"   Max Left Fin: {max_fin_deflections[2]:.1f}°")
    print(f"   Max Right Fin: {max_fin_deflections[3]:.1f}°")
    print(f"📡 Radar Performance:")
    print(f"   Target in Radar Range: {radar_percentage:.1f}% of time")
    print("=" * 60)


def create_animation(traj, target_history, defense_history, projectile_history, target_type, result, best_speed,
                     best_elev):
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))

    # 3D trajectory plot
    ax1 = fig.add_subplot(236, projection='3d')

    # Fin control plots
    ax2 = fig.add_subplot(231)
    ax3 = fig.add_subplot(232)
    ax4 = fig.add_subplot(233)
    ax5 = fig.add_subplot(234)

    # Guidance acceleration plot
    ax6 = fig.add_subplot(235)

    # Set up 3D plot
    all_positions = np.vstack([traj, target_history])
    max_x = np.max(all_positions[:, 0]) * 1.2
    max_y = np.max(all_positions[:, 1]) * 1.2
    max_z = np.max(all_positions[:, 2]) * 1.3

    ax1.set_xlim(0, max_x)
    ax1.set_ylim(0, max_y)
    ax1.set_zlim(0, max_z)

    # Create ground plane
    xx, yy = np.meshgrid(np.linspace(0, max_x, 10), np.linspace(0, max_y, 10))
    zz = np.zeros_like(xx)
    ax1.plot_surface(xx, yy, zz, alpha=0.2, color='green', cmap='terrain')

    # Initialize trajectory and missile
    trajectory_line, = ax1.plot([], [], [], 'b-', linewidth=2, alpha=0.8, label='Missile Trajectory')
    target_point, = ax1.plot([], [], [], 'go', markersize=8, label='Target')
    target_trail, = ax1.plot([], [], [], 'g--', alpha=0.5, label='Target Path')

    # Add defense systems
    defense_scatters = []
    defense_colors = [DEFENSE_TYPES[defense.type]["color"] for defense in result.get('defense_systems', [])]
    for i, defense_pos in enumerate(result.get('defense_positions', [])):
        scatter = ax1.scatter([defense_pos[0]], [defense_pos[1]], [defense_pos[2]],
                              color=defense_colors[i],
                              marker=DEFENSE_TYPES[result.get('defense_types', [])[i]]["marker"],
                              s=DEFENSE_TYPES[result.get('defense_types', [])[i]]["size"],
                              label=DEFENSE_TYPES[result.get('defense_types', [])[i]]["label"])
        defense_scatters.append(scatter)

    # Add defense projectiles
    projectile_scatters = []
    for i in range(len(projectile_history)):
        scatter = ax1.scatter([], [], [], color='yellow', marker='*', s=50,
                              alpha=0.7, label='Defense Projectiles' if i == 0 else "")
        projectile_scatters.append(scatter)

    # Add launch site
    ax1.scatter([0], [0], [0], color='green', marker='^', s=200, label='Launch Site')

    # Initialize radar range indicator
    radar_points = ax1.scatter([], [], [], c=[], s=50, alpha=0.3, cmap='RdYlGn',
                               vmin=0, vmax=1, label='Radar Range')

    # Store missile visualization objects
    missile_parts = []

    # Set up fin control plots
    # Set up fin control plots
    time_data = result['time_history']
    fin_data = result['fin_history']

    # Ensure fin_data has the right shape
    if len(fin_data) > len(time_data):
        fin_data = fin_data[:len(time_data)]
    elif len(time_data) > len(fin_data):
        time_data = time_data[:len(fin_data)]

    # Calculate guidance acceleration magnitude for plotting
    guidance_data = result['guidance_history']
    guidance_magnitude = np.linalg.norm(guidance_data, axis=1)

    # Ensure guidance data matches time data length
    if len(guidance_magnitude) > len(time_data):
        guidance_magnitude = guidance_magnitude[:len(time_data)]
    elif len(time_data) > len(guidance_magnitude):
        time_data = time_data[:len(guidance_magnitude)]

    fin_labels = ['Up Fin', 'Down Fin', 'Left Fin', 'Right Fin']
    fin_axes = [ax2, ax3, ax4, ax5]

    fin_lines = []
    for i, ax in enumerate(fin_axes):
        ax.set_xlim(0, np.max(time_data) if len(time_data) > 0 else 10)
        ax.set_ylim(-35, 35)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Deflection (°)')
        ax.set_title(f'{fin_labels[i]} Control')
        ax.grid(True, alpha=0.3)
        line, = ax.plot([], [], 'b-', linewidth=2)
        fin_lines.append(line)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axhline(y=30, color='r', linestyle='--', alpha=0.5, label='Max Limit')
        ax.axhline(y=-30, color='r', linestyle='--', alpha=0.5)
        ax.legend(loc='upper right')

    # Set up guidance acceleration plot
    ax6.set_xlim(0, np.max(time_data) if len(time_data) > 0 else 10)
    ax6.set_ylim(0, np.max(guidance_magnitude) * 1.1 if len(guidance_magnitude) > 0 else 100)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Acceleration (m/s²)')
    ax6.set_title('Guidance Acceleration')
    ax6.grid(True, alpha=0.3)
    guidance_line, = ax6.plot([], [], 'r-', linewidth=2)

    # Add info text
    info_text = ax1.text2D(0.02, 0.98, "", transform=ax1.transAxes, fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

    # Precompute radar range visualization points
    def create_radar_points(missile_pos, radar_range=1500, num_points=30):
        """Create points for radar range visualization"""
        # Create points in a sphere around the missile
        phi = np.linspace(0, 2 * np.pi, num_points)
        theta = np.linspace(0, np.pi, num_points)

        points = []
        for p in phi:
            for t in theta:
                x = missile_pos[0] + radar_range * np.sin(t) * np.cos(p)
                y = missile_pos[1] + radar_range * np.sin(t) * np.sin(p)
                z = missile_pos[2] + radar_range * np.cos(t)
                points.append([x, y, z])

        return np.array(points)

    # Get missile designer for geometry
    missile_designer = get_missile_designer()
    missile_geo = missile_designer.get_missile_geometry()

    # Function to rotate points around origin
    def rotate_points(points, axis, angle):
        """Rotate points around given axis by angle (radians)"""
        axis = axis / np.linalg.norm(axis)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        # Rotation matrix
        rot_matrix = np.array([
            [cos_angle + axis[0] ** 2 * (1 - cos_angle),
             axis[0] * axis[1] * (1 - cos_angle) - axis[2] * sin_angle,
             axis[0] * axis[2] * (1 - cos_angle) + axis[1] * sin_angle],
            [axis[1] * axis[0] * (1 - cos_angle) + axis[2] * sin_angle,
             cos_angle + axis[1] ** 2 * (1 - cos_angle),
             axis[1] * axis[2] * (1 - cos_angle) - axis[0] * sin_angle],
            [axis[2] * axis[0] * (1 - cos_angle) - axis[1] * sin_angle,
             axis[2] * axis[1] * (1 - cos_angle) + axis[0] * sin_angle,
             cos_angle + axis[2] ** 2 * (1 - cos_angle)]
        ])

        return np.dot(points, rot_matrix.T)

    # Function to calculate scale factor based on distance to target
    def calculate_scale_factor(missile_pos, target_pos):
        """Calculate scale factor based on distance to target"""
        # Calculate distance to target
        distance = np.linalg.norm(missile_pos - target_pos)

        # Base scale factor (larger when closer to target)
        # Use a logarithmic scale to make the missile larger as it approaches the target
        min_scale = 0.5  # Minimum scale factor
        max_scale = 10  # Maximum scale factor
        scale_range = 1000.0  # Distance over which scaling occurs (meters)

        # Calculate scale factor (inverse relationship with distance)
        if distance > scale_range:
            scale_factor = min_scale
        else:
            # Logarithmic scaling for smoother transitions
            scale_factor = max_scale - (max_scale - min_scale) * (distance / scale_range)

            # Ensure scale factor is within bounds
            scale_factor = max(min_scale, min(max_scale, scale_factor))

        return scale_factor

    # Function to create missile geometry at specific position and orientation
    def create_missile_geometry(position, direction, target_pos, fin_deflections=None):
        """Create missile geometry at given position with proper orientation"""
        # Calculate scale factor based on distance to target
        scale_factor = calculate_scale_factor(position, target_pos)

        # Default fin deflections if not provided
        if fin_deflections is None:
            fin_deflections = np.zeros(4)

        # Normalize direction
        if np.linalg.norm(direction) < 1e-9:
            direction = np.array([1, 0, 0])
        else:
            direction = direction / np.linalg.norm(direction)

        # Get base missile geometry
        nose_x, nose_y, nose_z = missile_geo['nose']
        body_x, body_y, body_z = missile_geo['body']
        fins = missile_geo['fins']

        # Scale the geometry based on distance to target
        nose_x = nose_x * scale_factor
        nose_y = nose_y * scale_factor
        nose_z = nose_z * scale_factor

        body_x = body_x * scale_factor
        body_y = body_y * scale_factor
        body_z = body_z * scale_factor

        # Apply fin deflections
        fin_angles = [0, 90, 180, 270]
        fin_colors = ['#8B0000', '#00008B', '#006400', '#8B008B']
        fin_objects = []

        for i, (angle, fin_vert) in enumerate(zip(fin_angles, fins)):
            x, y, z = fin_vert

            # Scale fin based on distance to target
            x = np.array(x) * scale_factor
            y = np.array(y) * scale_factor
            z = np.array(z) * scale_factor

            # Apply fin deflection
            deflection_rad = np.radians(fin_deflections[i % len(fin_deflections)])

            # Rotate fin based on its position and deflection
            if angle == 0:  # Up fin
                y = y * np.cos(deflection_rad) - z * np.sin(deflection_rad)
                z = y * np.sin(deflection_rad) + z * np.cos(deflection_rad)
            elif angle == 90:  # Right fin
                x = x * np.cos(deflection_rad) - z * np.sin(deflection_rad)
                z = x * np.sin(deflection_rad) + z * np.cos(deflection_rad)
            elif angle == 180:  # Down fin
                y = y * np.cos(-deflection_rad) - z * np.sin(-deflection_rad)
                z = y * np.sin(-deflection_rad) + z * np.cos(-deflection_rad)
            elif angle == 270:  # Left fin
                x = x * np.cos(-deflection_rad) - z * np.sin(-deflection_rad)
                z = x * np.sin(-deflection_rad) + z * np.cos(-deflection_rad)

            # Combine all points
            fin_points = np.stack([x, y, z], axis=-1)
            fin_objects.append((fin_points, fin_colors[i]))

        # Combine all missile parts
        nose_points = np.stack([nose_x.flatten(), nose_y.flatten(), nose_z.flatten()], axis=1)
        body_points = np.stack([body_x.flatten(), body_y.flatten(), body_z.flatten()], axis=1)
        all_points = np.vstack([nose_points, body_points])

        # Calculate rotation to align with direction
        default_direction = np.array([0, 0, 1])  # Missile points along Z-axis by default in modeler3D
        direction = direction / np.linalg.norm(direction)

        # Calculate rotation axis and angle
        rotation_axis = np.cross(default_direction, direction)
        if np.linalg.norm(rotation_axis) < 1e-9:
            # Vectors are parallel, no rotation needed
            rotated_points = all_points
            rotated_fins = fin_objects
        else:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            rotation_angle = np.arccos(np.dot(default_direction, direction))

            # Rotate all points
            rotated_points = rotate_points(all_points, rotation_axis, rotation_angle)

            # Rotate fins
            rotated_fins = []
            for fin_points, color in fin_objects:
                rotated_fin_points = rotate_points(fin_points, rotation_axis, rotation_angle)
                rotated_fins.append((rotated_fin_points, color))

        # Separate back into nose and body points
        nose_points_rot = rotated_points[:len(nose_points)].reshape(nose_x.shape + (3,))
        body_points_rot = rotated_points[len(nose_points):].reshape(body_x.shape + (3,))

        # Translate to position
        nose_points_rot[..., 0] += position[0]
        nose_points_rot[..., 1] += position[1]
        nose_points_rot[..., 2] += position[2]

        body_points_rot[..., 0] += position[0]
        body_points_rot[..., 1] += position[1]
        body_points_rot[..., 2] += position[2]

        # Translate fins
        translated_fins = []
        for fin_points, color in rotated_fins:
            fin_points[..., 0] += position[0]
            fin_points[..., 1] += position[1]
            fin_points[..., 2] += position[2]
            translated_fins.append((fin_points, color))

        return nose_points_rot, body_points_rot, translated_fins

    # Animation function
    def animate(i):
        artists = []

        # Remove previous missile visualization if it exists
        for part in missile_parts:
            if hasattr(part, 'remove'):
                part.remove()
        missile_parts.clear()

        # Update 3D trajectory
        if i < len(traj):
            # Update 3D trajectory line
            trajectory_line.set_data(traj[:i, 0], traj[:i, 1])
            trajectory_line.set_3d_properties(traj[:i, 2])
            artists.append(trajectory_line)

            # Create missile visualization at current position with proper rotation
            current_pos = traj[i]
            current_target_pos = target_history[i]

            # Calculate direction (use average of recent points for smoother rotation)
            if i > 2:
                # Use average of last few points for smoother direction
                direction = traj[i] - traj[i - 2]
            elif i > 0:
                direction = traj[i] - traj[i - 1]
            else:
                direction = np.array([1, 0, 0])  # Default direction

            if np.linalg.norm(direction) < 1e-9:
                direction = np.array([1, 0, 0])

            # Get current fin deflections
            current_fins = fin_data[i] if i < len(fin_data) else np.zeros(4)

            # Create rotated missile geometry with fin deflections
            nose_points, body_points, fins = create_missile_geometry(current_pos, direction, current_target_pos,
                                                                     current_fins)

            # Plot missile body and nose
            missile_nose = ax1.plot_surface(nose_points[:, :, 0], nose_points[:, :, 1], nose_points[:, :, 2],
                                            color='#708090', alpha=0.98, shade=True)
            missile_body = ax1.plot_surface(body_points[:, :, 0], body_points[:, :, 1], body_points[:, :, 2],
                                            color='#2F4F4F', alpha=0.95, shade=True)

            missile_parts.append(missile_nose)
            missile_parts.append(missile_body)

            # Plot fins with proper deflections
            for fin_points, color in fins:
                # Create a polycollection for each fin
                verts = [list(zip(fin_points[:, 0], fin_points[:, 1], fin_points[:, 2]))]
                poly = art3d.Poly3DCollection(verts, alpha=0.9, color=color,
                                              linewidths=1.5, edgecolors='black')
                ax1.add_collection3d(poly)
                missile_parts.append(poly)

            # Update target position
            target_point.set_data([target_history[i, 0]], [target_history[i, 1]])
            target_point.set_3d_properties([target_history[i, 2]])
            artists.append(target_point)

            # Update target trail
            target_trail.set_data(target_history[:i, 0], target_history[:i, 1])
            target_trail.set_3d_properties(target_history[:i, 2])
            artists.append(target_trail)

            # Update defense projectiles
            for j, scatter in enumerate(projectile_scatters):
                if j < len(projectile_history) and i < len(projectile_history[j]):
                    if not np.isnan(projectile_history[j][i][0]):
                        scatter._offsets3d = ([projectile_history[j][i][0]],
                                              [projectile_history[j][i][1]],
                                              [projectile_history[j][i][2]])
                        artists.append(scatter)

            # Update radar range indicator
            radar_pts = create_radar_points(traj[i])
            if len(radar_pts) > 0:
                radar_points._offsets3d = (radar_pts[:, 0], radar_pts[:, 1], radar_pts[:, 2])
                if i < len(result['radar_history']) and result['radar_history'][i]:
                    radar_points.set_array(np.ones(len(radar_pts)))
                else:
                    radar_points.set_array(np.zeros(len(radar_pts)))
                artists.append(radar_points)

            # Update fin control plots
            for j, fin_line in enumerate(fin_lines):
                fin_line.set_data(time_data[:i], fin_data[:i, j])
                artists.append(fin_line)

            # Update guidance acceleration plot
            guidance_line.set_data(time_data[:i], guidance_magnitude[:i])
            artists.append(guidance_line)

            # Update info text
            if i > 0:
                speed = np.linalg.norm(traj[i] - traj[i - 1]) / 0.01
                alt = traj[i, 2]
                dist = np.linalg.norm(target_history[i] - traj[i])
                status = "DESTROYED" if result.get('destroyed', False) and i >= len(traj) - 10 else "ACTIVE"
                info_text.set_text(
                    f"Time: {i * 0.01:.1f}s\nSpeed: {speed:.0f} m/s\nAlt: {alt:.0f}m\nDist: {dist:.0f}m\nStatus: {status}")
                artists.append(info_text)

        return artists

    # Create animation
    ani = animation.FuncAnimation(fig, animate, frames=min(len(traj), len(target_history)),
                                  interval=20, blit=False, repeat=False)

    # Add title and legend
    ax1.set_title(f"Missile Trajectory (Speed: {best_speed:.0f}m/s, Elevation: {best_elev:.1f}°)")
    ax1.legend(loc='upper right')

    plt.tight_layout()
    return ani


def create_missile_visualization():
    """Create and display the missile design visualization"""
    print("Creating missile design visualization...")

    # Get the missile designer without triggering output
    missile_designer = get_missile_designer()

    # Create the design visualization
    fig_3d, aero_fig = missile_designer.create_missile_design()
    missile_designer.display_design_specs()

    return fig_3d, aero_fig, missile_designer


def main():
    # Show missile design first
    missile_fig, aero_fig, missile_designer = create_missile_visualization()
    plt.pause(2)  # Briefly show the design

    # Continue with the simulation
    target, defense_systems, target_type = choose_target_scenario()
    params = VehicleParams()

    # Adjust parameters based on target type for better guidance
    if target_type in ["airborne", "naval"]:
        # Increase guidance gains for moving targets
        params.pn_gain = 45.0
        params.terminal_pn_gain = 70.0
        print("📈 Increased guidance gains for moving target")

    print("Optimizing launch parameters...")
    t0 = time.time()
    # For moving targets, we need to optimize against the initial position
    best_speed, best_elev, best_az, best_miss = optimize_launch_to_target(
        target.position, params, prefer_high_angle=True
    )
    t1 = time.time()

    print(f"⚡ Optimization complete in {t1 - t0:.1f}s")
    print(f"🎯 Launch solution: {best_speed:.0f}m/s, {best_elev:.1f}°, {best_az:.1f}°")

    missile = GuidedMissile(target, defense_systems, params, best_speed, best_elev, best_az)
    print("🚀 Running guided simulation...")
    result = missile.run_guided()

    # Add defense system info to result for animation
    result['defense_systems'] = defense_systems
    result['defense_positions'] = [defense.position for defense in defense_systems]
    result['defense_types'] = [defense.type for defense in defense_systems]

    display_performance_metrics(result, target, params, best_speed, best_elev, best_az)

    # Close the missile design figures to avoid clutter
    plt.close(missile_fig)
    plt.close(aero_fig)

    # Create and show animation
    print("🎬 Creating animation...")
    ani = create_animation(
        result['trajectory'],
        result['target_history'],
        result.get('defense_history', []),
        result.get('projectile_history', []),
        target_type,
        result,
        best_speed,
        best_elev
    )

    # Add aerodynamic analysis to the final display
    print("\n📊 Displaying aerodynamic performance analysis...")
    aero_fig, _ = missile_designer.create_missile_design()
    plt.show()

    return best_speed, best_elev, best_az, best_miss, result


if __name__ == "__main__":
    runs = 1  # Run one simulation at a time for better animation viewing
    for i in range(runs):
        print(f"\n--- Simulation {i + 1} ---")
        main()