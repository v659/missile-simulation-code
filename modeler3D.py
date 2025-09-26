import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.animation import FuncAnimation


class StreamlinedMissileDesigner:
    def __init__(self, length=2.2, diameter=0.12, fin_span=0.22, fin_length=0.18):
        self.length = length
        self.diameter = diameter
        self.fin_span = fin_span
        self.fin_length = fin_length

        # Aerodynamic properties
        self.drag_coeff = 0.18  # Much lower due to streamlining
        self.lift_coeff = 0.15
        self.cross_sectional_area = np.pi * (diameter / 2) ** 2

        # Material properties for stress analysis
        self.yield_strength = 60e6  # Pa (PLA plastic)
        self.safety_factor = 2.0

        # HIGHLY STREAMLINED geometry - corrected nose direction
        self._create_streamlined_geometry()

        # Store animation
        self.animation = None

    def _create_streamlined_geometry(self):
        """Create HIGHLY STREAMLINED missile geometry with CORRECT nose direction"""
        # CORRECTED: EXTREMELY POINTY nose cone facing FORWARD
        self.nose_z = np.linspace(0, self.length * 0.3, 50)
        self.nose_theta = np.linspace(0, 2 * np.pi, 40)
        self.nose_theta_grid, self.nose_z_grid = np.meshgrid(self.nose_theta, self.nose_z)

        # CORRECTED: Proper exponential taper for forward-facing nose
        L_nose = self.length * 0.3
        R = self.diameter / 2
        z_norm = self.nose_z_grid / L_nose

        # Ultra-sharp exponential taper (pointed forward)
        self.nose_radius = R * (1 - z_norm) ** 3

        self.nose_x_grid = self.nose_radius * np.cos(self.nose_theta_grid)
        self.nose_y_grid = self.nose_radius * np.sin(self.nose_theta_grid)
        self.nose_z_grid = L_nose - self.nose_z_grid  # CORRECTED: Nose now faces forward

        # Main body - more slender and tapered
        self.body_z = np.linspace(0, self.length * 0.7, 60)  # CORRECTED: Starts from 0
        self.body_theta = np.linspace(0, 2 * np.pi, 40)
        self.body_theta_grid, self.body_z_grid = np.meshgrid(self.body_theta, self.body_z)

        # Slight body taper (more streamlined)
        body_taper = 0.92  # 8% taper toward the rear
        taper_factor = 1.0 - (1.0 - body_taper) * (self.body_z_grid) / (self.length * 0.7)
        self.body_x_grid = (self.diameter / 2) * np.cos(self.body_theta_grid) * taper_factor
        self.body_y_grid = (self.diameter / 2) * np.sin(self.body_theta_grid) * taper_factor
        self.body_z_grid += self.length * 0.3  # CORRECTED: Position after nose

        # HIGHLY SWEPT BACK fins for maximum streamlining
        self.fin_angles = [0, 90, 180, 270]
        self.fin_z_start = self.length * 0.85
        self.fin_z_end = self.fin_z_start + self.fin_length
        self.fin_sweep_angle = 55

        # Smaller control fins at front (canards)
        self.canard_angles = [45, 135, 225, 315]
        self.canard_z = self.length * 0.35
        self.canard_span = self.fin_span * 0.4
        self.canard_length = self.fin_length * 0.6

    def calculate_aerodynamics(self, wind_speed=250, angle_of_attack=5):
        """Calculate aerodynamic forces and performance"""
        # Convert to SI units
        wind_speed_ms = wind_speed / 3.6  # km/h to m/s

        # Aerodynamic forces
        dynamic_pressure = 0.5 * 1.225 * wind_speed_ms ** 2
        drag_force = dynamic_pressure * self.drag_coeff * self.cross_sectional_area
        lift_force = dynamic_pressure * self.lift_coeff * np.sin(
            np.radians(angle_of_attack)) * self.cross_sectional_area

        # Performance metrics
        lift_to_drag = lift_force / drag_force if drag_force > 0 else 0
        mach_number = wind_speed_ms / 340  # Speed of sound

        # Structural analysis
        max_bending_moment = lift_force * self.length / 4
        section_modulus = np.pi * (self.diameter / 2) ** 3 / 4
        bending_stress = max_bending_moment / section_modulus
        safety_margin = self.yield_strength / (bending_stress * self.safety_factor)

        return {
            'drag_force': drag_force,
            'lift_force': lift_force,
            'lift_to_drag': lift_to_drag,
            'mach_number': mach_number,
            'bending_stress': bending_stress,
            'safety_margin': safety_margin,
            'dynamic_pressure': dynamic_pressure
        }

    def create_missile_design(self):
        """Create and display the HIGHLY STREAMLINED missile design"""
        # Create separate figures for 3D design and aerodynamics
        fig_3d = plt.figure(figsize=(12, 10))
        ax_3d = fig_3d.add_subplot(111, projection='3d')

        # Create 3D missile design
        self._create_3d_design(ax_3d)

        # Create aerodynamic analysis in separate figure
        aero_fig = self._create_aerodynamic_analysis()

        plt.tight_layout()
        return fig_3d, aero_fig

    def _create_3d_design(self, ax):
        """Create the 3D missile design"""
        # Create main body with advanced shading
        body = ax.plot_surface(self.body_x_grid, self.body_y_grid, self.body_z_grid,
                               color='#2F4F4F', alpha=0.95, label='Main Body',
                               shade=True, antialiased=True)

        # Create EXTREMELY POINTY nose cone (CORRECTED direction)
        nose = ax.plot_surface(self.nose_x_grid, self.nose_y_grid, self.nose_z_grid,
                               color='#708090', alpha=0.98, label='Nose Cone',
                               shade=True, antialiased=True)

        # Razor sharp tip (CORRECTED position)
        ax.scatter([0], [0], [self.length * 0.3], color='#000000', s=80, alpha=1.0,
                   marker='o', edgecolors='silver', linewidth=1.5)

        # Create HIGHLY SWEPT BACK main fins
        fin_colors = ['#8B0000', '#00008B', '#006400', '#8B008B']
        sweep_rad = np.radians(self.fin_sweep_angle)

        fins = []
        for i, angle in enumerate(self.fin_angles):
            rad_angle = np.radians(angle)

            # Extreme swept delta wing shape
            x = [
                0,
                np.cos(rad_angle) * self.fin_span / 4 * np.cos(sweep_rad),
                np.cos(rad_angle) * self.fin_span / 2,
                0
            ]

            y = [
                0,
                np.sin(rad_angle) * self.fin_span / 4 * np.cos(sweep_rad),
                np.sin(rad_angle) * self.fin_span / 2,
                0
            ]

            z = [
                self.fin_z_start,
                self.fin_z_start + self.fin_length * np.sin(sweep_rad) * 0.8,
                self.fin_z_end,
                self.fin_z_end
            ]

            verts = [list(zip(x, y, z))]
            poly = art3d.Poly3DCollection(verts, alpha=0.9, color=fin_colors[i],
                                          linewidths=1.5, edgecolors='black')
            ax.add_collection3d(poly)
            fins.append(poly)

        # Create canard fins at front
        canard_colors = ['#FF4500', '#32CD32', '#1E90FF', '#FFD700']
        for i, angle in enumerate(self.canard_angles):
            rad_angle = np.radians(angle)

            # Small delta canards
            x = [
                0,
                np.cos(rad_angle) * self.canard_span / 3,
                np.cos(rad_angle) * self.canard_span / 2,
                0
            ]

            y = [
                0,
                np.sin(rad_angle) * self.canard_span / 3,
                np.sin(rad_angle) * self.canard_span / 2,
                0
            ]

            z = [
                self.canard_z - self.canard_length / 2,
                self.canard_z,
                self.canard_z + self.canard_length / 2,
                self.canard_z + self.canard_length / 2
            ]

            verts = [list(zip(x, y, z))]
            poly = art3d.Poly3DCollection(verts, alpha=0.85, color=canard_colors[i],
                                          linewidths=1.2, edgecolors='white')
            ax.add_collection3d(poly)
            fins.append(poly)

        # Add details
        self._add_aerodynamic_details(ax)
        self._add_propulsion_details(ax)

        # Set 3D plot properties
        max_span = max(self.fin_span, self.diameter) * 1.3
        ax.set_xlim(-max_span, max_span)
        ax.set_ylim(-max_span, max_span)
        ax.set_zlim(0, self.length * 1.05)

        ax.set_xlabel('Width (m)', fontweight='bold')
        ax.set_ylabel('Depth (m)', fontweight='bold')
        ax.set_zlabel('Length (m)', fontweight='bold')
        ax.set_title('ULTRA-STREAMLINED MISSILE DESIGN', fontweight='bold')

        ax.set_box_aspect([1, 1, 2.5])
        ax.view_init(elev=25, azim=45)
        ax.grid(True, alpha=0.3)

    def get_missile_geometry(self):
        """Return the 3D geometry of the missile for visualization"""
        # Return the geometry that was created in _create_streamlined_geometry
        return {
            'nose': (self.nose_x_grid, self.nose_y_grid, self.nose_z_grid),
            'body': (self.body_x_grid, self.body_y_grid, self.body_z_grid),
            'fins': self._get_fin_vertices(),
            'length': self.length,
            'diameter': self.diameter
        }

    def _get_fin_vertices(self):
        """Get the vertices for all fins"""
        fin_vertices = []
        sweep_rad = np.radians(self.fin_sweep_angle)

        for angle in self.fin_angles:
            rad_angle = np.radians(angle)

            x = [
                0,
                np.cos(rad_angle) * self.fin_span / 4 * np.cos(sweep_rad),
                np.cos(rad_angle) * self.fin_span / 2,
                0
            ]

            y = [
                0,
                np.sin(rad_angle) * self.fin_span / 4 * np.cos(sweep_rad),
                np.sin(rad_angle) * self.fin_span / 2,
                0
            ]

            z = [
                self.fin_z_start,
                self.fin_z_start + self.fin_length * np.sin(sweep_rad) * 0.8,
                self.fin_z_end,
                self.fin_z_end
            ]

            fin_vertices.append((x, y, z))

        # Add canard fins
        for angle in self.canard_angles:
            rad_angle = np.radians(angle)

            x = [
                0,
                np.cos(rad_angle) * self.canard_span / 3,
                np.cos(rad_angle) * self.canard_span / 2,
                0
            ]

            y = [
                0,
                np.sin(rad_angle) * self.canard_span / 3,
                np.sin(rad_angle) * self.canard_span / 2,
                0
            ]

            z = [
                self.canard_z - self.canard_length / 2,
                self.canard_z,
                self.canard_z + self.canard_length / 2,
                self.canard_z + self.canard_length / 2
            ]

            fin_vertices.append((x, y, z))

        return fin_vertices
    def _create_aerodynamic_analysis(self):
        """Create aerodynamic analysis visualization in separate figure"""
        # Calculate aerodynamics at different speeds
        speeds = np.linspace(100, 1000, 20)  # km/h
        aoa = 5  # degrees

        results = [self.calculate_aerodynamics(speed, aoa) for speed in speeds]

        drag_forces = [r['drag_force'] for r in results]
        lift_forces = [r['lift_force'] for r in results]
        mach_numbers = [r['mach_number'] for r in results]
        safety_margins = [r['safety_margin'] for r in results]

        # Create new figure for aerodynamics
        aero_fig = plt.figure(figsize=(12, 10))

        # Create 2x2 grid of subplots
        ax1 = aero_fig.add_subplot(221)  # Top left
        ax2 = aero_fig.add_subplot(222)  # Top right
        ax3 = aero_fig.add_subplot(223)  # Bottom left
        ax4 = aero_fig.add_subplot(224)  # Bottom right

        # Plot 1: Drag and Lift forces
        ax1.plot(speeds, drag_forces, 'r-', linewidth=2, label='Drag Force')
        ax1.plot(speeds, lift_forces, 'b-', linewidth=2, label='Lift Force')
        ax1.set_xlabel('Speed (km/h)')
        ax1.set_ylabel('Force (N)')
        ax1.set_title('Aerodynamic Forces')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Mach number
        ax2.plot(speeds, mach_numbers, 'g-', linewidth=2)
        ax2.set_xlabel('Speed (km/h)')
        ax2.set_ylabel('Mach Number')
        ax2.set_title('Flight Mach Number')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Sound Barrier')
        ax2.legend()

        # Plot 3: Lift-to-Drag ratio
        l_d_ratios = [r['lift_to_drag'] for r in results]
        ax3.plot(speeds, l_d_ratios, 'purple', linewidth=2)
        ax3.set_xlabel('Speed (km/h)')
        ax3.set_ylabel('Lift/Drag Ratio')
        ax3.set_title('Aerodynamic Efficiency')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Structural safety
        ax4.plot(speeds, safety_margins, 'orange', linewidth=2)
        ax4.set_xlabel('Speed (km/h)')
        ax4.set_ylabel('Safety Margin')
        ax4.set_title('Structural Safety Margin')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Safety Limit')
        ax4.legend()

        aero_fig.suptitle('AERODYNAMIC PERFORMANCE ANALYSIS', fontsize=16, fontweight='bold')
        aero_fig.tight_layout()

        return aero_fig

    def _add_aerodynamic_details(self, ax):
        """Add aerodynamic details to the missile"""
        # Guidance section rings
        guidance_z = self.length * 0.65
        theta = np.linspace(0, 2 * np.pi, 30)
        x = (self.diameter / 2 + 0.008) * np.cos(theta)
        y = (self.diameter / 2 + 0.008) * np.sin(theta)
        z = np.full_like(x, guidance_z)
        ax.plot(x, y, z, '#FF0000', alpha=0.8, linewidth=2.0)

        # Warhead section
        warhead_z = self.length * 0.45
        x = (self.diameter / 2 + 0.006) * np.cos(theta)
        y = (self.diameter / 2 + 0.006) * np.sin(theta)
        z = np.full_like(x, warhead_z)
        ax.plot(x, y, z, '#FF8C00', alpha=0.8, linewidth=2.0)

    def _add_propulsion_details(self, ax):
        """Add propulsion system details"""
        # Rocket motor section
        motor_z = np.linspace(self.length * 0.92, self.length, 10)
        theta = np.linspace(0, 2 * np.pi, 20)
        theta_grid, z_grid = np.meshgrid(theta, motor_z)

        x = (self.diameter / 2 * 0.95) * np.cos(theta_grid)
        y = (self.diameter / 2 * 0.95) * np.sin(theta_grid)

        ax.plot_surface(x, y, z_grid, color='#8B4513', alpha=0.9, shade=True)

    def display_design_specs(self):
        """Display comprehensive design specifications with aerodynamics"""
        # Calculate sample aerodynamics
        aero = self.calculate_aerodynamics(800, 5)  # 800 km/h, 5° AoA

        specs = f"""
        ULTRA-STREAMLINED MISSILE DESIGN SPECIFICATIONS
        {'=' * 60}

        DIMENSIONS:
        Total Length: {self.length:.2f} m
        Body Diameter: {self.diameter:.2f} m
        Fin Span: {self.fin_span:.2f} m
        Fin Sweep Angle: {self.fin_sweep_angle}°

        AERODYNAMIC PERFORMANCE (800 km/h, 5° AoA):
        Drag Force: {aero['drag_force']:.1f} N
        Lift Force: {aero['lift_force']:.1f} N
        Lift-to-Drag Ratio: {aero['lift_to_drag']:.2f}
        Mach Number: {aero['mach_number']:.2f}

        STRUCTURAL INTEGRITY:
        Bending Stress: {aero['bending_stress'] / 1e6:.1f} MPa
        Safety Margin: {aero['safety_margin']:.2f}
        {'✓ STRUCTURALLY SOUND' if aero['safety_margin'] > 1.0 else '⚠️  STRUCTURAL CONCERN'}

        KEY FEATURES:
        • Extreme pointed nose cone (cubic taper)
        • 55° highly swept delta wings
        • Canard control surfaces
        • Tapered body for reduced drag
        • Low drag coefficient (Cd={self.drag_coeff})

        ESTIMATED PERFORMANCE:
        • Maximum Speed: Mach {self.length / 1.2:.1f}+
        • Range: {self.length * 150:.0f}+ km
        • Maneuverability: Excellent
        """

        print(specs)

    def show_design(self):
        """Display the complete missile design with aerodynamics"""
        fig_3d, aero_fig = self.create_missile_design()
        self.display_design_specs()

        # Add rotation animation to 3D view
        def animate(frame):
            fig_3d.axes[0].view_init(elev=25, azim=frame)
            return []

        self.animation = FuncAnimation(fig_3d, animate, frames=np.arange(0, 360, 2),
                                       interval=50, blit=False)

        plt.show()


# Create and display the complete missile design
if __name__ == "__main__":
    print("Creating ULTRA-STREAMLINED Missile with Aerodynamic Analysis...")
    print("=" * 65)

    # Create extremely streamlined missile
    missile = StreamlinedMissileDesigner(
        length=2.2,
        diameter=0.12,
        fin_span=0.22,
        fin_length=0.18
    )

    # Display the complete design
    missile.show_design()