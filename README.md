# Advanced Missile Guidance Simulation System

A comprehensive Python-based simulation of an advanced missile guidance system with realistic aerodynamics, defense evasion, and 3D visualization.

## Overview

This project simulates a sophisticated missile guidance system featuring:

- Real-time 3D animation with detailed missile visualization
- Radar homing guidance with predictive targeting
- Aerodynamic fin control with realistic dynamics
- Moving targets with randomized evasion behaviors
- Defense systems that actively engage the missile
- Obstacle avoidance algorithms for threat evasion
- Terminal guidance phase for precision strikes

## System Architecture

### Core Components

| Component          | Description                      | Key Features                                          |
|--------------------|----------------------------------|-------------------------------------------------------|
| GuidedMissile      | Main missile control system      | Radar homing, defense evasion, predictive guidance    |
| FinControlSystem   | Aerodynamic control surfaces     | Realistic fin dynamics, speed-based effectiveness     |
| VehicleParams      | Physical and control parameters  | Mass, propulsion, guidance gains, aerodynamics        |
| MovingTarget       | Dynamic target simulation        | Multiple movement patterns, smooth transitions        |
| DefenseSystem      | Anti-missile defenses            | SAM sites, AAA, radar-guided weapons                  |
| 3D Modeler         | Missile visualization            | Streamlined design, aerodynamic analysis              |

### Guidance Modes

- **Ballistic Phase** - Open-loop trajectory optimization
- **Mid-course Guidance** - Radar homing with threat avoidance
- **Terminal Phase** - High-precision guidance for final approach

## Key Features

### Advanced Guidance

- **Proportional Navigation** with adaptive gains
- **Predictive Targeting** - aims where target will be
- **Threat-based Avoidance** - balances evasion with mission objectives
- **Energy Management** - optimizes propulsion usage

### Realistic Physics

- **Aerodynamic Drag** with altitude-based air density
- **Fin Control Dynamics** - second-order system with damping
- **Wind Effects** - environmental factors
- **Structural Limits** - maximum deflection and acceleration

### Defense Systems

- **Multiple Threat Types**: SAM, AAA, Radar-guided
- **Projectile Simulation** - incoming defense fire
- **Collision Detection** - missile destruction conditions
- **Threat Evaluation** - weighted avoidance vectors

## Project Structure

```
missile_guidance_system/
├── main.py                          # Main simulation orchestrator
├── guidance/
│   ├── GuidedMissile.py            # Core missile guidance logic
│   ├── VehicleParams.py            # Physical parameters dataclass
│   └── FinControlSystem.py         # Aerodynamic control system
├── target_system/
│   ├── MovingTarget.py             # Dynamic target behaviors
│   └── target_defense/
│       ├── DefenseSystem.py        # Anti-missile defense systems
│       └── DefenseProjectile.py    # Defense projectile simulation
├── modeler3D.py                    # 3D missile visualization
```

## Usage

### Running Simulations

**Launch the System:**

```python
python main.py
```

**Select Target Scenario:**

- Choose from 6 predefined scenarios
- Each with different target types and defense layouts

**View Results:**

- Real-time 3D animation
- Performance metrics dashboard
- Aerodynamic analysis

### Custom Scenarios

Create custom scenarios by modifying the target and defense configurations in `main.py`:

```python
custom_scenario = {
    "pos": [x, y, z],           # Target position
    "type": "airborne",         # Target type
    "defenses": [               # Defense systems
        ([def_x, def_y, def_z], "sam"),
        ([def_x, def_y, def_z], "aaa")
    ]
}
```

## Configuration

### Key Parameters (VehicleParams.py)

| Parameter            | Description                      | Default Value |
|----------------------|----------------------------------|---------------|
| mass                 | Missile mass                     | 2.0 kg        |
| thrust_accel         | Main propulsion acceleration     | 200 m/s²      |
| fin_max_deflection   | Maximum fin angle                | 25°           |
| pn_gain              | Proportional navigation gain     | 35.0          |
| radar_range          | Target detection range           | 1500 m        |

### Target Types

- **Ground**: Stationary targets
- **Airborne**: Smooth random movement
- **Naval**: Evasive maneuvers
- **Bunker**: Fortified stationary targets

### Defense Types

- **SAM**: Long-range, high threat
- **AAA**: Medium-range, moderate threat
- **Radar**: Long-range detection, low threat

## Output & Visualization

### Real-time Displays

- **3D Trajectory Plot**: Missile path with defense systems
- **Fin Control Graphs**: Individual fin deflection over time
- **Guidance Acceleration**: Commanded acceleration profile
- **Radar Status**: Target lock and range information

### Performance Metrics

- Flight time and maximum altitude
- Speed profiles and acceleration
- Fin deflection utilization
- Radar lock percentage
- Impact accuracy

## Technical Details

### Guidance Algorithm

```python
# Predictive targeting
predicted_target_pos = target_pos + target_velocity_estimate * prediction_time

# Balanced guidance with threat avoidance
guidance_dir = (1.0 - threat_level) * los_unit + threat_level * avoidance_vector
```

### Fin Control Dynamics

- Second-order system with damping
- Rate limiting based on physical constraints
- Speed-dependent effectiveness
- Realistic response time (0.08s)

### Aerodynamic Modeling

- Drag forces proportional to velocity squared
- Lift generation from fin deflections
- Altitude-based air density variation
- Wind and turbulence effects

## Simulation Scenarios

1. **Standard Ground Target** - Basic stationary target with SAM defense
2. **Airborne Target** - Moving air target with radar defense
3. **Naval Target** - Evasive ship target with mixed defenses
4. **Mountain Bunker** - Heavily defended stationary target
5. **Close-range Test** - Simple validation scenario
6. **Advanced Air Target** - Fast-moving target with layered defenses

## Optimization Features

### Launch Parameter Optimization

- Automated speed, elevation, and azimuth calculation
- Grid search and differential evolution methods
- Preference for high-angle trajectories
- Real-time optimization progress

### Adaptive Guidance

- Distance-based gain adjustment
- Terminal phase activation (100m range)
- Threat-level based maneuver aggressiveness
- Energy management for extended range

## Limitations & Assumptions

### Current Limitations

- Simplified atmospheric model
- Basic radar propagation (no multipath)
- Idealized sensor measurements
- 2D planar defense engagement

### Key Assumptions

- Constant gravitational field
- Exponential air density decay
- Ideal fin actuation (no mechanical delays)
- Perfect radar detection within range

## Future Enhancements

Planned improvements:

- Enhanced atmospheric modeling
- Advanced radar simulation (clutter, multipath)
- More sophisticated threat evaluation
- Multi-missile coordination
- Real-world terrain integration
- Hardware-in-the-loop capability
