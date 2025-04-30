# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 17:26:09 2025

@author: alexa
"""



import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


x0, y0, z0 = 4604.49276873138, 1150.81472538679, 4694.55079634563   # km
vx0, vy0, vz0 = -5.10903235110107 , -2.48824074138143 ,5.62098648967432   # km/s

# Pack initial state vector
X0 = [x0, y0, z0, vx0, vy0, vz0]  


def TBP_ECI(t, state_X, mu):
    x, y, z, vx, vy, vz = state_X  # Unpack state vector
    r = np.sqrt(x**2 + y**2 + z**2)  # Compute radius
    ax, ay, az = -mu * x / r**3, -mu * y / r**3, -mu * z / r**3  # Acceleration components
    return [vx, vy, vz, ax, ay, az]  # Return derivatives


# Constants
mu_earth = 398600.4418  # km^3/s^2
r_earth = 6378.137  # km (Earth radius)

r0 = np.linalg.norm(X0[:3])  # Initial distance from Earth's center (km)
v0 = np.linalg.norm(X0[3:])  # Initial speed (km/s)
a = 1 / (2 / r0 - v0**2 / mu_earth)  # Semi-major axis (km)
T = 2 * np.pi * np.sqrt(a**3 / mu_earth)  # Orbital period (s)

# Set integration time span for two orbital periods
t_start = 0
t_end = 2 * T  # Two orbital periods
time_step = 10  # Output every 10 seconds
t_eval = np.arange(t_start, t_end, time_step)  

# Solve the system using solve_ivp with strict tolerances
solution = solve_ivp(
    TBP_ECI, (t_start, t_end), X0, t_eval=t_eval, method='RK45',
    args=(mu_earth,), rtol=1e-12, atol=1e-12
)

# Extract components
x, y, z = solution.y[0], solution.y[1], solution.y[2]
vx, vy, vz = solution.y[3], solution.y[4], solution.y[5]

r = np.sqrt(x**2 + y**2 + z**2)



#v_lin = np.sqrt(vx**2 + vy**2 + vz**2)



# 3D Trajectory Plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot([0,0], [0,0], [-2*r0, 2*r0], color='k', linestyle='--', label="Earth rotation axis (z)")
#Plot Earth as a sphere
earth_radius = 6378  # km (mean Earth radius)
u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
X_earth = earth_radius * np.cos(u) * np.sin(v)
Y_earth = earth_radius * np.sin(u) * np.sin(v)
Z_earth = earth_radius * np.cos(v)
ax.plot_surface(X_earth, Y_earth, Z_earth, color='b', alpha=0.3)

# Plot the orbit
ax.plot(x, y, z, label="Orbit Path", color='r')

def is_behind_earth(point, view_direction):
    # Calculate vector from Earth center to the point
    earth_to_point = np.array(point)
    # Calculate dot product with viewing direction
    dot_product = np.dot(earth_to_point, view_direction)
    # Calculate distance from Earth center
    distance = np.linalg.norm(earth_to_point)
    # If dot product is negative and distance less than Earth radius, point is behind Earth
    return dot_product < 0 and distance < earth_radius

# Create segments for visible and hidden parts
# We'll determine visibility based on camera's view direction
visible_x, visible_y, visible_z = [], [], []
hidden_x, hidden_y, hidden_z = [], [], []

# We'll use segments to break the orbit into visible and hidden parts
segments = []
current_segment = {"visible": True, "points": []}

# Function to check if a point is behind Earth from the current view
def is_point_behind_earth(x, y, z, camera_pos):
    # Vector from camera to point
    point = np.array([x, y, z])
    earth_center = np.array([0, 0, 0])
    
    # Vector from camera to point and from camera to earth center
    cam_to_point = point - camera_pos
    cam_to_earth = earth_center - camera_pos
    
    # Normalize vectors
    cam_to_point_norm = cam_to_point / np.linalg.norm(cam_to_point)
    cam_to_earth_norm = cam_to_earth / np.linalg.norm(cam_to_earth)
    
    # Angle between vectors
    dot_product = np.dot(cam_to_point_norm, cam_to_earth_norm)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    
    # Distance from earth center to the point
    earth_to_point = point - earth_center
    distance_from_earth_center = np.linalg.norm(earth_to_point)
    
    # Distance from camera position to point
    distance_from_camera = np.linalg.norm(cam_to_point)
    
    # Distance from camera to earth center
    distance_to_earth = np.linalg.norm(cam_to_earth)
    
    # Calculate if point is behind earth
    # Using law of cosines to determine if earth blocks the point
    # Earth blocks if the point is farther than the earth's center and 
    # the angle is small enough that the line of sight passes through earth
    angle_threshold = np.arcsin(earth_radius / distance_to_earth)
    
    return (angle < angle_threshold and 
            distance_from_camera > distance_to_earth and 
            distance_from_earth_center < distance_from_camera)

# Define the camera position (viewer's perspective)
# Positioning the camera far away on the positive z-axis for this example
camera_pos = np.array([0, -20000, 5000])

# Process orbit points to determine visibility
current_visible = None
current_segment = []

for i in range(len(x)):
    # Check if current point is behind Earth
    behind = is_point_behind_earth(x[i], y[i], z[i], camera_pos)
    
    # If visibility status changed, start a new segment
    if current_visible is None:
        current_visible = not behind
        current_segment = [(x[i], y[i], z[i])]
    elif (behind and current_visible) or (not behind and not current_visible):
        # Visibility changed, add current segment to segments list
        segments.append({"visible": current_visible, "points": current_segment})
        # Start new segment
        current_visible = not behind
        current_segment = [(x[i], y[i], z[i])]
    else:
        # Continue current segment
        current_segment.append((x[i], y[i], z[i]))

# Add the last segment
if current_segment:
    segments.append({"visible": current_visible, "points": current_segment})

# Plot orbit segments
for segment in segments:
    points = np.array(segment["points"])
    if len(points) > 0:
        if segment["visible"]:
            ax.plot(points[:, 0], points[:, 1], points[:, 2], color='r', linewidth=2)
        else:
            ax.plot(points[:, 0], points[:, 1], points[:, 2], color='r', linewidth=1, alpha=0.2, linestyle='--')

# Labels and title
ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_zlabel("Z (km)")
ax.set_title("Orbital Trajectory in ECI Frame (Polar Orbit)")

# Set equal aspect ratio
ax.set_box_aspect([1, 1, 1])

# Set limits to give a clear view
limit = 1.5 * r0
ax.set_xlim([-limit, limit])
ax.set_ylim([-limit, limit])
ax.set_zlim([-limit, limit])

# Add a custom view angle to better show the polar nature of the orbit
ax.view_init(elev=25, azim=-45)

plt.tight_layout()
plt.show()