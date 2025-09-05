import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

# Set up the figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left plot: Complex plane showing e^(ix) tracing unit circle
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-1.5, 1.5)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)
ax1.set_xlabel('Real', fontsize=12)
ax1.set_ylabel('Imaginary', fontsize=12)
ax1.set_title(r'$e^{ix}$ traces the unit circle in the complex plane', fontsize=14)

# Draw unit circle
circle = Circle((0, 0), 1, fill=False, edgecolor='gray', linestyle='--', alpha=0.5)
ax1.add_patch(circle)

# Generate points for the path
theta = np.linspace(0, 2*np.pi, 1000)
x_path = np.cos(theta)
y_path = np.sin(theta)

# Plot the full path
ax1.plot(x_path, y_path, 'b-', alpha=0.3, linewidth=2)

# Highlight special points
ax1.plot(1, 0, 'go', markersize=10, label=r'$e^{i \cdot 0} = 1$')
ax1.plot(-1, 0, 'ro', markersize=10, label=r'$e^{i\pi} = -1$')
ax1.plot(0, 1, 'mo', markersize=8, label=r'$e^{i\pi/2} = i$')
ax1.plot(0, -1, 'co', markersize=8, label=r'$e^{i3\pi/2} = -i$')

# Add annotations
ax1.annotate(r'$e^{i\pi} = -1$', xy=(-1, 0), xytext=(-1.3, 0.3),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=12, color='red', weight='bold')

# Right plot: Real and imaginary parts vs angle
ax2.set_xlim(0, 2*np.pi)
ax2.set_ylim(-1.2, 1.2)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.set_xlabel(r'Angle $x$ (radians)', fontsize=12)
ax2.set_ylabel('Value', fontsize=12)
ax2.set_title(r'Real and Imaginary parts of $e^{ix}$', fontsize=14)

# Plot cos(x) and sin(x)
x_angles = np.linspace(0, 2*np.pi, 1000)
ax2.plot(x_angles, np.cos(x_angles), 'b-', label=r'$\cos(x)$ (Real part)', linewidth=2)
ax2.plot(x_angles, np.sin(x_angles), 'r-', label=r'$\sin(x)$ (Imaginary part)', linewidth=2)

# Mark special angles
special_angles = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
special_labels = ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']

for angle, label in zip(special_angles, special_labels):
    ax2.axvline(x=angle, color='gray', linestyle=':', alpha=0.5)
    ax2.text(angle, -1.15, label, ha='center', fontsize=10)

# Highlight π
ax2.axvline(x=np.pi, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax2.plot(np.pi, -1, 'ro', markersize=10)
ax2.plot(np.pi, 0, 'ro', markersize=10)

# Add text box with Euler's identity
textstr = r'$e^{i\pi} + 1 = 0$'
props = dict(boxstyle='round', facecolor='yellow', alpha=0.8)
ax2.text(0.5, 0.95, textstr, transform=ax2.transAxes, fontsize=16,
        verticalalignment='top', bbox=props, weight='bold')

# Legends
ax1.legend(loc='upper right', fontsize=10)
ax2.legend(loc='lower right', fontsize=10)

# Main title
fig.suptitle("Euler's Identity: The Most Beautiful Equation in Mathematics", 
             fontsize=16, weight='bold')

plt.tight_layout()
plt.show()

# Additional visualization: Animation showing the rotation
fig2, ax3 = plt.subplots(figsize=(8, 8))
ax3.set_xlim(-1.5, 1.5)
ax3.set_ylim(-1.5, 1.5)
ax3.set_aspect('equal')
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='k', linewidth=0.5)
ax3.axvline(x=0, color='k', linewidth=0.5)
ax3.set_xlabel('Real', fontsize=12)
ax3.set_ylabel('Imaginary', fontsize=12)
ax3.set_title(r'Animation: $e^{ix}$ rotating around the unit circle', fontsize=14)

# Draw unit circle
circle2 = Circle((0, 0), 1, fill=False, edgecolor='gray', linestyle='--', alpha=0.5)
ax3.add_patch(circle2)

# Initialize objects for animation
line, = ax3.plot([], [], 'b-', linewidth=2)
point, = ax3.plot([], [], 'ro', markersize=10)
radius_line, = ax3.plot([], [], 'g-', linewidth=1.5)
text = ax3.text(0.1, 1.3, '', fontsize=12, weight='bold')

def init():
    line.set_data([], [])
    point.set_data([], [])
    radius_line.set_data([], [])
    text.set_text('')
    return line, point, radius_line, text

def animate(frame):
    t = frame * 0.05  # Adjust speed
    
    # Generate path up to current point
    theta_current = np.linspace(0, t, max(int(t * 50), 1))
    x_current = np.cos(theta_current)
    y_current = np.sin(theta_current)
    
    # Current point
    x_point = np.cos(t)
    y_point = np.sin(t)
    
    line.set_data(x_current, y_current)
    point.set_data([x_point], [y_point])
    radius_line.set_data([0, x_point], [0, y_point])
    
    # Update text
    if abs(t - np.pi) < 0.05:
        text.set_text(r'$e^{i\pi} = -1$, so $e^{i\pi} + 1 = 0$')
        text.set_color('red')
    else:
        text.set_text(f'$e^{{i \\cdot {t:.2f}}} = {x_point:.2f} + {y_point:.2f}i$')
        text.set_color('black')
    
    return line, point, radius_line, text

# Create animation
anim = FuncAnimation(fig2, animate, init_func=init, frames=126, 
                    interval=50, blit=True, repeat=True)

plt.show()

print("\nEuler's Identity demonstrates that:")
print("- When we raise e to the power of i*π, we get -1")
print("- This connects 5 fundamental constants: e, i, π, 1, and 0")
print("- It emerges from the fact that e^(ix) traces a unit circle in the complex plane")
print("- At x = π, we've rotated exactly halfway around the circle to reach -1")