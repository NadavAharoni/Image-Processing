import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations

def draw_color_cube():
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Define the 8 corners of the cube (0 or 1 for each axis)
    r = [0, 1]
    corners = list(product(r, r, r))
    
    # 2. Draw spheres at each corner colored by their RGB value
    for corner in corners:
        # The (x,y,z) coordinates are the RGB values [0.0 to 1.0]
        print(f"corner={corner}")
        ax.scatter(corner[0], corner[1], corner[2], 
                   color=corner, s=200, edgecolors='black', alpha=1)

    # 3. Draw dashed edges between adjacent corners
    # Edges that connect the origin with the "main" colors are
    # a bit thicker
    for start, end in combinations(corners, 2):
        print(*zip(start, end)) # for debugging / verbosity
        # Only connect corners that share an edge (distance of exactly 1)
        if np.sum(np.abs(np.array(start) - np.array(end))) == 1:
            if (0, 0, 0) in (start, end):
                # Main RGB Axes: Solid and Bold
                ax.plot3D(*zip(start, end), color="black", linestyle="-", linewidth=2)
            else:
                # Other edges: Dashed and Subtle
                ax.plot3D(*zip(start, end), color="black", linestyle="--", linewidth=1, alpha=0.5)

    # 4. Draw the "brightness" vector (from 0,0,0 to 1,1,1)
    # The first three args are start point; next three are direction vectors
    draw_brightness_vector = True
    if draw_brightness_vector:
        ax.quiver(0, 0, 0, 1, 1, 1, 
                color='black', 
                linewidth=1.5, 
                arrow_length_ratio=0.03,
                linestyle='dashed',
                facecolor='none', 
                label='Luminance Vector')

    # Add example colors:
    example_colors = False
    if example_colors:
        origin = (0,0,0)
        example_colors = [
            [200, 100, 50],
            [200, 150, 20],
            [200, 200, 80],
            [200,  50, 100],
            [255, 128, 0] # orange
        ]
        for c in example_colors:
            color = np.array(c) / 255.0
            ax.plot3D(*zip(origin, color), color="black", linestyle="--", linewidth=1)
            ax.scatter(color[0], color[1], color[2],
                    color=color, s=200, edgecolors='black', alpha=1)

    # 5. Final plot adjustments
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    ax.set_title('3D RGB Color Cube (2026)')
    
    # Set axis limits and equal aspect ratio
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    # ax.margins(0.15)  # Add 15% margin to prevent clipping during rotation
    ax.set_box_aspect([1,1,1]) 

    plt.show()

if __name__ == "__main__":
    draw_color_cube()
