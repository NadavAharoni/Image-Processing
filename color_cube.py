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
    for start, end in combinations(corners, 2):
        # Only connect corners that share an edge (distance of exactly 1)
        if np.sum(np.abs(np.array(start) - np.array(end))) == 1:
            print(*zip(start, end))
            ax.plot3D(*zip(start, end), color="black", linestyle="--", linewidth=1)

    # 4. DRAW THE VECTOR (from 0,0,0 to 1,1,1)
    # The first three args are start point; next three are direction vectors
    ax.quiver(0, 0, 0, 1, 1, 1, 
              color='black', 
              linewidth=1, 
              arrow_length_ratio=0.01, 
              label='Luminance Vector')

    # 5. Final plot adjustments
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    ax.set_title('3D RGB Color Cube (2026)')
    
    # Set axis limits and equal aspect ratio
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_box_aspect([1,1,1]) 

    plt.show()

if __name__ == "__main__":
    draw_color_cube()
