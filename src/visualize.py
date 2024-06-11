import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple

# this function takes a 33x33x33 voxel (as tensor) and visualizes it.
def visualize(voxel, int_range=[0.6,1], title='Plot'):
    """
    Visualizes the magnitudes in a 3D scatter plot.

    Parameters:
    - voxel: tensor
        The intensity data (magnitudes) to be visualized.
    - int_range: list, optional
        The range of magnitude values to be displayed, everything else is cut off. The default values [0.6,1] work pretty well for my tests, but adjust if needed.
    - title: string, optional
        The title of the plot.

    Returns:
    None, displays the 3D plot. 

    Info:
    '%matplotlib widget' makes the plot interactive, could not be solved via if-statement, so comment or uncomment it manually.
    """

    # make interactive, if wanted
    #%matplotlib widget   

    # create meshgrid, better data structure for 3D plotting
    x, y, z = np.meshgrid(
        np.arange(voxel.shape[0]),
        np.arange(voxel.shape[1]),
        np.arange(voxel.shape[2])
    )

    # copy 
    voxel_cut = voxel.clone().detach().numpy()

    # set values that are not in int_range
    voxel_cut[voxel < int_range[0]] = np.nan
    voxel_cut[voxel > int_range[1]] = np.nan

    # Plot the intensity data in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=voxel_cut.flatten())

    # Add labels to the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # add title
    ax.set_title(title)

    # add legend with colorbar
    plt.colorbar(ax.scatter(x, y, z, c=voxel_cut.flatten()), ax=ax, label='Magnitude')
    plt.show()

def visualize_cloud(voxel, angles: Tuple[float, float] = None):
    """
    Visualize a 3D point cloud and a line defined by angles passing through the center of the plot.

    Parameters:
    voxel (numpy.ndarray): A numpy array of shape (N, 3) representing the 3D points.
    angles (tuple): A tuple of two angles (theta, phi) defining the line direction.
    """
    points = voxel

    # Extract x, y, z coordinates from the points tensor
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    alfa = 0.2

    # Create a new figure for the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a colormap with transparency
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([(0, 0, 1, alfa)])  # RGBA: blue with 50% transparency

    # Plot the points as a 3D scatter plot with the transparent colormap
    ax.scatter(x, y, z, marker='o', alpha=alfa, cmap=cmap)

    # Set labels for each axis
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set axis limits to match the dimensions of the data
    ax.set_xlim(0, 33)
    ax.set_ylim(0, 33)
    ax.set_zlim(0, 33)

    # Set title for the plot
    ax.set_title('3D Scatter Plot of Points')

    if angles != None:
        # Define the center of the plot
        center = np.array([16, 16, 16])

        # Extract the angles
        theta, phi = angles

        # Convert angles from degrees to radians
        theta = np.radians(theta)
        phi = np.radians(phi)

        # Define the direction of the line
        direction = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])

        # Define the line points
        line_length = 20  # Define the length of the line
        line_points = np.array([
            center - line_length * direction,
            center + line_length * direction
        ])

        # Plot the line
        ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], color='r', linewidth=3)

    # Show the plot
    plt.show()