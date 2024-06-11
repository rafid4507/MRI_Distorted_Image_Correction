import numpy as np
import torch
import matplotlib.pyplot as plt

# this function takes a 33x33x33 voxel (as tensor) and visualizes it.
def visualize(voxel, int_range=[0.6,1], scale=True, title='Plot'):
    """
    Visualizes the magnitudes in a 3D scatter plot.

    Parameters:
    - voxel: tensor
        The intensity data (magnitudes) to be visualized.
    - int_range: list, optional
        The range of magnitude values to be displayed, everything else is cut off. The default values [0.6,1] work pretty well for my tests, but adjust if needed.
    - scale: bool, optional
        Whether to scale the axes to the size of the voxel. Default is True.
    - title: string, optional
        The title of the plot.

    Returns:
    None, displays the 3D plot. 

    Info:
    Add "import PyQt6 & %matplotlib qt" to make the plot a interactive window.
    """ 

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
    ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')

    # set ranges to 0 - len(voxel)
    if scale: 
        ax.set_xlim([0, len(voxel)])
        ax.set_ylim([0, len(voxel)])
        ax.set_zlim([0, len(voxel)])

    # add title
    ax.set_title(title)

    # add legend with colorbar
    plt.colorbar(ax.scatter(x, y, z, c=voxel_cut.flatten()), ax=ax, label='Magnitude')
    plt.show()

# gets the magnitudes
def get_mags(voxel):
    """
    Calculate the normalized magnitudes of a voxel.

    Parameters:
        voxel (torch.Tensor): The input voxel.

    Returns:
        torch.Tensor: The magnitudes of the voxel (3D tensor with values from 0 to 1).
    """

    # cet the magnitudes and normalize them
    # mags = torch.sqrt(torch.sum(voxel**2, dim=3))
    # mags = (mags - mags.min()) / (mags.max() - mags.min())
    # mags / mags.max()
    mags = torch.norm(voxel, dim=3)
    
    return mags

# noise generation
def gen_noise(dim=33, log=False, seed=None):
    """
    Generate (normalized) 3D noise using a Gaussian distribution and random values.

    Parameters:
    - dim (int): The dimension of the noise cube (default: 33).
    - log (bool): Whether to print log information (default: False).
    - seed (int): The seed value for reproducibility (default: None).

    Returns:
    - noise (tensor): The generated 3D noise tensor.
    """

    # only for logging, then the test point is the same
    # if seed != None:
    #     np.random.seed(seed)

    ## STEP 01 ##
    # generate the 3d gaussian distribution
    x, y, z = (torch.linspace(-1, 1, dim) for _ in range(3))
    # form to meshgrid (better for 3d)
    x, y, z = torch.meshgrid(x, y, z)

    # Compute Gaussian values
    #gauss3d = torch.exp(-2 * (x**2 + y**2 + z**2)) 
    #gauss3d = torch.exp(- (x**2 + y**2 + z**2) / 2) / (2 * torch.tensor(np.pi))**(3/2)
    gauss3d = gauss3d = torch.exp(-((x**2 + y**2 + z**2) / (2)))

    # Normalize the distribution (sum is 2 in the end because [val, val])
    gauss3d = gauss3d / gauss3d.sum()

    if log:
        print('\n\n--- gauss3d ---')
        # find max value position
        print(gauss3d.max())
        print('cords of max value =', torch.where(gauss3d == gauss3d.max()))
        print('value at max cords =', gauss3d[16, 16, 16])

        # find min value position
        print(gauss3d.min())
        print('cords of min value =', torch.where(gauss3d == gauss3d.min()))
        print('value at min cords =', gauss3d[0, 0, 0], end='\n\n\n')


    # duplicate each field to list
    gauss3d = torch.stack([gauss3d, gauss3d], dim=3)

    # log if needed
    if log: 
        print('--- log for 01: gauss ---') 
        print('sum =', gauss3d.sum())
        print('max =', gauss3d.max())
        print('min =', gauss3d.min())
        print()
        print('edge =', gauss3d[0, 0, 0])
        print('cntr =', gauss3d[dim//2, dim//2, dim//2])

    ## STEP 02 ##
    # set seed to make it reproducible, if added
    if seed != None:
        torch.manual_seed(seed)
    
    # generate random values in same size
    a = torch.rand(dim, dim, dim) * 2 - 1
    b = torch.rand(dim, dim, dim) * 2 - 1

    # create complex tensor
    random_noise = torch.stack([a, b], dim=3)

    if log: 
        print('\n\n--- log for 02: random ---') 
        print('shape =', random_noise.shape)
        print('cooridnates =', 10, 12, 14)
        print('random field =', random_noise[10, 12, 14])
    
    ## STEP 03 ##
    # overlap the 3d gauss and the random noise
    noise = random_noise * gauss3d

    if log: 
        print('\n\n--- log for 03: noise ---')
        print('shape =', noise.shape)
        print('cooridnates =', 10, 12, 14)
        print('random field =', noise[10, 12, 14])

    # normalize between -1 and 1
    noise = (noise - noise.min()) / (noise.max() - noise.min()) * 2 - 1
    return noise

# noise application
def apply_noise(voxel, noise, factor=0.4, log=False):
    """
    Applies given noise to a data-voxel.

    Parameters:
    voxel (tensor): The voxel data.
    noise (tensor): The generated noise data.
    factor (float): The scaling factor for the noise (default: 0.4, good for my tests).
    log (bool): Whether to print log information (default: False).

    Returns:
    tensor: The voxel data with applied noise.

    Raises:
    ValueError: If the shapes of voxel and noise are not the same.
    """

    # check if the shape is the same
    if voxel.shape != noise.shape: 
        raise ValueError('Shapes of voxel and noise are not the same, they are {} and {}'.format(voxel.shape, noise.shape))
    
    # normalize the noise and the voxel 
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    voxel = (voxel - voxel.min()) / (voxel.max() - voxel.min())

    if log:
        print('\n--- from apply_noise ---')
        print('min =', voxel.min())
        print('max =', voxel.max())
        print('mean =', voxel.mean())

        # unnecassary
        print('==========noise')
        print(noise[0,0,0], 'at 0,0,0')
        print(noise[16,16,16], 'at 16,16,16')
        print(noise[31,31,31], 'at 31,31,31')

    # adjusting the ranges of the real and imaginary part
    # noise[:, :, :, 0] = noise[:, :, :, 0]
    noise[:, :, :, 1] = noise[:, :, :, 1] * voxel[:, :, :, 1].mean()        # ???

    if log: 
        print('====================voxel')
        print(voxel[:, :, :, 0].mean())
        print(voxel[:, :, :, 1].mean())
        # does not work yet (the adjustment)
        print('====================noise')
        print(noise[:, :, :, 0].mean())
        print(noise[:, :, :, 1].mean())

    # print a test real and imaginary value
    if log:
        print('\n--- from apply_noise - 0 ---')
        cords = (10, 10, 10)
        print('voxel at cords =', voxel[cords])
        print('noise at cords =', noise[cords])

    # print some test values
    if log:
        print('\n--- from apply_noise - 1 ---')
        cords = (10, 10, 10)
        print('voxel at cords =', voxel[cords])
        print('noise at cords =', noise[cords])

    # get variance and mean of the voxel
    if log:
        print('\n--- from apply_noise - 2 ---')
        print('var_voxel =', voxel.var())
        print('mean_voxel =', voxel.mean())
        # get variance and mean of the noise
        print('var_noise =', noise.var())
        print('mean_noise =', noise.mean())

    # apply generated noise in size of voxel to the data
    noised =  voxel + (noise * factor)
    
    if log:
        print('\n--- from apply_noise ---')
        print('min =', noised.min())
        print('max =', noised.max())

    # normalize the noised data
    noised = (noised - noised.min()) / (noised.max() - noised.min())
    return noised
