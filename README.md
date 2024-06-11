# Hough-Transformations
Hugh-Approximation

Explanation of what we are trying to achieve.

This is our output from Hough:

npoints=119, a=(28.294538,-37.813383,479.314286), b=(0.301528,0.570895,0.763649)

npoints=62, a=(95.670981,69.383966,517.677419), b=(0.506133,0.689802,-0.517690)

npoints=22, a=(33.465249,9.660581,744.581818), b=(0.118717,0.038881,-0.992167)

npoints=2, a=(-98.164700,59.576900,1267.200000), b=(0.232210,-0.147954,-0.961347)

Each line represented by  a(a point) and b (a direction vector) could be a segment of a fiber tract.

By connecting these points and following their direction vectors, we can construct fiber tracts. The number of points   indicates  the level of detail of the tract.

<img width="456" alt="image" src="https://github.com/Project-THI/Hough-Transformations/assets/121182611/fa771300-5f7a-48d5-b331-57e02922c9a0">


## Setup

We do not load the data onto the GitHub Repository. That is why everone should create a `data` folder in their working directory, which is included in `.gitignore`. The `.mat` files must be placed their. Additionally, when we write function, which transform the data, and we want to save the transformed data, the functions will store it inside the `data` folder.

## General Guideline

The file `src/loaders.py` contains important and usefully function for loading and transforming the data. Please use them over the course of our project.
The functions located in `loaders.py` help us to insure data consistency, while we all work on our different subtasks.

Importand functions include:
```
def load_mat(
    name: str,
    split_real_imaginary: bool = True,
    include_metadata: List[Literal["fiber_fractions"]] = []
    ) -> Tuple[Tensor, Tensor]:

def load_point_clouds(name: str) -> Tuple[Dict[int, Tensor], Dict[int, Tensor]]:

def get_mags(voxel: torch.Tensor) -> torch.Tensor:

def visualize(voxel, int_range=[0.6,1], title='Plot'):
```
Upcoming ones:
```
def save_tensors()

def load_tensors()
```

# Table of Contents
- [Hough-Transformations](#hough-transformations)
    - [Approach](#approach)
    - [Data](#data)
    - [Hugh](#hugh)
    - [Noising](#noising)
    - [PCA](#pca)
  - [Notes](#notes)
  - [Questions](#questions)


### Approach

> [!NOTE]
> 1) Hugh Transform on clean data -> Gets us the lines
> 2) Train a Neural Network, which approximates the HughTransform (Reason: More Robust than Standard HughTransform); taking the lines from step 1) as Ground truth
> 3) Make data more noisy, which should Show that the HughTransform is less robust as the NN (proofing our assumption on 2) ); In other words, the HughTransform could be our baseline which we want to beat
> 4) Train the NN to its best!


### Data 
Features: `33x33x33` Array per Sample <br>
target: Linienfunktion

### Hugh
Use a Python script like [this](https://github.com/JingLin0/3D-Hough-Transform). 

### PCA

In using PCA for analyzing the direction of "freedom of movement" we find the eigenvectors . The direction of these eigenvectors tells us about the principal axes of our data distribution. A larger eigenvalue means more variance, which translates to a broader spread of data points along its corresponding eigenvector. This is the direction in which our data can "move" or vary the most, implying a higher degree of freedom in that dimension and this is what we need to follow higher variance. In the other hand the calculation of the angle as we discussced with Mauritz , the angle corresponds to the angle between eigenvector axis and a 3 DoF axis (can be x or z ,...) . This gives us insights into how these principal directions are oriented relative to our established coordinate system. This information can be useful for aligning, rotating, or transforming our data in further analyses or graphical representations but not what we need.



<details><summary>Algorithm Explanation</summary>
Lines in 2D and 3D Spaces



**Line Representation in 2D Space**

Slope-Intercept Form:  y=mx+c

Parametric Form:  r ⃗ =p ⃗+tv ⃗  , where p is a point on the line, v is a direction vector, and t is a scalar that allows the line to extend infinitely in both directions.

**Line Representation in 3D Space**

Parametric Form: The standard way to represent a line in 3D is  r ⃗ = N ⃗ + t B ⃗   , where  N ⃗    is a point on the line,  B ⃗    is a direction vector, and  t is a scalar. 

Now the idea is that we need to calculate these parameters to find the points that define the straight line . A very efficient way is calculating B by using two other parameters: 

θ - Azimuth angle, which is the horizontal angle measured in the xy-plane from the positive x-axis.

ϕ - Elevation angle, which is the vertical angle measured from the positive z-axis downward.

Now we these two parameter which are calculated , we can find the direction vector and our 3d Point in line which is closest to the origin to find the N . The approach is here using the sphere coordiantes to estimate our line. 

To go further with this first we apply an edge detection filter which highlights our edges or potential lines. Then we set up the Hough space using these parameters :

r=xcos(ϕ)+ycos(ϕ)+zsin(ϕ) //here the potential line calcualated throguh each edge points.

Now when for each edge points the r is calculated we accumulate the parameters that are most shown which translates to a potential line.

</details>


<details><summary>Code Explanation</summary>
...
</details>

Research :
Try to calculate or find the orientation of the oval shape. This will infer to where the line is headed to and as output we have the line and the orientation. 
Here is the approach : 
To process the image and find the ellipse's boundary, we convert the image to grayscale and then apply the Canny edge detector. The Canny algorithm is effective at detecting edges with high precision, which is crucial for accurately outlining the ellipse.

After detecting edges, we find contours in the image. A contour is a curve joining all continuous points along the boundary that have the same color or intensity. We select the largest contour as it is expected to correspond to our ellipse.

Now we need to find the rotation matrix and the endpoitns and inferr the line . 


### Noising
Stage wise apply noise (5%, 10%, etc.) on the data.
- Blur
- etc.

<details><summary>People</summary>

Matthias (research data noising)

</details>

### Notes
## Meeting with Fr. Menzel, 2024/04/30: Take Away
We talk about the approach in total and if its fine. She told me that the way in which I presented it, that we are on the right path.

**Line Detection**

The approach is interesting because of the fact that we only want to predict the lines. You could said, that it is a subproblem of the greater picture. Which does not make it less important.
We need to create our on labels (the line orientation), which creates another step for us. She told me about the RADON Transformation, which which is basically approximated be the approach I am going for in the `HughImpl` Branch.

Sometimes, there will be distributions, which make it very difficult for us to detect if there are multiple oval shape overlapping with each other. One might be to big and eat up the other one. This case can happen, and we should simply accept it.

Techincally, the number of fibers (and the fraction of each fiber) is stated in the metadata of the samples. We could use that as reference. Technically, since we work with symphatic data, we could as her to produce us data, which has the orientations (our 2 angles) in the metadata. But that should only be concidered if we are having trouble producing our own labels.

**Noising and Data Representation**

The data consists of a reel and an imaginary part. We should use the magnitude of the complex numbers as values that make up our 33x33x33 Tensor (look here as reference: https://mriquestions.com/real-v-imaginary.html). This means we must transfrom the data first.

More interestingly, noise must be first applied on the complex values, then the magnitude shall be calculated.

$Q + a * N = G$

Q = qspace tensor

a = Noise Factor

N = Gaussian Noise Distribution

G = qspace with noise

Q is a distribution, which means that the sum of all the values in the tensor needs to be 1!

N is a Tensor which has has random values between [-1; 1] in each cell at first. The sum of all values has to be 0, thou. Then, it is multiple be a Gaussion Distribution. This makes sure that there is more noising of the data in the center (16x16x16). The result is N, our noise tensor.

<details><summary>People</summary>

Author of these notes: Moritz

</details>

## Questions
1. Are the values in `qspace` supposed to be complex numbers (and very large) like `tensor(4.2486e+14-0.1041j, dtype=torch.complex128)`? (I'll check the `.mat` file before asking, maybe there is a mistake on my side.)

    Jup , its because of FT(fourier). I guess this is the encoding of 2 values ( i think its intensity and phase or magnitude or something). Here you can find more to read: https://mriquestions.com/real-v-imaginary.html

2. question 2

   anser 2
