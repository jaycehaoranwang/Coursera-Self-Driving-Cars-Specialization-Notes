# Basics of 3D Computer Vision
## The Camera Sensor - Pinhole Camera Model
- A camera is a passive exteroceptive sensor. It uses an imaging sensor to capture information conveyed by light rays emitted from objects in the world.
- We want to place a barrier in front of the imaging sensor with a tiny hole or **aperture** in its center. The barrier allows only a small number of light rays to pass through the aperture, reducing the blurriness of the image. This model is called the __pinhole camera model__ and describes the relationship between a point in the world and it's corresponding projection on the image plane.
- The two most important parameters in a pinhole camera model are:
	- Distance between the pinhole and the image plane which we call the **focal length** $f$. 
	- The **focus length** defines the size of the object projected onto the image and plays an important role in the camera focus when using lenses to improve camera performance. The coordinates of the center of the pinhole, which we call the camera center, these coordinates define the location on the imaging sensor that the object projection will inhabit.
### Camera Projective Geometry
- Camera projective geometry deals with the question of how to project a 3D point in the world frame to the image frame using coordinate system transformations
- Light travels from the point in the world through the aperture to the sensor's surface. Because the aperture forces the straight light rays through a pinhole, the resulting image on the camera sensor will be flipped. upside down. To avoid this, **we usually define a virtual image plane in front of the camera center** so that the point can be projected on the virtual image plane without being flipped as the light rays theoretically did not pass through the aperture yet. 
- Now from the camera centre, we want to project the point in the real world $O_{world}=[X,Y,Z]^T$ to the 2D virtual image plane $O_{image} = [u,v]^T$.
- Now we define characteristics of the camera:
	- Select a **world frame** in which to define the coordinates of all objects and the camera
	- Define the **camera coordinate frame** as the coordinate frame attached to the center of our lens aperture known as the optical sensor. We will use a corresponding translation vector and rotation matrix to convert between the world frame and the camera frame. These parameters of the camera pose are **extrinsic parameters** as they are external to the camera and specific to the location of the camera in the world frame.
	- Define our __image coordinate frame__ as the coordinate frame attached to our virtual image plane emanating from the optical centre. Note that the center of the image pixel coordinate system is attached to the **top-left corner** of the virtual image plane, so we'll need to adjust the pixel locations to the image coordinate frame.
	- Define the **focal length** as the distance between the camera and the image plane along the z-axis of the camera coordinate frame.
	- **Intrinsic Parameters**: Allows a mapping between camera coordinates and pixel coordinates in the image frame. Parameters are internal and fixed to a particular camera/digitization setup. E.g. focal length, image center, geometric lens distortion parameters
	- **Extrinsic Parameters**: Defines the location and orientation of the camera with respect to the world frame. External to the camera and may change with respect to the world frame.
- With these definitions, our problem simplifies to:
	1. Project from world coordinates to camera coordinates/optical center
	2. Project from camera coordinates to image coordinates
	3. Then transform image coordinates to pixel coordinates through scaling and offset.
- World -> Camera:
	- $O_{camera} = [R | t]O_{world}$ where $[R | t]$ is a combined matrix with rotation matrix R and translation vector t put together
- Camera -> Image:
	- $O_{image} = KO_{camera}$ where K is a 3x3 camera intrinsic parameter matrix e.g. $\begin{bmatrix}  
f & 0 & u_0\\  
0 & f & v_0 \\
0 & 0 & 1 \\
\end{bmatrix}$
- World -> Image: 
	- Can combine transformations as P = K [R | t]
	- $O_{image} = PO = K[R|t]O_{world}$ but $O_{world}$ must be in homogeneous coordinate system, meaning it becomes a 4x1 vector with 1 at the end: $O_{world} = [X,Y,Z,1]^T$ since P is a 3x4 matrix
- Image Coordinates -> Pixel Coordinates
	- divide image coordinates by z: $\frac{1}{z}[x,y,z]^T = [u,v,1]^T$
	- Pixel coordinates can be multiplied by a scaling factor s for future definitions
### Camera Calibration
- Camera calibration allows us to obtain the camera matrix P and extract the intrinsic and extrinsic parameters from the camera matrix P
- Mathematically defined as finding the elements of the P matrix then decomposing it into intrinsic and extrinsic parameters
- Linear Method:
	- Linear method to solve camera calibration involves using an image of a known geometry (checkerboard with known square sizes). This way, we know the 3D position of a given point (vertex of squares on checkerboard) as well as the pixel coordinate and with enough points, we can construct a system of homogeneous linear equations that can be solved with SVD or LS
	- Advantages: Easy to formulate, closed for solution
	- Disadvantages: 
		- Does not directly provide camera parameters (intrinsic, and extrinsic pose)
		- Does not model radial distortion and other complex phenomena
		- Does not allow for constraints such as known focal length to be imposed
- Better Method: RQ Factorization of matrix P -  Camera Parameters to extract **intrinsic** and **extrinsic** parameters
	- $P = K[R|t] = K[R|-RC] = [KR|-KRC]$, let M = KR, $P = [M|-MC], M=\mathcal{R}\mathcal{Q}$ we use the fact that any square matrix can be factored into an upper triangular matrix R and an orthogonal basis to decompose M into upper triangular $\mathcal{R} \in \mathbb{R}^3$ and orthogonal basis $\mathcal{Q} \in \mathbb{R}^3$. Note that $\mathcal{R}$ is NOT the rotation matrix, its the output of the RQ factorization.
		- $C$ is the 3D camera center, is the point that projects to 0 when multiplied by P: $PC$ = 0
	- From here, we have $M = \mathcal{R} \mathcal{Q} = KR$ 
		- Intrinsic Calibration Matrix: $K=\mathcal{R}$
		- Rotation Matrix: $R = \mathcal{Q}$
		- Translation Vector: $t = -K^{-1}P[:,4] = -K^{-1}MC$
### Visual Depth Perception
- We can use a stereo Camera model with two cameras imitating human eyes to perceive depth information from a scene with two slight offset images. The two camera frames are completely identical in terms of orientation except that they are offset along the aligned x axis, where the positive z axis points toward the scene away from the image plane, and positive y points up from the x axis.
- ![](https://lh4.googleusercontent.com/hffvlrp_CwBboaKlia9bfF8vYWNF3TYzw51_WlbbE6Y-eXCkphjaYFhWJRJvYUx45TaifcBFJFV3ks4tJce92LYJhHhDaPfhs5tHSoSifSP8pKlFPtAIpqD5RVPgDNw7uiHBgHt86T_8jfLUgOrfdsw)
- Define params (Diagram provided in C3M1Lesson3Part1 slides):
	- **focal length $f$**: the distance between the camera center and the image plane
	- **baseline $b$**: the distance along the shared x-axis between the left and right camera centers. By defining a baseline to represent the transformation between the two camera coordinate frames, we are assuming that the rotation matrix is identity and there is only a non-zero x component in the translation vector. The R and T transformation therefore boils down to a single baseline parameter B
- We want to compute the x and z coordinates of the 3D point O in the scene with respect to the left camera frame (Note the coordinates are the same in the right camera frame except the X value is offset by baseline b). The y coordinate can be estimated easily after the x and z coordinates are computed.
	- We are given the baseline, focal length, and coordinates of the projection of the point O onto the left and right image planes
	- We define the __disparity__ d to be the difference between the image coordinates of the same pixel in the left and right images $d = x_L - x_R$. We can easily obtain $x_L ,  x_R$ from transforming them from pixel coordinates using x and y offsets: $x_L = u_L - u_0, x_R = u_R - u_0, y_L = v_L - v_0$
	- ![](https://lh4.googleusercontent.com/pY73f_qVu5qezGQKA1n_iZsFn1wyxUMtqnFLr1tQiKcP_4H892A7sIMpYxzfteJFD7bxSI4eTc0H2DOVWrB6aiCLh_H0kC_JC9YAwRD24956kBzUSpJUldf1MbJDtI2hd9J1AVFupLI1AUxbEeEMUFg)
	- Using similar triangle analysis from birds-eye view of the scene, we can solve for the values of $Z$ and $X$ as:
		- $Z = fb/d$, $X = Zx_L/f$, $Y=Zy_L/f$
- Before applying this solution, we need to address:
	- use stereo camera calibration to know $f,b,u_0, v_0$ 
	- Use disparity computation algorithms to find corresponding $x_R$ for each $x_L$
### Computing the Disparity
- **Disparity**: is the difference in the image location of the same 3D point as observed by two different cameras. To compute the disparity we need to be able to find the same point in the left and right stereo camera images. This problem is known as the stereo correspondence problem.
- We want to define a __Epipolar Constraint__ for Correspondence so that we only need to search for a pixel in the left image along a 1D line in the right image instead of over the entire 2D image.
	- The epipolar constraint works off the fact that if we move a point in the scene along the line that connects the point in the scene to the left camera center through the left image plane, its projection in the left image plane does NOT change, but the projection moves along a horizontal line on the RIGHT image plane. This line is called the __epipolar line__.
	- We now constraint our pixel search to be ONLY along this 1D epipolar line in the right image plane to find the position of the same desired point in the right image plane
	- Note that horizontal epipolar lines only occur if the optical axes of the two cameras are parallel/the image planes of the two cameras are in/part of the same 3D plane. If not, the epipolar line will NOT be horizontal, instead it will be skewed/angled. To in this case, we can use standard __stereo rectification__ algorithms to warp images originating from two cameras with non-parallel optical axes to force epipolar lines to be horizontal.
- Basic Stereo Matching Algorithm:
	- Given: rectified Images and Stereo Calibration
	- For each epipolar line:
		- Take each pixel on this line in the left image
		- Compare these left image pixels to every pixel in the right image on the same epipolar line
		- Pick the pixel that has minimum cost (e.g. squared difference in pixel intensities)
		- Compute disparity, d for pixel by subtraction
- This algorithm creates what we know as stereo matching, which can take two images from stereo cameras can produce a depth map of the pixels using the left image as the base as pixels of objects closer to the cameras will have less disparity than pixels of objects that are further away from the cameras.
- Note that more complex algorithms exist to improve the efficiency of the stereo matching process, and typically benchmarked on the Middlebury Stereo Benchmark.
### Image Filtering
- Image filtering is typically done through cross-correlation and convolution operations to reduce noise (e.g. salt and pepper noise in images where pixels are randomly converted to either white or black)
- __Cross-Correlation__ is built from a mean smoothing method based on the neighbourhood of pixels around an outlier noise pixel (compute the mean value of the pixels around and including the outlier pixel then divide by 1/(sum of kernel values)) but generalized to add a weight to every pixel in the neighbourhood before computing the mean, this weight matrix is called the __kernel__ and this generalized form is called __cross-correlation__
	- E.g. A mean filter uses a kernel of all 1s while a gaussian filter uses a kernel with higher weights in the middle and lower in the corners of the 3x3 matrix
	- Applications: 
		- Useful for Template matching, where given a small square section of an image and the original image, we can run cross-correlation to see which pixel has the highest response. This pixel with the highest cross-correlation response is the center pixel of the given square query image section.
- __A convolution__ is a cross-correlation where the filter __is flipped both horizontally and vertically__ before being applied to the image.
	- Convolution is superior because unlike cross-correlation, convolution is associative, meaning we can precompute and stack all filter convolutions ahead of time before applying it once to the image to reduce runtime
	- To implement a 2D convolution, we effectively compute a weighted mean of an area where the weights are given by the values in the kernel, and they are multiplied elementwise (after the horizontal and vertical flips) to the values in the actual image that the kernel is currently on top of, finally the middle value (pixel at center of the kernel) is replaced with the weighted mean of the kernel pixels divided by 1/(sum of element values of the kernel).
	- Depending on how the values in the kernel are distributed/chosen, the convolution can result in different operations such as a gaussian filter, blurring, sobel filter/gradient filter...etc
	- Applications:
		- Image gradients can be computed by a convolution with a kernel that performs finite difference. We can rotate our kernel in different orientations to get vertical, horizontal or even diagonal gradients of an image at a pixel.
		- Image gradients are extremely useful for detection of edges and corners, and are used extensively in self-driving for image feature and object detection
# Visual Features - Detection, Description and Matching
### Feature Detection
- __Features__ are __points of interest__ in an image
- __Points of Interest__ should have the following characteristics:
	- __Saliency__: Distinctive, identificable, and different from its immediate neighborhood
	- __Repeatability__: can be found in multiple images using same operations
	- __Locality__: Occupies a relatively small subset of image space
	- __Quantity/Numerous__: enough points represented in the image
	- __Efficiency__: reasonable time to compute and generate
- Open Source Algorithms:
	- Harris {corners}: Easy to compute, but not scale invariant
	- Harris-Laplace {corners}: Same procedure as Harris detector, addition of scale selection based on Laplacian. Scale invariance
	- Features from Accelerated Segment Test (FAST) {corners}: Machine Learning approach for fast corner detection.
	- Laplacian of Gaussian (LOG) detector {blobs}: Uses the concept of scale space in a large neighborhood (blob) somewhat scale invariant
	- Difference of Gaussian (DOG) detector {blobs}: Approximates LOG but is faster to compute
	- When a feature detector is __scale-invariant__, it means that the detector can detect the same visual features or keypoints in an image, regardless of their size or scale. In other words, the detector is capable of finding features at different levels of detail or different zoom levels in the image. In the context of image processing and computer vision, images can have objects or structures that vary in size due to factors like distance, perspective, or object size itself. A scale-invariant feature detector is essential in scenarios where objects can appear at different scales. For example, consider an image of a car on a road. If the car is far away, it will appear smaller in the image compared to when it's closer. A scale-invariant feature detector can detect the same distinctive keypoints on the car (e.g., corners, edges) at both the far and close distances, regardless of the car's size in the image.
### Feature Extraction
- Repetive textureless patches are challenging to detect consistently
- Patches with large contrast changes (gradients) are easier to detect (edges)
- Gradients in at least two (significantly) different orientations are the easiest to detect (corners)

### Feature Descriptors
- __Feature__: Point of interest in an image defined by its pixel coordinates $[u,v]$
- __Descriptor__: An __N-Dimensional__ vector that provides a __summary__ of the image information around the detected __feature__
- Feature Descriptors should be:
	- __Repeatability__: manifested as __robustness and invariance__ to translation, scale, rotation, and illumination changes
	- __Distinctiveness__: should allow us to distinguish between two close by features, very important for matching later on
	- __Compactness & Efficiency__: Compact and efficient to compute
- Designing Invariant Descriptors: __SIFT (Scale Invariance Feature Transform) Descriptors__:
	- Given a feature in an image, SIFT takes a 16x16 window of pixels around detected feature. This window is known as the feature's local neighbourhood.
	- Separate window into 4, 4x4 cells, each comprised of 4x4 patch of pixels
	- Compute edges and edge orientation of each pixel in each cell using the gradient filters we discussed in module one. For stability of the descriptor, we suppress weak edges using a predefined threshold as they are likely to vary significantly in orientation with small amounts of noise between images.
	- Construct 32 dimensional histogram of orientations for each cell, then concatenate to get 128 dimensional __descriptor__. Note that each dimension is a specific gradient orientation in the local neighborhood of each respective cell. 
- Note there are a mass variety of other feature detection and descriptor algorithms. Specifically for feature descriptors, some other popular ones are: SURF, GLOH, BRIEF, ORB
### Feature Matching
- General Image Feature Pipeline for Perception:
	- Identify Image Features, distinctive points in our images
	- Associate a descriptor for each feature from its neighborhood.
	- Use descriptors to match features across two or more images
- Baseline feature matching problem: Given a feature and it's descriptor in image one, we want to try to find the best match for the feature in image two.
- __Brute Force Feature Matching__: Define a distance function $d(f_i, f_j)$ that compares the two descriptors
	- For every feature descriptor $f_i$ in image 1:
		- Compute $d(f_i, f_j)$ with all features $f_j$ in image 2
		- Find the __closest__ match $f_c$, the match that has the minimum distance. But set a delta threshold such that any closest match feature with distance beyond threshold is not considered.
	- Distance functions for $d(f_i, f_j)$ include 
		- Sum of Squared Differences (SSD), which computes the squared error between each dimension/element of the feature vectors. Makes it sensitive to large variations in the descriptors, but insensitive to small ones.
		- Sum of absolute Differences (SAD), which computes the absolute difference between each dimension/element of the feature vectors. Penalizes all variation equally.
	- BF matching is suitable when the number of features we want to match is reasonable, but has quadratic time complexity, making it very non-scalable for large number of features.
- __Handling Ambiguity in Matching__: What should we do if we find two matching features both with the same minimum distance below threshold?
	- Modifying the BF Feature Matching Algorithm, after we find the closest match $f_c$, we also compute the second closest match $f_s$.
	- Find how better the closest match is than the second closest match using a distance ratio $0\le \frac{d(f_i,f_c)}{d(f_i,f_s)} \le1$
	- Instead of defining distance threshold, we define instead distance ratio threshold $\rho$ (usually set soemwhere around 0.5), and our new __algorithm__ becomes:
	- For every feature $f_i$ in Image 1:
		- Compute $d(f_i, f_j)$ with all features f_j in image 2
		- Find the closest match $f_c$ and the second closest match $f_s$
		- Compute distance ratio $\frac{d(f_i,f_c)}{d(f_i,f_s)}$
		- Keep matches with distance ratio $\lt \rho$
	- If the distance ratio is close to one, it means that according to our descriptor and distance function, fi matches both fs and fc. In this case, we don't want to use this match in our processing later on, as it clearly is not known to our matcher which location in image two corresponds to the feature in image one.
	- Setting $\rho$ typically to be around 0.5 means we require our best match to be at least twice as close as our second best match to our initial features descriptor.
### Outlier Rejection: Image Features: Localization Problem & RANSAC
- Given any two images of the same scene from different perspectives, find the translation $T = [t_u, t_v]$, between the coordinate system of the first image shown in red, and the coordinate system of ,the second image shown in green. In practice, we'd also want to solve for the scale and skew due to different viewpoints.
- Begin by computing features and their descriptors in image one and image two. We then match these features using the BF matcher previously developed.
- For every matched feature pairs in images 1 and 2:
	- $f^{(1)}_i, f^{(2)}_i | i \in [0,...,N]$, note we are using features vectors that contain the pixel location of the feature $f^{(1)}_i = (u^{(1)}_i, v^{(2)}_i)$ 
- Our math model becomes:
	- $u^{(1)}_i + t_u = u^{(2)}_i$, $v^{(1)}_i + t_v = v^{(2)}_i$
- To find the translation  $T = [t_u, t_v]$, we can solve using least squares:
	- $t_u = (1/N)*\sqrt{\sum_i(u^{(1)}_i-u^{(2)}_i)^2}$
	- $t_v = (1/N)*\sqrt{\sum_i(v^{(1)}_i-v^{(2)}_i)^2}$
- For some feature pairs after matching, some feature pairs may be invalid/incorrect as the two features that are paired not are not the same feature even if we use distance ratio BF method. These incorrect feature pairs are called __outliers__
- Outliers can be handled using amodel-based outlier rejection method called __Random Sample Consensus (RANSAC)__ algorithm, which is one of the most used model-based methods for outlier rejection in robotics:
	- __Initialization__: Given a __model__, find the smallest number of samples , __M__ from which the model can be computed
	- __Main Loop__:
		- From your data, randomly select __M__ samples
		- Compute model parameters using the seleccted __M__ samples
		- Check how many samples from the rest of your data actually fits the model. We call this number the number of __inliers C__
		- If __C > inlier ratio threshold or maximum iterations reached__, terminate and return the best inlier set. Else, go back to step 2 (first step of main loop)
	- __Final Step__: Recompute model parameters from entire best inlier set
- In context of feature mapping/outlier rejection, our model is the translation parameters $T=[t_u, t_v]$, __M__ = 1, and dataset samples are our feature pairs. For a given translation $T$ computed from one feature pair, our inliers are computed by applying the SAME translation to all of our other samples and seeing which ones also correctly translate between feature pairs $f^{(1)}_i$ to $f^{(2)}_i$.
- This way, computing our parameters from our inlier set eliminates the outliers as the outliers will not be included in our inlier set. 
### Visual Odometry (VO)
- VO is the process of incrementally estimating the pose of the vehicle by examining the changes that motion induces on the images of its onboard cameras
- VO Benefits:
	- Not affected by wheel slip in uneven terrain, rainy/snowy weather, or other adverse conditions
	- More accurate trajectory estimates compared to wheel odometry because of larger quantity of information from an image
- VO Cons:
	- Usually need an external sensor to estimate __absolute scale__.  What this means is that estimation of motion produced from one camera can be stretched or compressed in the 3D world without affecting the pixel feature locations by adjusting the relative motion estimate between the two images. As a result, we need at least one additional sensor, often a second camera or an inertial measurement unit, to be able to provide accurately scaled trajectories when using VO.
	- Camera is a __passive sensor__, might not be very robust against weather conditions and extreme illumination changes (night, headlights, streetlights)
	- As with any form of odometry (incremental state estimation), it will drift over time es estimation errors accumulate. As a result, we often quote VO performance as a percentage error per unit distance travelled.
- Mathematical Formulation of VO Problem:
	- Estimate the camera motion $T_k$ between consecutive images $I_{k-1}$ and $I_k$: $T_k = \begin{bmatrix}  
R_{k,k-1} & t_{k,k-1} \\  
0 & 1 
\end{bmatrix}$ defined by translation vector $t_{k,k-1}$ and rotation matrix $R_{k,k-1}$
	- Concatenating these single movements allows the recovery of the full trajectory of the camera, given frames $C_1,......,C_m$
- __Complete VO Algorithm__:
	- __Given__: two consecutive image frames $\bold I_{k-1}$ and $\bold I_k$
	- __Apply feature detection -> feature description -> and then match features into feature pairs__ $\bold f_{k-1}$ and $\bold f_{k}$ between the two consecutive image frames
	- Perform __Motion Estimation Step__ to get $\bold T_k$
- The motion estimation method also depends on what type of feature representation we have/Feature Correspondence types:
	- __2D-2D__: both $\bold f_{k-1}$  and $\bold f_{k}$ are defined in Image coordinates. 
	- __3D-3D__: both $\bold f_{k-1}$  and $\bold f_{k}$ are defined in 3D. Requires ability to locate new image features in 3D space, therefore used with depth cameras, stereo cameras, which can provide depth/3D information
	- __3D-2D__: $\bold f_{k-1}$ is specified in 3D world coordinates, and $\bold f_{k}$ are their corresponding projection in 2D image coordinates
- __3D-2D__ Motion estimation
	- __Given__:
		- 3D world coordinates of features in frame k-1
		- 2D image coordinates of same features in new frame k
		- Note that since we cannot recover the scale for a monocular visual odometry directly, we include a scaling parameter S when forming the homogeneous feature vector from the image coordinates.
	- We can use the same camera projection equations to solve this problem where: $[su,sv,s]^T = K[R|t][X,Y,Z,1]^T$
		- A simplifying distinction to note between calibration and VO is that the camera intrinsic calibration matrix k is already __known__
		- Problem reduces to estimating transformation components $R$ and $t$ from the system of equations constructed using all of our matched features.
	- We will solve for $R$ and $t$ using the popular __Perspective-n-Point Algorithm (PnP)__ 
		- Given feature locations in 3D, their corresponding projection in 2D, and the camera intrinsic calibration matrix k, PnP solves for the extrinsic transformations by:
			- Solve for initial guess of $[R|t]$
 using __Direct Linear Transform (DLT)__ , which forms a linear model and solves for $[R|t]$, with methods such as SVD
			 - However, the equations we have are nonlinear in the parameters of R and t, so we have to __refine our initial DLT solution__ with an iterative nonlinear optimization technique such as the __Levenberg-Marquardt (LM) Method__
			 - The PnP algorithm requires at least three features to solve for R and t. Need at __least 3 points to solve (P3P)__, 4 if we don't want ambiguous solutions.
			 - Finally, RANSAC can be incorporated into PnP by assuming that the transformation generated by PnP on four points is our model. We then choose a subset of all feature matches to evaluate this model and check the percentage of inliers that result to confirm the validity of the point matches selected.
			 - The PnP method only uses only a subset of the available matches to compute the solution. We can improve on PnP by applying the batch estimation techniques you studied in course two. By doing so, we can also incorporate additional measurements from other onboard sensors and incorporate vision into the state estimation pipeline.
		 - OpenCV has a robust implementation of the PnP method:
			 - `cv2.solvePnP()`: Solves for camera position given 3D points in frame k-1, their projection in frame k, and the camera intrinsic calibration matrix
			 - `cv2.solvePnPRansac()`: Same as above, but uses RANSAC to handle outliers
# Feedforward NNs and CNNs
- Functions to Estimate with FNNs:
	- __Object Classification__: Image -> Label
	- __Object Detection__: Image -> Label + Location
	- __Depth Estimation__: Image -> Depth for every pixel
	- __Semantic Segmentation__: Image -> Label for every pixel
- Review of Mathematical Formulation  of Hidden Units:
	- $h_n = g(W^Th_{n-1} + b)$
	- Where $g()$ is the nonlinear activation function, W is the weight matrix, $h_{n-1}$ is the input received from the previous layer (output of the previous layer), linear bias b, 
- __Classification__: Given Input x map it to one of k classes or categories (image classification, semantic segmentation)
- __Regression__: Given input x map it to a real number (Depth prediction, bounding box estimation)
- __Softmax Output layers__ are the most often used as the output of a classifier, to represent the  __probability distribution__ over K different classes
	- Given the output of last hidden layer $z = W^Th+b$
	- Compute the softmax probability distribution as : $\text{softmax}(z_i) = \frac{\text{exp}(z_i)}{\sum_j \text{exp}(z_j)}$
	- Note that the sum of all probability values across each class after using the softmax will sum to 1 as it computes a probability distribution of how likely the input image belongs to each class across all the classes.
	- The standard __loss function__ to be used with the softmax output layer is the __cross-entropy loss__, which is formed by taking the negative log of the softmax function
		- $L(\theta) = -log(softmax(z_i)) = -z_i + log(\sum_jexp(z_j))$
		- When minimizing this loss function, the negative of the class logit z_i encourages the network to output a large value for the probability of the correct class. The second term on the other hand, encourages the output of the affine transformation to be small. In other words, it will heavily penalize erroneous/large outputs for incorrect classes logits regardless of the logit for the correct class, teaching the network to make very discriminative decisions where it should only be confident in one class. __The two terms together encourages the network to minimize the difference between the predicted class probabilities and the true class probability.__
- The __linear output layer__ is mostly used for regression tasks to model statistics of common probability distributions. The linear output layer is simply comprised of a single affine transformation without any non-linearity.
	- The loss function to be used with linear output layers is generally Mean Squared error (MSE).
### Neural Network Training with Gradient Descent
- Given thousands of training example pairs $[x,f^*(x)]$, the loss function computed over all N training examples is termed the training loss and can be written as the mean of the gradient of the individual losses over every training example:
	- $J(\theta) = (1/N)\sum^N_{i=1}L[f(x_i,\theta),f^*(x_i)]$
- Batch gradient descent is an __iterative first order__ optimization procedure,
	-  __iterative__ means that it starts from an initial guess of parameters theta and improves on these parameters iteratively
	- __First order__ means that the algorithm only uses the first order derivative to improve the parameters theta
- __Batch Gradient Descent Algorithm__:
	- Initialize Parameters $\bold \theta$
	- While __Stopping condition__ is ___Not met__:
		- Compute gradient of loss function $\nabla_\theta J(\theta)$ over __all__ training examples
		- Update parameters according to: $\theta \leftarrow \theta - \epsilon \nabla_\theta J(\theta)$ with some defined learning rate $\epsilon$
- Parameter Initialization:
	- __Weights__: Generally initialized by randomly sampling from a standard normal distribution, although other more advanced actively researched methods exist 
	- __Biases__: Generally initialized to 0,  although other more advanced actively researched methods exist 
- Stopping Conditions:
	- Number of training epochs/iterations
	- Change in $\theta$ weight value thresholding
	- Change in training loss $J(\theta)$ thresholding
- Batch Gradient Descent algorithm is generally __NEVER__ used in practise with large datasets as it is extremely expensive/slow as the backpropagation will have to be computed over the entire training set, which is NOT ideal.
- To fix this issue, in practise we use __Stochastic (minibatch) Gradient Descent__, where instead of computing gradient of loss function over ALL training samples, we __sample a present number of examples known as the minibatch size from the training data__ instead.
- Note that many variants of SGD exist, but generally Adam is the most popular choice due to its robustness to choice of learning rate and other hyperparams
	- To pick a minibatch size, we want to pick values that are __powers of 2__ to match GPU computing architecture and use resources more efficiently
	- Large batch sizes > 256:
		- Hardware underutilized with very small batch sizes
		- More accurate estimate of the gradient, but with less than linear returns in accuracy improvement
	- Small batch size < 64:
		- Small batches can offer a __regularizing effect__. The best generalization error is often achieved with batch size of 1.
		- Smaller batch sizes allow for faster convergence, as the algorithm can compute the parameter updates rapidly
- Always make sure to __shuffle__ the dataset before sampling minibatch
### Data Splits and Neural Network Performance Evaluation
- Split the data into Training, Validation, and Test sets
	- Training set is used for training the model
	- Validation set is to observe initial generalization performance, and perform hyperparameter tuning for best results
	- Test set is untouched until hyper-parameters are tuned and it is time to simulate performance in deployment for pure generalization. We do not do testing on the validation set because the results would be biased as we are tuning the hyperparameters to fit the validation data set. Our ultimate goal should be to get the best testing performance as possible
- For small datasets, 60/20/20 split is ok, but for millions of data points, we can do a 98/1/1 split or 90/5/5
- Tackling Underfitting (Training Loss is high):
	- Train longer
	- More layers or more parameters per layer (adding model complexity)
	- Change architecture if above doesnt help
- Tackling Overfitting (Generalization gap between training and validation loss is high)
	- Collect more training data
	- __Regularization__ (parameter norm penalities, dropout, early stopping)	
		- Parameter norm penalities are applied to the loss function to penalize weight (not biases) magnitudes: $J(\theta)_{reg} = J(\theta) + \alpha \Omega(W)$ where $\alpha$ is a hp that weights the relative contribution of the norm penalty to the value of the loss function and $\Omega()$ is usually a Lp Norm where p is usually 2: $\Omega(W) = \frac{1}{2}||W||^2_2=\frac{1}{2}W^TW$
		- __Dropout__ is a regularization specific to neural networks and is inexpensive, effective, and non-limiting way to regularize neural networks. Given a keep probability $P_{keep}$,  At every training iteration, this probability is used to choose a subset of the network nodes to keep in the network. These nodes can be either hidden units, output units, or input units. 
			- We then proceed to evaluate the output y after cutting all the connections coming out of this unit. Since we are removing units proportional to the keep probably,$P_{keep}$ , we multiply the final weights by $P_{keep}$ at the ending of training. This is essential to avoid incorrectly scaling the outputs when we switch to inference for the full network.
			- Dropout can be intuitively explained as forcing the model to learn with missing input and hidden units. Or in other words, with different versions of itself.
		- __Early Stopping__ can be used to prevent overfitting by ending training when the validation loss begins to increase instead of decrease for a present number of iterations/epochs. Lowest priority method to combat overfitting as it limits training time.
	- Change Architecture
## Convolutional Neural Networks (CNNs)
- Used for processing data defined on a grid
	- 1D Time series data, 2D images, 3D videos
- Two major types of Layers: __Convolution Layers__, __pooling layers__
- Although counter-intuitive, convolutional layers use __cross-correlation__ (but with the filter flipped both vertically and horizontally) not convolutions for their linear operator instead of general matrix multiplication. The logic behind using cross-correlation is that if the parameters are learned, it does not matter if we flip the output or not. Since we are learning the weights of the convolutional layer, the flipping does not affect our results at all. This results in what we call sparse connectivity
- Each input element to the convolutional layer only affects a few output elements, thanks to the use of a limited size kernel for the convolutional operation.
- We perform the convolution operations through a set of kernels or filters. Each Filter is comprised of a set of weights and a single bias. The number of channels of the kernel needs to correspond to the number of channels of the input volume. 
- __Our output channel size for the conv layer corresponds with the number of filters we use as each kernel will produce one channel of output for the current conv layer__.This is different than the channel of the filter/kernel itself as each filter/kernel must correpond to the channel number of the previous convolution or input layer.
- Assuming we have filter size __mxm__, __K__ filters, Stride __S__, and padding __P__, we can calculate our output volume shape as:
	- __Width__: $W_{out} = \frac{W_{in} -m+2\times p}{S}+1$
	- __Height__: $H_{out} = \frac{H_{in} -m+2\times p}{S}+1$
	- __Depth__: $D_{out} = K$
- __A pooling layer__ uses pooling functions to replace the output of the previous layer with a summary statistic of the nearby outputs. Pooling helps make the representations become invariant to small translations of the input. If we translate the input a small amount, the output of the pooling layer will not change. This is important for object recognition for example, as if we shift a car a small amount in an image space, it should still be recognizable as a car.
	- Most notably, __max pooling__ is used, it summarizes output convolution volume patches with the max function by replacing an area of the output volume by the highest value in that area in the pooling output layer.
		- For pool size __nxn__ with stride __S__, we can calculate output pooling layer size as:
			- 	__Width__: $W_{out} = \frac{W_{in} -n}{S}+1$
			- __Height__: $H_{out} = \frac{H_{in} -n}{S}+1$
			- __Depth__: $D_{out} = D_{in}$
- Conv layers have __less parameters__ than fully connected layers, reducing the chances of overfitting and being much more computationally efficient to process dense image information than FCNNs
- __More importantly__, CNNs are far superior for processing images than FCNNs because of __translation invariance__ property. Conv layers use the same parameters to process every block of the image, along with pooling layers, this leads to __translation invariance__, which is very important for image understanding. ConvNets are capable of detecting an object or classifying a pixel even if it is shifted with a translation on the image plane. This means we can detect cars wherever they appear.
# 2D Object Detection
-  __The Object Detection Problem__
	- Given a 2D image's input, we are required to estimate the location defined by a bounding box and the class of all objects in the scene. Usually, we choose classes that are relevant to our application. For self-driving cars, we usually are most interested in object classes that are dynamic, that is ones that move through the scene.
-  __Challenges__
	- Extent of objects is not fully observed
		- **Occlusion**: Background objects covered by foreground objects
		- **Truncation**: Objects are out of image boundaries
	- **Scale**: Object sizes get smaller as the object moves farther away
	- Illumination Changes
		- Too Bright
		- Too Dark
- __Mathematical Problem Formulation__:
	- Given an input image x, we want to find the function $f(x;\theta)$ that produces an output vector that includes:
		- Coordinates of the top left of the box __x_min, y_min__
		- Coordinates of the lower right corner of the box: __x_max, y_max__
		- And class scores __S_class1 to S_classk__ where S_classi specifies how confident our algorithm is that the object belongs to the class i, where i ranges from 1 to k, where k is the number of classes of interest
- __Evaluation Metrics__
	- Given the output of a 2D object detector, we want to compare how well it fits the true output the ground truth bounding box
	- First, compare our detector localization output to the ground truth boxes via the __Intersection-Over-Union (IOU)__ metric. IOU is defined as the area of the intersection of two polygons divided by the area of their union
	- To account for class scores during evaluation, we define:
		- __True Positive (TP)__: Object class score > score threshold AND IOU > IOU threshold
		- __False Positives (FP)__: Object class score > score threshold AND IOU < IOU threshold
			- This can be easily computed as the total number of detected objects after the application of the score threshold minus the number of true positives.
		- __False Negatives (FN)__: Number of ground truth objects not detected by the algorithm
		- __Precision__: TP/(TP+FP)
		- __Recall__: TP/(TP+FN)
		- __Precision Recall Curve (PR-Curve)__: Use multiple object class score thresholds to compute precision and recall. Plot the values with precision on y-axis, and recall on x-axis
		- __Average Precision (AP)__: Area under PR-Curve for a single class. Usually approximated using 11 recall points. The value of the average precision of the detector can be thought of as an average of performance over all score thresholds allowing objective comparison of the performance of detectors without having to consider the exact score threshold that generated those detections.
		- For a detector with precision of 1 and recall of 0.5, The detector in this case is a high precision low recall detector. This means that the detector misses some objects in the scene, but when it does detect an object, it makes very few mistakes in, category classification and bounding box location.
### 2D Object Detection with CNNs

- The CNN 2D Object Detection Pipeline involves:
	- Image -> Feature Extractor + Prior Boxes -> Output Layers => Non-Maximal Suppression -> Output detections
	- __Feature extractors__ are the most computationally expensive component of the 2D object detector because this is where we use SOTA CNN architectures for feature extraction like VGG, ResNet...etc.
	- To generate 2D bounding boxes, we usually do __not__ start from scratch and estimate the corners of the bounding box without any prior. We assume that we do have a prior on where the boxes are in image space and how large these boxes should be. These priors are called __anchor boxes__ and are manually defined over the whole image usually on an equally-spaced grid
		- Let's assume that we have a set of anchors close to our ground-truth boxes. During training, the network learns to take each of these anchors and tries to move it as close as possible to the ground truth bounding box in both the centroid location and box dimensions. This is termed __residual learning__ and it takes advantage of the notion that it is easier to nudge and existing box a small amount to improve it rather than to search the entire image for possible object locations. 
		- In practice, residual learning has proven to provide much better results than attempting to directly estimate bounding boxes without any prior.
	- After we have a MxNxD feature map/tensor from the CNN, we can generate anchor boxes using various methods such as the Faster R-CNN interaction method explained in the video. Methodology is omitted in these notes. But the idea is that the network will output a __regressed/predicted vector of residuals__ that need to be added to the anchor at hand to get the ground truth bounding box (since we are not directly estimating box location and size) and a __softmax classification layer to estimate score per object class.__
- How to handle multiple detections per object and formulate a loss function since our network will output multiple bounding boxes for each object prediction while we only have one ground truth bounding box for each object/class
	- during training through __minibatch selection__:
		- Given a set of anchor bounding boxes in a grid with a certain stride based on the feature extractor resolution reduction (e.g. factor of 32 will place anchor box centers with stride 32), we __categorize anchor boxes into two categories based on their IOU with ground truth bounding boxes.__ 
			- If IOU > positive member threshold, set anchor as positive anchor
			- if IOU < negative threshold, set anchor as negative.
			- Any anchor with IOU in between the positive and negative thresholds are discarded
		- Negative Anchors Prediction target:
			- __Classification__: Background class (Background is usually a class we add to our classes of interest to describe anything non-included in these classes.)
		- Positive Anchors target:
			- __Classification__: Category of the ground truth bounding box (we want the neural network to assign ground truth class to any positive anchor intersecting that ground truth.)
			- __Regression__: Align box parameters with highest IOU ground truth bounding box. (For regression, we want to shift the parameters of the positive anchor to be aligned with those of the ground truth bounding box. The negative anchors are not used in bounding box regression as they are assumed to be background.)
		- __Problem__: Majority of anchors are negatives results in NN will label all detections as background
		- __Solution__: instead of using all anchors to compute the loss function, we sample the chosen minibatch size with a three to one ratio of negative to positive anchors. The negatives are chosen through a process called online hard negative mining, in which negative minibatch members are chosen as the negative anchors with the __highest classification loss.__ This  means that we are training to fix the biggest errors in negative classification
			- E.g. minibatch size 64 -> 48 negative anchors, 16 positive anchors
		- __Classification Loss__: The total classification loss is the average of the cross entropy loss of all anchors in the minibatch.
		- For __regression loss__, we use the L2 norm loss in a similar manner. However, __we only attempt to modify an anchor if it is a positive anchor__. This is expressed mathematically with a multiplier Pi on the L2 norm. It is 0 if the anchor is negative and 1 if the anchor is positive.
			- Remember that we don't directly estimate box parameters, but rather, we modify the anchor parameters by an additive residual or a multiplicative scale. So bi must be constructed from the estimated residuals.
	- during inference, we use the __non-maximum suppression (NMS)__ algorithm to ensure that we only output a single bounding box per object. The NMS algorithm is omitted here but can be easily found in the lecture slides, video or online.
### Using 2D Object Detectors for Self-Driving Cars
- Self-driving cars require scene understanding in 3D to be able to safely traverse their environment. Knowing where pedestrians, cars, lanes, and signs are around the car entirely defines what actions can be taken safely when autonomous. This means that detecting objects in the image plane is not enough. We need to __extend the problem from 2D to 3D, and locate the detected objects in the world frame.__
- For a given 3D object detected, we want to estimate:
	- Category classification of the object
	- Position of the centroid in 3D [x,y,z]
	- Extent/dimensions of the object in 3D [l,w,h] (length, width, height of object)
	- Orientation of object (since our car can be safely assumed to be always on the road plane, we only care about the yaw angle of the car $\theta$ from the roll, pitch, and yaw angles expressed as $[\phi, \psi, \theta]$)
- To get from 2D bounding box to 3D estimation of location and extent of object, the most common and successful way is to use LiDAR point clouds.
	- Given a 2D bounding box in an image space and a 3D LiDAR point cloud, we can use the inverse of the camera projection matrix to project the corners of the bounding box as rays into the 3D space. The polygon intersection of these lines is called a __frustum__, and usually contains points in 3D that correspond to the object in our image. 
	- We then take the data in this frustum from the LiDAR, transform it to a representation of our choice, and train a small neural network to predict the seven parameters ($[x,y,z,l,w,h,\theta]$) required to define our bounding box in 3D.
	- LiDAR Point Representations to use can include:
		- Raw point cloud data
		- Normalize point cloud data with respect to some fixed point such as the center of the frustum
		- One could also preprocess the points to build fixed length representations such as a histogram of x, y, z points, making their use as an input to CNN much more convenient
	- At the end, we are expected to get results in the form of oriented 3D bounding boxes. 
	- The above procedure is only one way of many of perfoming 2D->3D object detection
- __Advantages__ of 2D -> 3D Object Detection:
	- Allows exploitation of mature 2D object detectors, with high precision and recall
	- Class already determined from 2D detection
	- Does NOT require prior scene knowledge, such as groud plane location
- __Disadvantages__ of 2D -> 3D Object Detection:
	- The performance of the 3D estimator is bounded by the performance of the 2D detector
	- Occlusion and truncation are hard to handle from 2D only
	- 3D estimator needs to wait for 2D detector, inducing __latency__ in our system
- __Object Tracking__ is another important application of 2D to 3D object detection.
	- Assumptions made in object tracking include:
		- Camera is not moving instantly to new viewpoint
		- Objects do not disappear and reappaear in different places in the scene
		- If the camera is moving, there is a gradual change in pose between camera and scene
	- We use a kalman filter like process to perform object tracking
	- Given a detected object in the first frame along with their velocity vectors, we begin by predicting where the objects will end up in the second frame if we model their motion using the velocity vector.
	- We then get new detections in the second frame using our 2D object detector, which we call our measurements, we correlate each detection with a corresponding measurement, and then update our object prediction using the correlated measurement and kalman filter framework.
	- We initiate new tracks if we get 2D detector results not correlated to any previous prediction.
	- Similarly, we terminate inconsistent tracks, if a predicted object does not correlate with a measurement for a preset number of frames.
	- Finally, we need to note that by defining IOU in 3D, we can use the same methodology for 3D object tracking.
- The described 2D Object Detection framework can be similarly used to detect and determine state of traffic signs and signals which is obviously very important for driving on roads
# Semantic Segmentation
- Given an input image, we want to classify each pixel into a set of preset categories. The categories can be static road elements such as roads, sidewalk, pole, traffic lights, and traffic signs or dynamic obstacles such as cars, pedestrians, and cyclists. Also, we always have a background class that encompasses any category we do not include in our preset categories.
- Given an image, we take every pixel as an input and output a vector of class scores per pixel. A pixel belongs to the class with the highest class score.
- Semantic segmentation also has a major problem specific difficulty. This difficulty is caused by an __ambiguity of boundaries in image space__, especially for thin objects such as poles, similar looking objects such as a road and a sidewalk and far away objects
- __Evaluation Metrics__:
	- __True Positives (TP)__: The number of correctly classified pixels belonging to class X
	- __False Positive (FP)__: Number of pixels that do not belong to class X in ground truth but are classified that class by the algorithm
	- __False Negative (FN)__: The number of pixels that do belong to class X in ground truth, but are not classified as that class by the algorithm
	- __IOU_class__: TP / (TP+FP+FN) Class IOU over all the data is calculated by __computing the sum of TP, FP, FN for ALL images first__. Computing the IOU per image and then averaging will actually give you an __incorrect__ class IOU score
		- Averaging the class IOU is usually not a very good idea because a global IOU measure (over all classes) is biased toward object incidences that cover a large image area

### ConvNets for Semantic Segmentation
- A basic convnet model for semantic segmentation involves:
	1. Given an input image, pass it through a standard CNN to obtain an expressive, deep, but low resolution feature map/tensor as output.
	2. Then we use a upsampling decoder CNN to take the feature map as input and upsamples it to get a final feature map of equal resolution to the original input image with less depth
	3. Finally, a linear layer followed by a softmax function generates the class ConvNets one-hot vector for each pixel in the input image.
- A cross entropy classification loss across all classified pixels in all the images of a mini-batch is used. For each pixel, the cross entropy loss would take as input the true one-hot encoded vector of the correct class, and the predicted softmax one-hot output vector for that pixel.
### Semantic Segmentation for Road Scene Understanding 
- __3D Drivable Surface Estimation__:
	1. Generate semantic segmentation output
	2. Associate 3D point coordinates with 2D image pixels by converting pixels to 3D points in the camera frame given the depth map of the image and camera intrinsic matrix K using equations: 
		- $z = depth$
		- $x = \frac{(u - u_c) * z}{f}$
		- $y = \frac{(v - u_v) * z}{f}$

		- Here, $c_u$, $c_v$, and $f$ are the intrinsic calibration parameters found in the camera calibration matrix K such that:

$$K = \begin{pmatrix} f & 0 & u_c \\ 0 & f & u_v \\ 0& 0 & 1 \end{pmatrix}$$
	3. Choose 3D points belonging to the __driveable surface category__ using semantic segmentation map of the image and correlating which pixels/3D points belong to road class. In implementation, this is done using a boolean mask from the semantic segmentation map
	4. Estimate 3D drivable surface model using the 3D points belonging to the __driveable surface category__
		- This is done using the plane model of $ax + by + z = d$, and using a LS formulation to solve for the parameters a, b, d using measurements/points x, y and z
		- Minimum number of points to estimate the model: 3 non-collinear points
		- We can use the __RANSAC algorithm__ to robustly fit a drivable surface plane even with semantic segmentation errors that provide 3D points that are not actually on the drivable surface plane
	- RANSAC Algorithm:
		1. From data, randomly select 3 points
		2. Compute model parameters a, b, and d using LS estimation
		3. Compute number of inliers, N. where inliers are the 3D points that satisfy these model parameters as most of the outliers are a result of the erroneous segmentation ouputs located at the boundaries
		4. If N>threshold of satisfaction, terminate and return the computed plane parameters. Else go back to step 1.
		5. Recompute the model parameter using all the inliers in the largest inlier set
- __Semantic Lane Estimation__:
	1. Given the output of a semantic segmentation model, we first extract a binary segmentation mask from pixels belonging to lane separators such as lane __markings__ or __curbs__
	2. Extract edges from this segmentation mask using an __edge detector__ such as Canny. The output are pixels classified as edges that will be used to estimate the lane boundaries.
	3. The final step is to determine which model is to be used to estimate the lanes. Here we choose a linear lane model, where each lane is assumed to be made up of a single line. To detect lines in the output edge map, we need a line detector. The Hough transform line detection algorithm is widely used and capable of detecting multiple lines in an edge map.
		- Given an edge map, the Hough transform can generate a set of lines that link pixels belonging to edges in the edge map. Then hyperparameter tuning can be done to eliminate lines that we know cannot be useful to lane detection, e.g. lines that are not long enough, horizontal lines if camera is placed forward facing direction of motion, lines that does not belong to the drivable surface
	4. The last step would be to determine the classes that occur at both sides of the lane, which can easily be done by considering the classes on both sides of the estimated lane lines.
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTExMTEwMjgwMTEsMTQ3OTAyODIxMSwtNj
Y3NzAxMDkxLC0xMjEzNTM5NzIsLTIwNDY1NDU1NSwtMTQ4NTYw
NTA3OCwxMTk0MzQ3NSwtMTI1MDY2NzE5MSwtMjA2NDY1NTU0Ni
wtMTA3ODEwOTg3OSwxNjM5MjE4Njk1LDEzMjM3NDE0MDMsMTkz
ODI2Mjc3NSwxMDA5ODQ0ODM5LC0yMTIzNjc5NTE5LDE5MTA1MT
E5NTEsLTI2ODY3MTU2Miw4NTcyNTAzMTAsMTAwMzA2MjM4MCwt
MTkzOTM2NDA4Ml19
-->