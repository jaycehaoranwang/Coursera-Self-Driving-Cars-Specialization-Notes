# Squared Error Criterion and the Method of Least Squares
**Method of Least Squares**: The most probable value of an unknown parameter is that which minimizes the sum of squared errors between what we observe and what we expect. 

This means for a given set of measurements made to estimate an unknown variable, we create a loss equation that sums the difference between the measured value and the unknown value (the error) squared, then compute the minimum of the loss equation using calculus
### Linear Equally Weighted Method of Least Squares
- Assuming linear relationship and measurements are equally weighted/equally significant between the measurements $y$ and true value of an unknown $x$ with error term $v$, we can model the relationship as $y_i = x_i + v_i$ for each measurement $i$ made and $v_i$ is assumed to be i.i.d.
- The vectorized formulation can be written as:
	- $\bold e = \bold y - \bold H \bold x$ where $\bold H$ is the Jacobian matrix of size m x n where m is the number of measurements and n is the number of unknown variables.
	- Then we can write the loss as $\mathcal{L}(x) = \bold e^T\bold e$
	- Substituting, expanding, then minimizing the loss function wrt $x$, we get the analytical minimizing solution as : $\hat x_{LS} = (\bold H^T\bold H)^{-1}\bold H^T \bold y$
	- Note this solution only exists if the inverse exists, which means we must have equal or more measurements than unknown parameters (m >= n).

### Linear Weighted Least Squares
- We may want to trust some measurements more than others if they come from a more reliable/accurate/better sensor
- We may now assume that the noise term on each measurement is independent but of difference variance, so measurements with lower noise variance should be weights more strongly since we care more about measurements made from lower error measurements
- We can then define the weighted LS criterion as: $\mathcal{L}_{WLS}(x) = \bold e^T\bold R^{-1}\bold e$ where $\bold R$ is the square diagonal matrix where the diagonal elements correspond to the variance/standard deviation squared of each measurement. (remember variance  = std^2)
- Substituting, expanding, then minimizing the loss function wrt $x$, we get the analytical minimizing solution as : $\hat x_{WLS} = (\bold H^T\bold R^{-1}\bold H)^{-1}\bold H^T \bold R^{-1}\bold y$
- Again we assume a linear relationship where m >= n and variances are all > 0 

### Linear Recursive Least Squares
- Recursive LS allows us to compute a running estimate of the least squares solution as measurements stream in, instead of using a complete batch of measurements at once
- To do this, we use a **Linear Recursive Estimator** that produces a new estimate of the unknown parameter $x$ in the new timestep $k$ using the previous best estimate $\hat x_{k-1}$. This update can be written as $\bold{\hat x_k} = \bold{\hat x_{k-1}} + \bold K_k(\bold y_k - \bold H_k \bold{\hat x_{k-1}} )$ 
- The term in brackets is called the innovation and quantifies how well our current measurement matches our previous best estimate
- $\bold K_k$ is a gain matrix that can be computed using the estimator covariance matrix $\bold P_k$ as $\bold K_k = \bold P_{k-1}\bold H^T_k (\bold H_k \bold P_{k-1} \bold H^T_k + \bold R_k )^{-1}$
- We can also formulate an expression for $\bold P_k = (\bold 1 - \bold K_k \bold H_k)\bold P_{k-1}$ to show that the covariance shrinks with each measurement, meaning Intuitively, you can think of this gain matrix as balancing the information we get from our prior estimate and the information we receive from our new measurement
- The RLS algorithm can be sumamrized as:
	1. Initialize the estimator and covariance $\bold{\hat x}_0 = \mathbb{E}[\bold x]$,  $\bold{P}_0 = \mathbb{E}[(\bold x - \bold{\hat  x_0})(\bold x - \bold{\hat  x_0})^T]$
	2. Setup the measurement model, defining/picking values for the Jacobian and measurement covariance matrix: $\bold y_k  = \bold H_k \bold x + \bold v_k$
	3. Everytime a new measurement is recorded we update our estimates in order of:
		4. $\bold K_k = \bold P_{k-1}\bold H^T_k (\bold H_k \bold P_{k-1} \bold H^T_k + \bold R_k )^{-1}$
		5. $\bold{\hat x_k} = \bold{\hat x_{k-1}} + \bold K_k(\bold y_k - \bold H_k \bold{\hat x_{k-1}} )$ 
		6. $\bold P_k = (\bold 1 - \bold K_k \bold H_k)\bold P_{k-1}$

### LS and Maximum Likelihood
- Instead of writing down a loss, we can approach the problem of optimal parameter estimation by asking which parameters make our recorded measurements the most likely.
- By converting our measurements to gaussian probability density functions where the mean is the measurement and the variance is the variance of the measurement $p(y | x) = \mathcal{N}(y; x,\sigma^2)$ (note that for multiple independent measurements we can multiply all the different gaussians together) we can apply maximal likelihood estimate (MLE) to ask the value of our unknown parameter that maximizes the probability of generating the associated probability density function $\hat x_{MLE} = \text{argmax}_x p(\bold y | x)$
- By applying logs to simplify the expression and using the fact that finding the argmax is the same as finding the argmin of the negative of the function, we can derive that the maximum likelihood estimate given additive gaussian noise is **equivalent** to the LS or WLS solutions derived earlier. 
- This result is important because there are many independent sources of noise in a self driving system, by applying the **Central Limit Theorem** that tells us that when combining all of these errors together, they can reasonably be modeled by a single Gaussian error distribution. As a result, we are able to model our system probabilistically and yet maintain simplicity in calculations knowing if the errors are gaussian, we can compute the best maximum likelihood estimate of the parameters using standard LS or WLS solutions. 
- Note that due to the gaussian nature of LS, the solution is very sensitive to **outlier measurements** (since outliers are assumed to appear with less than 5% chance according to gaussian distribution) and will significantly skew the parameter estimate. 

# State Estimation -  Linear and Nonlinear Kalman Filters
**The Kalman filter** is very similar to the linear recursive least squares filter. While recursive least squares updates the estimate of a static parameter, the Kalman filter is able to update an estimate of an evolving state. The goal of the Kalman filter is to take a probabilistic estimate of this state and update it in real time using two steps; **prediction** and **correction**.
### Linear Kalman Filter
- Prediction is done via a motion model of the system 
- Correction is done using an observation model that produces measurements of the vehicle and is fused with the prediction to produce the final estimate at a given time step
- Note the initial state, predicted state, and final corrected state are all random gaussian variables (probability density functions) defined using means and variances to progress from $\mathcal{N}(\hat x_{k-1}, \hat P_{k-1})$ to the corrected updated estimate $\mathcal{N}(\hat x_{k}, \hat P_{k})$
- Linear Kalman Filter requires the following motion and measurement models:
	- Motion Model: $\bold x_k = \bold F_{k-1} \bold x_{k-1} + \bold G_{k-1} \bold u_{k-1} + \bold w_{k-1}$ ($\bold u_{k-1}$ is input and $\bold w_{k-1}$ is process or motion noise)
	- Linear Measurement Model = $\bold y_k = \bold H_k \bold x_k + \bold v_k$ ($\bold v_{k}$ is measurement noise)
	- With Noise properties $\bold v_k \sim \mathcal{N}(\bold 0, \bold R_k)$, $\bold w_k \sim \mathcal{N}(\bold 0, \bold Q_k)$
	- Applying a similar algorithm to RLS but including a motion model that tells how the state evolves over time, the **kalman filter algorithm** is:
		1. Prediction:
			-  $\bold {\check x_k} = \bold F_{k-1} \bold x_{k-1} + \bold G_{k-1} \bold u_{k-1}$
			-  $\bold {\check P_k} = \bold F_{k-1} \bold{\hat P_{k-1}} \bold F^T_{k-1} + \bold Q_{k-1}$
		2. Optimal Gain (optimally fusing our measurements):
			- $\bold K_k = \bold {\check P_{k}}\bold H^T_k (\bold H_k \bold {\check P_{k}} \bold H^T_k + \bold R_k )^{-1}$
		3. Correction:
			- $\bold{\hat x_k} = \bold{\check x_{k}} + \bold K_k(\bold y_k - \bold H_k \bold{\check x_{k}} )$ 
			- $\bold {\hat P_k} = (\bold 1 - \bold K_k \bold H_k)\bold {\check P_{k}}$
		- Note that $\bold {\check x_k}$ is the prediction (given motion model) at time k and $\bold {\hat x_k}$ is the corrected prediction (given measurement) at time k

### Kalman Filter and the Bias (Best Linear Unbiased Estimator) BLUEs
- We say an estimator or filter is **unbiased** if it produces an "average" error of zero at a particular time step k over many trials, note that the error is calculated assuming knowledge of truth information of the unknown variable
	- The difference between the mean of estimated position values (under a gaussian distribution) and the true position is known as the **bias**. This value should approach 0 for an unbiased filter
	- It can be analytically shown (in the slides) that the Linear Kalman filter is unbiased, that the expected value of the prediction and corrected error terms ($\bold{\check e_k}$, $\bold{\hat e_k}$) are both 0 so long as the inital state estimate is unbiased ($\mathbb{E} [\bold{\check e_0] = 0}$) and the noise is white, uncorrelated, and zero mean ($\mathbb{E} [\bold{v] = 0}$, $\mathbb{E} [\bold{w] = 0}$).
- The Kalman filter is also **Consistent**, meaning for all timesteps k, the filter covariance $\hat p_k$ matches the expected value of the square of our error $\sqrt(\mathbb{E}[\hat e^2_k])$
	- More formally: $\mathbb{E}[\hat e^2_k] = \mathbb{E}[(\hat p_k - p_k)^2] = \hat P_k$
	- For scalar parameters this means that the empirical variance of our estimate should match the variance reported by the filter.
	- Practically, this means that our filter is neither overconfident, nor underconfident in the estimate it has produced. Being inconsistent is bad or dangerous because it may place too much emphasis on its own estimate and will be less sensitive to future measurement updates, which may provide critical information
	- Again we assume that the initial estimate is consistent and the error is white, uncorrelated, and zero mean

### Nonlinear Extended Kalman Filter (EKF)
- The linear Kalman filter cannot be used directly to estimate states that are non-linear functions of either the measurements or the control inputs.
- To use the Kalman Filter on nonlinear systems, we need to linearize a nonlinear system by choosing an operating point a and approximate the nonlinear function by a tangent line at that point
	- Mathematically, we compute this linear approximation using a first-order Taylor expansion, meaning we only take the first order linear terms of the Taylor expansion
- Choose the most recent state estimate as the operating point of the Taylor expansion
	- This means around timestep k-1 for the motion model but around timestep **k** for the measurement model
- Most important part is to compute the Jacobian matrices of the first order derivatives of the noise and taylor difference term correctly
	- Recall the Jacobian matrix is the matrix of all first order partial derivatives of a vector-valued function
- Using the Jacobian matrices and first order taylor expansion terms, we can write a linearized motion and measurement model as (note taylor expansion about $\bold x_{k-1}$):
	- Linearized Motion Model:
		- $\bold x_k = \bold f_{k-1} (\bold{\hat x_{k-1}}, \bold u_{k-1}, \bold 0) + \bold F_{k-1} (\bold x_{k-1} - \bold{\hat x_{k-1}}) + \bold L_{k-1} \bold w_{k-1}$ 
	- Linearized measurement model:
		- $\bold y_k = \bold h_k (\bold{\check x_k}, \bold 0 ) + \bold H_k (\bold x_k - \bold{\check x_k}) + \bold M_k \bold v_k$
	- Prediction:
		- $\bold{\check x_k} =  \bold f_{k-1} (\bold{\hat x_{k-1}}, \bold u_{k-1}, \bold 0)$
		- $\bold{\check P_k} = \bold F_{k-1} \bold{\hat P_{k-1}}\bold F^T_{k-1} + \bold L_{k-1} \bold Q_{k-1} \bold L^T_{k-1}$
	- Optimal Gain:
		- $\bold K_k = \bold {\check P_{k}}\bold H^T_k (\bold H_k \bold {\check P_{k}} \bold H^T_k + \bold M_k \bold R_k \bold M^T_k)^{-1}$
	- Correction:
		-  $\bold{\hat x_k} = \bold{\check x_{k}} + \bold K_k(\bold y_k - \bold h_k (\bold{\check x_{k}}, \bold 0 ))$ NOTE that we are using the NON-LINEAR measurement model function $\bold h_k (\bold{\check x_{k}}, \bold 0 )$ to compute the measurement residual, NOT the linearized version
		- $\bold {\hat P_k} = (\bold 1 - \bold K_k \bold H_k)\bold {\check P_{k}}$
	- Note that $\bold F_{k-1}, \bold L_{k-1},\bold H_k,\bold M_k$ are the respective jacobian matrices of the motion and measurement models
	
### Improved EKF - The Error State EKF (ESEKF)
- The Error state formulation separates the state into a "large" nominal state and a "small" error state: $\bold x = \bold{\hat x} + \delta \bold x$ (true state = nominal "large" state + error state "small")
- ESEKF uses local linearization to estimate the error state and uses it to correct the nominal state, this means formulating EKF equations with error terms instead of estimated true terms: $\delta \bold x_k = \bold x_k -  \bold f_{k-1} (\bold{\hat x_{k-1}}, \bold u_{k-1}, \bold 0)$, $\delta \bold x_{k-1} = \bold x_{k-1} - \bold{\hat x_{k-1}}$ for motion model and $\delta \bold x_{k} = \bold x_{k} - \bold{\check x_{k}}$ for measurement model
- **ESEKF Loop**:
	- Update Nominal State and uncertainty with motion model repeatedly with the current best estimate (note this step does not require measurement for correction step)
		- $\bold {\check x_k} =\bold f_{k-1} (\bold{ x_{k-1}}, \bold u_{k-1}, \bold 0)$ Note $\bold{ x_{k-1}}$ could be $\check x$ or $\hat x$ 
	- Propagate Uncertainty
		- $\bold{\check P_k} = \bold F_{k-1} \bold{P_{k-1}}\bold F^T_{k-1} + \bold L_{k-1} \bold Q_{k-1} \bold L^T_{k-1}$ Again P could be predicted value or corrected value
	- If a measurement is available:
		- Compute Kalman Gain:  $\bold K_k = \bold {\check P_{k}}\bold H^T_k (\bold H_k \bold {\check P_{k}} \bold H^T_k + \bold R )^{-1}$
		- Compute best estimate of error state: $\delta \bold{\hat x_k} = \bold K_k (\bold y_k - \bold h_k (\bold{\check x_k, \bold 0}))$
		- Correct Nominal State: $\bold{\hat x_k} = \bold{\check x_k} + \delta \bold{\hat x_k}$
		- Correct State Covariance: $\bold{\hat P_k} = (\bold 1 - \bold K_k \bold H_k)\bold{\check P_k}$
- We want to use ES-EKF because:
	- Better performance compared to vanilla EKF
	- Easy to work with constrained quantities (e.g. rotations in 3D)

### Limitations of the EKF
- Since Nonlinear Kalman Filter depend on linearization of a nonlinear system about a point, EKF is very prone to large linearization errors for very nonlinear transformations that are not well represented by a linearization about certain point
- The EKF is prone to linearization error when
	- The system dynamics are highly nonlinear
	- The sensor sampling time is slow relative to how fast the system is evolving
- This has two important consequences:
	- The estimated mean state can become very different from the true state
	- The estimated state covariance can fail to capture the true uncertainty in the state
- **Linearization error can cause the estimator to be overconfident in a wrong answer, diverging errors, and cause inrecoverable state estimation divergence due to error**
- Obviously a huge problem for safety for self-driving cars
- Other Limitations/Challenges of using EKF in practise/implementation include computing jacobian matrices for complicated nonlinear functions
	- Analytical differentiation is prone to human error
	- Numerical differentiation can be slow and unstable
	- Automatic differentiation (at compile time) can also behave unpredictably
	- State models may NOT be differentiable 

### Alternative to EKF - Unscented Kalman Filter (UKF)
- Is an alternative approach to Nonlinear Kalman Filtering, that relies on something called the Unscented Transform, to pass probability distributions through nonlinear functions
- Unscented Transform gives us much higher accuracy than analytical EKF style linearization, for a similar amount of computation, and without needing to compute any Jacobians. It is recommended to use **UKF** whenever possible in projects. (ES-EKF is second best if UKF cannot be used for any reason)
- UKF Algorithm Involves:
	- Compute Sigma Points: Sigma points are points on the probability distribution that are deterministic samples chosen to be a certain number of standard deviations away from the mean. **For a N-dimensional PDF $\mathcal{N}(\mu_x, \Sigma_{xx})$, we need 2N+1 sigma points**
		- First take the Cholesky Decomposition of the covariance matrix: $\bold L \bold L^T = \Sigma_{xx}$ ($\bold L$ lower triangular)
		- Second, calculate the sigma points where the first is at the mean of the distribution:
			- $\bold x_0 = \bold{\mu_x}$
			- $\bold x_i = \bold{\mu_x}+/- \sqrt(N + \kappa) \text{col}_i\bold L$ where $i=1,...N$
			- $\kappa$ is a tuning parameter that is free to set, but for gaussian distributions, it is conventionally $\kappa = 3-N$
		- Transform the 2N+1 sigma points through the non-linear function to get a new set of transformed sigma points and compute the mean and covariance of the output PDF:
			- $\bold y_i = \bold h(\bold x_i)$, $i=1,...,2N$
		- Caculated updated:
			- Mean: $\bold \mu_y = \sum^{2N}_{i=0}{\alpha_i \bold y_i}$
			- Covariance: $\bold \Sigma_{yy} = \sum^{2N}_{i=0}{\alpha_i(\bold y_i - \bold{\mu_y})(\bold y_i - \bold \mu_y)^T} + \bold Q_{k-1}$ (note mean should be a vector too, bold not working on mu) (Q is additive process noise)
				- with weights $\alpha_i = \frac{\kappa}{N + \kappa}$ if i = 0
				- $\alpha_i = \frac{1}{2(N + \kappa)}$ otherwise
	- To propogate the state from time (k-1) to time k, apply the Unscented transform using the current best guess for the mean and covariance:
		- Compute sigma points of the system at k-1 as above
		- For prediction step: Propogate sigma points through nonlinear prediction state $\bold{\check x^{(i)}_{k}} = \bold f_{k-1}(\bold{\hat x^{(i)}_{k-1}, \bold u_{k-1}, \bold 0)}$ for $i = 0...2N$ to get the predicted sigma points through the non-linear motion model
		- Compute predicted mean and covariance
			- $\bold{\check x_k}= \sum^{2N}_{i=0}{\alpha_i \bold{\check x^{(i)}_k}}$
			- $\bold{\check P_k} = \sum^{2N}_{i=0}\alpha_i(\bold{\check x^{(i)}_{k}}-\bold{\check x_{k}})(\bold{\check x^{(i)}_{k}}-\bold{\check x_{k}})^T  + \bold Q_{k-1}$
		- For correction step: Predict measurements from propogated sigma points by passing sigma points through the nonlinear measurement model 
			- $\bold{\hat y_k}= \sum^{2N}_{i=0}{\alpha_i \bold y^{(i)}_k}$
		- Estimate mean and covariance of predicted measurements using same method as above with additive measurement noise $\bold R_k$ instead of Q
			-  $\bold{\hat y_k}= \sum^{2N}_{i=0}{\alpha_i \bold{\hat y^{(i)}_k}}$
			- $\bold{P_y} = \sum^{2N}_{i=0}\alpha_i(\bold{\hat y^{(i)}_{k}}-\bold{\hat y_{k}})(\bold{\hat y^{(i)}_{k}}-\bold{\hat y_{k}})^T  + \bold R_{k-1}$
		- Compute cross-covariance (between predicted state x and predicted measurements y ) and kalman gain which tells us how the measurements are correlated with the state
			- $\bold{P_{xy}} = \sum^{2N}_{i=0}\alpha_i(\bold{\check x^{(i)}_{k}}-\bold{\check x_{k}})(\bold{\hat y^{(i)}_{k}}-\bold{\hat y_{k}})^T$
			- $\bold K_k = \bold P_{xy} \bold P^{-1}_y$
		- Compute corrected mean and covariance 
			- $\bold{\hat x_{k}} = \bold{\check x_{k}} + \bold K_k(\bold{y_{k}}-\bold{\hat y_{k}})$
			- $\bold{\hat P_k} = \bold{\check P_k} - \bold K_k \bold P_y \bold K^T_k$
			- 
# GNSS/INS Sensing for Pose Estimation
## 3D Geometry and Reference Frames
- One vector can be expressed in different coordinate frames, and the coordinates of the vector are related through a **rotation matrix**
- For reference frames a and b, we can convert the a vector in reference frame a as: $\bold r_b = \bold C_{ba} \bold r_a$ where $\bold C_{ba}$ takes coordinates in frame a and rotates them into frame b, subscript read from right to left
- For transformations of vectors, we must ensure when adding or subtracting that all vectors are in the same reference frame
### How can we represent a rotation?

- 3x3 rotation matrix "direction cosine matrix" (DCM)
	- Requires 9 parameters and has constraints, but doesnt suffer from singularities
- Unit quaternions. A unit quaternion can be represented as a four-dimensional vector of unit length, the parameterizes a rotation about an axis defined by the vector u and an angle phi about that vector.
	- Can convert unit quaternions to rotation matrix with algebraic expression
	- Don't suffer from singularities and only need 4 parameters instead of 9
- Euler Angles
	- Requires 3 angle parameters. These angles represent an arbitrary rotation as the composition of three separate rotations about different principal axes.
	- Suffers from singularities/Gimbal Lock
## Reference Frames
- Earth Centered Inertial Frame (ECIF)
	- ECIF is a fixed coordinate frame, meaning that the axes do not move with the planet's rotation, x and y axes are fixed wrt the stars, and z is true north always. 
	- The Earth in this frame rotates about the z-axis
- Earth-Centered Earth-Fixed Frame (ECEF)
	- Coordinate axes are fixed to the Earth as it spins, X is fixed to the prime meridian (on equator), Z is fixed to true north and Y is determined by X,Z right hand rule
	- The Earth in this frame rotates about the z-axis
- Navigation Frame/Local Tangent Frame (for car applications)
	- NED Frame: x points to True North, y points True East, and Z points inward to Earth
	- ENU Frame: x points to True East, y points True North, and Z points outward to Earth
- Sensor Frames are dependent on placement of each sensor on the car, and if we can track the sensor, we should be able to track any point on vehicle, with proper calibration
## Inertial Measurement Unit (IMU)
Composed of:
- Gyroscopes (3 to measure angular rotation rates about three separate axes)
	- Expensive, mechanical, and very accurate gyroscope uses a very fast spinning disk in center that resists orientation and gimbal frames outside it to track orientation. But strap down IMUs are also mechanical and can be used in self driving cars but do not use a spinning disk
	- Cheap microelectromechanical gyros measure rotational rates instead of orientation directly, are very noisy, drift overtime, and introduce lots of error as finding the actual orientation requires numerical integration of rotational rates
	- Mathematically measures $\omega(t) = \omega_s(t)+\bold b_{gyro}(t) + \bold n_{gyro}(t)$ where $\omega_s(t)$ is angular vel. of the sensor expressed in the sensor frame, $\bold b_{gyro}(t)$ is a slowly evolving bias, and $\bold n_{gyro}(t)$ is a noise term
- Accelerometers (3 to Measure accelerations along three orthogonal axes) 
	- Accelerometers measure acceleration relative to free-fall, also called the proper acceleration or specific force. (e.g., sitting still, a person's proper acceleration is g upwards as the normal force of the chair is holding you up relative to free fall)
	- In localization, we care about "coordinate acceleration", relative to a fixed reference frame, which can be computed using fundamental equation for accelerometers in a gravity field: $\bold f + \bold g = \ddot r_i$ (The second derivative opposition, computed in a fixed frame, is the sum of the specific force, and the acceleration due to gravity.)
	- Mathemtically measures: $\bold a(t) = \bold C_{sn}(t)(\bold{\ddot r_n^{sn}(t)-\bold g_n}) + \bold b_{accel}(t) + \bold n_{accel}(t)$ (need to explicitly remove effect of gravity to measure coordinate acceleration instead of proper acceleration relative to free-fall
	- NOTE that if we inaccurately keep track of the orientation/rotation matrix $\bold C_{sn}(t)$, we incorporate components of gravity and lead to terrible estimates of position ($\bold r_n^{sn}(t)$)
	
## Global Navigation Satellite System (GNSS)
- Works through trilateration via pseudoraning from at least 4 satellites (for a 3D position fix)
	- Trilateration works as the receiver computes a distance to each visible satellite by comparing its own internal clock with that of the time of transmission. The time difference is converted to a distance using knowledge that electromagnetic signals propagate at the speed of light. To compute a 3D position, the ranging equations require at least four visible satellites.
	- For each satellite, we measure the **pseudorange** using a mathematical equation (not listed), and with 4 satellites and 4 equations, we can solve for the receiver's 3D position and clock error explicitly
- Sources of Error:
	- Ionospheric Delay: charged ions in the atmosphere affect signal propagation
	- Multipath effects: Surrounding terrain, buildings can caused unwanted deflections
	- Ephemeris & clock errors: a tiny clock error can have a catastrophic error in position estimates
	- Geometric Dilutin of Precision (GDOP): Configuration of the visible satellites relative to the receiver can also affect position estimation. Ideally, you want the satellites to be spread out across the sky relative to the receiver for a good config, a poor config is if all satellites are clustered together in the sky.
- GNSS accuracy can be improved using (but can be costly to implement):
	- Differential GPS (DGPS): Differential GPS can correct receiver positioning estimates by making use of the more accurately known positions of one or more fixed base stations. Corrections are broadcast on separate frequencies to the GNSS receiver in the moving vehicle.
	- Real-Time Kinematic GPS (RTK): Real-Time Kinematic, or RTK GPS makes use of carrier phase information to improve positioning accuracy down to two centimeters in some cases.
# Light Detection and Ranging Sensors (LIDAR)
- LIDAR sensors use laser pulses and time-of-flight to measure distances to objects along a specific direction. Knowing the speed of light and the round-trip time of a laser that is sent out and bounced back on a surface, you can easily calculate the distance toward that point in space. Because it is an active sensor and emits its own light, it can see in the dark
- To measure distances in 2D, standard LIDARs shoot a laser into a angled rotating mirror that rotates and sends the laser out in a 2D horizontal plane, which allows you to get distance points in that 2D horizontal plane.
- For 3D point measurements, standard LIDARs either use a series of 2D scans pointed at different angles to get a collection of 2D scans that maps the 3D environment with horizontal scans, or one mirror itself dynamically changes its reflecting angle, but this is less common.
- 3D LIDARs report the position of points in 3D using spherical coordinates, range ($r$) (distance to 3D point), elevation angle ($\epsilon$) (angle from sensor's xy plane to the point), and azimuth ($\alpha$) angle (angle measured CCW from sensor's X-axis), also can report the intensity.
	- Elevation and azimuth angles tells you the direction in the sensor frame of the detected point. Measured using encoders that tells you the orientation of the mirror
	- Range  tells you how far the point is from the sensor's origin. Measured using time of flight described previously
	- Intensity is am measure of the return strength of the laser pulse for that point.
	- To convert spherical to Cartesian coordinates and vise versa, we use a standard equations:
		- $[x,y,z]^T = \bold h^{-1}(r,\alpha,\epsilon) = [r\text{cos}\alpha\text{cos}\epsilon, r\text{sin}\alpha\text{cos}\epsilon, r\text{sin}\epsilon]^T$ (Inverse Sensor Model, spherical -> cartesian)
		- $[r,\alpha,\epsilon] = \bold h(x,y,z) = [\sqrt{x^2+y^2+z^2}, \text{tan}^{-1}(y/x), \text{sin}^{-1}(\frac{z}{\sqrt{x^2+y^2+z^2}})]^T$ (Forward sensor model, cartesian -> spherical)
- Uncertainties/Sources of Measurement Noise:
	- Uncertainty in determining exact time of arrival of reflected signal
	- Uncertainty in measuring the exact orientation of mirror
	- Interaction with target (surface absorption, specular reflection, etc)
	- Variation of Propagation speed (e.g. through materials)
	- Are typically accounted for by assuming additive zero mean gaussian noise added to the forward sensor model with a empirically determined or manually tuned covariance

### LIDAR Sensor Models and Point Clouds
- Common solution to storing point cloud information is to use a matrix, and index each point to create a matrix with columns as the different points in the point cloud, allows us to do fast matrix computations on the point cloud
- **Translational Operations** on Point Clouds
	- Occurs when we detect a point P in a reference frame as the vehicle moves/translates, and we want the coordinates of point P in the new frame. Note the point P does not move, our view moves.
	- This can be done easily with matrix subtraction as you only need to subtract each point in the first frame point cloud by the negative of the translation vector needed to reach the new frame. If we stack the translation vector column-wise into a large matrix of the same size as the point cloud matrix, we can complete the entire point cloud translation in one matrix subtraction 
- **Rotation Operations** on Point Clouds 
	- Occurs when we rotate our frame of reference (FOR) with the same origin while observing the point P such that our new frame of reference is defined with a new set of basis vectors
	- This can be done using a rotation matrix that represents the change from the first FOR to the new rotated FOR. This rotation matrix can be multiplied for each point or directly on the entire point cloud matrix $\bold P_{s'} = \bold C_{s's} \bold P_s$
- **Scaling Operations** on Point Clouds
	- Works similarly to rotation but instead of changing the direction of the basis vectors, we are changing their lengths/magnitudes
	- Mathematically, this just means pre-multiplying the coordinates of each point by a diagonal matrix S whose non-zero elements are simply the desired scaling factors along each dimension. This can again be directly multiplied on the point cloud matrix to scale the entire point cloud.
- We can mathematically combine all 3 operations in one equation to easily apply any operations on a point cloud at once: $\bold P_{s'} = \bold S_{s's}\bold C_{s's}(\bold P_s- \bold R^{s's}_s)$
- We can use point clouds for useful self-driving tasks, like fitting a 3D plane to find the road surface from the point cloud. This can be done with least-squares optimization where we want to fit parameters a,b,c for a 3D-plane equation $z = a + by + cy$ where we have measurements of $x,y,z$.
	- Define the error to be the difference between the true z-value of the road surface plane and our measurements ($e_j = \hat z_j - z_j = (\hat a + \hat b x_j + \hat c y_j) - z_j$ for all j = 1...n).
	- Stack all measurement errors into matrix form where $\bold e$ is a column vector of errors, $\bold A$ is a matrix where each row is $[1, x_j, y_j]$ for all j=1...n, $\bold b$ is a column vector of our measured $z_j$ values, and $\bold x$ to be the column vector of our unknowns a,b,c.
	- Minimizing the squared-error criterion and using the method of least squares, we can get the optimal parameters either analytically using the pseudoinverse as : $\bold{\hat x} = (\bold A^T \bold A)^{-1}\bold A^T \bold b$
	- Point Cloud Library (PCL) is a very popular tool used in industry for working with point clouds in C++, with Python bindings that exist.
### State/Pose Estimation From LIDAR Data via Point Set Registration Problem
- The point set registration problem says: given two point clouds in two different coordinate frames, and with the knowledge that they correspond to or contain the same object in the world, how shall we align them to determine how the sensor must have moved between the two scans? In general, we don't know which points correspond to each other.
- The most popular algorithm for solving this problem is the **Iterative Closest Point (ICP)** algorithm. 
	- __Intuition__: When the optimal motion is found, corresponding points will be closer to each other than to other points in a euclidean sense.
	- **Heuristic**: For each point, the best candidate for a corresponding point is the point that is closest to it right now. This heuristic should give us the correspondences that let us make our next guess for the translation and rotation, that's a little bit better than our current guess, and this repeats until convergence
- **ICP Algorithm Procedure**:
	1. Get an initial guess for the transformations: $\bold{\check C_{s's}}, \bold{\check r^{s's}_s}$
		- Must have a good initial guess to not get stuck in local minima. Common sources of guesses include a motion model for the car as supplied by IMUs, wheel odometry, or constant velocity estimate. Complexity of guess depends on how smoothly or not the car is driving,
	2. Transform the initial guess from its reference frame to the new frame and then associate/map each point in $\bold P_{s'}$ with the nearest point in $\bold P_s$. Note that there's nothing stopping two points in one cloud from being associated with the same point in the other cloud.
	3. Take all of the matched points and find the optimal transformation that minimizes the sum of squared distances between the corresponding points. 
		- Use Least Squares to solve for optimal $\bold{\hat C_{s's}}, \bold{\hat r^{s's}_s}$ at each step. Formulation and equations are in video slides, omitted here in notes. But a nice closed form analytical solution exists:
		- To compute the **optimal rotation matrix** from S to S'
			1. Compute centroids of each point cloud
			2. Compute a matrix capturing the spread of the two point clouds using the centroids: $\bold W_{s's} = (1/n) \sum^n_{j=1}(\bold p^{(j)}_s - \bold{\mu_s})(\bold p^{(j)}_{s'} - \bold{\mu_{s'}})^T$ for all j points j=1...n
			3. Use the singular value decomposition of the spread matrix, set the diagonal singular values to 1, 1, and detUdetV (for bottom right value), and multiply the three matrices in the same order to get the optimal rotation matrix $\bold{\hat C_{s's}}$
		- To compute the **optimal translation matrix** after finding the optimal rotation matrix we just rotate the centroid of the new frame into the old frame using the optimal rotation matrix, then subtract the centroids to find the optimal translation vector: $\bold{\hat r^{s's}_s = \bold \mu_s - \bold{\hat C^T_{s's}}} \bold \mu_{s'}$
	4. Repeat until convergence with our new found translation and rotation as our initial guess
- Note the above algorithm is a **point-to-point** ICP that minimizes the Euclidean distance between each point in the new frame/point cloud and the nearest point in the original frame/point cloud
- **Point-to-plane** ICP minimizes the perpendicular distance between each point in the new frame/point cloud to the nearest plane in the original frame/point cloud, this tends to work well in structured environments like cities. This requires a number of planes to be fit to the origin point cloud to represent walls and surfaces. The challenge is to figure out which valid planes are and to use.
- Note ICP requires assumption of a stationary world, as objects that move with our sensor will have 0 relative velocity to us and skew our measurements, So we need to be careful to exclude or mitigate the effects of outlying points that violate our assumptions of a stationary world.
	- Mitigation options include:
		- Fusing ICP motion estimates with GPS estimates
		- Identify and ignore moving objects with computer vision
		- Choosing a different loss function that is less sensitive to large errors induced by outliers than our standard squared error loss. This class of loss functions are called Robust Loss/Cost Functions (Squared error, Absolute error, Cauchy loss, Huber Loss) but require additional complexity as we need to add an iterative optimization step instead of having a nice analytical solution for computing optimal rotation/translation matrices in the ICP algorithm.
# State Estimation in Practise
## Multisensor Fusion for State Estimation
Develop an Error State extended Kalman Filter for estimating position, velocity, and orientation using an IMU, GNSS sensor, and LIDAR
- Benefits of using GNSS with IMU and LIDAR:
	- Error dynamics are completely different and uncorrelated
	- IMU provides "smoothing" of GNSS, fill-in during outages due to jamming or maneuvering
	- GNSS provides absolute positioning information to mitigate IMU drift
	- LIDAR provides accurate local positioning within known maps
- For EKF State Estimation, Estimator Coupling can be Tight or Loose:
	- Tight: Uses raw Pseudo-ranges to satellites & LIDAR point clouds, potentially higher accuracy, but higher complexity/more tedious to implement and requires alot of tuning
	- Loose: Assume GNSS/LIDAR data has been preprocessed to produce a noisy position estimate as input, potentially lower accuracy but much lower complexity to implement
- System Design:
	- Use IMU measurements as noisy inputs to our motion model, updated at a high rate (>100Hz)
	- At a lower rate(< 1-10Hz), incorporate GNSS/LIDAR measurements whenever available.
	- Combine the predicted state via the Motion Model ($\bold{\check x_k}$) and GNSS/LIDAR position observations in Kalman Fusion to obtain a corrected state $\bold{\hat x_k}$, which is sent back to the motion model for future updates
- State & Inputs:
	- __Vehicle State__ consists of a 10-dim vector that consists of 3D position, 3D velocity, and 4D parameterization of orientation using a unit quaternion
	- __Motion model input__ will consist of 6-dim vector containing 3D specific force ($\bold f_k$) and 3D rotational rates ($\bold \omega_k$) from our IMU (assume unbiased, gyroscope and accelerometer biases are omitted, NOT a realistic assumption)/Could also be called/known as an axis angle as it denotes the rate of change of the axes of the IMU,, and the $\bold q(\theta)$ function below converts it to a quaternion so that it can be applied to change a previous quaternion estimate of the orientation.
	- Motion model equations are provided in slides as well if image below doesnt work:
	- ![](https://lh4.googleusercontent.com/NLep18HQwuKyQsCHw08niWa7BJpiGUk-gzX-qIeEG66q0PTF8ZQpTav4X5iO2dHf4eFPUF8dExuBBLDIhjI0d9tmsxPjYxi5eQpJZBIa6SEa9Wij2paM9A-Kq2WWCYOnVC88Jayn8-97i6wOtnx2K68)
	- NOTE: that the $\bold q(\bold \theta)$ function takes in a 3D vector $\theta$ and outputs a 4D parameterized quaternion where the top cos term is a scalar, and the bottom sin term produces a elementwise scaled 3D vector of $\theta$. Note that $|\theta|$ denotes the L2 norm of the 3D $\theta$ vector.
	- Cns is the rotation matrix associated with quaternion q_k-1, just convert the quaternion to a rotation matrix representation to get Cns
	- The __error state__ consists of a 9-dim vector, 3D position error, 3D velocity error, and a 3D orientation error in the global/navigation frame: $\delta \bold{x_k} = \bold K_k (\bold y_k - \bold{\check p_k})) = [\bold{\delta p_k},\bold{\delta v_k},\bold{\delta \phi_k}]^T$
	- The __Measurement Model__ for GNSS and LIDAR can be expressed as:
		- $\bold y_k = \bold h(\bold x_k) + \bold{\nu_k} = \bold H_k \bold x_k + \bold{\nu_k} = [\bold 1 \bold 0 \bold 0]\bold x_k + \bold{\nu_k} = \bold p_k + \bold{\nu_k}$
		- where $\bold{\nu_k} \sim \mathcal{N}(\bold 0, \bold R_{GNSS}) or \sim \mathcal{N}(\bold 0, \bold R_{LIDAR})$ is the normal gaussian position measurement noise for the GNSS and LIDAR, and we assume LIDAR and GNSS supplied measurements in the same coordinate frame
		- Note we only take the first 3 elements of $\bold x_k$which represent position as $\bold y_k$ measurement
	- Algorithm:
		- Loop:
			- Update State $\bold{\check x_k} = [\bold{\check p_k},\bold{\check v_k},\bold{\check q_k}]^T$ with motion model using IMU inputs  $\bold{\check x_{k-1}} = [\bold{\check p_{k-1}},\bold{\check v_{k-1}},\bold{\check q_{k-1}}]^T$ via equations above. Note that $[\bold{\check p_{k-1}},\bold{\check v_{k-1}},\bold{\check q_{k-1}}]^T$ can be either corrected or uncorrected depending on whether GNSS/LIDAR measurements were available
			- Propagate Uncertainty: $\bold{\check P_k} = \bold F_{k-1} \bold{P_{k-1}}\bold F^T_{k-1} + \bold L_{k-1} \bold Q_{k-1} \bold L^T_{k-1}$ Again P could be predicted value or corrected value
			- If GNSS or LIDAR position available:
				- Compute Kalman Gain: $\bold K_k = \bold {\check P_{k}}\bold H^T_k (\bold H_k \bold {\check P_{k}} \bold H^T_k + \bold R )^{-1}$ where $\bold R$ can be GNSS or LIDAR
				- Compute error state: $\delta \bold{x_k} = \bold K_k (\bold y_k - \bold{\check p_k})) = [\bold{\delta p_k},\bold{\delta v_k},\bold{\delta \phi_k}]^T$
				- Correct predicted state:
					- $\bold{\hat p_k} = \bold{\check p_k} + \bold{\delta p_k}$
					- $\bold{\hat v_k} = \bold{\check v_k} + \bold{\delta v_k}$
					- $\bold{\hat q_k} = \bold{q(\delta \bold \phi)} \bigotimes \bold{\check q_k}$ note that the quaternion is a constrained quantity that is not a simple vector. We have chosen to use a global orientation error meaning the orientation update involves multiplying by the error state quaternion on the left, which is different from the propagation step in the motion model. The $\bigotimes$ symbol represents quaternion multiplication.
				- Compute corrected covariance: $\bold{\hat P_k} = (\bold 1 - \bold K_k \bold H_k)\bold{\check P_k}$
	- The above formulation also assumes our sensors are spatially and temporally aligned
## Sensor Calibration - A Necessary Evil
### Intrinsic Calibration
- In intrinsic calibration, we want to determine the fixed parameters of our sensor models, so that we can use them in an estimator like an extended Kalman filter, and are typically expected to be constant.
- Example: Radius of wheel for wheel encoders, elevation angle of a scan line in a LIDAR sensor
- Sources of intrinsic parameter estimation:
	- Manufacturer specifications
	- Measure by hand if possible
	- Estimate as part of the vehicle state as a special calibration step before or during operation
### Extrinsic Calibration
- We want to figure out how do we determine the relative poses of all the sensors? we're interested in determining the relative poses of all of the sensors usually with respect to the vehicle frame.
- Very important for fusing information from multiple sensors
- For example, we need to know the relative pose of the IMU and the LiDAR. So, the rates reported by the IMU are expressed in the same coordinate system as the LiDAR point clouds.
- Sources of extrinsic parameter estimation:
	- CAD model of vehicle and sensors
	- Estimate as part of the vehicle state, but difficult to do
### Temporal Calibration
- How do we determine te relative time deplays of all the sensors and/or synchronize them?
- Straightforward solution is to timestamp each measurement when the onboard computer receives it and match up the measurements that are closest to each other. But this solution suffers from unknown time delays between the recorded measurement by the sensor and the time the computer receives the measurement
- If w want a really accurate state estimate, we need to think about how well our sensors are actually synchronized:
	- Simplest and most common thing to do is just to assume the delay is zero. But less than accurate than better temporal calibration methods
	- Can use hardware timing signals to synchronize the sensors, but only an option for more expensive sensor setups
	- Estimate as part of the state, but can get EXTREMELY complicated, active area of research
## Loss of one or more sensors
- In order to build a safe vehicle, it's crucial to understand and characterize what happens when one or more sensors malfunctions, and the minimal sensing we need to maintain safe operation.
- Most cars have long, medium, and short range sensing. If one of these malfunctions, it's often important to restrict movement so the malfunction doesn't affect safety. And consider the minimal allowable sensing equipment necessary to perform different actions
- Multiple sensors are crucial to robust localiztion in varied environments. If possible, consider independent backup systems, sensors, and frequently sensor calibration
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTExMTk3MDk1MDMsMTk5Mzg0MDkwOCwxOT
E3ODM5NzgzLC00NTEzMTk1ODUsLTQwMDkwODAzOCw2MjI1ODYz
ODQsLTE5OTU0NDE3MzMsNjE0NDcwNTgsMTYwNjEyMjMyNSwxNj
YzMDA3NDMwLDEyOTk3OTI4MjIsLTE4NDE1Njc0MTUsLTE1Mjc1
OTQwNTMsLTczNjYzNzc5MywxOTkwNjU2NTYzLDk2NTAxMjYzNi
wtMjAxOTc3MzUwNiw4MTg0NzQxMTYsNzUzNDU4NjI2LC0xOTg2
MTgxMDddfQ==
-->