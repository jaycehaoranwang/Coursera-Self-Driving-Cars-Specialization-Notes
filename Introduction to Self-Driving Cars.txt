
# Week 1: Taxonomy of Driving
### Main Driving Task
 - Perception, perceiving the environment
 - Motion Planning: planning how to reach from point A to point B
 - Controlling the Vehicle
 ###  Operational Design Domain
 - Domain/environment which the car is designed to operate safely within
 ###  How to classify driving system Automation?
- Driver Attention requirements
- Driver Action Requirements
### What Makes up a driving Task?
-   Lateral Control - steering
-   Longitudinal Control - braking, acceleration
-   Object and Event Detection and Response (OEDR): detection, reaction
-   Planning: Long term/Short Term
-   Miscellaneous: signalling
### Autonomous Capabilities
-   Automated lateral control?
-   Automated longitudinal control?
-   OEDR
-   Automatic Emergency Response
-   Driver supervision
-   Complete VS restricted ODD
### Levels of Autonomy
-   Level 0: Regular human driver
-   Level 1: Driving Assistance - longitudinal or lateral control assists, EITHER but NOT both. E.g. Adaptive Cruise control, lane keeping assistance
-   Level 2: Partial Driving Automation - Both longitudinal and lateral assists, but requires driver attention at all times
-   Level 3: Conditional Driving Automation - Longitudinal, Lateral, and OEDR but requires driver attention and takeover in event of failure
-   Level 4: High Driving Automation: Level 3 but handles emergencies autonomously, does NOT require driver attention, but in a defined ODD
-   Level 5: High Driving Automation: Level 4 but in an unlimited ODD
- ![](https://lh3.googleusercontent.com/rqSCl_9oLMWUIGDc8UV3xbd9W1IrVZPHhg1bv9Vx0l4QLyFR0wYHvIFKIvfFDsf6rwc4mBoQ0taWRwRrS5e7poNn0XdCfeD3v-4AHwuovslmdBMW0cb3WVWkHPp-ub4wIIPDvV6tjtYm2IWy-sHmfU8)
- ## Perception: Making Sense of the Environment

Goals for Perception:

-   Identify Static Objects:
    -   Roads/Lane markings (on-road)
    -   Curbs
    -   Traffic Lights
    -   Road signs
    -   Construction signs, obstructions
-   Identify Dynamic Objects that move:
    -   Vehicles
        -   4 Wheelers
        -   2 wheelers
    -   Pedestrians
-   Ego Localization:
    -   Position
    -   Velocity, acceleration
    -   Orientation, angular motion

Challenges to Perception:

-   Robust detection and segmentation
-   Sensor uncertainty/corrupted measurements
-   Occlusion, reflection
-   Illumination, lens flare, GPS outage
-   Weather, precipitation

## Lesson: Driving Decisions and Actions

Making Decisions:

### Long Term

-   How to navigate from A to B

### Short Term

-   Can I change my lane?
-   Can I pass this intersection to join the left road

### Immediate

-   Can I stay on track on the curved road?
-   Accelerate or brake, by how much?

Rule Based Planning:

-   Involved decision trees
-   In reactive rule-based planning, rules consider the current state of ego and other objects to make decisions
    -   Example rules:
        -   If there is a pedestrian on the road, stop
        -   If the speed limit changes, adjust speed to match it

Predictive Planning:

-   Make predictions about other vehicles and how they are moving
-   Use these predictions to inform our decisions
-   Example predictions:
    -   That car has been stopped for the last 10 seconds. It is going to be stopped for the next few seconds
    -   Pedestrian is jaywalking, she will enter our lane by the time we reach her
-   Predominant planning for self-driving as it greatly increases the scenarios we can handle safely

# Week 2: Sensors and Computing Hardware

### Sensors

#### Exteroceptive Sensors (Surroundings)

-   Camera
    -   Resolution
    -   Field of View
    -   Dynamic Range
-   Stereo Cameras
    -   Enables depth estimation from a pair of image data
-   LIDAR (Light Detection and Ranging)
    -   Detailed 3D scene geometry from LIDAR point cloud
    -   Shoots and measures returned light beams
    -   Comparison Metrics:
        -   Number of beams
        -   Points per second
        -   Rotation Rate
        -   Field of View
-   Solid State LIDAR (Upcoming)
-   Radar (Radio Detection and Ranging)
    -   Robust object detection and relative speed estimation
    -   Useful in adverse weather conditions
    -   Comparison Metrics:
        -   Range
        -   Field of View
        -   Position and speed accuracy
-   Ultrasonic (Sonars, Sound Navigation and Ranging)
    -   Short-range all-weather distance measurement
    -   Ideal for low-cost parking solutions
    -   Comparison Metrics:
        -   Range
        -   Field of View
        -   Cost

#### Proprioceptive Sensors (Internal)

-   Global Navigation Satellite Systems (GNSS)
    -   Direct measurement of ego vehicle states
    -   Position, velocity
    -   Varying accuracies based on calculation methods
-   Inertial Measurement Units (IMU)
    -   Angular rotation rate
    -   Acceleration
    -   Heading (IMU + GPS)
-   Wheel Odometry
    -   Tracks wheel velocities and orientation
    -   Calculates overall speed and orientation of the car
    -   Accuracy:
        -   Speed accuracy
        -   Position drift

### Computing Hardware

-   Main brain computer:
    -   Takes in all sensor data and computes actions
    -   Existing advanced systems for self-driving car processing
-   Specific hardware for parallelizablecomputation:
    -   GPUs
    -   FPGAs
    -   ASICs
-   Synchronization hardware needed to synchronize different modules and provide a common clock
    -   Often GPS is used as an appropriate reference clock

## Hardware Configuration Design

Assumptions that will drive sensor range requirements:

-   Aggressive deceleration: 5m/s
-   Comfortable deceleration: 2m/s (norm unless otherwise stated)
-   Simplified Stopping distance: $d = v^2/2a$ ($v$ = velocity, $a$ = deceleration rate)
# Software Architecture
![](https://lh6.googleusercontent.com/i8UaQlkY6Hww6vhWn_Et8QvYCqhpWNAXuNUMsCz99nqOT_fEiRJ5fC0u1picNKVdtbLpYA2fA3ReWFn-c6xW0CFd8VP6pFqMkxsjJdfLWn0_NaHbOcdLTxYBjb2Eoc6e4-2SGs1BRSW7biDC-aKwmKc)
## Environment Perception

### Localization of ego-vehicle in space

-   Takes in: GPS/IMU/Wheel Odometry
-   Outputs: Vehicle position

### Classifying and locating important elements of the environment

-   Dynamic Object Detection
    -   Takes in: LIDAR/Cameras/Radar, GPS/IMU/Wheel Odometry
    -   Outputs: Bounding Boxes and encoded information of detected objects
-   Dynamic Object Tracking
    -   Takes in: Bounding boxes
    -   Provides: Current position of dynamic object and history of its path through the environment
-   Object Motion Prediction
    -   Uses history of path with roadmaps to predict future path of dynamic objects
-   Static Object Detection
    -   Takes in: LIDAR/Cameras/Radar, HD Road Map
    -   Identifies significant static objects in the scene, including current lane and location of regulatory objects such as signs and traffic lights

## Environment Mapping

### Occupancy Grid Map

-   Takes in: Object tracks
-   Represents all static objects in the environment surrounding the vehicle
-   LIDAR is predominantly used to construct the occupancy grid map
-   A set of filters is applied to the LIDAR data to make it usable by the occupancy grid:
    -   Remove drivable surface points and dynamic object points
-   The occupancy grid map represents the environment as a set of grid cells and associates a probability that each cell is occupied

### Localization Map

-   Constructed from LIDAR or camera data
-   Used by the localization module to improve ego state estimate
-   Sensor data is compared to this map while driving to determine the motion of the car relative to the localization map
-   Motion is combined with other proprioceptor sensor information to accurately localize the ego vehicle

### Detailed Road Map

-   Provides a map of road segments representing the driving environment
-   Captures signs and lane markings for motion planning
-   Traditionally a combination of prerecorded data and incoming information from the current static environment gathered by the perception stack

The environment mapping and perception modules interact significantly to improve the performance of both modules. For example, the perception module provides the static environment information needed to update the detailed road map, which is used by the prediction module to create more accurate dynamic object predictions.

## Motion Planning

### Mission Planner

-   Handles long-term path planning over the entire driving task
-   Determines optimal driving path/navigation
-   Takes in: Current goal, detailed road map, and vehicle position

### Behavior Planner

-   Takes in: Detailed road map, Mission path, Dynamic Objects, Occupancy Grid
-   Establishes a set of safe actions/maneuvers to be executed while traveling along the mission path (short-term planning)
-   Provides a set of constraints to execute with each action

### Local Planner

-   Performs immediate or reactive planning
-   Takes in: Occupancy grid, Behavior constraints, Dynamic Objects
-   Outputs a planned trajectory, which is a combined desired path and velocity profile for a short period of time into the future

## Vehicle Controller

-   Separated into different controllers
-   Longitudinal controller regulates throttle, gears, and braking system to achieve correct velocity
-   Lateral controller outputs the steering angle required to maintain the planned trajectory
-   Both controllers calculate current errors and tracking performance of the local plan and adjust the current actuation commands to minimize the errors going forward

## System Supervisor

-   Continuously monitors all aspects of the autonomous car
-   Gives appropriate warnings in the event of a subsystem failure
-   Software Supervisor:
    -   Validates the software stack to ensure all elements are running as intended at the right frequencies and providing complete outputs
    -   Analyzes inconsistencies between outputs of all modules
-   Hardware Supervisor- Continuously monitors all hardware components
-   Checks for any faults, such as a broken sensor, a missing measurement, or degraded information
-   Analyzes the hardware outputs for inconsistent hardware outputs within the domain

## Environment Representation

### Map types:

-   Localization of vehicle in the environment
    -   Localization point cloud or feature map
    -   Collects continuous sets of LIDAR data
    -   Difference between LIDAR maps is used to calculate the movement of the autonomous vehicle
-   Collision avoidance with static objects
    -   Occupancy grid map
    -   Discretized fine-grain grid map (can be 3D or 2D)
    -   Represents occupancy by static objects and non-drivable surfaces that affect the car's plan and movement
    -   All dynamic objects must FIRST be removed by perception step
    -   Only LIDAR points from relevant static objects remain but are not perfect and provide a probabilistic grid

### Path Planning

-   Detailed road map
-   Contains information regarding the lanes of a road, as well as any traffic regulation elements that may affect them
-   Best way is to be created using both online and offline methods

# Week 3: Safety for Self-Driving Cars
Safety: "absence of unreasonable risk of harm"
Hazard: "potential source of unreasonable risk of harm"
Major Hazard Sources can include any of: Mechanical, Electrical, Hardware, Software, Sensors, Behavioral, Fallback, Cybersecurity
### NHTSA: Safety Framework
- Systems engineering approach to Safety
- Autonomy Design
	- ODD
	- OEDR
	- Fallback
	- Traffic Laws
	- Cybersecurity
	- Human Machine Interface (HMI)
- Testing & Crash Mitigation
	- Testing (simulation closed-track testing, public driving)
	- Crashworthiness
	- Post Crash Behavior 
	- Data recording function
	- Consumer Education and training

### Waymo: Safety Levels
- Behavioral Safety (Safe Decision making in all scenarios)
- Functional Safety (Redundancies in functioning and systems of car
- Crash Safety (Ensure minimal damage inside car in even tof crash)
- Operational Safety (Interfaces are usable, convenient, and intuitive for passengers)
- Non-Collision Safety (Minimize danger to people that interact with the system)
### Safety Procedure
- Identify Hazard scenarios and mitigations
- Use Hazard assessment methods to define safety requirements
- Extensive testing (Simulation, Closed-course testing, public tests)

Generic Safety Framework
 - 
 Probabilistic Fault Tree Analysis 
	- Top down deductive failure analysis 
	- Boolean Logic
	- Assign probabilities to fault "leaves"
	- Use Logic gates to construct failure tree (P(A) or P(B) = P(A) + P(B), P(A) AND P(B) = P(A) * P(B))
	
Failure Mode and Effect Analysis (FMEA)
- Bottom up process to identify all the effects of faults in a system
- Failure Mode: Modes or ways system may fail
- Effects Analysis: Analyzing effects of the failure modes on the operation of the system
- Idea: Categorize failure modes by priority
	- How serious are their effects
	- How frequently do they happen?
	- How easily can they be detected?
- Eliminate or reduce failures, starting with top priority
- Steps: Compute a Risk Priority Number (RPN) = S * O * D for all listed failure modes and address accordingly
	- S = Severity Rating
	- O = Occurrence rating
	- D = How difficult is the failure to detect (10 impossible, 1 guaranteed)

# Week 4: Vehicle Dynamic Modelling
## Kinematic Modelling in 2D
### Kinematic vs Dynamic Modelling
- At low speeds, it is often sufficient to look only at kinematic models of vehicles 
- Dynamic modelling is more involved, but captures vehicle behaviour more precisely over a wide operating range
### Coordinate Frames
- Right handed by convention
- **Inertial Frame**: Fixed, usually relative to earth
- **Body Frame**: Attached to vehicle, origin at vehicle center of gravity, or center of rotation
- **Sensor Frame**: Attached to sensor, convenient for expressing sensor measurements
- We need to attach several coordinates to our moving system and represent elements from these frames in the inertial frame, so we need to transform variables from one coordinate frame to another
- **Coordinate Transformation**: Conversion between inertial frame and body coordinates is done with a translation vector and rotation matrix

In general, to transform one point from one coordinate frame to the other coordinate frame, body to inertial and vice versa, requires two terms:
- The translation from the origin $O_{AB}$ or $O_{BA}$ 
- Rotation matrix between the two coordinate frames

For Example: for a point $P$ in space with a robot with body frame $B$:
- Location of point $P$ in body frame $B$ is:
	- $P_B = C_{EB}(\theta)P_E + O_{EB}$
- Location of point $P$ in inertial frame $E$ is:
	- $P_E = C_{BE}(\theta)P_B + O_{BE}$

where $\theta$ is the orientation angle of the robot 
### Homogeneous Coordinate Form
- A 2D vector in homogeneous form: $P = [x  y]^T$ -> $\bar P =  [x y 1]^T$
- With homogeneous forms, we can transform a point from body to inertial coordinates with a matrix multiplication instead of equations:
	- $\bar P_E = [C_{EB}(\theta) | O_{EB}] \bar P_B$

### 2D Kinematic Modelling
- The kinematic constraint is **nonholonomic**: means it restricts the rate of change of the position of our robot. So our robot can roll forward and turn while rolling, but cannot move sideways directly.
	- A constraint on rate of change of degrees of freedom
	- Vehicle velocity always tangent to current path
	-![](https://lh3.googleusercontent.com/GTj9QRyLcbSzd8FNgibDX62UQh019TFxNYbYVALIZhI6Ki1octniR1zPym_eo2JeL27-7Rv66w1I73ELI_WiQH1SwuCLK0BS-6U18uMcQLITH8cHMNBarglgYlJRcw9Mefv38dEW6dJvuWBai2NVTJc)**State** Definition: a set of variables often arranged in the form of a vector that fully describe the system at the current time.
### Simple Robot Motion Kinematics
![](https://lh6.googleusercontent.com/LWUk6OPQg7OUwvLGGOzBhL4beWJq8auERlfowDtzXqpsLSLMoEnnA9fnLCFVm303tt34jt64F52r_-KlR4nWd1dZSXa2sux245XijpyRDclRKT_Xd4NzHrCCZhZm2lsE12jUY8-HysbrEtE7Zx19O2k)
### Two-Wheeled Robot Kinematics
![](https://lh4.googleusercontent.com/i8itc7aGR-C70FaqBzQAlJ2jXBMVNts4JqmqC68KA40QERheH8mCWc2CdaCAMek4QNXCLxNaEmih233unKmcpJb99wQy7hlQYNZw_0RHri2YKth5_17I43Su9KNsIJsHtBrnRKoX2bbZ7p7qsnKhORQ)
- Assuming no slip with wheels, we can write velocity of individual wheel as $v_i = rw_i$
- Velocity is the average of the two wheel velocities: $v = \frac{v_1 + v_2}{2} = \frac{rw_1 + rw_2}{2}$
- For wheels spinning at different speeds, we can define a **instantaneous center of rotation (ICR)** and find the angular rate of rotation using wheel velocities as:
- ![](https://lh5.googleusercontent.com/o5y8bLp7nfChywTzTDRoxSMErd4jaNMNQZqm7e-9J-MbWIHpJUao2q-yGWHnvTzRjbxoTEr4fmjb99xOLSj5QJFzH8SENy_6dXVEfKHEASzK5nOF8taSOkTVmn0SjhZRLkgEo4lhe043YOAu-VJphKg)Finally, we can write:
- ![](https://lh5.googleusercontent.com/3lanVGNE-bqJrTBlKhedROCULOI8FucDZdUIHLYop2DdudsQn1GoN3b2M65osSQhotBgS2z7RU8XSwtn5LCYrs_DZ1ss2Mc0TOzbV-TSTNbN2er2lQgK2pu-oBbORKLWWm7ItOml9X2MCKRG1XE__WY)
## The Kinematic Bicycle Model
Learning about slip angles and develop the kinematic bicycle models to represent the nonholonomic constraints of a car, the front bike wheel represent the front two wheels of the car and same for the back wheels
- Front wheel steering and assume operation on a 2D plane
- $\theta$ is heading
- **Slip Angle**: slip angle or sideslip angle is the angle between the direction in which a wheel is pointing and the direction in which it is actually traveling
- Reference point of the bicycle can be either:
	- The front axle of the front wheel
	- Rear axle of the rear wheel
	- Center of gravity
- The equations will vary depending on reference point selected. 
 **Rear Wheel Reference Point $(x_r, y_r)$**:
	-	$\dot x_r = v cos \theta$
	-	$\dot y_r = v sin \theta$
	-	$\dot \theta = \frac{vtan\delta}{L}$
	-	Where $L$ is the length between front and rear wheels, $\theta$ is the heading of the vehicle from inertial x-frame, $\delta$ is the steering angle of front wheel relative to heading, $v$ is the velocity of the wheel
 
	 **Front Wheel Reference Point $(x_f, y_f)$**:
	-	$\dot x_f = v cos (\theta+\delta)$
	-	$\dot y_f = v sin (\theta+\delta)$
	-	$\dot \theta = \frac{vsin\delta}{L}$

	 **Center of gravity Wheel Reference Point $(x_c, y_c)$**:
	-	$\dot x_c = v cos (\theta+\beta)$
	-	$\dot y_c = v sin (\theta+\beta)$
	-	$\dot \theta = \frac{vcos\beta tan\delta}{L}$
	-	$\beta = tan^{-1}(\frac{l_r tan\delta}{L})$
	-	Note $l_r$ is length of center of mass from the rear axle
	
![](https://lh5.googleusercontent.com/yV62BGLJH2r8GL5x-gMpTyxi1-iCRPzwGBILsBUpd7vwRiNV92pS6SfAeKvYojyj-2wfb3sB8CI9PgZbADjtxbk_Cf3ZygvMfuLYDl2SiCJXUL1sRw0wcszl6_NEYto5RiD5Bhd65yrUA8UJR4i9OCY)
## Dynamic Modelling in 2D 

Steps to build a typical dynamic model
- Coordinate frames
- Lumped dynamic elements
- Free body diagram
- Dynamic equations using Newton's second law $\sum F = M * a$
- For Rotational/Torsional Systems:
	- Inertia J (akin to mass), torsional force $\tau$, rotational acceleration $\alpha$: $\sum \tau = J * \alpha$
		- Forces resisting torsional force
			- Spring force
			- Damping force
			- Inertia force
		- Sum torques $\tau$ about each axis of rotation and lump

# Rest of this week can be found in slides

# Week 5: Vehicle Longitudinal Control/PID Control
- Proportional gain directly modifies the response proportional to the error. Increasing K_p:
	- Decreases Rise time
	- increases overshoot
	- Small change in settling time
	- decreases steady state error
- Integral gain modifies the response based on accumulated error of the error signal over time, more error accumulated -> larger the response. Increasing K_I:
	- Decreases rise time
	- Increases overshoot
	- Increases settling time
	- Completely eliminates steady state error
- Derivative gain modifies the response based on the rate of change of the error, so if the error is rapidly increasing due to disturbance or other factors, it will be quick to correct it back and increase the response. Increasing K_d:
	- Small change in rise time
	- Decrease in overshoot
	- Decrease in settling time
	- Small change in steady state error
## Feedforward & Feedback Control
- Feedback - Closed loop control structure works based off of the error between the reference and output as input 
- Feedforward - open-loop control directly uses the reference as input by modelling the plant process and applying inputs directly without considering error signals
- Because PID controllers need existence of errors to correct itself, using a pure PID controller will lag behind on a reference signal, whereas using a PID+ Feedforward controller will very accurately follow the reference signal, but still not perfect, which is why a feedback controller is used in conjunction to combat existence of any tracking errors

# Week 6: Lateral Control
## Introduction to Lateral Vehicle Control
The Reference Path is fundamental interace between planning system and lateral control and can be defined in multiple ways:
- **Straight Line segments** can be very compact and easy to construct assuming environment allows for mostly straight line motion, but includes heading discontinuities which make precise tracking a challenge
- A refinement of the line segment approach is to provide a series of **tightly spaced waypoints**. This spacing is usually fixed in terms of distance or travel time. The relative position of the waypoints can be restricted to satisfy an approximate curvature constraint. Waypoint paths are very common, as they are easy to work with and can be directly constructed from state estimates or GPS waypoints collected in earlier runs of a particular route.
- It is also possible to define a path using a sequence of **continuous parameterized curves**, which can be eithe drawn from a fixed set of motion primitives or can be identified through optimization during planning. These curves provide the benefit of continuously varying motion, and can be constructed to have smooth derivatives to aid in the consistency of error and error rate calculations.
### Lateral Controller Design
- Geometric Controllers: rely on the geometry and coordinates of the desired path and the kinematic models of the vehicle:
- Generically, it is any controller that tracks a reference path using only the geometry of the vehicle kinematics and the reference path. In the case of self-driving cars, a geometric path tracking controller is a type of lateral controller that ignores dynamic forces on the vehicles and assumes the no-slip condition holds at the wheels. It relies on a kinematic bicycle model and the error measures defined in the previous video to construct a steering command rule that achieves path tracking.
- Note this method suffers greatly when the no slip assumption is violated in aggressive vehicle movements or lateral acceleration
	- Pure Pursuit (carrot following)
		- In the pure pursuit method, the core idea is that a reference point can be placed on the path a fixed distance ahead of the vehicle, and the steering commands needed to intersect with this point using a constant steering angle can be derived and computed from a single formula
		- Pure Pursuit Controller Formula is: $\delta = tan^{-1}(	\frac{2L sin\alpha}{K_{pp}v_f})$
		- $\delta$ is the steering angle, 
		- $L$ is the length from rear wheels to front wheels of car, 
		- $\alpha$ is the angle between the heading of the car and the line pointing from rear wheel center to the reference lookahead point on the reference path, 
		- $K_{pp}$ is the pure pursuit gain that can be tuned to set how aggressively close or far the reference lookahead point is relative to the velocity 
		- $v_f$ is the forward velocity of the vehicle
		- PP controller is similar to the proportional controller in terms of how the steering angle aggressiveness is proportional to the error/distance from the reference path the vehicle is
	- Stanley
		- Similar to PP but uses the center of the front axle is a reference point
		- Looks at both the error in heading and the error in position relative to the closest point on the path
		- Defines Intuitive steering laws to 
			- Correct heading Error: Steering angle is proportional with desired heading 
			- Correct position error:
				- eliminate cross track error, a proportional control is added, whose gain is scaled by the inverse of the forward velocity. The control is then passed through an inverse tan function which maps the proportional control signal to the angular range of minus Pi to Pi
			- Obey max steering angle bounds: Capped within a range
		- **Stanley Control Law**: $\delta (t) = \psi(t) + tan^{-1} (\frac{ke(t)}{v_f(t)}), \delta (t)  \in [\delta_{min}, \delta_{max}]$
		- $k$ is the proportional gain that is tuned
		- $e(t)$ is the crosstrack error
		- $\psi(t)$ is the heading error
		- From the error dynamic analysis of the stanley control law, we can see that the controller will exhibit an exponential decaying error control toward the reference path. 
			- For a large initial crosstrack error, For slower vehicles, there will be more aggressive steering angle before exhibiting an exponentially decaying steering profile to converge toward the reference path, for faster vehicles, the decay takes longer and steering angles will not be as aggressive to safely converge onto the path as well
			- For small crosstrack but large heading error (vehicle pointing in wrong direction on reference path), the controller will drive away from the reference path in a circle at large steering angle to correct its heading, then as crosstrack error grows, it enters the exponentially decaying phase of the crosstrack error and steeering profile
	- Adjustment factors can be added to the PP and stanley control laws to allow smoother and safer operation for the riders and aggressiveness
- Dynamic Controllers

	- Model Predictive Controller (MPC): performs a finite horizon optimization to identify the control command to apply. MPC is commonly used because of its ability to handle a wide variety of constraints and to identify optimized solutions that consider more than just the current errors.
- Lateral Controller Error Terms:
	- **Heading Error**: Angle error between heading of vehicle and heading of reference path. Desired heading is zero
	- **Crosstrack Error**: Distance from the center of front axle to the closest point on path, crosstrack error line should be perpendicular to reference path at the closest point. Closest point can be computed in a variety of ways mathematically, not in scope
	- The combination and minimization of the above errors together ensures our vehicle stays exactly on the reference path without being parallel to it or straying away from it
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTcxNzYzMjQ4MCwtMTEzNjI0MTQ1NSw3NT
EyMDI4NDIsLTQzNDE3OTAyNSwtMTI3MTk4MTA0OCwtMTk2MDc1
OTQ5OSwxMjk0Njc3MTc3LDE3OTQ1MTk0NjEsNDIyNTEwMDQ2LC
0xNzc5OTY4MDc3LDEwMTQyNDMyNTAsLTE5MTE3MTk2NzAsMjEw
NTczOTQwMywtMTQxMTI5NDYxLC0xOTgwNDQ1MjA4LC00NDE2OD
k5ODksLTE5NTM4NDk4NCwtMTY0OTg0NTY5OSwtMzYyNTkzOTI5
LC03NzUyNjcxMTRdfQ==
-->