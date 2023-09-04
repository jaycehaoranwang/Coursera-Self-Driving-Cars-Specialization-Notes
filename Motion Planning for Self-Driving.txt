# Motion Planning for Self-Driving Cars
## The Planning Problem
- High Level Autonomous Driving Mission Planning is to navigate from point A to point B on a map. Low-level details are abstracted away. Goal is to find most efficient path (in terms of time or distance)
- Road Structure Scenarios
	- Driving Straight and following in a lane
	- Lane change manoeuvres
	- Left and right turn scenarios at intersections and drivelanes 
	- U-Turns if possible/legal
- Obstacle Scenarios
	- Static obstacles restrict which locations our path can occupy
	- Dynamic obstacles require maintaining safe gap while executing manoeuvres (cars, pedestrians, cyclists...etc)
	- Different time windows exist depending on locations and speed and other dynamic obstacles, and need to use estimation and prediction to calculate these windows of opportunity
- Required High Level Behaviours (not exhaustive):
	- Speed Tracking
	- Deceleration to Stop
	- Staying Stopped
	- Yielding
	- Emergency Stop
- Challenges include considering  edge cases where predicted agents/dynamic variables violate their predicted rules (jaywalking, motorcyclists lane splitting)
- We __must__ break down the driving mission and scenarios into hierarchy of optimization problems to enable different objectives, constraints, optimization and solutions taliored to the correct scope and level of abstraction
	
### Motion Planning Constraints
 - Kinematics Simplified to bicycle model, which imposes __non-holonomic Curvature constraints__ on our path planning process. 
	 - A non-holonomic constraint refers to a limitation on the possible motions of a system that cannot be fully described by a set of independent scalar equations, or equivalently, cannot be integrated into a set of closed-form constraints on the system's coordinates and velocities. making the control and planning much more complex. 
 - Vehicle Dynamics require manoeuvres that keep the friction force of the tires of the car to be within the friction ellipse, which denotes the maximum magnitude of the tire forces before stability loss. For non-emergency situations, we often use the comfort rectangle range of the friction ellipse as the constraint for the vehicle dynamics/longitudinal or lateral acceleration
	 - When we are generating a velocity profile for our autonomous vehicle, we have to take the curvature of the path as well as the vehicles maximum lateral acceleration into consideration. Velocity and curvature relation equations are in the slides.
- Static Obstacles constrain portions of the ego vehicle's drivable workspace, and require purposed vehicle paths to check for collision with static obstacles and be altered to safely avoid them
- Dynamic Obstacles require additional prediction of dynamic variable paths to assess a feasible path for our ego vehicle. However we must tradeoff a super conservative approach that leads to no solutions if we overestimate the possible paths of predicted dynamic obstacles versus an unsafe under estimated approach for possible behaviorals of dynamic obstacles
- Rules of the Road and Regulatory elements
### Objective Functions for Motion Planning
- Efficiency:
	- __Path Length__: Minimize the arc length (distance of path parameterized as arcs) of a path to generate the shortest path to the goal. Also includes reference tracking that penalize deviation from the reference path or speed profile
	- __Travel Time__: Minimize time to destination while following the planned path while minimizing the the deviation from the reference speed profile using __Hinge Loss__ to penalize speed limit violations severely. The hinge loss only activates when we overspeed.
- __Smoothness__: Want to minimize the amount of jerk along our trajectory by computin the integral over the path to calculate and minimize the accumulated absolute value of jerk 
- __Curvature__: Recall that paths with high curvature constrained the maximum velocity along the path to a low value in order to remain within the friction ellipse of the vehicle. To ensure that we avoid points of high curvature along our path, we need to formulate some penalty for large absolute curvature values, and objective function that is commonly used to represent this penalty is known as the bending energy of the path. Essentially, it is the square curvature integrated along the path, where once again Kappa denotes the curvature. This objective distributes the curvature more evenly along the path, preventing any one along the path from reaching too high a total curvature value. __This makes the ride more comfortable for passengers__
### Hierarchical Motion Planning
- __Mission Planner__ (Map-level navigation) 
	- Focuses on selecting feasible paths for navigation from start to destination
	- Can be solved with graph-based methods (Dijkstra's, A*)
- __Behavioural Planner__ (Focuses on other agents, rules of the road, driving behaviours, deciding on manoeuvres)
	- The behavior planner is a portion of the motion planner that focuses on high level decision making required to follow the rules of the road, and recognize which maneuvers are safe to make in a given driving scenario
	- Deciding how to take the input and decide on a decision is an active area of research. Currently, there are 3 prevailing architectures:
		- __Finite State Machines__: States are the possible decisions to make based on perception of surroundings, and Transitions are based on inputs from the driving scenario
			- FSMs are memoryless but straightforward to implement
		- __Rule-Based System__: Uses a hierarchy of rules to determine output behaviour, rules are evaluated based on input from driving scenario and go through all the rules considering  priority of rules to make a final decision.
			- Rule-based system can be complex to maintain, and ensure logical coherence, and does not scale well to unseen/unconsidered scenarios
		- __Reinforcement Learning__: Active area of research but has the potential to be very robust to unseen scenarios as well as common scenarios, but can be a black box method depending on implementation
- __Local Planner__: Path Planner, Velocity Profile Generator (Focuses on generating feasible, collision-free paths for both steering and acceleration profiles)
	- Given the decided manoeuvre from the behavioural planner, generate feasible, collision-free paths and comfortable velocity profiles to execute the manoeuvre. There are three main categories of path planners:
		- __Sampling-Based Planners__: Randomly sample the control inputs to quickly explore the workspace. Collision-checking is performed as new points are added to the explored space. Often very fast but can generate poor-quality paths. E.g the Rapidly exploring random tree (RRT) algorithm by generating points in randomly sampled locations and finding paths to the desired location from the current node that are collision free to create a big tree of possible paths.
		- __Variational Planners__: Variational planners combine both path planning and velocity planning into a single step and contains cost penalities for collision avoidance and robot dynamics. Is generally slower, more complex and their convergence to a feasible path is sensitive to the initial conditions. (E.G> Chomp algorithm, out of scope of this course)
		- __Lattice Planners__: Lattice planners constrain the search space by limiting the actions that the ego vehicle can take at any point in the workspace. This set of actions is known as the control set of the lattice planner. This control set, when paired with a discretization of the workspace, implicitly defines a graph. This graph can then be searched using a graph search algorithm such as Dijkstra's or A*, which results in fast computation of paths. Obstacles can set edges that cross them to infinite cost. So the graph search allows us to perform collision checking as well.
			- A common variance on the lattice planner is known as the conformal lattice planner. Where a goal points are selected some distance ahead of the car, laterally offset from one another with respect to the direction of the road and a path is optimized to each of these points. The path that best satisfies some objective while also remaining collision free is then chosen as the path to execute.
	- Velocity profile generation is usually set up as a constrained optimization problem. Generally, we would combine many of the velocity profile objectives described previously such as the goals of minimizing jerk or minimizing deviation from a desired reference.
- __Vehicle Control__ (Low-level control to follow the given paths and velocity profiles) Low level controllers were introduced in the first course, such as PID controllers for lateral and longitudinal control. 
## Mapping for Planning
### Occupancy Grid Maps
- An occupancy grid is a discretized grid which surrounds the current ego vehicle position. This discretization can be done in two or three dimensions. Each grid square of the occupancy grid indicates if a static or stationary object is present in that grid location. If so, that grid location is classified as occupied. Each square of the occupancy grid noted by Mi, maps to a binary value in which one indicates that the square is occupied by a static object, and zero indicates that it is not.
- For self driving, we use LIDAR to help construct our occupancy grid for our vehicle.
	- We must first filter the LIDAR data to exclude: 
		- Points on the ground plane/road surface
		- Points above the highest point of the vehicle (as they won't affect our driving)
		- Lidar Points of Non-Static objects need to be identified and removed
	- Once we complete our lidar filtering, we project the LIDAR points down to a 2D plane to construct our 2D occupancy grid. This projected 2D data from the LIDAR effectively provides us with an accurate long range 2D range sensor measuring distance to static objects 
- We must account for LIDAR sensor/processing noise and map uncertainties using a __probabilistic occupancy grid__  instead: Instead of of cell i storing a binary value for occupied/not, now each cell will store __a probability between 0 and 1, corresponding to the certainty that the given square is occupied__.
	- To use this set of probabilities, the occupancy grid can now be represented as a belief map where for each cell, the belief over the current cell $m^i$ is equal to the probability that the current cell $m^i$ is occupied given the sensor measurements for that cell location Y and the vehicle location X: $\text{belief}_t(m^i) = p(m^i|(y,x))$ 
	- To convert from belief map back to a binary map, a threshold value can be established at which a given belief is confident enough to be classified as occupied.
	- We iteratively and recursively improve the belief map robustness at some time-step using measurements and vehicle location of multiple past time-steps mathematically represented as  $\text{belief}_t(m^i) = p(m^i|(y,x)_{1:t})$. 
	- In practise, the above is implemented and updated using __bayes theorem__ (and markov assumption that all information for estimating current cell occupancy is captured in the previous timesteps) at each update step for each cell: $\text{belief}_t(m^i) = \eta * p(y_t|m^i)*\text{belief}_{t-1}(m^i)$
		- $\text{belief}_{t-1}(m^i)$ comes from the previous belief map
		- $p(y_t|m^i)$ is the probability of getting a particular measurement given cell $m^i$ is occupied, which is the measurement model
		- $\eta$ is a normalizing constant to scale results to ensure it remains a probability distribution
	- __To avoid the issue of multiplying extremely small numbers/numerical instability for small probabilities__ we instead store the log odds ratio rather than the probability, by converting probabilities using the logit function $log(\frac{p}{1-p})$ which takes probability values from 0 to 1 and maps them to the entire real axis -inf to + inf and can easily be converted back to probabilities using a simple formula
		- Inserting the logit function into the bayes update rule via some derivation omitted here (available in slides/video), our final __Bayesian log odds update__ equation is:
		- $\text{logit}(p(m^i|y_{1:t})) = \text{logit}(p(m^i|y_{t})) + \text{logit}(p(m^i|y_{1:t-1})-\text{logit}(p(m^i))$
		-  $l_{t,i} = \text{logit}(p(m^i|y_{t})) + l_{t-1,i} - l_{0,i}$
			- The $\text{logit}(p(m^i|y_{t}))$ term is the logit formed using new measurement information. The probability distribution $p(m^i|y_t)$ given yt is known as the __inverse measurement model.__
			- $l_{t-1,i}$ is the previous belief at time t-1 for cell i
			-  $l_{0,i}$ is the initial belief at time zero for the same cell. The initial belief represents the baseline belief that a grid cell is occupied, which is usually set to 0.5 uniformly as we don't expect to have prior information that improves on this value.
		- The Bayesian Log odds update is __numerically stable__ and __computationally efficient__ as it relies exclusively on addition for updates instead of multiplication
- __Inverse Measurement Module__: For our bayesian log odds update rule, we need to know the state of the occupancy grid given a measurement: $p(m^i|y_t)$, Thus we need an inverse measurement model.
	- Detailed explanation and visualizations are in the lecture video and slides, refer to there for indepth explanation. 
	- At a high level, we start knowing that our lidar emits lasers 360degrees around the vehicle  at set angles between beams and each laser extends to a set range.
	- We construct a temporary occupancy grid that encompasses the maximum range of the beams in all directions
		- The coordinate frame for this measurement grid uses the occupancy grid map frame and so we define a position x1 t and x2 t and an orientation x3 t of the sensor in the occupancy grid frame. In practice, this occupancy grid frame is set to the vehicle frame and the map is transformed at each step based on our state estimates.
	- Within this temporary occupancy grid, we will have 3 distinct types of cells:
		- Cells in the __no information__ area where the beams cannot reach to collect information
		- Cells in the __low probability__ of object close to/around the car since the beam was able to pass through that area to get to its maximum range or hit a surface to reflect from
		- Cells in the __high probability__ of object area, which is always a barrier of cells between the no information area and low probability of object areas since the high probability area is where the beam reflected from and return a non-maximum range value.
	- Each cell contains values of __relative range__ and __relative bearing__ to the sensor/LIDAR in the temporary occupancy grid at the current timestep. __Relative range__ is simply euclidean distance and __bearing__ is simply calculated using tan trig ratios.
	- We can associate each cell to its closest relative LIDAR beam by computing the minimum error between its beam angle and cell bearing.
	- We now define two parameters $\alpha, \beta$, where 
		- $\alpha$ defines the affect range for high probability
		- $\beta$ defines the affected angle for low and high probability
	- These parameters allow us to formulate conditional statements in an algorithm to categorize each cell (written at high level, mathematical conditional statements in slides):
		- No Information: cell relative range is outside maximum lidar beam range OR outside the angle beta sized cone for the measurements/LIDAR beams associated with it
		- High probability: defines cells that fall within $\alpha/2$ of the range measurement and within $\beta/2$ of the angle of the measurement associated with it.
		- Low Probability: defined by cells that have a range less than the measured range minus $\alpha/2$ and lie within the $\beta$ sized cone about the measurement.
	- We can use a ray-tracing algorithm such as Bresenham's line algorithm to improve computational efficiency to avoid having to update every single cell in the temporary occupancy grid. Instead we perform updat eon each beam from the LIDAR rather then each cell on the grid.
		- This performs far fewer updates (ignores no information zone)
		- Much cheaper per computation
### Occupancy Grid Updates for Self-Driving Cars - Converting 3D Lidar to 2D data for occupancy grids
- __Filtering of 3D LIDAR Data__
	- __Downsampling__: Reducing/removing the number of redundant/overlapping LIDAR points to save on computation complexity. 
		- Downsampling algorithms readily available in PCL or OpenCV. Some example downsampling techniques include: systematic filter that keeps every nth point along the lidar scan ring, or apply image downsampling techniques in the range image, and search spatially in a 3D grid replacing collections of points with a single occupancy measurement.
	- __Removal of overhanging objects__: Trivially remove all LIDAR points that are above a given threshold of the height limit of the car as these do not affect our driving at all
	- __Removal of Ground Plane Points__: We want to remove points that fall on the ground plane we estimated using our perception modules as the car otherwise will assume they are static obstacles that impede movement. However there are challenges/complications to consider:
		- Differing Road geometries
		- Curbs/lane boundaries are ambiguous 
		- Small objects on the road that are obstacles are also hard to distinguish.
		- The best way to resolve these challenges is to leverage our semantic segmentation map, and remove all the points that fall on our segmentation map's identified drivable surface, leaving points for detected objects that are actually obstacles on the road 
	- __Removal of Dynamic Objects__: We must leverage the perception stack which must detect and track all dynamic objects in the scene. The 3D bounding box of the detected dynamic object is used to remove all the points in the affected area. A small threshold is also added to the size of the bounding box used to account for any small mistakes in the perception algorithm object location estimate increasing robustness of the point remove filter.
		- Challenges:
		- Some vehicles while detected as dynamic, may be actually static/parked and should be considered static obstacles instead of dynamic. 
			- To handle this issue, perception needs to use dynamic object tracks to identify those objects that are currently static
		- Due to computing time of perception stack, the dynamic object is only detected after some delay, resulting in the occupancy grid using out of date object positions that can lead to bounding box missing large portions of the lidar points on a dynamic vehicle
			- To handle this issue, we rely on predictions of the moving objects motion based on their object tracks, and __shift__ the bounding box forward accordingly along the predicted path before associating and removing dynmic object lidar points
- __Projection of LIDAR onto 2D Plane Simple Solution__:
	- Collapse all points by zeroing the z-coordinate
	- Then sum up the number of LIDAR points in each grid location, more points indicate greater chance of occupation of that grid cell, used as measure of occupancy belief
### High Definition Road Maps using Lanelets
- Lanelets are a data structure and an efficient representation framework that allows road maps to be efficiently organized, stored, and used with various operations
- A Lanelet map contains two key components: __lanelet element__, __intersection element__
- __Lanelet element__: stores all information connected to a small longitudinal segment of a lane on a road which it represents. Specifically a lanelet __Stores__:
	- Left and right boundaries of the given lane
		- Lane boundaries are stored as a set of points creating a continuous polygonal line. Each point stores its x,y, and z GPS coordinates. The distance between points can be as fine as a few centimeters, or as course as a few meters depending on the smoothness of the polyline in question.
			- This boundary point information allows us to easily calculate:
				- ordering of the points defines the direction of travel and heading for the lanelet
				- The road curvature of any part of the lanelet
				- A center line between the two boundaries can be interpolated, which can be used as the desired path of travel for the autonomous vehicle in that lane.
	- Any regulatory:
		-  __Elements__ that might be present __at the end__ of the lanelet element, such as a stop sign line or a static sign line. A decision must be made. Note that we only store a line for any regulatory element as this is the point thatthe autonomous vehicle treats as the active location for that regulatory element.
		- __Attributes__ that might affect the particular section of the road such as a speed limit, or whether this lanelet crosses another lanelet as in an intersection or merge
	- __Connectivity of itself to other lanelet elements around it__. This allows for easy traversal and calculations through the graph created by the set of lanelets in an HD map.
		- Each lanelet has four possible connections: 
			- Lanelets directly to the left, 
			- lanelets directly to the right, 
			- lanelet preceding it, 
			- lanelet following it. 
		- The entire lanelet structure is connected in a __directed graph__, where each directed edge has an index that denotes the relationship between the source to the target lanelet. It should be noted there could be more than one lanelet preceding the current lanelet, or more than one following lanelet as in the case of an intersection
	- Each lanelet ends as a regulatory element or a change to a regulatory attribute is encountered. This means that a lanelet element can be as short as only a few meters in the case of a lanelet, which is part of an intersection, or can be hundreds of meters long for a highway road segment.
- __Intersection Element__: stores all lanelet elements which are part of a single intersection for simple retrieval during motion planning tasks.
	- The intersection elements simply holds a pointer to all regulatory elements, which make up the intersection of interest. All lanelet elements which are part of an intersection also point to this intersection element.
- The best way to __create a lanelet map__ is to create it offline by driving the road network several times and collecting information and then fusing both segmentation as well as localization information to improve the accuracy of the map. Then update the lanelet map online during self-driving when new elements/changes are perceived
## Mission Planning in Driving Environments
### Creating a Road Network Graph
- We represent the road network/high level road map as a __directed__ graph where each node is an intersection and each edge is a one-way road way. Using a directed graph allows us to represent both one way and dual way roads.
	- We can easily add weights to edges based on live traffic, road distance...etc.
- Using the weighted directed graph to represent the road network, we can easily find the optimal path from one node to another using graph traversal algorithms like __Dijkstra's Algorithm__ (For a simple non-weighted graph, we can use __breadth-first search__).  Specific pseudocode for graph BFS is shown in lecture slides.
- When we leverage a heuristic in our search, we can get an even more computationally efficient search using the __A* shortest path algorithm__, which is similar to Dijkstra's algorithm but considers an additional heuristic function in the cost/weight beyond the pure weight of the edges. This allows us to avoid searching a large amount of useless edges.
	-  A __heuristic__ is an estimate of the remaining cost to reach the destination vertex from any given vertex in the graph. Of course, any heuristic we use won't be exact as that would require knowing the answer to our search problem already. Instead, we rely on the structure of the problem instance to develop a reasonable estimate that is __fast to compute__ and __exploits structure of the problem__.
		- For example, euclidean distance heuristic is a commonly used heuristic for path planning in mobile robotics.
		- Note that this estimate is always an __underestimate of the true distance__ to reach the goal, since the shortest path between any two points is a straight line. __This is an important requirement for A* search__, and heuristics that satisfy this requirement are called __admissible heuristics.__
- Note that for self-driving mission planning on roads, traffic, speed limits, and weather affect the planning such that __time estimate of driving down a road rather than distance is a better edge weight for mission planning road network graphs__.

## Dynamic Object Interactions
### Motion Prediction Overview
- Motion prediction attempts to estimate the future positions, headings, and velocities of all dynamic objects in the environment over some finite horizon.
	- Allows us to anticipate and plan a set of maneuvers to correctly interact with dynamic objects
	- Avoid collisions on a planned trajectory
- Requirements for Motion Prediction Models
	- Mandatory Requirements:
		- Identifying/knowing class of dynamic objects
		- Current position, heading, and velocity associated with each dynamic object
	- Optional Requirements:
		- History of the position, heading, and velocity
			- Requires object tracking between identifications over a set amount of time
		- Current high definition roadmap
		- Image of the current dynamic object
- We can make certain assumptions to simplify the motion prediction task:
	- Physics-based Assumptions: Vehicles must follow a set of physical constraints governing their movement.
	- Maneuver-based Assumptions: We assume that all vehicles on the road are made up of and can execute a finite set of maneuvers in a restricted domain in the driving environment. E.g. vehicles on the road will follow driving rules and execute a predictable set of maneuvers in most situations (lane change, left/right turns, going straight)
	- Interaction-aware Assumptions: Similar to maneuver based assumptions but we incorporate the assumption that the dynamic objects will react and interact with each other accordingly in a given scenario. 
- Similar assumptions can be made for pedestrians, but key differences to note:
	- Physics-based assumptions: Pedestrians have a very low top speed comparatively so their range of positions a pedestrian can reach in a short time frame is limited, but can change their direction/heading very quickly.
	- Maneuver-based assumptions: While pedestrians for the most part do not interact with vehicles as they walk on sidewalks and cross the road using intersections/designated crosswalks, must be ready for pedestrians to travel without following the rules when jaywalking..etc
	- Pedestrians ultimately have the right of way and it is the self-driving cars duty to stop when necessary.
### Map-Aware Motion Prediction
- Map-aware algorithms make two broad categories of assumptions to improve the motion predictions particularly for vehicles. __Position-based assumptions__ to improve the position component of the vehicle state, and __velocity-based assumptions__ to improve the velocity component.
- Positional Assumptions:
	- Vehicles driving down a given lane usually follow the given drive lane
	- Changing drive lanes is usually prompted by an indicator signal
- Velocity Assumptions:
	- Vehicles usually modify their velocity when approaching restrictive geometry (tight turns), thus when vehicles approach a turn with high curvature, it is likely to slow down to avoid exceeding lateral acceleration limits
	- Vehicles usually modify the velocity when approaching regulatory elements that we can identify using road maps/lanelets
- In a scenario where a dynamic vehicle has multiple possible paths it can take, we want to adopt a __multi-hypothesis prediction__ approach where we consider all possible paths/maneuvers and associate probabilities that the vehicle will take each path and update that possibility in real-time as we get new state inputs. __These probabilities can be learned from training data or be engineered and refined from real world testing.__
	- This approach is also safer and more robust to unexpected maneuvers from human drivers
- Important to keep in mind that vehicles do __not__ always stay within their lane, stop at regulartory elements, follow speed limits...etc. Further, they may react to information not yet available to the prediction system, such as a potholes in the road ahead or a bouncing ball. They may simply not observe a regulatory element as occurs when a vehicle accidentally runs a red light.
-  All these variations must be accounted for which can be done to some extent with the multi-hypothesis approach. The best approach is therefore to track the evolution of beliefs over the set of hypotheses, and to update based on evidence from the perception stack at every time step.
### Time to Collision
- The time to collision provides a valuable measure of behavioral safety in a self-driving vehicle, and is heavily used in assessing potential maneuvers, based on the current driving environment. We assume all dynamic objects continue along their predicted path for collision and time checking.
- Time to collision is comprised of:
	- Collision point between the two dynamic objects
	- Prediction of the time to arrive to the collision point
- Requirements for Accuracy:
	- Accurate estimation/predicted trajectories for all dynamic objects (position, heading and velocity)
	- Accurate estimation of dynamic objects geometries (3D bounding boxes/borders)
- __Two basic approaches to calculating time to collision__:
	- Simulation Approach
		- At each simulation time step, new position, heading, and occupancy extent predictions are calculated for every dynamic object. This is done by propagating forward the predicted trajectory. No rough estimations are made as exact calculations are done
		- Once the position of all dynamic objects is known for a given time in the simulated future, a check is conducted to determine if any two or more dynamic objects have collided with each other. If they have collided, the location and time to the collision is then noted.
			- The collision check is done through polygonal intersection analysis where the cars are estimated using a set number of circles (e.g. 3). Each circle/polygon representing one vehicle is cross checked with polygons in the other approaching vehicle to assess all possible collision scenarios.
			- The number of polygons used to estimate each vehicle determine our computational complexity (number of computations) versus accuracy tradeoff
		- Drawback is that this approach is __Computationally expensive__ but provide __higher accuracy if simulated with high fidelity__.
		- This method is better used in __offline applications (dataset evaluations or simulations)__
		- Simulation approach pseudocode is provided in the slides.
	- Estimation Approach
		- Estimation-based approaches function by computing the evolution of the geometry of each vehicle as it moves through its predicted path instead of having to compute new state information for every dynamic object. The results is a swath for each vehicle that can then be compared for potential collisions. Once the swath has been computed, their intersection can be explored for potential collision points by computing if any two or more dynamic objects will occupy the same space at the same time.
		- This method traditionally makes many simplifying assumptions to accelerate the calculations. These assumptions include: identifying collision point locations based on object path intersection points, estimating object space occupancy based on simple geometric primitives like bounding boxes, and estimating the time to reach a collision point based on a constant velocity profile.
		- This approach is __computationally inexpensive in terms of both memory footprint and computational time__ but is __less accurate__ due to approximations and estimations
		- This method is better used in __realtime applications__
## Principles of Behaviour Planning
- Behaviour Planner should have a set of Driving Maneuvers to execute, for example non-exhaustive list:
	- Track Speed: Maintain current speed of road
	- Follow Leader: Match the speed of the leading vehicle and maintain a safe distance
	- Decelerate to stop: Begin decelerating to stop before a given space
	- Stop: remain stopped in the current position
	- Merge: join or switch onto a new drive lane
- Behaviour Planner Outputs:
	- Driving maneuver to be executed
	- Set of constraints which must be obeyed by the planned trajectory of the self driving car:
		- Ideal Path
		- Speed Limit
		- Lane Boundaries
		- Future Stop Locations
		- Set of interest vehicles
- Input Requirements:
	- High definition road map
	- Mission path (navigation information)
	- Localization information
	- Perception Information:
		- All observed dynamic objects:
			- Predicted future movement
			- Collision points/time to collision
		- All observed static objects
		- Occupancy Grid
- A basic behaviour planner approach is to use a rule-based __Finite State Machine__
	- Each state is a driving maneuver + constraint
	- Transition conditions between states
	-  Advantage is that it is simple to implement and easy to check rules. But the complexity explosion for a large number of states and inability to handle uncertain states make it not scalable for advanced autonomy systems, but will work well for well constrained/simple driving environments.
- We can use a hierarchical FSM to account for and switch between lower-level FSM decision making for multiple scenarios. We represent each high-level scenario as a single state and make state transitions based on rules that dictate how high level scenarios change.
	- We can define states within the low level scenario FSM as key exit states that have the ability if conditions are met to exit out of the current scenario in the high level FSM and transition to a different scenario
	- Advantages include: Decrease in computation time and simpler to create and maintain
	- Disadvantages include: rule explosion and repetition of many rules in the low level state machines since eacch low level FSM is an individual FSM. Still hard to scale rapidly for uncertain scenarios
- Major State Machine/Rule-Based Behaviour Planner Issues:
	- Rule-explosion when dealing with complex scenarios
	- Dealing with a noisy environment/sensors/inputs
	- Hyperparameter Tuning for rules and transitions...etc
	- Incapable of dealing with Unencountered/scenarios/states that are not accounted for
 - Advanced Methods/Techniques for more Robust Behaviour Planning:
	 - Deal with environmental noise: Incorporating Fuzzy logic in  a system by which a set of crisp, well-defined values are used to create a more __continuous__ set of fuzzy states. Still suffers from rule-based restrictions
	 - __Reinforcement Learning for behaviour planning__:
		 - Hierarchical Reinforcement Learning
		 - Model-Based Reinforcement Learning 
		 - Inverse Reinforcement Learning
		 - End-to-End Approaches 
		 - Suffers from simulation fidelity/simplicity versus computational cost tradeoff, black box behaviours have safety concerns
## Reactive Planning in Static Environments
### Difference Between Kinematic and Dynamic Motion Models
- Particle Kinematic Model:
	- Disregards mass and inertia of the robot
	- Uses linear and angular velocities (and/or derivatives) as input
	- For path planning and trajectory optimization, we often focus on kinematic models to make the motion planning problem more computationally tractable and leave the issues raised by the simplification of the dynamics to the controller.
- Particle Dynamic Model
	- Takes mass and inertia into consideration
	- Uses forces and torques as inputs at the cost of being more complex
- Using given/derived equations of motion for a kinematic model and control inputs (velocities), we can calculate how a trajectory will evolve with time.
	- This is done in practise and computationally efficiently through discretizing the kinematic model differential equations and computing them recursively
	- By focusing on the discrete model, it allows us to easily and efficiently propagate trajectories for a given sequence of control inputs.
	- If we now apply this discretization and step through an entire control input sequence, we will get an accurate approximation of the trajectory that the robot will follow. While this is useful for trajectory planning, it is also useful for motion prediction, where we know a kinematic model of a different agent in the driving scenario and we have an educated guess on the control inputs they will take. From this, we can estimate their future trajectory and plan our motion to avoid collisions
### Collision Checking
- Swath Computation: In exact form, collision checking amounts to rotating and translating the footprint of a vehicle along every point of a given path. Each point in the footprint is rotated by the heading at each path point and translated by the position of that same path point.
	- After performing this for every point along the path, the resulting swath of the car along the path is given by the union of each rotated and translated footprint. We then check this entire set to see if there are any obstacles inside it.
	- In implementation, we must discretize the representation of the car's footprint and the path in terms of the occupancy grid. 
		- The cars footprint contains K points and the path is  N points long. Algorithmically, computing the swath requires us to rotate and translate all K points in the cars footprint, N times one for each point in the path.
		- Because this swath-based method is often computationally expensive, it is often useful when we are exploiting a lot of repetition in our motion planning algorithm as in the case in lattice planners. Since we are constrained to a small set of control actions each step of a lattice planner, we are also constrained to a small set of swaths, so their union can be pre-computed offline. When performing collision checking online, the problem reduces to multiple array lookups.
	- To mitigate imperfect information and computational requirements, we use conservative circular body approximations to collision checking instead of exact vehicle geometry points.
		- Using circles to approximate the body of a vehicle provides a good conservative collision buffer while being easy to check for collisions as we simply need to check if another object's points is within the radius of any body circle.
		- However, depending on circle approximation size, too conservative approximations may result in a stuck motion planner with no solution due to approximated collisions or generating a path that is not efficient. 
	- Discretization resolution along a path (number of points on a path) for computing swath computations also trades off collision checking accuracy with computational resources. 
### Trajectory Rollout Algorithm
- __Trajectory Set Generation__: First step is to generate a set of possible trajectories at each time step by uniformly sampling fixed inputs across the range of available input values. 
	- We can tune the number and range of inputs we sample from to tradeoff exploration/possible trajectory optimalities with computation overhead
- __Trajectory Propagation__: Using the sampled fixed inputs, we propagate future states along the trajectory by using the kinematic model. Using the two inputs velocity and steering angle, if we hold velocity constant and vary the steering angle across the range -pi/4 to pi/4, we can generate a set of arcs as candidate trajectories by evaluating the kinematic equations recursively 
	- Then check to see which of our arcs are collision free using the swath computation method assuming we are given an occupancy grid that represents a discretization of the vehicles workspace.
- __Trajectory Scoring__: Using our collision free kinematically feasible trajectories, we score each using our objective function to see which one is most optimal for our context. 
	- The primary element that every objective function needs is some way of rewarding progress towards some goal point or region, which is the ultimate goal of our motion planning problem. A simple and effective way to do this is to have a term in the objective function that is proportional to the distance from the end of the candidate trajectory to the goal node.
	- We want to consider additional behaviors in our objective function including minimizing curvature and deviation from lane/path centerline...etc
- The trajectory planning cycle in practise should be shorter than the trajectory length such that the initially calculated trajectory is not fully executed before new paths are generated to allow the vehicle flexibility in reacting to environment/context changes. We call this __receding horizon planner__ 
- This planner is __greedy__ (greedily sample sub paths until goal) and sub-optimal (short-sighted, get stuck in dead ends), but is fast enough to allow for online planning 
- We can apply __dynamic windows__ to constrain acceleration and change in steering angle to ensure a smoother ride but limits our manoeuvrability 

## Smooth Local Planning
- The local planner is the portion of the hierarchical planner that executes the maneuver requested by the behavior planner in a collision-free, efficient, and comfortable manner. 
	- This results in either a trajectory, which is a sequence of points in space at given times or 
	- a path and velocity profile, which is a sequence of points in space with the required velocities at each point.
- Fundamental Requirements of the path planning problem: Given a starting position, heading, and curvature, find a path to an ending position heading, and curvature that satisfies our kinematic constraints. 
	- In the context of an optimization, the starting and end values can be formulated as the boundary conditions of the problem, and the kinematic motion of the vehicle can be formulated as continuous time constraints on the optimization variables.
	- These boundary conditions will influence how we decide to set up the underlying structure of the optimization problem.
- We want to define our path as a parametric curve, which is a vector function that can be described by a set of parameterized equations where the parameter denotes path traversal, can be arc length or unitless
	- For example $\bold r(u) = <x(u), y(u)>, u \in [0,1]$ where $x(u) = \alpha_3u^3 + \alpha_2 u^2 + \alpha_1 u + \alpha_0$,  $y(u) = \beta_3u^3 + \beta_2 u^2 + \beta_1 u + \beta_0$
- For path optimization for autonomous driving, we often but not always require the path to be a parametric curve
	- This is because parameteric curves allow for optimizing over parameter space, which simplifies optimization formulation
	- In contrast, non-parametric paths represent the trajectory and the path with a sequence of points in space instead of a parametric representation. 
- The two primary parameterizations that we can select to represent our curves are __quintic splines__ or __cubic spirals__.
	- __Quintic Splines__: x and y are defined by 5th order splines (polynomials parameterized by our path parameter)
		- $x(u) = \alpha_5u^5 +\alpha_4u^4 +\alpha_3u^3 + \alpha_2 u^2 + \alpha_1 u + \alpha_0$,  
		- $y(u) =\beta_5u^5 +\beta_4u^4 + \beta_3u^3 + \beta_2 u^2 + \beta_1 u + \beta_0$
		- Advantage: 
			- Closed form solution is immediately available for the spline coefficients that satisfy the given $(x,y,\theta, \kappa)$ boundary conditions (x,y,heading,curvature) and is cheaper to evaluate than generating a path using an iterative optimization method/more computationally efficient
		- Disadvantage:
			- Hard to constrain curvature within a certain set of bounds as is often required in autonomous driving due to nature of spline's curvature and potential discontinuities. This is because looking at the derived the curvature equation from x(u) and y(u), the function will not in general be a polynomial, introducing discontinuities that make it difficult to approximately satisfy curvature constraints across the entire domain of the spline.
	- __Polynomial Spirals__: These curves offer a closed form equation for the curvature of the curve along each point of its arc length. Curvature as a function of arc length. For autonomous driving, it is common to select a cubic polynomial as our curvature function of arc length. (Spiral Equations written in lecture slides, omitted here)
		- Advantages:
			- Their structure is highly conducive to satify the approximate curvature constraints that are often required by the path planning problem. 
			- Since a spiral is a polynomial function of curvature, the curvature value will not change extremely quickly like it can in the case of quintic splines. This means we can constrain the curvature of only a few points in the spiral and the spiral will very likely satisfy the curvature constraints across the entire curve 
			- This is highly useful when performing path optimization, as the number of constraints greatly increases the computational effort of each optimization step.
		- Disadvantages:
			- Spiral position and heading does __not__ have a closed form solution. So an iterative optimization must be performed in order to generate a spiral that satisfies our boundary conditions. 
			- The position equations results in __Fresnel integrals__, which have no closed form solution. So numerical approximation is needed to compute the final end points of the spiral. We can approximate these Fresnel integrals using __Simpson's rule.__
		- Note we can also include the bending energy function as part of the cost/objective function to be optimized to ensure that curvature is evenly distributed along spiral to promote comfort
		- Full integration/implementation/addressing challenges example is given in lesson 2 of this module: Path Planning Optimization
### Conformal Lattice Planning
- The conformal lattice planner exploits the structured nature of roads, to speed up the planning process while avoiding obstacles. By focusing on only those smooth path options that swerve slightly to the left or right of the goal path, the conformal lattice planner produces plans that closely resemble human driving. In addition, this keeps the search space computationally tractable as we only care about paths that would result in forward progress to our current vehicle. 
- The conformal lattice planner chooses a central goal state of going straight on the road as well as a sequence of alternate goal states that are formed by laterally offsetting from the central goal state, with respect to the heading of the road.
- How do we select goal states for our planner to generate paths toward?
	- Short lookahead improves computation time, but reduces ability to avoid future obstacles/see longer into the time horizon. Can be very problematic at higher speeds where the car covers more distance between planning cycles
	- Goal point is dynamically calculated based on factors such as car speed and weather conditions.
	- For a simple implementation, you we take the goal point as the point along the center line of the lane, that is a distance ahead equal to the fixed goal horizon distance
- Once these goals states have been found, we can then calculate the __cubic spirals__ (using path optimization as previously described) required to reach each one of them. At this point, we don't worry about whether the paths are collision free, we just want kinematically feasible paths to each of our goal states and discard non kinematically feasible goal points
- Once the optimization produces the optimal parameters, undo any parameter transformations to ensure you have the optimized spiral coefficients of the spiral function. Then, we can use the known spiral function to sample points along the spiral to get a discrete representation of the entire path.
	- However, since we don't have a closed form solution of the position along the spiral, we need to perform efficient numerical integration using the __trapezoid rule__ which is more efficient than Simpson's rule for this context to get the points of the positions along the path.
- Once we have a set of paths, we can perform collision checking using circle based or swath-based methods to see which paths are infeasible due to collisions. Then from the remaining set of feasible, collision-free paths, we select the best path based on some reward/objective function that defines the desireability of paths
- This process repeats to continually generate paths that can avoid obstacles while converging to a desired goal point in the road toward where we want the car to go.
### Velocity Profile Generation

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE4NDAyMTYzODAsOTAzOTAyNjAxLC0yMz
E5ODc0OTAsMTU1MDU2ODc5NywtNzcxMjA4NzksLTM3MTQ4ODUw
NCwtNTg3NDg0NTM4LC0zMTk0NjkyNiwtMTUyMjgyMzkwMCwxNz
E3MjYyNzcsLTQ0MzQ1OTExNiwxOTk0NjEwNjI1LDEzMDE2MzY0
NzIsLTIwMTQ4ODcwNTIsMTE2NTcwNDIzNiw1MzI4ODUyODQsLT
E4NDkzNTMxOTcsMjAxNjI1MTc0OCwtMzU5MDIyOTU4LDU0ODA5
NDkxXX0=
-->