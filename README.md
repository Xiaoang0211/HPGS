## HPGS-ROS2

The current version is integrated with ROS2 for interaction with VSLAM systems

Compared to the baseline method (GSFusion) the new features are:

+ Additional keypoints from the VSLAM System for Gaussian primitive initialization.
+ Faster method for quadtree generation.
+ Growth control for the keypoint-based Gaussian primitives.

Other new implementation features:
+ New output for the render function: projected 2D Gaussian positions on the image plane.
+ ROS2 Support for integration with VLSLAM Systems.
+ Buffer for depth images