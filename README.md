## HPGS-ROS2

The current version is integrated with ROS2 for interaction with VSLAM systems

Compared to the baseline method ([GSFusion](https://github.com/smartroboticslab/GSFusion)) the new features are:

+ Additional keypoints from the VSLAM System for Gaussian primitive initialization.
+ Faster method for quadtree generation.
+ Growth control for the keypoint-based Gaussian primitives.

Other new implementation details:
+ New output for the render function: projected 2D Gaussian positions on the image plane.
+ ROS2 Support for integration with VSLAM Systems.
+ Buffer for depth images.

Note that this branch only has a listener and there is no ROS2-free main function. Therefore it must used togather with a VSLAM system.