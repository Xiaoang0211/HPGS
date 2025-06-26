## HPGS2

The this is the branch for our novel densification approach. ROS2 is not yet integrated. We call it HPGS2 for now.

Compared to the baseline method (GSFusion) the new features are:

+ Depth rendering within the alpha-blending pipeline.
+ Depth loss that ignores invalid pixels in the ground truth depth image.
+ Integration of ["Revising Densification in Gaussian Splatting"](https://arxiv.org/abs/2404.06109).

Other new implementation features:
+ New output for the render function: projected 2D Gaussian positions on the image plane.
+ New output for the render function: a "pseudo image", from which we derive the densification metric based on pixel-error.
+ Buffer for depth images.