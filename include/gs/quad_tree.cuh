#ifndef GS_QUAD_TREE_HPP
#define GS_QUAD_TREE_HPP

#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

namespace gs {

// Quadtree node dividing points until at most one per leaf
class Node {
public:
    Node(int x0, int y0, int width, int height, const std::vector<std::tuple<Eigen::Vector2f, Eigen::Vector3f, float, int>>& keypoints)
        : x0_(x0), y0_(y0), width_(width), height_(height), keypoints_(keypoints)
    {}

    std::vector<Node> children;

    inline int getOriginX() const { return x0_; }
    inline int getOriginY() const { return y0_; }
    inline int getWidth()   const { return width_; }
    inline int getHeight()  const { return height_; }
    inline float getNodeSizeF() const { 
        return static_cast<float>(std::max(width_, height_)); 
    }

    inline void setNodeKeypoint(const Eigen::Vector2f& keypoint) { node_keypoint_ = keypoint; }
    inline void setNodeMappoint(const Eigen::Vector3f& mappoint) { node_mappoint_ = mappoint; }

    inline Eigen::Vector2f getNodeKeypoint() const { return node_keypoint_; }
    inline Eigen::Vector3f getNodeMappoint() const { return node_mappoint_; }
    const  std::vector<std::tuple<Eigen::Vector2f, Eigen::Vector3f, float, int>>& getPoints() const { return keypoints_; }
    bool   has_keypoint_{false};

private:
    int x0_, y0_, width_, height_;
    std::vector<std::tuple<Eigen::Vector2f, Eigen::Vector3f, float, int>> keypoints_;
    Eigen::Vector2f node_keypoint_;
    Eigen::Vector3f node_mappoint_;
};

// Quadtree managing root and building leaves
class QTree {
public:
    // Constructor
    QTree(int min_pixel_size,
          const cv::Mat& image,
          const std::vector<std::tuple<Eigen::Vector2f, Eigen::Vector3f, float, int>>& keypoints,
          int min_leaves,
          bool post_subdivide);

    // Build quadtree: subdivide until leaves have <= 1 keypoint
    void subdivide();

    // Retrieve all leaf pointers (each with 0 or 1 keypoint)
    const std::vector<Node*>& getAllLeafPtrs() const { return all_leaves_ptrs_; }

    // Render quadtree: draw rectangles over leaves on provided image
    void renderImg(cv::Mat& image, int thickness = 1, cv::Scalar color = cv::Scalar(0,0,255));

    // Visualize before/after: shows original and overlayed rectangle image
    void visualize(const cv::Mat& image, int thickness = 1, cv::Scalar color = cv::Scalar(0,0,255));

private:
    Node root_;
    int min_pixel_size_;
    int min_leaves_;
    bool post_subdivide_;

    // Store pointers to leaf nodes for fast access
    std::vector<Node*> all_leaves_ptrs_;
    std::unordered_map<Node*, size_t>    leaf_index_;

    // Utilities
    inline int quadrantIndex(const std::tuple<Eigen::Vector2f, Eigen::Vector3f, float, int>& p, 
                             int xm, int ym);
    inline std::array<int,4> childBounds(int idx, int x0, int y0, int w, int h);
    void subdivideLeaf(Node& node);
    inline void addLeaf(Node* p);
    inline void removeLeaf(Node* p);

    std::vector<Node*> find_leaves_ptr(Node& node);
    void recursive_subdivide(Node& node);
};

} // namespace gs

#endif // GS_QUAD_TREE_HPP