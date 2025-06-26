#include "gs/quad_tree.cuh"
#include <tuple>
#include <array>
#include <queue>

namespace gs {

struct LeafEntry {
    Node* node;
    size_t area;
    bool operator<(const LeafEntry& rhs) const { return area < rhs.area; }
};

QTree::QTree(int min_pixel_size,
                const cv::Mat& image,
                const std::vector<std::tuple<Eigen::Vector2f, Eigen::Vector3f>>& keypoints,
                int min_leaves,
                bool post_subdivide)
: root_(0, 0, image.cols, image.rows, keypoints),
  min_pixel_size_(min_pixel_size),
  min_leaves_(min_leaves),
  post_subdivide_(post_subdivide)
{
    all_leaves_ptrs_.reserve(min_leaves_);
}

// helper to add in O(1)
inline void QTree::addLeaf(Node* p) {
    leaf_index_[p] = all_leaves_ptrs_.size();
    all_leaves_ptrs_.push_back(p);
}

// helper to remove in O(1)
inline void QTree::removeLeaf(Node* p) {
    size_t idx = leaf_index_[p];
    Node* back = all_leaves_ptrs_.back();
    all_leaves_ptrs_[idx]      = back;
    leaf_index_[back]          = idx;
    all_leaves_ptrs_.pop_back();
    leaf_index_.erase(p);
}

void QTree::subdivide() {
    // 1) clear old leaves/map
    all_leaves_ptrs_.clear();
    leaf_index_.clear();

    // 2) initial pass: fill leaves via recursive_subdivide()
    //    make sure recursive_subdivide calls addLeaf(&node)
    recursive_subdivide(root_);

    if (!post_subdivide_)
        return;

    size_t curr_leaves = all_leaves_ptrs_.size();

    // Keep doing “rounds” until we reach min_leaves_ or no more splits are possible
    while (curr_leaves < size_t(min_leaves_)) {
        // Take a snapshot of the *existing* leaves before this round
        std::vector<Node*> round = all_leaves_ptrs_;
        bool anySplit = false;

        for (Node* leaf : round) {
            // stop early if we hit min_leaves_
            if (curr_leaves >= size_t(min_leaves_))
                break;

            // only split if big enough and still a leaf
            if (leaf->children.empty() &&
                leaf->getWidth()  > min_pixel_size_ &&
                leaf->getHeight() > min_pixel_size_)
            {
                // remove this leaf
                removeLeaf(leaf);

                // split it, adding 4 children
                subdivideLeaf(*leaf);
                curr_leaves += 3;  // one leaf → four children = +3 net

                // add the brand-new children to the global leaf list
                for (auto& child : leaf->children) {
                    addLeaf(&child);
                }

                anySplit = true;
            }
        }

        // if we made no progress in this entire round, give up
        if (!anySplit)
            break;
    }
}

void QTree::subdivideLeaf(Node& node) {
    if (!node.children.empty()) return;
    int x0 = node.getOriginX();
    int y0 = node.getOriginY();
    int w = node.getWidth();
    int h = node.getHeight();

    const auto& pts = node.getPoints();
    int xm = x0 + w/2, ym = y0 + h/2;
    std::vector<std::tuple<Eigen::Vector2f, Eigen::Vector3f>> bins[4];
    for (const auto& p : pts) {
        bins[quadrantIndex(p, xm, ym)].push_back(p);
    }
    for (int i = 0; i < 4; ++i) {
        auto b = childBounds(i, x0, y0, w, h);
        node.children.emplace_back(b[0], b[1], b[2], b[3], bins[i]);
    }

    // Make sure keypoints are marked in children!
    for (auto& child : node.children) {
        const auto& pts = child.getPoints();
        if (pts.size() == 1) {
            child.has_keypoint_ = true;
            child.setNodeKeypoint(std::get<0>(pts[0]));
            child.setNodeMappoint(std::get<1>(pts[0]));
        }
    }
}

void QTree::renderImg(cv::Mat& image, int thickness, cv::Scalar color) {
    if (all_leaves_ptrs_.empty()) subdivide();
    for (Node* leaf : all_leaves_ptrs_) {
        int x = leaf->getOriginX();
        int y = leaf->getOriginY();
        int w = leaf->getWidth();
        int h = leaf->getHeight();
        cv::rectangle(image,
                        cv::Point(x, y),
                        cv::Point(x + w, y + h),
                        color, thickness);
    }
}

void QTree::visualize(const cv::Mat& image, int thickness, cv::Scalar color) {
    cv::Mat before = (image.channels() == 1 ? cv::Mat() : cv::Mat());
    if (image.channels() == 1)
        cv::cvtColor(image, before, cv::COLOR_GRAY2BGR);
    else
        before = image.clone();

    for (const auto& p : root_.getPoints()) {
        Eigen::Vector2f p_2d = std::get<0>(p); // indices 0,1
        cv::circle(before,
                    cv::Point2f(p_2d.x(), p_2d.y()),
                    thickness + 2,
                    cv::Scalar(0, 0, 255),
                    cv::FILLED);
    }

    cv::imshow("Quadtree Before", before);

    cv::Mat over = (image.channels() == 1 ? cv::Mat() : cv::Mat());
    if (image.channels() == 1)
        cv::cvtColor(image, over, cv::COLOR_GRAY2BGR);
    else
        over = image.clone();

    renderImg(over, thickness, color);
    cv::imshow("Quadtree After", over);
    cv::waitKey(0);
}

inline int QTree::quadrantIndex(const std::tuple<Eigen::Vector2f, Eigen::Vector3f>& p, 
                                int xm, int ym) {
    Eigen::Vector2f p_2f = std::get<0>(p);
    return (p_2f.x() >= xm ? 1 : 0) + (p_2f.y() >= ym ? 2 : 0);
}

inline std::array<int,4> QTree::childBounds(int idx, int x0, int y0, int w, int h) {
    int w1 = w/2, h1 = h/2;
    switch (idx) {
        case 0: return {x0, y0, w1, h1};
        case 1: return {x0 + w1, y0, w - w1, h1};
        case 2: return {x0, y0 + h1, w1, h - h1};
        case 3: return {x0 + w1, y0 + h1, w - w1, h - h1};
    }
    return {0,0,0,0};
}

void QTree::recursive_subdivide(Node& node) {
    const auto& pts = node.getPoints();
    if (pts.size() <= 1 ||
        node.getWidth() <= min_pixel_size_ ||
        node.getHeight() <= min_pixel_size_) {
        all_leaves_ptrs_.push_back(&node);
        if (pts.size() == 1) {
            node.has_keypoint_ = true;
            node.setNodeKeypoint(std::get<0>(pts[0])); // set Vector2f
            node.setNodeMappoint(std::get<1>(pts[0])); // set Vector3f
        }
        return;
    }

    int x0 = node.getOriginX(), y0 = node.getOriginY();
    int w = node.getWidth(), h = node.getHeight();
    int xm = x0 + w/2, ym = y0 + h/2;

    std::vector<std::tuple<Eigen::Vector2f, Eigen::Vector3f>> bins[4];
    for (const auto& p : pts) {
        bins[quadrantIndex(p, xm, ym)].push_back(p);
    }

    for (int i = 0; i < 4; ++i) {
        auto b = childBounds(i, x0, y0, w, h);
        node.children.emplace_back(b[0], b[1], b[2], b[3], bins[i]);
    }

    for (auto& c : node.children)
        recursive_subdivide(c);
}

} // namespace gs