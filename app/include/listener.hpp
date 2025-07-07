#pragma once
#include <opencv2/imgproc.hpp>
#include <se/supereight.hpp>
#include <thread>
#include <mutex>
#include <memory>
#include <condition_variable>
#include <tuple>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <torch/torch.h>
#include <Eigen/Dense>
#include <cassert>
#include <cmath>

#include "config.hpp"
#include "gs/gaussian.cuh"
#include "gs/gaussian_utils.cuh"
#include "reader.hpp"
#include "se/common/filesystem.hpp"
#include "se/common/system_utils.hpp"

// ROS2 includes
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/bool.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <cv_bridge/cv_bridge.h>

// ROS2 image transport
#include <image_transport/image_transport.hpp>
#include <image_transport/subscriber_filter.hpp>

// message filters for synchronized subscription
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>

namespace fs = std::filesystem;


template<typename MessageType>
struct Queue
{
    std::queue<MessageType> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cond_var_;
    std::thread thread_;
    bool stop_flag_ = false;

    // Get current queue length (thread-safe)
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
};

class VSLAMListener : public rclcpp::Node
{
public:
    VSLAMListener(const std::string& node_name,
                  const std::string& cfg_file_path,
                  const int integration_rate,
                  std::shared_ptr<se::TSDFColMap<se::Res::Single>> map,
                  std::shared_ptr<const se::PinholeCamera> sensor,
                  std::shared_ptr<gs::GaussianModel> gs_model,
                  std::shared_ptr<std::vector<gs::Camera>> gs_cam_list,
                  std::shared_ptr<std::vector<torch::Tensor>> gt_img_list,
                  std::shared_ptr<std::vector<torch::Tensor>> gt_depth_list,
                  std::shared_ptr<gs::DataQueue> data_queue,
                  std::shared_ptr<unsigned int> frame,
                  std::shared_ptr<float> mean_fps);

    ~VSLAMListener() override;

    bool readConfigFile(const std::string& filePath);
    bool vslam_activated_=false;
    bool last_frame_received_=false;
    bool online_optimization_finished_=false;

    void initSync();

    // For gs mapping thread
    void insertGSMappingInput(const Eigen::Matrix4f& T_WS, 
                              const se::Image<se::rgb_t>& input_colour_img, 
                              const se::Image<float>& input_depth_img,
                              const std::vector<std::tuple<Eigen::Vector2f, Eigen::Vector3f, float, int>>& keypoints, 
                              const int current_frame_id);
    void checkTerminationCondition();
    void Start();
    void Stop();
    void set_training_views_list_path(const std::string& path) {
        training_views_list_path_ = path;
        // Open the training views list file for writing
        training_views_list_.open(training_views_list_path_, std::ios::out | std::ios::trunc);
        if (!training_views_list_.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open training_views_list.txt for writing.");
        }
    }

    void close_log_files() {
        // Close the training views list file if it is open
        if (training_views_list_.is_open()) {
            training_views_list_.close();
            RCLCPP_INFO(this->get_logger(), "Closed training_views_list.txt successfully.");
        } else {
            RCLCPP_WARN(this->get_logger(), "training_views_list.txt was not open.");
        }
        // Additional other log files can be closed here if needed
    }

private:

    // training views list file and path
    std::ofstream training_views_list_;
    std::string training_views_list_path_;

    // qos
    rmw_qos_profile_t sub_qos_profile_;

    // member objects
    std::shared_ptr<se::TSDFColMap<se::Res::Single>> map_; // tsdf volumetric map
    std::shared_ptr<const se::PinholeCamera> sensor_; 
    std::shared_ptr<gs::GaussianModel> gs_model_; 

    // member variables
    bool use_gt_pose_;
    bool has_logged_mapping_start_ = false;
    bool readin_config_ = false;
    int integration_rate_;
    std::shared_ptr<std::vector<gs::Camera>> gs_cam_list_; 
    std::shared_ptr<std::vector<torch::Tensor>> gt_img_list_; 
    std::shared_ptr<std::vector<torch::Tensor>> gt_depth_list_; 
    std::shared_ptr<gs::DataQueue> data_queue_; 
    std::shared_ptr<unsigned int> frame_;
    std::shared_ptr<float> mean_fps_;

    // publisher/subscriber topics names
    std::string vslamDriverTopic;
    std::string ackTopic;
    std::string imageFilenameTopic;
    std::string rgbImageTopic;
    std::string depthImageTopic;
    std::string mapPointsTopic;
    std::string keypointsTopic;
    std::string odometryTopic;
    std::string odometryGTTopic;
    std::string lastFrameFlagTopic;
    std::string lastFrameAckTopic;

    // handshake publisher and subscriber
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr ack_pub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr vslam_driver_sub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr last_frame_ack_pub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr last_frame_flag_sub_;

    // subscribers
    image_transport::SubscriberFilter rgb_image_sub_;
    image_transport::SubscriberFilter depth_image_sub_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>> map_points_sub_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>> keypoints_sub_;
    std::shared_ptr<message_filters::Subscriber<nav_msgs::msg::Odometry>> odometry_sub_;
    std::shared_ptr<message_filters::Subscriber<nav_msgs::msg::Odometry>> odometry_gt_sub_;

    // message synchronizer for all subscribed messages             
    using msgSyncPolicy = message_filters::sync_policies::ExactTime<sensor_msgs::msg::Image,
                                                                    sensor_msgs::msg::Image,
                                                                    sensor_msgs::msg::PointCloud2,
                                                                    nav_msgs::msg::Odometry,
                                                                    nav_msgs::msg::Odometry>;
    using msgSyncPolicyTmp = message_filters::sync_policies::ExactTime<sensor_msgs::msg::PointCloud2,
                                                                       nav_msgs::msg::Odometry,
                                                                       nav_msgs::msg::Odometry>;
                                                        

    std::shared_ptr<message_filters::Synchronizer<msgSyncPolicy>> msg_sync_;
    std::shared_ptr<message_filters::Synchronizer<msgSyncPolicyTmp>> msg_sync_tmp_;

    // mutex for the callback function
    std::mutex callback_mutex_;
    void driverCallback(const std_msgs::msg::Bool& msg);
    void msgCallback(const sensor_msgs::msg::Image::ConstSharedPtr& rgb_image_msg,
                     const sensor_msgs::msg::Image::ConstSharedPtr& depth_image_msg,
                     const sensor_msgs::msg::PointCloud2::ConstSharedPtr& keypoints_msg,
                     const nav_msgs::msg::Odometry::ConstSharedPtr& odometry_msg,
                     const nav_msgs::msg::Odometry::ConstSharedPtr& odometry_gt_msg);
    
    void msgCallbackTmp(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& keypoints_msg,
                        const nav_msgs::msg::Odometry::ConstSharedPtr& odometry_msg,
                        const nav_msgs::msg::Odometry::ConstSharedPtr& odometry_gt_msg);
    void lastFrameFlagCallback(const std_msgs::msg::Bool::ConstSharedPtr& last_frame_flag_msg);

    fs::path rgb_image_base_;
    fs::path depth_image_base_;
    
    // helper functions for type casting input data of GSFusion
    Eigen::Matrix4f odomToMatrix(const nav_msgs::msg::Odometry::ConstSharedPtr& msg);
    void setDepthImage(se::Image<float>& depth_image, cv::Mat& depth_data);
    void setColourImage(se::Image<se::rgb_t>& colour_image, cv::Mat& colour_data);
    void gsOnlineMapping(const Eigen::Matrix4f& T_WS, 
                         const se::Image<se::rgb_t>& input_colour_img, 
                         const se::Image<float>& input_depth_img,
                         const std::vector<std::tuple<Eigen::Vector2f, Eigen::Vector3f, float, int>>& keypoints,
                         const int current_frame_id);
    void pointcloud2_to_eigen(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg,
                              std::vector<std::tuple<Eigen::Vector2f, Eigen::Vector3f, float, int>>& points2d3d);

    // For GSFusion online optimization thread
    Queue<std::tuple<Eigen::Matrix4f, 
                     se::Image<se::rgb_t>, 
                     se::Image<float>, 
                     std::vector<std::tuple<Eigen::Vector2f, Eigen::Vector3f, float, int>>, 
                     int>> gs_mapping_queue_;

    template<typename MessageType, typename Function>
    void startThread(Queue<MessageType>& q, Function function) 
    {
        q.thread_ = std::thread([&q, function]() {
            while (true) {
                std::unique_lock<std::mutex> lock(q.mutex_);
                q.cond_var_.wait(lock, [&q]() {
                return !q.queue_.empty() || q.stop_flag_;
                });
                if (q.stop_flag_ && q.queue_.empty()) break;
                auto msg = std::move(q.queue_.front());
                q.queue_.pop();
                lock.unlock();
                std::apply(function, msg);
            }
        });
    }

    template<typename MessageType>
    void stopThread(Queue<MessageType>& q) {
        {
            std::lock_guard<std::mutex> lock(q.mutex_);
            q.stop_flag_ = true;
        }
        q.cond_var_.notify_all();
        if (q.thread_.joinable()) q.thread_.join();
    }
};