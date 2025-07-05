/*
* SPDX-FileCopyrightText: 2021 Smart Robotics Lab, Imperial College London, Technical University of Munich
* SPDX-FileCopyrightText: 2021 Nils Funk
* SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
* SPDX-FileCopyrightText: 2024 Smart Robotics Lab, Technical University of Munich
* SPDX-FileCopyrightText: 2024 Jiaxin Wei
* SPDX-License-Identifier: BSD-3-Clause
*/

#include "listener.hpp"
#include "gs/eval_utils.cuh"

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60
using namespace std::chrono_literals;

void printProgress(double percentage)
{
    int val = (int) (percentage * 100);
    int lpad = (int) (percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
    if (val == 100) {
        printf("\n");
    }
    fflush(stdout);
}

VSLAMListener::VSLAMListener(const std::string& node_name,
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
                            std::shared_ptr<float> mean_fps)
: Node(node_name), 
  integration_rate_(integration_rate), 
  map_(map), 
  sensor_(sensor), 
  gs_model_(gs_model), 
  gs_cam_list_(gs_cam_list), 
  gt_img_list_(gt_img_list), 
  gt_depth_list_(gt_depth_list),
  data_queue_(data_queue), 
  frame_(frame), 
  mean_fps_(mean_fps)
{  
    readin_config_ = readConfigFile(cfg_file_path);

    declare_parameter<std::string>("image_transport", "compressed");

    sub_qos_profile_ = rmw_qos_profile_default;
    sub_qos_profile_.reliability = RMW_QOS_POLICY_RELIABILITY_RELIABLE;
    sub_qos_profile_.durability = RMW_QOS_POLICY_DURABILITY_VOLATILE;
    sub_qos_profile_.depth = 10;

    // ROS2 message publishing and subscription
    // For handshake with vslam
    vslam_driver_sub_ = this->create_subscription<std_msgs::msg::Bool>(vslamDriverTopic, 10, std::bind(&VSLAMListener::driverCallback, this, std::placeholders::_1));
    ack_pub_ = this->create_publisher<std_msgs::msg::String>(ackTopic, 10);

    // handshake for finishing online optimization
    last_frame_flag_sub_ = this->create_subscription<std_msgs::msg::Bool>(lastFrameFlagTopic, 10, std::bind(&VSLAMListener::lastFrameFlagCallback, this, std::placeholders::_1));
    last_frame_ack_pub_ = this->create_publisher<std_msgs::msg::String>(lastFrameAckTopic, 10);

    // synchronized subscription for all subscribed messages
    keypoints_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>>(this, keypointsTopic, sub_qos_profile_);
    odometry_sub_ = std::make_shared<message_filters::Subscriber<nav_msgs::msg::Odometry>>(this, odometryTopic, sub_qos_profile_);
    odometry_gt_sub_ = std::make_shared<message_filters::Subscriber<nav_msgs::msg::Odometry>>(this, odometryGTTopic, sub_qos_profile_);

    // msg_sync_tmp_ = std::make_shared<message_filters::Synchronizer<msgSyncPolicyTmp>>(msgSyncPolicyTmp(300), *keypoints_sub_,
        //                                                                                                      *odometry_sub_,
        //                                                                                                      *odometry_gt_sub_);
    // msg_sync_tmp_->registerCallback(std::bind(&VSLAMListener::msgCallbackTmp, this, std::placeholders::_1,                                                   
    //                                                                                 std::placeholders::_2,
    //                                                                                 std::placeholders::_3));
}

VSLAMListener::~VSLAMListener()
{
    Stop();
}

void VSLAMListener::initSync() {
    auto self = this->shared_from_this();
    image_transport::ImageTransport it(self);

    std::string transport;
    get_parameter("image_transport", transport);
    image_transport::TransportHints hints(self.get(), transport);

    rgb_image_sub_.subscribe(
        this,
        rgbImageTopic,
        transport,
        sub_qos_profile_,
        rclcpp::SubscriptionOptions());

    RCLCPP_INFO(this->get_logger(), "Subscribed to RGB topic: %s", rgbImageTopic.c_str());


    depth_image_sub_.subscribe(
        this,
        depthImageTopic,
        transport,
        sub_qos_profile_,
        rclcpp::SubscriptionOptions());

    RCLCPP_INFO(this->get_logger(), "Subscribed to depth topic: %s", depthImageTopic.c_str());

    msg_sync_ = std::make_shared<message_filters::Synchronizer<msgSyncPolicy>>(
                msgSyncPolicy(10),        // queue size = 10
                rgb_image_sub_,
                depth_image_sub_,
                *keypoints_sub_,
                *odometry_sub_,
                *odometry_gt_sub_
    );


    msg_sync_->registerCallback(
        std::bind(&VSLAMListener::msgCallback, this, std::placeholders::_1,                                                   
                                                     std::placeholders::_2,
                                                     std::placeholders::_3,
                                                     std::placeholders::_4,
                                                     std::placeholders::_5));
                                                    
    RCLCPP_INFO(this->get_logger(), "Synchronized subscription to all topics are initialized");
     // Callback for all subscribed messages
    if (!vslam_activated_) {
        RCLCPP_INFO(this->get_logger(), "========= Waiting for finishing the handshake with the VSLAM system ...... =========");
    }
}

bool VSLAMListener::readConfigFile(const std::string& filePath) {
    cv::FileStorage fs(filePath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        RCLCPP_ERROR(this->get_logger(), "Could not read config file: %s", filePath.c_str());
        return false;
    }

    RCLCPP_INFO(this->get_logger(), "RosPublisher configured from file: %s", filePath.c_str());

    try {
        std::string use_gt_pose_str;
        std::string rgb_image_path_str;
        std::string depth_image_path_str;
        fs["topic_vslam_driver"] >> vslamDriverTopic;
        fs["topic_gsfusion_acknowledge"] >> ackTopic;
        fs["topic_img_filename"] >> imageFilenameTopic;
        fs["topic_rgb"] >> rgbImageTopic;
        fs["topic_depth"] >> depthImageTopic;
        fs["topic_map_points"] >> mapPointsTopic;
        fs["topic_keypoints"] >> keypointsTopic;
        fs["topic_odometry"] >> odometryTopic;
        fs["topic_odometry_gt"] >> odometryGTTopic;
        fs["topic_last_frame_flag"] >> lastFrameFlagTopic;
        fs["topic_last_frame_ack"] >> lastFrameAckTopic;
        fs["use_gt_pose"] >> use_gt_pose_str;
        fs["rgb_image_path"] >> rgb_image_path_str;
        fs["depth_image_path"] >> depth_image_path_str;
        if (use_gt_pose_str == "true") {
            use_gt_pose_ = true;
        } else if (use_gt_pose_str == "false") {
            use_gt_pose_ = false;
        } else {
            RCLCPP_ERROR(this->get_logger(), "Config parameter use_gt_pose must be set as 'true' or 'false'");
            return false;
        }
        
        rgb_image_base_ = fs::path(rgb_image_path_str);
        depth_image_base_ = fs::path(depth_image_path_str);

        // Log the values
        RCLCPP_INFO(this->get_logger(), "Config: topic_vslam_driver = %s", vslamDriverTopic.c_str());
        RCLCPP_INFO(this->get_logger(), "Config: topic_gsfusion_acknowledge = %s", ackTopic.c_str());
        RCLCPP_INFO(this->get_logger(), "Config: topic_img_filename = %s", imageFilenameTopic.c_str());
        RCLCPP_INFO(this->get_logger(), "Config: topic_rgb = %s", rgbImageTopic.c_str());
        RCLCPP_INFO(this->get_logger(), "Config: topic_depth = %s", depthImageTopic.c_str());
        RCLCPP_INFO(this->get_logger(), "Config: topic_map_points = %s", mapPointsTopic.c_str());
        RCLCPP_INFO(this->get_logger(), "Config: topic_keypoints = %s", keypointsTopic.c_str());
        RCLCPP_INFO(this->get_logger(), "Config: topic_odometry = %s", odometryTopic.c_str());
        RCLCPP_INFO(this->get_logger(), "Config: topic_odometry_gt = %s", odometryGTTopic.c_str());
        RCLCPP_INFO(this->get_logger(), "Config: topic_last_frame_flag = %s", lastFrameFlagTopic.c_str());
        RCLCPP_INFO(this->get_logger(), "Config: topic_last_frame_ack = %s", lastFrameAckTopic.c_str());
        RCLCPP_INFO(this->get_logger(), "Config: use_gt_pose = %s", use_gt_pose_ ? "true" : "false");
    }
    catch (const cv::Exception& e) {
        RCLCPP_WARN(this->get_logger(), "Failed to parse config file: %s", e.what());
        return false;
    }
    return true;
}

void VSLAMListener::Start() 
{   
    // gs mapping thread
    startThread(gs_mapping_queue_, [this](const Eigen::Matrix4f& T_WS, 
                                          const se::Image<se::rgb_t>& input_colour_img, 
                                          const se::Image<float>& input_depth_img,
                                          const std::vector<std::tuple<Eigen::Vector2f, Eigen::Vector3f>>& keypoints, 
                                          const int current_frame_id)
    {
        this->gsOnlineMapping(T_WS, input_colour_img, input_depth_img, keypoints, current_frame_id);
    });
}

void VSLAMListener::Stop()
{
    stopThread(gs_mapping_queue_);
}

Eigen::Matrix4f VSLAMListener::odomToMatrix(const nav_msgs::msg::Odometry::ConstSharedPtr& msg)
{
    // Extract translation
    const auto& pos = msg->pose.pose.position;
    float x = pos.x;
    float y = pos.y;
    float z = pos.z;

    // Extract quaternion
    const auto& q = msg->pose.pose.orientation;
    Eigen::Quaternionf quat(q.w, q.x, q.y, q.z);

    // Convert to transformation matrix
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T.block<3,3>(0,0) = quat.toRotationMatrix(); // rotation
    T.block<3,1>(0,3) = Eigen::Vector3f(x, y, z); // translation

    return T;
}

void VSLAMListener::setDepthImage(se::Image<float>& depth_image, cv::Mat& depth_data)
{   
    depth_data.convertTo(depth_data, CV_32FC1, 1/5000.0f); // hard-coded for tum rgbd
    cv::Mat wrapper_mat(depth_data.rows, depth_data.cols, CV_32FC1, depth_image.data());
    depth_data.copyTo(wrapper_mat);
}

void VSLAMListener::setColourImage(se::Image<se::rgb_t>& colour_image, cv::Mat& image_data)
{      
    cv::Mat colour_data;
    cv::cvtColor(image_data, colour_data, cv::COLOR_BGR2RGB);
    cv::Mat wrapper_mat(colour_data.rows, colour_data.cols, CV_8UC3, colour_image.data());
    colour_data.copyTo(wrapper_mat);
}

void VSLAMListener::driverCallback(const std_msgs::msg::Bool& msg)
{   
    vslam_activated_ = msg.data;
    if (vslam_activated_) {
        auto acknowledge = std_msgs::msg::String();
        acknowledge.data = "Let's begin!";

        RCLCPP_INFO(this->get_logger(), "Sent response: %s", acknowledge.data.c_str());
        ack_pub_->publish(acknowledge);
    }
}

void VSLAMListener::msgCallbackTmp( const sensor_msgs::msg::PointCloud2::ConstSharedPtr& keypoints_msg,
                                    const nav_msgs::msg::Odometry::ConstSharedPtr& odometry_msg,
                                    const nav_msgs::msg::Odometry::ConstSharedPtr& odometry_gt_msg)
{    
    if (!has_logged_mapping_start_) {
        RCLCPP_INFO(this->get_logger(), "GSFusion mapping begins!");
        has_logged_mapping_start_ = true;
    }

    std::lock_guard<std::mutex> lock(callback_mutex_);
  
    *frame_ += 1;
    se::perfstats.setIter(*frame_);


    RCLCPP_INFO(this->get_logger(), "Number of VSLAM Frames received: %d", *frame_);

    // Setup input pose (two choices: ground truth pose/vslam estimated pose)
    Eigen::Matrix4f T_WB = use_gt_pose_ ? odomToMatrix(odometry_gt_msg) : odomToMatrix(odometry_msg);
    Eigen::Matrix4f T_WS = T_WB * sensor_->T_BS;

    // input images, depth image message is already scaled by the VSLAM module
    fs::path rgb_image_path = rgb_image_base_ / odometry_msg->header.frame_id;
    fs::path depth_image_path = depth_image_base_ / odometry_gt_msg->header.frame_id;

    // save training image to training_views_list.txt
    if (training_views_list_.is_open()) {
        training_views_list_ << odometry_msg->header.frame_id << std::endl;
    } else {
        RCLCPP_ERROR(this->get_logger(), "Failed to open output file for writing.");
    }

    cv::Mat bgr_image = cv::imread(rgb_image_path.string(), cv::IMREAD_COLOR);
    cv::Mat depth_image = cv::imread(depth_image_path.string(), cv::IMREAD_UNCHANGED);

    // Setup input images, assuming depth and color image resolutions have the same size and are aligned.
    Eigen::Vector2i input_img_res(bgr_image.cols, bgr_image.rows);
    se::Image<float> input_depth_img(input_img_res.x(), input_img_res.y());
    se::Image<se::rgb_t> input_colour_img(input_img_res.x(), input_img_res.y(), {0, 0, 0});

    // Setup input keypoints
    std::vector<std::tuple<Eigen::Vector2f, Eigen::Vector3f>> keypoints;
    this->pointcloud2_to_eigen(keypoints_msg, keypoints);
 
    this->setDepthImage(input_depth_img, depth_image);
    this->setColourImage(input_colour_img, bgr_image);
    this->insertGSMappingInput(T_WS, input_colour_img, input_depth_img, keypoints, *frame_);
}


void VSLAMListener::pointcloud2_to_eigen(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg,
    std::vector<std::tuple<Eigen::Vector2f, Eigen::Vector3f>>& points2d3d)
{
    size_t n_pts = msg->width * msg->height;
    points2d3d.clear();
    points2d3d.reserve(n_pts);

    sensor_msgs::PointCloud2ConstIterator<float> u_it(*msg, "u");
    sensor_msgs::PointCloud2ConstIterator<float> v_it(*msg, "v");
    sensor_msgs::PointCloud2ConstIterator<float> x_it(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> y_it(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> z_it(*msg, "z");

    for (; u_it != u_it.end(); ++u_it, ++v_it, ++x_it, ++y_it, ++z_it)
    {
        if (!std::isfinite(*u_it) || !std::isfinite(*v_it) ||
            !std::isfinite(*x_it) || !std::isfinite(*y_it) || !std::isfinite(*z_it))
            continue;

        Eigen::Vector2f keypoint(*u_it, *v_it);
        Eigen::Vector3f mappoint(*x_it, *y_it, *z_it);
        points2d3d.emplace_back(keypoint, mappoint);
    }
}



void VSLAMListener::msgCallback(const sensor_msgs::msg::Image::ConstSharedPtr& rgb_image_msg,
                                const sensor_msgs::msg::Image::ConstSharedPtr& depth_image_msg,
                                const sensor_msgs::msg::PointCloud2::ConstSharedPtr& keypoints_msg,
                                const nav_msgs::msg::Odometry::ConstSharedPtr& odometry_msg,
                                const nav_msgs::msg::Odometry::ConstSharedPtr& odometry_gt_msg)
{   
    if (!has_logged_mapping_start_) {
        RCLCPP_INFO(this->get_logger(), "GSFusion mapping begins!");
        has_logged_mapping_start_ = true;
    }

    std::lock_guard<std::mutex> lock(callback_mutex_);

    // Setup input pose (two choices: ground truth pose/vslam estimated pose)
    Eigen::Matrix4f T_WB = use_gt_pose_ ? odomToMatrix(odometry_gt_msg) : odomToMatrix(odometry_msg);
    Eigen::Matrix4f T_WS = T_WB * sensor_->T_BS;

    se::perfstats.setIter((*frame_)++);
    RCLCPP_INFO(this->get_logger(), "===================================");
    RCLCPP_INFO(this->get_logger(), "Number of VSLAM Frames received: %d", *frame_);

    // input images, depth image message is already scaled by the VSLAM module
    cv_bridge::CvImagePtr rgb_ptr, depth_ptr;
    
    try
    {
        rgb_ptr = cv_bridge::toCvCopy(rgb_image_msg);
        depth_ptr = cv_bridge::toCvCopy(depth_image_msg);
    }
    catch(cv_bridge::Exception& e)
    {
        RCLCPP_ERROR(this->get_logger(), "Error reading image");
        return;
    }

    // Get the image frame_id from the header
    std::string frame_id = rgb_image_msg->header.frame_id;
    // append the frame id to training_views_list.txt
    if (training_views_list_.is_open()) {
        training_views_list_ << frame_id << std::endl;
    } else {
        RCLCPP_ERROR(this->get_logger(), "Failed to open output file for writing.");
    }

    // Setup input images, assuming depth and color image resolutions have the same size and have been aligned.
    Eigen::Vector2i input_img_res(rgb_ptr->image.cols, rgb_ptr->image.rows);
    se::Image<float> input_depth_img(input_img_res.x(), input_img_res.y());
    se::Image<se::rgb_t> input_colour_img(input_img_res.x(), input_img_res.y(), {0, 0, 0});

    // Setup input keypoints
    std::vector<std::tuple<Eigen::Vector2f, Eigen::Vector3f>> keypoints;
    this->pointcloud2_to_eigen(keypoints_msg, keypoints);

    this->setDepthImage(input_depth_img, depth_ptr->image);
    this->setColourImage(input_colour_img, rgb_ptr->image);
    this->insertGSMappingInput(T_WS, input_colour_img, input_depth_img, keypoints, *frame_);
}

void VSLAMListener::lastFrameFlagCallback(const std_msgs::msg::Bool::ConstSharedPtr& last_frame_flag_msg)
{   
    auto last_frame_ack_msg = std_msgs::msg::String();
    RCLCPP_INFO(this->get_logger(), "Got the last frame flag message: %s", last_frame_flag_msg->data ? "true" : "false");    
    last_frame_received_ = true;
    last_frame_ack_msg.data = "Last frame received!";
    last_frame_ack_pub_->publish(last_frame_ack_msg);
}

void VSLAMListener::insertGSMappingInput(const Eigen::Matrix4f& T_WS, 
                                         const se::Image<se::rgb_t>& input_colour_img, 
                                         const se::Image<float>& input_depth_img,
                                         const std::vector<std::tuple<Eigen::Vector2f, Eigen::Vector3f>>& keypoints, 
                                         const int current_frame_id)
{
    std::lock_guard<std::mutex> lock(gs_mapping_queue_.mutex_);
    gs_mapping_queue_.queue_.emplace(T_WS, input_colour_img, input_depth_img, keypoints, current_frame_id);
    gs_mapping_queue_.cond_var_.notify_one();
}

void VSLAMListener::gsOnlineMapping(const Eigen::Matrix4f& T_WS, 
                                    const se::Image<se::rgb_t>& input_colour_img, 
                                    const se::Image<float>& input_depth_img,
                                    const std::vector<std::tuple<Eigen::Vector2f, Eigen::Vector3f>>& keypoints, 
                                    const int current_frame_id) 
{
    TICK("integration")
    double s = PerfStats::getTime();
    if (current_frame_id % integration_rate_ == 0) {
        se::integrator::integrate(*map_, *gs_model_, 
                                  *gs_cam_list_, *gt_img_list_, *gt_depth_list_,
                                  *data_queue_, 
                                  input_depth_img, input_colour_img, 
                                  keypoints,
                                  *sensor_, T_WS, 
                                  current_frame_id);
    }
    double e = PerfStats::getTime();
    *mean_fps_ += (1 / (e - s));
    TOCK("integration")
}

void VSLAMListener::checkTerminationCondition()
{
    if (last_frame_received_ && gs_mapping_queue_.size() == 0) {
        // flag for shuttinng down the ros2 node spinning because the online optimization is finished
        online_optimization_finished_ = true;
    }
}

int main(int argc, char** argv)
{
    try {
        if (argc != 3) {
            std::cerr << "Usage: ros2 run GSFusion gsfusion_ros2 -- <gsfusion_config_yaml> <vslam_ros2_config_yaml>\n";
            return 3;
        }

        auto mem_before = gs::getGPUMemoryUsage();

        // ========= Config & I/O INITIALIZATION  =========
        const std::string config_filename = argv[1];
        const se::Config<se::TSDFColDataConfig, se::PinholeCameraConfig> config(config_filename);

        // Create the mesh output directory
        if (!config.app.mesh_path.empty()) {
            stdfs::create_directories(config.app.mesh_path);
        }
        if (!config.app.slice_path.empty()) {
            stdfs::create_directories(config.app.slice_path);
        }
        if (!config.app.structure_path.empty()) {
            stdfs::create_directories(config.app.structure_path);
        }

        const std::string ros2_config_filename = argv[2];

        // Setup input images
        const Eigen::Vector2i input_img_res(config.sensor.width, config.sensor.height);

        // ========= Map INITIALIZATION  =========
        // Setup the single-res TSDF map w/ default block size of 8 voxels
        auto map = std::make_shared<se::TSDFColMap<se::Res::Single>>(config.map, config.data);

        // ========= Sensor INITIALIZATION  =========
        // Create a pinhole camera
        std::shared_ptr<const se::PinholeCamera> sensor = std::make_shared<se::PinholeCamera>(config.sensor);

        // ========= Gaussian Model INITIALIZATION  =========
        auto optimParams = gs::param::read_optim_params_from_json(config.app.optim_params_path);
        auto gs_model = std::make_shared<gs::GaussianModel>(optimParams, config.app.ply_path);
        auto gs_cam_list = std::make_shared<std::vector<gs::Camera>>();
        auto gt_img_list = std::make_shared<std::vector<torch::Tensor>>();
        auto gt_depth_list = std::make_shared<std::vector<torch::Tensor>>();

        // Write cfg_args file
        const std::string cfg_args_file = stdfs::path(config.app.ply_path).parent_path() / "cfg_args";
        std::ofstream fs(cfg_args_file, std::ios::out);
        if (!fs.good()) {
            std::cerr << "Failed to open cfg_args for writing!" << std::endl;
        }

        std::string image_folder_name;
        if (se::reader_type_to_string(config.reader.reader_type) == "ScanNetpp") {
            image_folder_name = "undistorted_images_2";
        }
        else if (se::reader_type_to_string(config.reader.reader_type) == "Replica") {
            image_folder_name = "results";
        }

        fs << "Namespace("
        << "eval=True, "
        << "images="
        << "\"" << image_folder_name << "\", "
        << "model_path=" << stdfs::path(config.app.ply_path).parent_path() << ", "
        << "resolution=-1, "
        << "sh_degree=" << gs_model->optimParams.sh_degree << ", "
        << "source_path="
        << "\"" << config.reader.sequence_path << "\", "
        << "white_background=False)";
        fs.close();

        // ========= GUI INITIALIZATION  =========
        auto data_queue = std::make_shared<gs::DataQueue>();
        // std::atomic<bool> stop_signal(false);
        // GUI gs_gui(data_queue, stop_signal, input_img_res.x(), input_img_res.y());
        // std::thread gui_thread([&]() { gs_gui.run(); });

        // ========= Integrator INITIALIZATION  =========
        auto frame = std::make_shared<unsigned int>(0);
        auto mean_fps = std::make_shared<float>(0.0f);
        int integration_rate = config.app.integration_rate;

        // ROS2 for online optimization
        rclcpp::init(argc, argv);
        auto vslam_listener_node = std::make_shared<VSLAMListener>(std::string("vslam_listener"), 
                                                                   ros2_config_filename, 
                                                                   integration_rate,
                                                                   map, sensor, gs_model, 
                                                                   gs_cam_list, gt_img_list, gt_depth_list, 
                                                                   data_queue, 
                                                                   frame, 
                                                                   mean_fps);
        // set the output path for the training views list
        std::string training_views_list_path = config.reader.sequence_path + "/training_views_list.txt";
        vslam_listener_node->set_training_views_list_path(training_views_list_path);
        vslam_listener_node->initSync(); // initialize the synchronized subscriptions involving image transport
        vslam_listener_node->Start(); // start the online gs mapping thread

        rclcpp::executors::SingleThreadedExecutor executor;
        executor.add_node(vslam_listener_node);
        
        // online optimization
        while (rclcpp::ok() && !(vslam_listener_node->last_frame_received_ && vslam_listener_node->online_optimization_finished_)) {
            executor.spin_some(); // Online optimization
            vslam_listener_node->checkTerminationCondition();
        }

        if (vslam_listener_node->online_optimization_finished_) {
            // shut down ros2
            std::cout << "Shutting down ROS2 ...\n"; 
            vslam_listener_node->Stop(); // stop the online gs mapping thread
            //stop writing to training_views_list.txt
            vslam_listener_node->close_log_files();
            // vslam_listener_node.reset();
            rclcpp::shutdown();
            std::cout << "Starting global optimization ...\n"; 
            double s = PerfStats::getTime();
            
            // Refresh GUI
            gs::DataPacket data_packet;
            data_packet.num_kf = gt_img_list->size();
            data_packet.rgb = cv::Mat(input_img_res.y(), input_img_res.x(), CV_8UC3, cv::Scalar(0, 0, 0));
            data_packet.depth = cv::Mat(input_img_res.y(), input_img_res.x(), CV_8UC3, cv::Scalar(0, 0, 0));
            data_packet.rendered_rgb = cv::Mat(input_img_res.y(), input_img_res.x(), CV_8UC3, cv::Scalar(0, 0, 0));
            data_queue->push(data_packet);

            // Global optimizaiton of reconstructed GS map (offline)
            auto lambda = gs_model->optimParams.lambda_dssim;
            auto iters = gs_model->optimParams.global_iters;
            for (int it = 0; it < iters; it++) {
                std::vector<int> indices = gs::get_random_indices(gt_img_list->size());
                for (int i = 0; i < indices.size(); i++) {
                    auto cur_gt_img = (*gt_img_list)[indices[i]];
                    auto cur_gs_cam = (*gs_cam_list)[indices[i]];

                    auto [image, viewspace_point_tensor, visibility_filter, radii] = gs::render(cur_gs_cam, *gs_model);

                    // Loss Computations
                    auto l1_loss = gs::l1_loss(image, cur_gt_img);
                    auto ssim_loss = gs::ssim(image, cur_gt_img, gs::conv_window, gs::window_size, gs::channel);
                    auto loss = (1.f - lambda) * l1_loss + lambda * (1.f - ssim_loss);

                    // Optimization
                    loss.backward();
                    gs_model->optimizer->step();
                    gs_model->optimizer->zero_grad(true);

                    if (i == indices.size() - 1) {
                        auto rendered_img_tensor = image.detach().permute({1, 2, 0}).contiguous().to(torch::kCPU);
                        rendered_img_tensor = rendered_img_tensor.mul(255).clamp(0, 255).to(torch::kU8);
                        auto cv_rendered_img = cv::Mat(image.size(1), image.size(2), CV_8UC3, rendered_img_tensor.data_ptr());
                        data_packet.rendered_rgb = cv_rendered_img;
                        data_packet.global_iter = it + 1;
                        data_queue->push(data_packet);
                    }
                }
            }
            torch::cuda::synchronize();
            double e = PerfStats::getTime();

            // Get GPU memory usage
            auto mem_after = gs::getGPUMemoryUsage();
            std::cout << "============== Global optimization finished ==============\n"; 
            std::cout << "Avg. fps: " << *mean_fps / *frame << std::endl;
            std::cout << "Global opt. time: " << e - s << " s" << std::endl;
            std::cout << "GPU memory usage: " << mem_after - mem_before << " MB" << std::endl;
            std::cout << "#Keyframes: " << gt_img_list->size() << std::endl;

            // Write mapping statistics to a file
            const std::string stats_file = stdfs::path(config.app.ply_path).parent_path() / "stats";
            std::ofstream fs(stats_file, std::ios::out);
            if (!fs.good()) {
                std::cerr << "Failed to open stats for writing!" << std::endl;
            }
            fs << "Avg. fps: " << *mean_fps / *frame << " Hz\n"
            << "Global opt. time: " << e - s << " s\n"
            << "GPU memory usage: " << mem_after - mem_before << " MB\n"
            << "#Keyframes: " << gt_img_list->size() << "\n";

            gs_model->Save_ply(gs_model->output_path, *frame, true);
        }

        // =========== Optional Evaluation ==============
        if (config.app.eval) {
            std::cout << "Starting evaluation ...\n";

            // read in the training views list in a vector
            std::ifstream training_views_file(config.reader.sequence_path + "/training_views_list.txt");
            if (!training_views_file.is_open()) {
                std::cerr << "Failed to open training views list file for reading." << std::endl;
                return EXIT_FAILURE;
            }   
            std::vector<std::string> training_views_list;
            std::string line;
            while (std::getline(training_views_file, line)) {
                training_views_list.push_back(line);
            }
            training_views_file.close();

            // Initialize the lists for evaluation metrics
            std::vector<float> psnr_training, ssim_training, lpips_training;
            std::vector<float> psnr_novel, ssim_novel, lpips_novel;

            // Create directory for rendered images
            const std::string rendered_dir = config.reader.sequence_path + "/rendered_color_hpgs";
            if (!stdfs::create_directories(rendered_dir)) {
                std::cerr << "Warning: could not create (or already existed) directory " 
                        << rendered_dir << std::endl;
            }

            // // Create directory for rendered depth images
            // const std::string rendered_depth_dir = config.reader.sequence_path + "/rendered_depth_hpgs";
            // if (!stdfs::create_directories(rendered_depth_dir)) {
            //     std::cerr << "Warning: could not create (or already existed) directory " 
            //             << rendered_depth_dir << std::endl;
            // }

            // Load the Lpips model for evaluation
            // initialize the LPIPS model
            torch::jit::script::Module lpips_model;
            gs::eval::loadLpipsModel(config.reader.lpips_model_path, lpips_model);
            
            // reset reader and frame count
            gs::Camera camera;
            *frame = 0;
            // ========= READER INITIALIZATION  =========
            se::Reader* reader = nullptr;
            reader = se::create_reader(config.reader);

            if (reader == nullptr) {
                return EXIT_FAILURE;
            }

            Eigen::Matrix4f T_WB = Eigen::Matrix4f::Identity();
            Eigen::Matrix4f T_BS = sensor->T_BS;
            Eigen::Matrix4f T_WS = T_WB * T_BS;

            // Setup input images and their file names
            se::Image<float> input_depth_img(input_img_res.x(), input_img_res.y());
            se::Image<se::rgb_t> input_colour_img(input_img_res.x(), input_img_res.y(), {0, 0, 0});
            std::string depth_image_name, colour_image_name;

            while (*frame != config.app.max_frames) {
                se::perfstats.setIter((*frame)++);
                se::ReaderStatus read_ok = reader->nextData(input_depth_img, &depth_image_name,
                                                            input_colour_img, &colour_image_name,
                                                            T_WB);
                if (read_ok != se::ReaderStatus::ok) {
                    break;
                }
                T_WS = T_WB * T_BS;

                // Construct gs::Camera used for rendering
                Eigen::Matrix4f T_SW = se::math::to_inverse_transformation(T_WS);
                torch::Tensor W2C_matrix = torch::from_blob(T_SW.data(), {4, 4}, torch::kFloat).clone().to(torch::kCUDA, true);
                torch::Tensor proj_matrix =
                    gs::getProjectionMatrix(input_colour_img.width(), input_colour_img.height(),
                                            sensor->model.focalLengthU(), sensor->model.focalLengthV(),
                                            sensor->model.imageCenterU(), sensor->model.imageCenterV())
                        .to(torch::kCUDA, true);
                camera.width = input_colour_img.width();
                camera.height = input_colour_img.height();
                camera.fov_x = sensor->horizontal_fov;
                camera.fov_y = sensor->vertical_fov;
                camera.T_W2C = W2C_matrix;
                camera.full_proj_matrix = W2C_matrix.mm(proj_matrix);
                camera.cam_center = W2C_matrix.inverse()[3].slice(0, 0, 3);
                // Rendering
                auto [image, viewspace_point_tensor, visibility_filter, radii] = gs::render(camera, *gs_model);
                
                // save the rendered rgb image
                auto rendered_img_tensor = image.detach().permute({1, 2, 0}).contiguous().to(torch::kCPU);
                rendered_img_tensor = rendered_img_tensor.mul(255).clamp(0, 255).to(torch::kU8);
                auto cv_rendered_img = cv::Mat(image.size(1), image.size(2), CV_8UC3, rendered_img_tensor.data_ptr());

                cv::Mat rendered_bgr;
                cv::cvtColor(cv_rendered_img, rendered_bgr, cv::COLOR_RGB2BGR);

                std::string color_image_file = rendered_dir + "/" + colour_image_name;
                if (!cv::imwrite(color_image_file, rendered_bgr)) {
                    std::cerr << "Failed to save rendered color image to " 
                            << color_image_file << std::endl;
                }

                // // Save the rendered depth image
                // const float min_d = 0.4f, max_d = 6.0f;
                // auto dep_t = depth_img.detach()
                //                 .permute({1,2,0})
                //                 .contiguous()
                //                 .to(torch::kCPU); // still float
                // // assume depth tensor is [H,W,1], so squeeze
                // dep_t = dep_t.squeeze();
                // // convert to uchar with range [0,255]
                // cv::Mat dep_f(image.size(1), image.size(2), CV_32FC1, dep_t.data_ptr());
                // cv::Mat dep_u8;
                // dep_f.convertTo(dep_u8, CV_8UC1,
                //                 255.0f / (max_d - min_d),
                //                 -255.0f * min_d / (max_d - min_d));
                // cv::applyColorMap(dep_u8, dep_u8, cv::COLORMAP_VIRIDIS);
                // cv::cvtColor(dep_u8, dep_u8, cv::COLOR_BGR2RGB);

                // const std::string out_depth_fn = rendered_depth_dir + "/" + depth_image_name;
                // if (!cv::imwrite(out_depth_fn, dep_u8)) {
                //     std::cerr << "Failed to save depth image to " << out_depth_fn << "\n";
                // }

                // Compute evaluation metrics
                // For evaluation, we assume the ground truth is available
                // Check if the current camera view is in the training views
                if (std::find(training_views_list.begin(), training_views_list.end(), colour_image_name) != training_views_list.end()) {
                    // If it is, we can compute the evaluation metrics
                    auto gt_training_view = cv::imread(config.reader.sequence_path + "/rgb/" + colour_image_name, cv::IMREAD_COLOR);

                    // Compute PSNR
                    float psnr = gs::eval::computePSNR(gt_training_view, rendered_bgr);

                    // Convert to grayscale for SSIM computation
                    cv::Mat gt_training_view_gray, rendered_bgr_gray;
                    cv::cvtColor(gt_training_view, gt_training_view_gray, cv::COLOR_BGR2GRAY);
                    cv::cvtColor(rendered_bgr, rendered_bgr_gray, cv::COLOR_BGR2GRAY);
                    float ssim = gs::eval::computeSSIM(gt_training_view_gray, rendered_bgr_gray);

                    // Compute LPIPS
                    cv::Mat gt_training_view_f, rendered_bgr_f;
                    gt_training_view.convertTo(gt_training_view_f, CV_32F, 1.0 / 255.0);
                    rendered_bgr.convertTo(rendered_bgr_f, CV_32F, 1.0 / 255.0);
                    float lpips = gs::eval::computeLPIPS(gt_training_view_f, rendered_bgr_f, lpips_model);

                    // Store the evaluation metrics
                    psnr_training.push_back(psnr);
                    ssim_training.push_back(ssim);
                    lpips_training.push_back(lpips);
                } else {
                    // If not, then it is a novel view
                    auto gt_novel_view = cv::imread(config.reader.sequence_path + "/rgb/" + colour_image_name, cv::IMREAD_COLOR);   
                    // Compute PSNR
                    float psnr = gs::eval::computePSNR(gt_novel_view, rendered_bgr);
                    // Compute SSIM  
                    cv::Mat gt_novel_view_gray, rendered_bgr_gray;
                    cv::cvtColor(gt_novel_view, gt_novel_view_gray, cv::COLOR_BGR2GRAY);
                    cv::cvtColor(rendered_bgr, rendered_bgr_gray, cv::COLOR_BGR2GRAY);               
                    float ssim = gs::eval::computeSSIM(gt_novel_view_gray, rendered_bgr_gray);
                    // Compute LPIPS
                    cv::Mat gt_novel_view_f, rendered_bgr_f;
                    gt_novel_view.convertTo(gt_novel_view_f, CV_32F, 1.0 / 255.0);
                    rendered_bgr.convertTo(rendered_bgr_f, CV_32F, 1.0 / 255.0);
                    float lpips = gs::eval::computeLPIPS(gt_novel_view_f, rendered_bgr_f, lpips_model);
                    // Store the evaluation metrics
                    psnr_novel.push_back(psnr);
                    ssim_novel.push_back(ssim);
                    lpips_novel.push_back(lpips);
                }
                torch::cuda::synchronize();
                printProgress(static_cast<double>(*frame) / (static_cast<double>(reader->numFrames()) - 1));
            }
            // Save the evaluation metrics to a file

            // Lambda function to compute the mean of a vector
            auto mean = [](const std::vector<float>& v) {
                return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
            };

            const std::string eval_file = config.reader.sequence_path + "/evaluation_results.txt";
            std::ofstream eval_fs(eval_file, std::ios::out);
            if (!eval_fs.good()) {
                std::cerr << "Failed to open evaluation results file for writing!" << std::endl;
            }

            // Always write training views (assuming they exist)
            if (!psnr_training.empty()) {
                eval_fs << "Training views evaluation metrics:\n";
                eval_fs << "PSNR: " << mean(psnr_training) << "\n";
                eval_fs << "SSIM: " << mean(ssim_training) << "\n";
                eval_fs << "LPIPS: " << mean(lpips_training) << "\n";
                
                std::cout << "============= Evaluation Results =============\n";
                std::cout << "Training views:\n";
                std::cout << "PSNR: " << mean(psnr_training) << "\n";
                std::cout << "SSIM: " << mean(ssim_training) << "\n";
                std::cout << "LPIPS: " << mean(lpips_training) << "\n";
            }

            // Only write novel views if they exist
            if (!psnr_novel.empty()) {
                eval_fs << "Novel views evaluation metrics:\n";
                eval_fs << "PSNR: " << mean(psnr_novel) << "\n";
                eval_fs << "SSIM: " << mean(ssim_novel) << "\n";
                eval_fs << "LPIPS: " << mean(lpips_novel) << "\n";
                
                std::cout << "Novel views:\n";
                std::cout << "PSNR: " << mean(psnr_novel) << "\n";
                std::cout << "SSIM: " << mean(ssim_novel) << "\n";
                std::cout << "LPIPS: " << mean(lpips_novel) << "\n";
            } else {
                eval_fs << "No novel views found for evaluation.\n";
                std::cout << "No novel views found for evaluation.\n";
            }

            if (!psnr_training.empty() || !psnr_novel.empty()) {
                std::cout << "==============================================\n";
            }

            eval_fs.close();
            std::cout << "Evaluation results saved to " << eval_file << "\n";
        }
        // stop_signal.store(true);
        // gui_thread.join();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
