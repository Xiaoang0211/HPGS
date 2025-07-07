// /*
//  * SPDX-FileCopyrightText: 2024 Smart Robotics Lab, Technical University of Munich
//  * SPDX-FileCopyrightText: 2024 Jiaxin Wei
//  * SPDX-License-Identifier: BSD-3-Clause
//  */

#include <queue>
#include <mutex>
#include <open3d/Open3D.h>
#include <opencv2/core.hpp>
#include <atomic>

#include "gs/gaussian_utils.cuh"

// namespace gui{

// struct DataPacket {
//     int ID = -1;
//     int global_iter = -1;
//     int num_kf;
//     int num_splats;
//     float fps;
//     cv::Mat rgb;
//     cv::Mat depth;
//     cv::Mat rendered_rgb;
// };

// class DataQueue {
// public:
//     void push(DataPacket data) {
//         std::lock_guard<std::mutex> lock(_mtx);
//         _queue.push(std::move(data));
//         _cv.notify_one();
//     }

//     DataPacket pop() {
//         std::unique_lock<std::mutex> lock(_mtx);
//         _cv.wait(lock, [this]() { return !_queue.empty(); });
//         DataPacket data = std::move(_queue.front());
//         _queue.pop();
//         return data;
//     }

//     int getSize() {
//         std::lock_guard<std::mutex> lock(_mtx); // 加锁
//         return _queue.size();
//     }

// private:
//     std::queue<DataPacket> _queue;
//     std::mutex _mtx;
//     std::condition_variable _cv;
// };

// class GUI {
//     public:
//     GUI(DataQueue& data_queue, std::atomic<bool>& stop_signal, int width, int height) : data_queue_(data_queue), stop_signal_(stop_signal), img_width_(width), img_height_(height)
//     {
//     }

//     void run();

//     private:
//     void initWidget();
//     void updateScene();
//     bool onWindowClose();
//     void cleanUp();

//     DataQueue& data_queue_;
//     std::atomic<bool>& stop_signal_;

//     int img_width_;
//     int img_height_;

//     std::shared_ptr<open3d::visualization::gui::Window> window_;
//     std::shared_ptr<open3d::visualization::gui::Widget> gs_panel_;
//     std::shared_ptr<open3d::visualization::gui::ImageWidget> gs_widget_;
//     std::shared_ptr<open3d::visualization::gui::Label> gs_info_;
//     std::shared_ptr<open3d::visualization::gui::Widget> panel_;
//     std::shared_ptr<open3d::visualization::gui::ImageWidget> rgb_widget_;
//     std::shared_ptr<open3d::visualization::gui::ImageWidget> depth_widget_;
//     std::shared_ptr<open3d::visualization::gui::Label> img_info_;
// };
// }
// #endif



class GUI {
public:
    GUI(gs::DataQueue& data_queue, std::atomic<bool>& stop_signal, int width, int height)
        : data_queue_(data_queue), stop_signal_(stop_signal), img_width_(width), img_height_(height) {}

    void run();

private:
    void show_images(const gs::DataPacket& packet);

    gs::DataQueue& data_queue_;
    std::atomic<bool>& stop_signal_;
    int img_width_;
    int img_height_;
};
