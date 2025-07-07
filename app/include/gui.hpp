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
