// /*
//  * SPDX-FileCopyrightText: 2024 Smart Robotics Lab, Technical University of Munich
//  * SPDX-FileCopyrightText: 2024 Jiaxin Wei
//  * SPDX-License-Identifier: BSD-3-Clause
//  */

// #include "gui.hpp"

// #include <Eigen/Core>
// #include <open3d/visualization/rendering/ColorGrading.h>
// #include <opencv2/imgproc.hpp>
// #include <sstream>
// #include <thread>


// void gui::GUI::run()
// {
//     auto& app = open3d::visualization::gui::Application::GetInstance();
//     app.Initialize("/app/GSFusion_ROS2_ws/third_party/open3d/install/share/resources");
//     std::cout<<"initWidget"<<std::endl;

//     initWidget();
//     std::cout<<"initWidget done "<<std::endl;
//     app.AddWindow(window_);
//     std::cout<<"AddWindow done "<<std::endl;

//     std::thread update_thread([this]() { this->updateScene(); });
//     update_thread.detach();
//     app.Run();
//     std::cout<<"app run  done "<<std::endl;

// }


// void gui::GUI::initWidget()
// {
//     window_ = std::make_shared<open3d::visualization::gui::Window>("GSFusion | Online RGB-D Mapping", 1280, 720);
//     float em = window_->GetTheme().font_size;

//     gs_panel_ = std::make_shared<open3d::visualization::gui::Vert>(0, open3d::visualization::gui::Margins(0.5f * em));
//     gs_panel_->AddChild(std::make_shared<open3d::visualization::gui::Label>("Rendered RGB"));
//     gs_info_ = std::make_shared<open3d::visualization::gui::Label>("Online optimization");
//     gs_panel_->AddChild(gs_info_);

//     auto black_img0 = std::make_shared<open3d::geometry::Image>();
//     black_img0->Prepare(img_width_, img_height_, 3, 1);
//     std::fill(black_img0->data_.begin(), black_img0->data_.end(), 0);
//     gs_widget_ = std::make_shared<open3d::visualization::gui::ImageWidget>(black_img0);
//     gs_panel_->AddChild(gs_widget_);

//     window_->AddChild(gs_panel_);

//     panel_ = std::make_shared<open3d::visualization::gui::Vert>(0, open3d::visualization::gui::Margins(0.5f * em));
//     panel_->AddChild(std::make_shared<open3d::visualization::gui::Label>("Input RGB-D sequence"));
//     img_info_ = std::make_shared<open3d::visualization::gui::Label>("Image ID: ---");
//     panel_->AddChild(img_info_);

//     // Add RGB and depth widgets
//     auto black_img1 = std::make_shared<open3d::geometry::Image>();
//     black_img1->Prepare(img_width_, img_height_, 3, 1);
//     std::fill(black_img1->data_.begin(), black_img1->data_.end(), 1);
//     rgb_widget_ = std::make_shared<open3d::visualization::gui::ImageWidget>(black_img1);
//     panel_->AddChild(rgb_widget_);

//     auto black_img2 = std::make_shared<open3d::geometry::Image>();
//     black_img2->Prepare(img_width_, img_height_, 3, 1);
//     std::fill(black_img2->data_.begin(), black_img2->data_.end(), 2);
//     depth_widget_ = std::make_shared<open3d::visualization::gui::ImageWidget>(black_img2);
//     panel_->AddChild(depth_widget_);

//     window_->AddChild(panel_);

//     // Set layout
//     float gs_width_ratio_ = 0.66;
//     auto contentRect = window_->GetContentRect();
//     int gs_width = static_cast<int>(contentRect.width * gs_width_ratio_);
//     panel_->SetFrame(open3d::visualization::gui::Rect(contentRect.x, contentRect.y, contentRect.width - gs_width, contentRect.height));
//     gs_panel_->SetFrame(open3d::visualization::gui::Rect(panel_->GetFrame().GetRight(), contentRect.y, gs_width, contentRect.height));
//     gs_info_->SetFrame(open3d::visualization::gui::Rect(panel_->GetFrame().GetRight(), contentRect.y, gs_width, em));

//     window_->SetOnClose([this]() { return this->onWindowClose(); });
// }


// void gui::GUI::updateScene()
// {
//     while (!stop_signal_.load()) {
//         if (data_queue_.getSize() == 0) {
//             continue;
//         }
//         DataPacket data_packet = data_queue_.pop();

//         auto vis_rgb = std::make_shared<open3d::geometry::Image>();
//         auto vis_depth = std::make_shared<open3d::geometry::Image>();
//         auto vis_rendered_rgb = std::make_shared<open3d::geometry::Image>();
//         vis_rgb->Prepare(data_packet.rgb.cols, data_packet.rgb.rows, 3, 1);
//         vis_depth->Prepare(data_packet.depth.cols, data_packet.depth.rows, 3, 1);
//         vis_rendered_rgb->Prepare(data_packet.rendered_rgb.cols, data_packet.rendered_rgb.rows, 3, 1);
//         memcpy(vis_rgb->data_.data(), data_packet.rgb.data, vis_rgb->data_.size());
//         memcpy(vis_depth->data_.data(), data_packet.depth.data, vis_depth->data_.size());
//         memcpy(vis_rendered_rgb->data_.data(), data_packet.rendered_rgb.data, vis_rendered_rgb->data_.size());

//         std::ostringstream gs_text;
//         std::ostringstream img_text;
//         if (data_packet.ID > 0) {
//             gs_text << std::fixed << std::setprecision(2) << "Online optimization | "
//                     << "#Gaussians=" << data_packet.num_splats << " | #KF=" << data_packet.num_kf << " | FPS=" << data_packet.fps;
//             img_text << "Image ID: " << data_packet.ID;
//         }
//         else if (data_packet.global_iter > 0) {
//             gs_text << "Global optimization | "
//                     << "#KF=" << data_packet.num_kf << " | Iter=" << data_packet.global_iter;
//             img_text << "Image ID: ---";
//         }
//         else {
//             gs_text << "Start global optimization...";
//             img_text << "Image ID: ---";
//         }

//         gs_info_->SetText(gs_text.str().c_str());
//         img_info_->SetText(img_text.str().c_str());

//         open3d::visualization::gui::Application::GetInstance().PostToMainThread(window_.get(), [this, vis_rgb, vis_depth, vis_rendered_rgb]() {
//             this->rgb_widget_->UpdateImage(vis_rgb);
//             this->depth_widget_->UpdateImage(vis_depth);
//             this->gs_widget_->UpdateImage(vis_rendered_rgb);
//         });
//     }

//     cleanUp();
//     std::cout << "Both online and offline mapping are finished. You can close the GUI now!" << std::endl;
// }


// bool gui::GUI::onWindowClose()
// {
//     if (stop_signal_) {
//         return true;
//     }
//     else {
//         return false;
//     }
// }


// void gui::GUI::cleanUp()
// {
//     window_.reset();
//     gs_panel_.reset();
//     gs_widget_.reset();
//     gs_info_.reset();
//     panel_.reset();
//     rgb_widget_.reset();
//     depth_widget_.reset();
//     img_info_.reset();
// }




#include "gui.hpp"
#include <opencv2/opencv.hpp>
#include <sstream>
#include <iomanip>
#include <thread>


void draw_text(cv::Mat& img, const std::string& text, int x, int y, double scale = 0.7, cv::Scalar color = {255, 255, 255}) {
    cv::putText(img, text, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, scale, color, 2, cv::LINE_AA);
}

cv::Mat colorize_depth(const cv::Mat& depth) {
    cv::Mat depth_norm, depth_color;
    if (depth.type() == CV_16U) {
        double minv, maxv;
        cv::minMaxLoc(depth, &minv, &maxv);
        depth.convertTo(depth_norm, CV_8U, 255.0 / (maxv - minv), -minv * 255.0 / (maxv - minv));
    } else if (depth.type() == CV_32F) {
        double minv, maxv;
        cv::minMaxLoc(depth, &minv, &maxv);
        depth.convertTo(depth_norm, CV_8U, 255.0 / (maxv - minv), -minv * 255.0 / (maxv - minv));
    } else if (depth.type() == CV_8U) {
        depth_norm = depth;
    } else {
        depth_norm = cv::Mat::zeros(depth.size(), CV_8U);
    }
    cv::applyColorMap(depth_norm, depth_color, cv::COLORMAP_JET);
    return depth_color;
}

void draw_fancy_panel(cv::Mat& canvas, const cv::Mat& img, const std::string& title, int x, int y, int w, int h) {
    cv::Mat roi = canvas(cv::Rect(x, y, w, h));
    img.copyTo(roi);

    // 画半透明底条
    cv::rectangle(canvas, cv::Rect(x, y+h-40, w, 40), cv::Scalar(30, 30, 30, 180), -1);
    // 画高亮边框
    cv::rectangle(canvas, cv::Rect(x, y, w, h), cv::Scalar(100, 220, 200), 3, cv::LINE_AA);

    // 标题栏
    cv::putText(canvas, title, cv::Point(x+12, y+h-12), cv::FONT_HERSHEY_DUPLEX, 1.0, cv::Scalar(200,255,200), 2, cv::LINE_AA);
}


/*
void GUI::show_images(const gs::DataPacket& packet) {
    cv::Mat rgb, depth_vis, rendered;
    int panel_w = img_width_, panel_h = img_height_;
    int pad = 30;

    // Resize, 并确保颜色顺序
    cv::resize(packet.rgb, rgb, cv::Size(panel_w, panel_h));
    cv::resize(packet.rendered_rgb, rendered, cv::Size(panel_w, panel_h));
    cv::cvtColor(rgb, rgb, cv::COLOR_RGB2BGR);
    cv::cvtColor(rendered, rendered, cv::COLOR_RGB2BGR);

    // 深度
    if (packet.depth.channels() == 3 && packet.depth.type() == CV_8UC3) {
        cv::resize(packet.depth, depth_vis, cv::Size(panel_w, panel_h));
        cv::cvtColor(depth_vis, depth_vis, cv::COLOR_RGB2BGR);
    } else {
        depth_vis = colorize_depth(packet.depth);
        cv::resize(depth_vis, depth_vis, cv::Size(panel_w, panel_h));
        cv::cvtColor(depth_vis, depth_vis, cv::COLOR_RGB2BGR);
    }

    // 大画布（深灰色）
    int canvas_w = panel_w * 3 + pad * 4;
    int canvas_h = panel_h + pad * 2 + 60;
    cv::Mat canvas(canvas_h, canvas_w, CV_8UC3, cv::Scalar(35, 40, 42));

    // 画三张panel
    draw_fancy_panel(canvas, rendered, "Rendered", pad, pad, panel_w, panel_h);
    draw_fancy_panel(canvas, rgb, "Input RGB", pad*2+panel_w, pad, panel_w, panel_h);
    draw_fancy_panel(canvas, depth_vis, "Depth", pad*3+panel_w*2, pad, panel_w, panel_h);

    // 在左上角/底部信息栏加全局文字
    std::ostringstream info1, info2;
    if (packet.ID > 0) {
        info1 << "Online optimization | #Gaussians=" << packet.num_splats << " | #KF=" << packet.num_kf << " | FPS=" << std::fixed << std::setprecision(2) << packet.fps;
        info2 << "Image ID: " << packet.ID;
    } else if (packet.global_iter > 0) {
        info1 << "Global optimization | #KF=" << packet.num_kf << " | Iter=" << packet.global_iter;
        info2 << "Image ID: ---";
    } else {
        info1 << "Start global optimization...";
        info2 << "Image ID: ---";
    }

    // 半透明信息栏
    cv::rectangle(canvas, cv::Rect(0, canvas_h-48, canvas_w, 48), cv::Scalar(20, 20, 20, 140), -1);
    draw_text(canvas, info1.str(), pad, canvas_h-20, 0.85, {255,255,255});
    draw_text(canvas, info2.str(), canvas_w-300, canvas_h-20, 0.85, {200,255,200});

    // 展示/保存
    cv::imshow("GSFusion Fancy Dashboard", canvas);
    cv::imwrite("/app/GSFusion_ROS2_ws/src/GSFusion/vis_folder/check_show_img_" + std::to_string(packet.ID) + ".png", canvas);
    cv::waitKey(10);
}
*/


void GUI::show_images(const gs::DataPacket& packet) {
    cv::Mat rgb, depth_vis, rendered;
    int panel_w = img_width_, panel_h = img_height_;
    int pad = 18;

    // resize & 颜色修正
    cv::resize(packet.rgb, rgb, cv::Size(panel_w, panel_h));
    cv::resize(packet.rendered_rgb, rendered, cv::Size(panel_w, panel_h));
    cv::cvtColor(rgb, rgb, cv::COLOR_RGB2BGR);
    cv::cvtColor(rendered, rendered, cv::COLOR_RGB2BGR);

    // depth可视化
    if (packet.depth.channels() == 3 && packet.depth.type() == CV_8UC3) {
        cv::resize(packet.depth, depth_vis, cv::Size(panel_w, panel_h));
        cv::cvtColor(depth_vis, depth_vis, cv::COLOR_RGB2BGR);
    } else {
        depth_vis = colorize_depth(packet.depth);
        cv::resize(depth_vis, depth_vis, cv::Size(panel_w, panel_h));
        cv::cvtColor(depth_vis, depth_vis, cv::COLOR_RGB2BGR);
    }

    // 整体画布 2x2
    int canvas_w = panel_w * 2 + pad * 3;
    int canvas_h = panel_h * 2 + pad * 3;
    cv::Mat canvas(canvas_h, canvas_w, CV_8UC3, cv::Scalar(35, 40, 42)); // 深灰

    // 画四个panel区域
    auto draw_panel = [&](const cv::Mat& img, const std::string& title, int col, int row) {
        int x = pad + col * (panel_w + pad);
        int y = pad + row * (panel_h + pad);
        cv::Rect roi(x, y, panel_w, panel_h);
        img.copyTo(canvas(roi));
        // 半透明底条
        cv::rectangle(canvas, cv::Rect(x, y+panel_h-40, panel_w, 40), cv::Scalar(30, 30, 30, 180), -1);
        // 高亮边框
        cv::rectangle(canvas, roi, cv::Scalar(100, 220, 200), 1, cv::LINE_AA);
        // 标题
        cv::putText(canvas, title, cv::Point(x+16, y+panel_h-16), cv::FONT_HERSHEY_DUPLEX, 0.9, cv::Scalar(210,250,210), 2, cv::LINE_AA);
    };

    draw_panel(rgb,       "Input RGB",    0, 0);   // 左上
    draw_panel(rendered,  "Rendered",     1, 0);   // 右上
    draw_panel(depth_vis, "Depth",        0, 1);   // 左下

    // 右下信息panel
    int info_x = pad + (panel_w + pad);
    int info_y = pad + (panel_h + pad);
    cv::Rect info_rect(info_x, info_y, panel_w, panel_h);
    cv::rectangle(canvas, info_rect, cv::Scalar(60, 65, 75), -1, cv::LINE_AA); // info区深灰
    cv::rectangle(canvas, info_rect, cv::Scalar(100, 220, 200), 1, cv::LINE_AA);

    // 文本内容
    std::ostringstream info1, info2, info3;
    if (packet.ID > 0) {
        info1 << "Online optimization";
        info2 << "#Gaussians=" << packet.num_splats << " | #KF=" << packet.num_kf;
        info3 << "FPS=" << std::fixed << std::setprecision(2) << packet.fps << " | Image ID: " << packet.ID;
    } else if (packet.global_iter > 0) {
        info1 << "Global optimization";
        info2 << "#KF=" << packet.num_kf << " | Iter=" << packet.global_iter;
        info3 << "Image ID: ---";
    } else {
        info1 << "Start global optimization...";
        info2 << "";
        info3 << "Image ID: ---";
    }
    int font_base = info_y + 48;
    cv::putText(canvas, info1.str(), cv::Point(info_x+32, font_base),     cv::FONT_HERSHEY_SIMPLEX, 1.1, cv::Scalar(255,255,255), 2, cv::LINE_AA);
    cv::putText(canvas, info2.str(), cv::Point(info_x+32, font_base+46),  cv::FONT_HERSHEY_SIMPLEX, 0.95, cv::Scalar(200,255,210), 2, cv::LINE_AA);
    cv::putText(canvas, info3.str(), cv::Point(info_x+32, font_base+92),  cv::FONT_HERSHEY_SIMPLEX, 0.85, cv::Scalar(180,220,255), 2, cv::LINE_AA);

    // 展示/保存
    cv::imshow("GSFusion 2x2 Dashboard", canvas);
    // cv::imwrite("/app/GSFusion_ROS2_ws/src/GSFusion/vis_folder/check_show_img_" + std::to_string(packet.ID) + ".png", canvas);
    cv::waitKey(10);
}




void GUI::run() {
    while (!stop_signal_.load()) {
        if (data_queue_.getSize() == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        gs::DataPacket packet = data_queue_.pop();
        show_images(packet);
    }
    cv::destroyAllWindows();
}

