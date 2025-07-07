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

