/*
 * SPDX-FileCopyrightText: 2021 Smart Robotics Lab, Imperial College London, Technical University of Munich
 * SPDX-FileCopyrightText: 2021 Nils Funk
 * SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
 * SPDX-FileCopyrightText: 2024 Smart Robotics Lab, Technical University of Munich
 * SPDX-FileCopyrightText: 2024 Jiaxin Wei
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <opencv2/imgproc.hpp>
#include <se/supereight.hpp>
#include <thread>
#include <torch/torch.h>

#include "config.hpp"
#include "gui.hpp"
#include "gs/gaussian.cuh"
#include "gs/gaussian_utils.cuh"
#include "gs/eval_utils.cuh"
#include "reader.hpp"
#include "se/common/filesystem.hpp"
#include "se/common/system_utils.hpp"


#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

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


int main(int argc, char** argv)
{
    try {
        if (argc != 2) {
            std::cerr << "Usage: " << argv[0] << " YAML_FILE\n";
            return 2;
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

        // Setup log stream
        std::ofstream log_file_stream;
        log_file_stream.open(config.app.log_file);
        se::perfstats.setFilestream(&log_file_stream);

        // Setup input images
        const Eigen::Vector2i input_img_res(config.sensor.width, config.sensor.height);
        se::Image<float> input_depth_img(input_img_res.x(), input_img_res.y());
        se::Image<se::rgb_t> input_colour_img(input_img_res.x(), input_img_res.y(), {0, 0, 0});

        // ========= Map INITIALIZATION  =========
        // Setup the single-res TSDF map w/ default block size of 8 voxels
        se::TSDFColMap<se::Res::Single> map(config.map, config.data);

        // ========= Sensor INITIALIZATION  =========
        // Create a pinhole camera
        const se::PinholeCamera sensor(config.sensor);

        // ========= Gaussian Model INITIALIZATION  =========
        auto optimParams = gs::param::read_optim_params_from_json(config.app.optim_params_path);
        gs::GaussianModel gs_model = gs::GaussianModel(optimParams, config.app.ply_path);
        std::vector<gs::Camera> gs_cam_list;
        std::vector<torch::Tensor> gt_img_list;
        std::vector<torch::Tensor> gt_depth_list;        

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
           << "sh_degree=" << gs_model.optimParams.sh_degree << ", "
           << "source_path="
           << "\"" << config.reader.sequence_path << "\", "
           << "white_background=False)";
        fs.close();

        // ========= GUI INITIALIZATION  =========
        gs::DataQueue data_queue;
        // std::atomic<bool> stop_signal(false);
        // GUI gs_gui(data_queue, stop_signal, input_img_res.x(), input_img_res.y());
        // std::thread gui_thread([&]() { gs_gui.run(); });

        // ========= READER INITIALIZATION  =========
        se::Reader* reader = nullptr;
        reader = se::create_reader(config.reader);

        if (reader == nullptr) {
            return EXIT_FAILURE;
        }

        Eigen::Matrix4f T_WB = Eigen::Matrix4f::Identity(); //< Body to world transformation
        Eigen::Matrix4f T_BS = sensor.T_BS;                 //< Sensor to body transformation
        Eigen::Matrix4f T_WS = T_WB * T_BS;                 //< Sensor to world transformation

        // ========= Integrator INITIALIZATION  =========
        int frame = 0;
        float mean_fps = 0.0f;

        // ========= Open file to store the training views  =========
        std::vector<std::string> training_views_list;
        std::string depth_image_name;
        std::string colour_image_name;
        
        std::size_t processed_frames = 0;

        while (frame != config.app.max_frames) {
            se::perfstats.setIter(frame++);
            // TICK("total")
            TICK("read")
            se::ReaderStatus read_ok = se::ReaderStatus::ok;
            if (config.app.enable_ground_truth || frame == 1) {
                read_ok = reader->nextData(input_depth_img, &depth_image_name,
                                           input_colour_img, &colour_image_name,
                                           T_WB);
                T_WS = T_WB * T_BS;
            }
            else {
                read_ok = reader->nextData(input_depth_img, &depth_image_name,
                                        input_colour_img, &colour_image_name);
            }
            if (read_ok != se::ReaderStatus::ok) {
                break;
            }
            TOCK("read")

            if (frame % config.app.stride == 0) {
                // Save the filenames of training views (colour images)
                training_views_list.push_back(colour_image_name);
                // training_views_file << colour_image_name << "\n";
                ++processed_frames;

                TICK("integration")
                double s = PerfStats::getTime();
                if (frame % config.app.integration_rate == 0) {
                    se::integrator::integrate(map, gs_model, gs_cam_list, gt_img_list, 
                                            gt_depth_list, 
                                            data_queue, 
                                            input_depth_img, input_colour_img,
                                            sensor, T_WS, frame);
                }
                double e = PerfStats::getTime();
                mean_fps += (1 / (e - s));
                TOCK("integration")
                // TOCK("total")
            }

            const bool last_frame = frame == config.app.max_frames || static_cast<size_t>(frame) == reader->numFrames();
            if (last_frame) {
                double s = PerfStats::getTime();

                // Refresh GUI
                gs::DataPacket data_packet;
                data_packet.num_kf = gt_img_list.size();
                data_packet.rgb = cv::Mat(input_img_res.y(), input_img_res.x(), CV_8UC3, cv::Scalar(0, 0, 0));
                data_packet.depth = cv::Mat(input_img_res.y(), input_img_res.x(), CV_8UC3, cv::Scalar(0, 0, 0));
                data_packet.rendered_rgb = cv::Mat(input_img_res.y(), input_img_res.x(), CV_8UC3, cv::Scalar(0, 0, 0));
                data_queue.push(data_packet);

                // Global optimizaiton of reconstructed GS map (offline)
                auto lambda = gs_model.optimParams.lambda_dssim;
                auto iters = gs_model.optimParams.global_iters;
                for (int it = 0; it < iters; it++) {
                    std::vector<int> indices = gs::get_random_indices(gt_img_list.size());
                    for (int i = 0; i < indices.size(); i++) {
                        auto cur_gt_img = gt_img_list[indices[i]];
                        auto cur_gt_depth = gt_depth_list[indices[i]];
                        auto cur_gs_cam = gs_cam_list[indices[i]];

                        auto [image, 
                              depth_img,
                              viewspace_point_tensor, visibility_filter, radii, err] = gs::render(cur_gs_cam, gs_model);

                        // auto valid_mask = (cur_gt_depth > 0); 
                        // auto diff = (depth_img - cur_gt_depth).abs();
                        // auto weighted_diff = diff * valid_mask;
                        // auto depth_loss = weighted_diff.sum() / valid_mask.sum();
                
                        // Loss Computations
                        auto l1_loss = gs::l1_loss(image, cur_gt_img);
                        auto ssim_loss = gs::ssim(image, cur_gt_img, gs::conv_window, gs::window_size, gs::channel);
                        auto loss = (1.f - lambda) * l1_loss + lambda * (1.f - ssim_loss); // + (1.f - lambda) * 0.5 * depth_loss;

                        // Optimization
                        loss.backward();
                        gs_model.optimizer->step();
                        gs_model.optimizer->zero_grad(true);

                        if (i == indices.size() - 1) {
                            auto rendered_img_tensor = image.detach().permute({1, 2, 0}).contiguous().to(torch::kCPU);
                            rendered_img_tensor = rendered_img_tensor.mul(255).clamp(0, 255).to(torch::kU8);
                            auto cv_rendered_img = cv::Mat(image.size(1), image.size(2), CV_8UC3, rendered_img_tensor.data_ptr());
                            data_packet.rendered_rgb = cv_rendered_img;
                            data_packet.global_iter = it + 1;
                            data_queue.push(data_packet);
                        }
                    }
                }
                torch::cuda::synchronize();
                double e = PerfStats::getTime();

                // Get GPU memory usage
                auto mem_after = gs::getGPUMemoryUsage();

                std::cout << "Avg. fps: " << mean_fps / processed_frames << std::endl;
                std::cout << "Global opt. time: " << e - s << " s" << std::endl;
                std::cout << "GPU memory usage: " << mem_after - mem_before << " MB" << std::endl;
                std::cout << "#Keyframes: " << gt_img_list.size() << std::endl;

                // Write mapping statistics to a file
                const std::string stats_file = stdfs::path(config.app.ply_path).parent_path() / "stats";
                std::ofstream fs(stats_file, std::ios::out);
                if (!fs.good()) {
                    std::cerr << "Failed to open stats for writing!" << std::endl;
                }
                fs << "Avg. fps: " << mean_fps / processed_frames << " Hz\n"
                   << "Global opt. time: " << e - s << " s\n"
                   << "GPU memory usage: " << mem_after - mem_before << " MB\n"
                   << "#Keyframes: " << gt_img_list.size() << "\n";

                gs_model.Save_ply(gs_model.output_path, frame, true);
            }

            // Save mesh if enabled
            if ((config.app.meshing_rate > 0 && frame % config.app.meshing_rate == 0) || last_frame) {
                if (!config.app.mesh_path.empty()) {
                    map.saveMesh(config.app.mesh_path + "/mesh_" + std::to_string(frame) + ".ply");
                }
                if (!config.app.slice_path.empty()) {
                    map.saveFieldSlices(config.app.slice_path + "/slice_x_" + std::to_string(frame) + ".vtk",
                                        config.app.slice_path + "/slice_y_" + std::to_string(frame) + ".vtk",
                                        config.app.slice_path + "/slice_z_" + std::to_string(frame) + ".vtk",
                                        se::math::to_translation(T_WS));
                }
                if (!config.app.structure_path.empty()) {
                    map.saveStructure(config.app.structure_path + "/struct_" + std::to_string(frame) + ".ply");
                }
            }

            se::perfstats.sample("memory usage", se::system::memory_usage_self() / 1024.0 / 1024.0, PerfStats::MEMORY);
            se::perfstats.writeToFilestream();
            printProgress(static_cast<double>(frame) / (static_cast<double>(reader->numFrames()) - 1));
        }

        // ========= Optional evaluation ==================
        if (config.app.eval) {
            // Initialize the lists for evaluation metrics
            std::vector<float> psnr_training, ssim_training, lpips_training;
            std::vector<float> psnr_novel, ssim_novel, lpips_novel;

            // Create directory for rendered images
            const std::string rendered_dir = config.reader.sequence_path + "/rendered_color";
            if (!stdfs::create_directories(rendered_dir)) {
                std::cerr << "Warning: could not create (or already existed) directory " 
                        << rendered_dir << std::endl;
            }

            // Create directory for rendered depth images
            const std::string rendered_depth_dir = config.reader.sequence_path + "/rendered_depth";
            if (!stdfs::create_directories(rendered_depth_dir)) {
                std::cerr << "Warning: could not create (or already existed) directory " 
                        << rendered_depth_dir << std::endl;
            }

            // Load the Lpips model for evaluation
            // initialize the LPIPS model
            torch::jit::script::Module lpips_model;
            gs::eval::loadLpipsModel(config.reader.lpips_model_path, lpips_model);
            
            // reset reader and frame count
            gs::Camera camera;
            frame = 0;
            // reset frame for evaluation
            reader->restart();

            while (frame != config.app.max_frames) {
                se::perfstats.setIter(frame++);
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
                                            sensor.model.focalLengthU(), sensor.model.focalLengthV(),
                                            sensor.model.imageCenterU(), sensor.model.imageCenterV())
                        .to(torch::kCUDA, true);
                camera.width = input_colour_img.width();
                camera.height = input_colour_img.height();
                camera.fov_x = sensor.horizontal_fov;
                camera.fov_y = sensor.vertical_fov;
                camera.T_W2C = W2C_matrix;
                camera.full_proj_matrix = W2C_matrix.mm(proj_matrix);
                camera.cam_center = W2C_matrix.inverse()[3].slice(0, 0, 3);
                // Rendering
                auto [image,
                      depth_img,
                      viewspace_point_tensor, visibility_filter, radii, err] = gs::render(camera, gs_model);
                
                // save the rendered rgb image
                auto rendered_img_tensor = image.detach().permute({1, 2, 0}).contiguous().to(torch::kCPU);
                rendered_img_tensor = rendered_img_tensor.mul(255).clamp(0, 255).to(torch::kU8);
                auto cv_rendered_img = cv::Mat(image.size(1), image.size(2), CV_8UC3, rendered_img_tensor.data_ptr());

                cv::Mat rendered_bgr;
                cv::cvtColor(cv_rendered_img, rendered_bgr, cv::COLOR_RGB2BGR);

                // Save the rendered image
                std::string color_image_file = rendered_dir + "/" + colour_image_name;
                if (!cv::imwrite(color_image_file, rendered_bgr)) {
                    std::cerr << "Failed to save rendered color image to " 
                            << color_image_file << std::endl;
                }
                // Save the rendered depth image
                const float min_d = 0.4f, max_d = 6.0f;
                auto dep_t = depth_img.detach()
                                .permute({1,2,0})
                                .contiguous()
                                .to(torch::kCPU); // still float
                // assume depth tensor is [H,W,1], so squeeze
                dep_t = dep_t.squeeze();
                // convert to uchar with range [0,255]
                cv::Mat dep_f(image.size(1), image.size(2), CV_32FC1, dep_t.data_ptr());
                cv::Mat dep_u8;
                dep_f.convertTo(dep_u8, CV_8UC1,
                                255.0f / (max_d - min_d),
                                -255.0f * min_d / (max_d - min_d));
                cv::applyColorMap(dep_u8, dep_u8, cv::COLORMAP_VIRIDIS);
                cv::cvtColor(dep_u8, dep_u8, cv::COLOR_BGR2RGB);

                const std::string out_depth_fn = rendered_depth_dir + "/" + depth_image_name;
                if (!cv::imwrite(out_depth_fn, dep_u8)) {
                    std::cerr << "Failed to save depth image to " << out_depth_fn << "\n";
                }

                // Compute eveluation metrics
                // For evaluation, we assume the ground truth is available
                // Check if the current camera view is in the training views
                if (std::find(training_views_list.begin(), training_views_list.end(), colour_image_name) != training_views_list.end()) {
                    // If it is, we can compute the evaluation metrics
                    auto gt_training_view = cv::imread(config.reader.sequence_path + "/rgb/" + colour_image_name, cv::IMREAD_COLOR);

                    // Compute PSNR
                    float psnr = gs::eval::computePSNR(gt_training_view, rendered_bgr);
                    // Compute SSIM
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
                    psnr_novel.push_back(psnr);
                    ssim_novel.push_back(ssim);
                    lpips_novel.push_back(lpips);
                }
                torch::cuda::synchronize();
                printProgress(static_cast<double>(frame) / (static_cast<double>(reader->numFrames()) - 1));
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
            eval_fs << "Training views evaluation metrics:\n";
            eval_fs << "PSNR: " << mean(psnr_training) << "\n";
            eval_fs << "SSIM: " << mean(ssim_training) << "\n";
            eval_fs << "LPIPS: " << mean(lpips_training) << "\n";
            eval_fs << "Novel views evaluation metrics:\n";
            eval_fs << "PSNR: " << mean(psnr_novel) << "\n";
            eval_fs << "SSIM: " << mean(ssim_novel) << "\n";
            eval_fs << "LPIPS: " << mean(lpips_novel) << "\n";
            eval_fs.close();

            std::cout << "Evaluation results saved to " << eval_file << "\n";

            std::cout << "============= Evaluation Results =============\n";
            std::cout << "Training views:\n";
            std::cout << "PSNR: " << mean(psnr_training) << "\n";
            std::cout << "SSIM: " << mean(ssim_training) << "\n";
            std::cout << "LPIPS: " << mean(lpips_training) << "\n";
            std::cout << "Novel views:\n";
            std::cout << "PSNR: " << mean(psnr_novel) << "\n";
            std::cout << "SSIM: " << mean(ssim_novel) << "\n";
            std::cout << "LPIPS: " << mean(lpips_novel) << "\n";
            std::cout << "==============================================\n";    
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
