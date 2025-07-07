/*
 * SPDX-FileCopyrightText: 2016-2019 Emanuele Vespa
 * SPDX-FileCopyrightText: 2021 Smart Robotics Lab, Imperial College London, Technical University of Munich
 * SPDX-FileCopyrightText: 2021 Nils Funk
 * SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
 * SPDX-FileCopyrightText: 2024 Smart Robotics Lab, Technical University of Munich
 * SPDX-FileCopyrightText: 2024 Jiaxin Wei
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef SE_SINGLERES_TSDF_GS_UPDATER_IMPL_HPP
#define SE_SINGLERES_TSDF_GS_UPDATER_IMPL_HPP

#include <algorithm>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cmath>
#include <opencv2/opencv.hpp>

#include "gs/loss_utils.cuh"
#include "gs/render_utils.cuh"

namespace se {

// Single-res TSDF updater
template<Colour ColB, Semantics SemB, int BlockSize, typename SensorT>
GSUpdater<Map<Data<Field::TSDF, ColB, SemB>, Res::Single, BlockSize>, SensorT>::GSUpdater(MapType& map,
                                                                                          const SensorT& sensor,
                                                                                          gs::GaussianModel& gs_model,
                                                                                          std::vector<gs::Camera>& gs_cam_list,
                                                                                          std::vector<torch::Tensor>& gt_img_list,
                                                                                          std::vector<torch::Tensor>& gt_depth_list,
                                                                                          gs::DataQueue& data_queue,
                                                                                          const Image<float>& depth_img,
                                                                                          const Image<rgb_t>* colour_img,
                                                                                          const std::vector<std::tuple<Eigen::Vector2f, Eigen::Vector3f, float, int>>& keypoints,
                                                                                          const Image<semantics_t>* class_img,
                                                                                          const Eigen::Matrix4f& T_WS,
                                                                                          const int frame) :
        map_(map),
        sensor_(sensor),
        gs_model_(gs_model),
        gs_cam_list_(gs_cam_list),
        gt_img_list_(gt_img_list),
        gt_depth_list_(gt_depth_list),
        data_queue_(data_queue),
        depth_img_(depth_img),
        colour_img_(colour_img),
        keypoints_(keypoints),
        class_img_(class_img),
        T_WS_(T_WS),
        frame_(frame),
        config_(map)
{
    // Construct torch::Tensor RGB image used for optimization
    start_time_ = PerfStats::getTime();
    for (size_t i = 0; i < colour_img_->size(); i++) {
        color_data_.push_back(colour_img_->data()[i].r);
        color_data_.push_back(colour_img_->data()[i].g);
        color_data_.push_back(colour_img_->data()[i].b);
    }

    torch::Tensor image_tensor = torch::from_blob(color_data_.data(), {colour_img_->height(), colour_img_->width(), 3}, {colour_img_->width() * 3, 3, 1}, torch::kUInt8);
    cur_gt_img_ = image_tensor.to(torch::kFloat32).permute({2, 0, 1}).clone() / 255.f;
    cur_gt_img_ = torch::clamp(cur_gt_img_, 0.f, 1.f).to(torch::kCUDA, true);
    gt_img_list_.push_back(cur_gt_img_);

    cur_gt_depth_ = torch::from_blob(
        const_cast<float*>(depth_img_.data()),
        {depth_img_.height(), depth_img_.width()},
        {depth_img_.width(), 1},
        torch::kFloat32
      )
      .clone()               // now owns its own CPU memory
      .unsqueeze(0)          // → (1, H, W)
      .to(torch::kCUDA, true);
    gt_depth_list_.push_back(cur_gt_depth_);

    // gray scale image for computing the image gradient for densification
    rgb_weights_ = torch::tensor({0.2989f, 0.5870f, 0.1140f}, torch::TensorOptions().dtype(torch::kFloat32).device(cur_gt_img_.device()));
    cur_gt_img_gray_ = (cur_gt_img_ * rgb_weights_.view({3, 1, 1})).sum(0); // (H, W), CUDA, float32, [0,1]

    // Construct gs::Camera used for rendering
    Eigen::Matrix4f T_SW = math::to_inverse_transformation(T_WS_);
    torch::Tensor W2C_matrix = torch::from_blob(T_SW.data(), {4, 4}, torch::kFloat).clone().to(torch::kCUDA, true);
    torch::Tensor proj_matrix =
        gs::getProjectionMatrix(colour_img_->width(), colour_img_->height(), sensor_.model.focalLengthU(), sensor_.model.focalLengthV(), sensor_.model.imageCenterU(), sensor_.model.imageCenterV())
            .to(torch::kCUDA, true);
    cur_gs_cam_.width = colour_img_->width();
    cur_gs_cam_.height = colour_img_->height();
    cur_gs_cam_.fov_x = sensor_.horizontal_fov;
    cur_gs_cam_.fov_y = sensor_.vertical_fov;
    cur_gs_cam_.T_W2C = W2C_matrix;
    cur_gs_cam_.full_proj_matrix = W2C_matrix.mm(proj_matrix);
    cur_gs_cam_.cam_center = W2C_matrix.inverse()[3].slice(0, 0, 3);
    gs_cam_list_.push_back(cur_gs_cam_);

    // Construct cv::Mat colored depth image for visualization
    std::vector<float> depth_data;
    for (size_t i = 0; i < depth_img_.size(); i++) {
        depth_data.push_back(depth_img_.data()[i]);
    }
    cv::Mat cv_src_depth(depth_img_.height(), depth_img_.width(), CV_32FC1, depth_data.data());

    float min_depth = 0.4;
    float max_depth = 6.0;
    cv_src_depth.convertTo(cv_src_depth, CV_8UC3, 255 / (max_depth - min_depth), -255 * min_depth / (max_depth - min_depth));
    cv::applyColorMap(cv_src_depth, cv_src_depth, cv::COLORMAP_VIRIDIS);
    cv::cvtColor(cv_src_depth, cv_src_depth, cv::COLOR_BGR2RGB);
    data_packet_.depth = cv_src_depth;
}


template<Colour ColB, Semantics SemB, int BlockSize, typename SensorT>
void GSUpdater<Map<Data<Field::TSDF, ColB, SemB>, Res::Single, BlockSize>, SensorT>::operator()(std::vector<OctantBase*>& block_ptrs)
{
    const bool has_colour = colour_img_;
    const bool has_semantics = class_img_;
    constexpr int block_size = BlockType::getSize();
    const Eigen::Matrix4f T_SW = math::to_inverse_transformation(T_WS_);
    const Eigen::Matrix3f C_SW = math::to_rotation(T_SW);

#pragma omp parallel for
    for (unsigned int i = 0; i < block_ptrs.size(); i++) {
        BlockType& block = *static_cast<BlockType*>(block_ptrs[i]);
        block.setTimeStamp(frame_);
        const Eigen::Vector3i block_coord = block.getCoord();
        Eigen::Vector3f point_base_W;
        map_.voxelToPoint(block_coord, point_base_W);
        const Eigen::Vector3f point_base_S = (T_SW * point_base_W.homogeneous()).head<3>();
        const Eigen::Matrix3f point_delta_matrix_S = C_SW * map_.getRes();

        for (unsigned int z = 0; z < block_size; ++z) {
            for (unsigned int y = 0; y < block_size; ++y) {
                for (unsigned int x = 0; x < block_size; ++x) {
                    // Set voxel coordinates
                    const Eigen::Vector3i voxel_coord = block_coord + Eigen::Vector3i(x, y, z);

                    // Set sample point in camera frame
                    const Eigen::Vector3f point_S = point_base_S + point_delta_matrix_S * Eigen::Vector3f(x, y, z);

                    if (point_S.norm() > sensor_.farDist(point_S)) {
                        continue;
                    }

                    // Project sample point to the image plane.
                    Eigen::Vector2f pixel_f;
                    if (sensor_.model.project(point_S, &pixel_f) != srl::projection::ProjectionStatus::Successful) {
                        continue;
                    }
                    const Eigen::Vector2i pixel = se::round_pixel(pixel_f);
                    const int pixel_idx = pixel.x() + depth_img_.width() * pixel.y();

                    // Fetch the image value.
                    const float depth_value = depth_img_[pixel_idx];

                    if (depth_value < sensor_.near_plane) {
                        continue;
                    }

                    // Update the TSDF
                    const float m = sensor_.measurementFromPoint(point_S);
                    const float sdf_value = point_S.norm() * (depth_value - m) / m;

                    if (sdf_value > -config_.truncation_boundary) {
                        DataType& data = block.getData(voxel_coord);

                        updateVoxel(data, sdf_value);

                        if constexpr (MapType::col_ == Colour::On) {
                            if (has_colour) {
                                updateVoxelColour(data, (*colour_img_)[pixel_idx]);
                            }
                        }
                        if constexpr (MapType::sem_ != Semantics::Off) {
                            if (has_semantics) {
                                updateVoxelSemantics(data, (*class_img_)[pixel_idx]);
                            }
                        }
                    }
                } // x
            }     // y
        }         // z
    }

    propagator::propagateTimeStampToRoot(block_ptrs);

    cv::Mat cv_src_img(colour_img_->height(), colour_img_->width(), CV_8UC3, color_data_.data());
    data_packet_.rgb = cv_src_img;

    gs::QTree qtree(gs_model_.optimParams.qtree_min_pixel_size, 
                    cv_src_img, // type cv::Mat
                    keypoints_, // type vector<Eigen::Vector2f>
                    gs_model_.optimParams.qtree_min_num_leaves, 
                    gs_model_.optimParams.post_subdivision);
                
    qtree.subdivide();

    std::vector<gs::Node*> nodes = qtree.getAllLeafPtrs();
    int num_nodes = nodes.size();
    int num_keypoints = keypoints_.size();

    std::vector<gs::Point> positions(num_nodes);
    std::vector<gs::Color> colors(num_nodes);
    std::vector<float> scales(num_nodes, 0);

    std::vector<gs::Point> positions_keypoints;
    std::vector<gs::Color> colors_keypoints;
    std::vector<float> scales_keypoints;
    
    positions_keypoints.reserve(num_keypoints);
    colors_keypoints.reserve(num_keypoints);
    scales_keypoints.reserve(num_keypoints);

#pragma omp parallel for
    for (int i = 0; i < nodes.size(); i++) {
        gs::Node* node_ptr = nodes[i];
        const bool keypoint_exists = node_ptr->has_keypoint_;
        const int num_observations = node_ptr->getKeypointObservations();

        // Backproject the cell center
        Eigen::Vector3f center;
        Eigen::Vector2f p2d(node_ptr->getOriginX() + 0.5 * node_ptr->getWidth(), node_ptr->getOriginY() + 0.5 * node_ptr->getHeight());

        // add position color scales of center point

        const Eigen::Vector2i pixel = se::round_pixel(p2d);
        const int pixel_idx = pixel.x() + depth_img_.width() * pixel.y();
        const float depth_value = depth_img_[pixel_idx];
        if (depth_value < sensor_.near_plane && !keypoint_exists) {
            continue;
        }

        // scale of the center gaussian
        float length = sqrt(pow(0.5 * node_ptr->getWidth(), 2) + pow(0.5 * node_ptr->getHeight(), 2));
        float scale = (depth_value * length) / sensor_.model.focalLengthU();

        if (depth_value >= sensor_.near_plane) {
            center *= depth_value;
            center = (T_WS_ * center.homogeneous()).head<3>();

            // Check the vicinity of the backprojected cell center in 3D space
            // this is only for reprojected depth points
            auto center_data = map_.getData(center);
            if (center_data.weight != 1) {
                continue;
            }

            scales[i] = scale;

            gs::Point center_p3d;
            center_p3d.x = center[0];
            center_p3d.y = center[1];
            center_p3d.z = center[2];
            positions[i] = center_p3d;

            auto center_rgb = (*colour_img_)[pixel_idx];
            gs::Color center_color;
            center_color.r = center_rgb.r;
            center_color.g = center_rgb.g;
            center_color.b = center_rgb.b;
            colors[i] = center_color;
        }

        if (keypoint_exists) {
            if (num_observations > gs_model_.optimParams.max_shared_observations) {
                continue; // skip nodes with too many redundant observations
            }
            // add position color scales of keypoint
            Eigen::Vector2f keypoint_2d = node_ptr->getNodeKeypoint();
            Eigen::Vector3f mappoint = node_ptr->getNodeMappoint();

            // keypoint scal1.5e
            float keypoint_scale = (node_ptr->getKeypointDepth() * 1.5) / sensor_.model.focalLengthU();
            scales_keypoints.push_back(keypoint_scale);

            // keypoint 3d position
            gs::Point keypoint_3d;
            keypoint_3d.x = mappoint[0];
            keypoint_3d.y = mappoint[1];
            keypoint_3d.z = mappoint[2];
            positions_keypoints.push_back(keypoint_3d);

            // keypoint color
            const int x = static_cast<int>(keypoint_2d.x());
            const int y = static_cast<int>(keypoint_2d.y());
            const int keypoint_pix_idx = x + depth_img_.width() * y;
            auto keypoint_rgb = (*colour_img_)[keypoint_pix_idx];
            gs::Color keypoint_color;
            keypoint_color.r = keypoint_rgb.r;
            keypoint_color.g = keypoint_rgb.g;
            keypoint_color.b = keypoint_rgb.b;
            colors_keypoints.push_back(keypoint_color);
        }
    }
    // Filter out invalid cells
    std::vector<gs::Point> valid_positions;
    std::vector<gs::Color> valid_colors;
    std::vector<float> valid_scales;
    for (int i = 0; i < scales.size(); i++) {
        if (scales[i] > 0) {
            valid_positions.push_back(positions[i]);
            valid_colors.push_back(colors[i]);
            valid_scales.push_back(scales[i]);
        }
    }

    // Control the growth rate introduced by new keypoints
    int P = gs_model_.Get_size() + valid_positions.size();
    int max_keypoints = static_cast<int>(gs_model_.optimParams.growth_rate * P);

    // Only subsample if needed
    if (num_keypoints > max_keypoints) {
        std::mt19937 rng(std::random_device{}());
        
        // Unrolled Fisher-Yates
        for (int i = positions_keypoints.size() - 1; i > 0; --i) {
            std::uniform_int_distribution<int> dist(0, i);
            int j = dist(rng);
            std::swap(positions_keypoints[i], positions_keypoints[j]);
            std::swap(colors_keypoints[i], colors_keypoints[j]);
            std::swap(scales_keypoints[i], scales_keypoints[j]);
        }
        positions_keypoints.resize(max_keypoints);
        colors_keypoints.resize(max_keypoints);
        scales_keypoints.resize(max_keypoints);
    }
    
    // Update keyframe list
    if (valid_positions.size() > gs_model_.optimParams.kf_thresh) {
        isKeyframe_ = true;
    }
    else {
        // Only keep non-keyframes for ScanNet++ dataset
        if (!gs_model_.optimParams.keep_all_frames) {
            gs_cam_list_.pop_back();
            gt_img_list_.pop_back();
        }
    }

    valid_positions.reserve(valid_positions.size() + positions_keypoints.size());
    valid_colors.reserve(valid_colors.size() + colors_keypoints.size());
    valid_scales.reserve(valid_scales.size() + scales_keypoints.size());

    // Use insert with move iterators
    valid_positions.insert(valid_positions.end(), 
                          std::make_move_iterator(positions_keypoints.begin()),
                          std::make_move_iterator(positions_keypoints.end()));
                        
    valid_colors.insert(valid_colors.end(),
                        std::make_move_iterator(colors_keypoints.begin()),
                        std::make_move_iterator(colors_keypoints.end()));
                    
    valid_scales.insert(valid_scales.end(),
                        std::make_move_iterator(scales_keypoints.begin()),
                        std::make_move_iterator(scales_keypoints.end()));

    updateGSModel(valid_positions, valid_colors, valid_scales);
}


template<Colour ColB, Semantics SemB, int BlockSize, typename SensorT>
void GSUpdater<Map<Data<Field::TSDF, ColB, SemB>, Res::Single, BlockSize>, SensorT>::updateVoxel(DataType& data, float sdf_value)
{
    weight::increment(data.weight, map_.getDataConfig().max_weight);
    const tsdf_t tsdf_value = math::clamp(sdf_value / config_.truncation_boundary, -1.0f, 1.0f) * tsdf_t_scale;
    data.tsdf = (data.tsdf * (data.weight - 1) + tsdf_value) / data.weight;
}


template<Colour ColB, Semantics SemB, int BlockSize, typename SensorT>
void GSUpdater<Map<Data<Field::TSDF, ColB, SemB>, Res::Single, BlockSize>, SensorT>::updateVoxelColour(DataType& data, rgb_t colour_value)
{
    // Use if instead of std::min to prevent overflow.
    if (data.rgb_weight < map_.getDataConfig().max_weight) {
        data.rgb_weight++;
    }
    // No overflow occurs due to integral promotion to int or unsigned int during arithmetic operations.
    data.rgb.r = (data.rgb.r * (data.rgb_weight - 1) + colour_value.r) / data.rgb_weight;
    data.rgb.g = (data.rgb.g * (data.rgb_weight - 1) + colour_value.g) / data.rgb_weight;
    data.rgb.b = (data.rgb.b * (data.rgb_weight - 1) + colour_value.b) / data.rgb_weight;
}


template<Colour ColB, Semantics SemB, int BlockSize, typename SensorT>
void GSUpdater<Map<Data<Field::TSDF, ColB, SemB>, Res::Single, BlockSize>, SensorT>::updateVoxelSemantics(DataType& data, semantics_t class_id)
{
    // Use if instead of std::min to prevent overflow.
    if (data.sem_weight < map_.getDataConfig().max_weight) {
        data.sem_weight++;
    }
    data.sem.merge(class_id, data.sem_weight);
}


template<Colour ColB, Semantics SemB, int BlockSize, typename SensorT>
void GSUpdater<Map<Data<Field::TSDF, ColB, SemB>, Res::Single, BlockSize>, SensorT>::updateGSModel(std::vector<gs::Point>& positions, std::vector<gs::Color>& colors, std::vector<float>& scales)
{
    // Spawn new primitives to the Gaussian Spaltting model
    if (positions.size() != 0) {
        torch::NoGradGuard no_grad;
        gs_model_.Add_gaussians(positions, colors, scales);
    }

    int iters = gs_model_.optimParams.kf_iters;
    if (!isKeyframe_) {
        iters = gs_model_.optimParams.non_kf_iters;
    }

    std::vector<int> kf_indices = gs::get_random_indices(gt_img_list_.size());

    // Start online optimization
    for (int iter = 0; iter < iters; iter++) {
        auto [image, viewspace_point_tensor, visibility_filter, radii] = gs::render(cur_gs_cam_, gs_model_);
        // Optimization
        // Loss Computations
        auto loss = gs::l1_loss(image, cur_gt_img_);
        loss.backward();
        gs_model_.optimizer->step();
        gs_model_.optimizer->zero_grad(true);

        // Store the cv::Mat rendered image for visualization
        if (iter == iters - 1) {
            auto rendered_img_tensor = image.detach().permute({1, 2, 0}).contiguous().to(torch::kCPU);
            rendered_img_tensor = rendered_img_tensor.mul(255).clamp(0, 255).to(torch::kU8);
            auto cv_rendered_img = cv::Mat(image.size(1), image.size(2), CV_8UC3, rendered_img_tensor.data_ptr());
            data_packet_.rendered_rgb = cv_rendered_img;
        }
    }

    if (!isKeyframe_) {
        int kf_iters = gs_model_.optimParams.random_kf_num;
        if (kf_indices.size() < kf_iters) {
            kf_iters = kf_indices.size();
        }
        for (int i = 0; i < kf_iters; i++) {
            auto kf_gt_img = gt_img_list_[kf_indices[i]];
            auto kf_gs_cam = gs_cam_list_[kf_indices[i]];

            auto [image, viewspace_point_tensor, visibility_filter, radii] = gs::render(kf_gs_cam, gs_model_);
            auto loss = gs::l1_loss(image, kf_gt_img);
            loss.backward();
            gs_model_.optimizer->step();
            gs_model_.optimizer->zero_grad(true);
        }
    }

    // Collect mapping statistics
    torch::cuda::synchronize();
    end_time_ = PerfStats::getTime();
    data_packet_.fps = 1 / (end_time_ - start_time_);
    data_packet_.ID = frame_;
    data_packet_.num_splats = gs_model_.Get_size();
    data_packet_.num_kf = gt_img_list_.size();
    data_queue_.push(data_packet_);
}

} // namespace se
#endif // SE_SINGLERES_TSDF_GS_UPDATER_IMPL_HPP
