#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <numeric>
#include <torch/script.h>

namespace fs = std::filesystem;

float computePSNR(const cv::Mat& I1, const cv::Mat& I2) {
    cv::Mat s1;
    absdiff(I1, I2, s1);    // |I1 - I2|
    s1.convertTo(s1, CV_32F);
    s1 = s1.mul(s1);
    float sse = sum(s1)[0] + sum(s1)[1] + sum(s1)[2];
    if (sse <= 1e-10) return 100;
    else {
        float mse = sse / (float)(I1.channels() * I1.total());
        float psnr = 10.0 * log10((255 * 255) / mse);
        return psnr;
    }
}

float computeSSIM(const cv::Mat& i1, const cv::Mat& i2) {
    cv::Mat I1, I2;
    i1.convertTo(I1, CV_32F);
    i2.convertTo(I2, CV_32F);
    float C1 = 6.5025, C2 = 58.5225;
    cv::Mat I1_2 = I1.mul(I1);        // I1^2
    cv::Mat I2_2 = I2.mul(I2);        // I2^2
    cv::Mat I1_I2 = I1.mul(I2);       // I1 * I2
    cv::Mat mu1, mu2;
    cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);
    cv::Mat mu1_2 = mu1.mul(mu1);
    cv::Mat mu2_2 = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);
    cv::Mat sigma1_2, sigma2_2, sigma12;
    cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;
    cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;
    cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;
    cv::Mat t1, t2, t3;
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2); // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2); // t1 = ((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
    cv::Mat ssim_map;
    divide(t3, t1, ssim_map);
    return mean(ssim_map)[0];
}

float computeSSIM_color(const cv::Mat& img1, const cv::Mat& img2) {
    std::vector<cv::Mat> channels1, channels2;
    cv::split(img1, channels1);
    cv::split(img2, channels2);
    float ssim = 0.0;
    for (int i = 0; i < 3; ++i)
        ssim += computeSSIM(channels1[i], channels2[i]);
    return ssim / 3.0;
}

// img1, img2: CV_32FC3, 0~1
float computeLPIPS(const cv::Mat& img1, const cv::Mat& img2, torch::jit::script::Module& lpips) {
    torch::Tensor input1 = torch::from_blob(img1.data, {1, img1.rows, img1.cols, 3}, torch::kFloat32).permute({0, 3, 1, 2}).clone();
    torch::Tensor input2 = torch::from_blob(img2.data, {1, img2.rows, img2.cols, 3}, torch::kFloat32).permute({0, 3, 1, 2}).clone();

    input1 = input1 * 2 - 1;
    input2 = input2 * 2 - 1;
    std::vector<torch::jit::IValue> inputs{input1, input2};
    auto out = lpips.forward(inputs).toTensor();
    return out.item<float>();
}

int main(int argc, char** argv) {
    
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <rendered_dir> <gt_dir> <lpips_vgg.pt>\n";
        return 1;
    }
    // assert if rendered and gt images are having the same name

    std::string render_dir = argv[1];
    std::string gt_dir = argv[2];
    std::string lpips_path = argv[3];

    torch::jit::script::Module lpips;
    try {
        lpips = torch::jit::load(lpips_path);
        lpips.to(torch::kCUDA);
        lpips.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model from " << lpips_path << std::endl;
        return 3;
    }

    std::vector<float> psnr_all, ssim_all, lpips_all;

    for (const auto& entry : std::filesystem::directory_iterator(render_dir)) {
        if (!entry.is_regular_file()) continue;
        std::string filename = entry.path().filename().string();
        std::string gt_path = gt_dir + "/" + filename;

        if (!fs::exists(gt_path)) {
            std::cerr << "Error: GT image " << gt_path << " not found for rendered image " << filename << ".\n";
            return 100;
        }

        cv::Mat img1 = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
        cv::Mat img2 = cv::imread(gt_path, cv::IMREAD_COLOR);
        if (img1.empty() || img2.empty() || img1.size() != img2.size()) {
            std::cout << "Skip " << filename << std::endl;
            continue;
        }

        float psnr = computePSNR(img1, img2);
        float ssim = computeSSIM_color(img1, img2);

        cv::Mat img1_f, img2_f;
        img1.convertTo(img1_f, CV_32FC3, 1.0 / 255.0);
        img2.convertTo(img2_f, CV_32FC3, 1.0 / 255.0);
        float lpips_val = computeLPIPS(img1_f, img2_f, lpips);

        psnr_all.push_back(psnr);
        ssim_all.push_back(ssim);
        lpips_all.push_back(lpips_val);

        std::cout << filename << ": PSNR=" << psnr << ", SSIM=" << ssim << ", LPIPS=" << lpips_val << std::endl;
    }

    // for (const auto& entry : fs::directory_iterator(gt_dir)) {
    //     if (!entry.is_regular_file()) continue;
    //     std::string filename = entry.path().filename().string();
    //     std::string render_path = render_dir + "/" + filename;
    //     if (!fs::exists(render_path)) {
    //         std::cerr << "Error: Rendered image " << render_path << " not found for GT image " << filename << ".\n";
    //         return 101; // 直接终止
    //     }
    // }

    auto mean = [](const std::vector<float>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    };

    std::cout << "Avg PSNR: " << mean(psnr_all) << std::endl;
    std::cout << "Avg SSIM: " << mean(ssim_all) << std::endl;
    std::cout << "Avg LPIPS: " << mean(lpips_all) << std::endl;
}