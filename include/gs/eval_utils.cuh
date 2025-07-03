#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>

namespace gs::eval {
/**
 * Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.
 *
 * \param I1 First image.
 * \param I2 Second image.
 * \return The PSNR value in dB.
 */
static inline float computePSNR(const cv::Mat& I1, const cv::Mat& I2)
{   
    // std::cout << "Computing PSNR..." << std::endl;
    // // Debug: Check image properties
    // std::cout << "I1 - Size: " << I1.size() << ", Channels: " << I1.channels() 
    //     << ", Type: " << I1.type() << ", Depth: " << I1.depth() << std::endl;
    // std::cout << "I2 - Size: " << I2.size() << ", Channels: " << I2.channels() 
    //     << ", Type: " << I2.type() << ", Depth: " << I2.depth() << std::endl;

    // // Check if images are empty
    // if (I1.empty() || I2.empty()) {
    //     std::cerr << "One or both images are empty!" << std::endl;
    //     return -1;
    // }

    // // Check if sizes match
    // if (I1.size() != I2.size()) {
    //     std::cerr << "Image sizes don't match!" << std::endl;
    //     return -1;
    // }

    // // Check if channel counts match
    // if (I1.channels() != I2.channels()) {
    //     std::cerr << "Channel counts don't match!" << std::endl;
    //     return -1;
    // }

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

/**
 * Compute the Structural Similarity Index (SSIM) between two images.
 *
 * \param i1 First image.
 * \param i2 Second image.
 * \return The SSIM value.
 */ 
static inline float computeSSIM(const cv::Mat& i1, const cv::Mat& i2)
{   
    // std::cout << "Computing SSIM..." << std::endl;
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

// /**
//  * Compute the SSIM for color images by averaging the SSIM values of each channel.
//  *
//  * \param img1 First color image.
//  * \param img2 Second color image.
//  * \return The average SSIM value across all channels.
//  */
// static inline float computeSSIM_color(const cv::Mat& img1, const cv::Mat& img2) {
//     std::vector<cv::Mat> channels1, channels2;
//     cv::split(img1, channels1);
//     cv::split(img2, channels2);
//     float ssim = 0.0;
//     for (int i = 0; i < 3; ++i)
//         ssim += computeSSIM(channels1[i], channels2[i]);
//     return ssim / 3.0;
// }

/**
 * Compute the Learned Perceptual Image Patch Similarity (LPIPS) between two images.
 *
 * \param I1 First image in float format.
 * \param I2 Second image in float format.
 * \return The LPIPS value.
 */
static inline float computeLPIPS(const cv::Mat& I1, const cv::Mat& I2, torch::jit::script::Module& lpips)
{
    auto input1 = torch::from_blob(I1.data,
                        {1, I1.rows, I1.cols, 3},
                        torch::kFloat32)
        .permute({0, 3, 1, 2})
        .clone()
        // move to GPU
        .to(torch::kCUDA);
    auto input2 = torch::from_blob(I2.data,
                        {1, I2.rows, I2.cols, 3},
                        torch::kFloat32)
        .permute({0, 3, 1, 2})
        .clone()
        // move to GPU
        .to(torch::kCUDA);

    // normalize into [-1,1]
    input1 = input1 * 2.0 - 1.0;
    input2 = input2 * 2.0 - 1.0;

    // forward on GPU
    auto out = lpips.forward({input1, input2}).toTensor();

    // bring result back to CPU for .item<float>() (optional: .item runs on CPU)
    out = out.to(torch::kCPU);
    return out.item<float>();
}



/**
 * Load the LPIPS model from the specified path.
 *
 * \param model_path Path to the LPIPS model.
 * \return The loaded LPIPS model.
 */
void loadLpipsModel(const std::string& model_path, torch::jit::script::Module& lpips)
{
    try {
        std::cout << "Attempting to load LPIPS from: '" << model_path << "'" << std::endl;
        if (!std::filesystem::exists(model_path)) {
            std::cerr << "Model file not found at: " << model_path << std::endl;
            throw std::runtime_error("LPIPS model file not found");
        }
        lpips = torch::jit::load(model_path);
        // Load the LPIPS model
        lpips = torch::jit::load(model_path);
        lpips.to(torch::kCUDA); // Move the model to GPU
        lpips.eval(); // Set the model to evaluation mode
    } catch (const c10::Error& e) {
        std::cerr << "Error loading LPIPS model: " << e.what() << std::endl;
        throw;
    }
    std::cout << "LPIPS model loaded successfully from " << model_path << std::endl;
}

} // namespace gs::eval
