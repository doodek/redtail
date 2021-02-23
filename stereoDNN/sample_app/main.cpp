// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Full license terms provided in LICENSE.md file.

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include <cuda_runtime_api.h>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <cudnn.h>
#include <opencv2/opencv.hpp>

#include "redtail_tensorrt_plugins.h"
#include "networks.h"


#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <vector>

#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#ifndef _WIN32
#include <unistd.h>
#endif

#define CAM_APPSINK (1 << 0)
#define CAM_UDPSINK (1 << 1)
#define CAM_TIMEOVERLAY (1 << 2)
#define CAM_IDOVERLAY (1 << 3)
#define CAM_FAKE (1 << 4)
#define CAM_NOT_NVIDIA (1 << 5)
#define CAM_NVIDIA (1 << 6)

using namespace std;

class GStCamera {
 public:
  unsigned int flags;
  unsigned int id;
  unsigned int width;
  unsigned int height;
  unsigned int framerate;
  string sinkFmt;

  string GStString;
  cv::VideoCapture cap;

  GStCamera(unsigned int id,
            unsigned int width,
            unsigned int height,
            unsigned int framerate,
            uint8_t flags,
            string sinkFormat,
            string udpParams);

  string UDPString(string recvIP, int port);
};

string GStCamera::UDPString(string recvIP, int port) {
  stringstream ss;

  ss << "host=" << recvIP << " port=" << port;

  return ss.str();
}

GStCamera::GStCamera(unsigned int id,
                     unsigned int width = 1920,
                     unsigned int height = 1080,
                     unsigned int framerate = 30,
                     uint8_t flags = CAM_NVIDIA,
                     string sinkFormat = "BGR",
                     string udpParams = "") {
  cout << "Creating cam" << endl;
  this->flags = flags;
  this->id = id;
  this->width = width;
  this->height = height;
  this->framerate = framerate;
  this->sinkFmt = sinkFormat;

  stringstream overlays_ss;

  if (flags & (CAM_TIMEOVERLAY)) {
    overlays_ss << " ! timeoverlay ";
    cout << "GST lib: added time overlay" << endl;
  }

  if (flags & (CAM_IDOVERLAY)) {
    overlays_ss << " ! textoverlay text =" << id
                << " valigment=top haligment=top font-desc=\"Sans, 72\"";
    cout << "GST lib: added cam ID overlay" << endl;
  }

  string overlays = overlays_ss.str();

  /*if(flags & (CAM_APPSINK | CAM_UDPSINK))
  {
      ss  <<  "nvarguscamerasrc sensor-id="
          << id
          << " ! video/x-raw(memory:NVMM), format=NV12, width="
          << width
          << ", height="
          << height
          << " tee name=t t. ! nvvidconv ! video/x-raw, width="
              << width
              << ", height="
              << height
              << ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)"
              << sinkFormat
              << " ! videoconvert "
              << overlays
              << " ! appsink "
          << "t. ! nvv4l2h264enc insert-sps-pps=true ! h264parse ! rtph264pay pt=96 ! udpsink "
          << udpParams;
  }    */

  if (1 == 0) /*(flags & (CAM_APPSINK | CAM_NOT_NVIDIA))*/ {
    this->cap = cv::VideoCapture(id);
    cout << "Not accelerated gstreamer" << endl;
  }

  else if (flags & (CAM_APPSINK | CAM_NVIDIA)) {
    stringstream ss;
    ss << "nvarguscamerasrc sensor-id=" << id
       << " ! video/x-raw(memory:NVMM), format=NV12, width=" << width << ", height=" << height
       << " ! nvvidconv ! video/x-raw, width=" << width << ", height=" << height
       << ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)" << sinkFormat
       << " ! videoconvert ! appsink ";
    string GStString = ss.str();
    cout << "Accelerated gstreamer " << endl;
    cout << GStString << endl;

    this->cap = cv::VideoCapture(GStString);
  } else if (flags & (CAM_APPSINK | CAM_FAKE)) {
    string GStString =
        " videotestsrc ! video/x-raw,width=1920,height=1080,framerate=30/1,format=BGR ! "
        " videoconvert ! appsink ";
    cout << "GST lib: Fake cam soruce " << endl;
    cout << GStString << endl;
    this->cap = cv::VideoCapture(GStString);
  }
}


#define UNUSED(x) ((void)(x))

#define CHECK(status) do {   \
    int res = (int)(status); \
    assert(res == 0);        \
    UNUSED(res);             \
} while(false)

using namespace nvinfer1;
using namespace redtail::tensorrt;

class Logger : public nvinfer1::ILogger
{
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) override
    {
        // Skip info (verbose) messages.
        // if (severity == Severity::kINFO)
        //     return;

        switch (severity)
        {
            case Severity::kINTERNAL_ERROR: std::cerr << "TRT INTERNAL_ERROR: "; break;
            case Severity::kERROR:          std::cerr << "TRT ERROR: "; break;
            case Severity::kWARNING:        std::cerr << "TRT WARNING: "; break;
            case Severity::kINFO:           std::cerr << "TRT INFO: "; break;
            default:                        std::cerr << "TRT UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
    }
};

static Logger gLogger;

class Profiler : public nvinfer1::IProfiler
{
public:
    void printLayerTimes()
    {
        float total_time = 0;
        for (size_t i = 0; i < profile_.size(); i++)
        {
            printf("%-60.60s %4.3fms\n", profile_[i].first.c_str(), profile_[i].second);
            total_time += profile_[i].second;
        }
        printf("All layers  : %4.3f\n", total_time);
    }

protected:
    void reportLayerTime(const char *layerName, float ms) override
    {
        auto record = std::find_if(profile_.begin(), profile_.end(), [&](const Record &r) { return r.first == layerName; });
        if (record == profile_.end())
            profile_.push_back(std::make_pair(layerName, ms));
        else
            record->second = ms;
    }

private:
    using Record = std::pair<std::string, float>;
    std::vector<Record> profile_;
};

static Profiler s_profiler;



std::unordered_map<std::string, Weights> readWeights(const std::string& filename, DataType data_type)
{
    assert(data_type == DataType::kFLOAT || data_type == DataType::kHALF);

    std::unordered_map<std::string, Weights> weights;
    std::ifstream weights_file(filename, std::ios::binary);
    assert(weights_file.is_open());
    while (weights_file.peek() != std::ifstream::traits_type::eof())
    {
        std::string name;
        uint32_t    count;
        Weights     w {data_type, nullptr, 0};
        std::getline(weights_file, name, '\0');
        weights_file.read(reinterpret_cast<char*>(&count), sizeof(uint32_t));
        w.count = count;
        size_t el_size_bytes = data_type == DataType::kFLOAT ? 4 : 2;
        auto p = new uint8_t[count * el_size_bytes];
        weights_file.read(reinterpret_cast<char*>(p), count * el_size_bytes);
        w.values = p;
        assert(weights.find(name) == weights.cend());
        weights[name] = w;
    }
    return weights;
}

int main(int argc, char** argv)
{
    if (argc < 8)
    {
        printf("\n"
               "Usage  : nvstereo_sample_app[_debug] <model_type> <width> <height> <path_to_weights_file> <path_to_left_image> <path_to_right_image> <disparity_output> [data_type]\n"
               "where  : model_type is the type of the DNN, supported are: nvsmall, resnet18, resnet18_2D\n"
               "         width and height are dimensions of the network (e.g. 1025 321)\n"
               "         weights file is the output of TensorRT model builder script\n"
               "         left and right are images that will be scaled to <width> x <height>\n"
               "         disparity output is the output of the network of size <width> x <height> (bin and PNG files are created)\n"
               "         data type(optional) is the data type of the model: fp32 (default) or fp16\n"
               "See <stereoDNN>/models directory for model files\n"
               "Example: nvstereo_sample_app nvsmall 1025 321 trt_weights.bin img_left.png img_right.png out_disp.bin\n\n");
        return 1;
    }
    //getchar();

    auto model_type = std::string(argv[1]);
    if (model_type != "nvsmall" && model_type != "resnet18" &&
        model_type != "resnet18_2D")
    {
        printf("Invalid model type %s, supported: nvsmall, resnet18, resnet18_2D.\n", model_type.c_str());
        exit(1);
    }

    DataType data_type = DataType::kFLOAT;
    if (argc >= 9)
    {
        auto d_type = std::string(argv[8]);
        if (d_type == "fp32")
            data_type = DataType::kFLOAT;
        else if (d_type == "fp16")
            data_type = DataType::kHALF;
        else
        {
            printf("Data type %s is not supported, supported types: fp32, fp16.\n", d_type.c_str());
            exit(1);
        }
    }
    printf("Using %s data type.\n", data_type == DataType::kFLOAT ? "fp32" : "fp16");

    // Read weights.
    // Note: the weights object lifetime must be at least the same as engine.
    std::string weights_file(argv[4]);
    auto weights = readWeights(weights_file, data_type);
    printf("Loaded %zu weight sets.\n", weights.size());

    //const int b = 1;
    const int c = 3;
    const int h = std::stoi(argv[3]);
    const int w = std::stoi(argv[2]);   
    printf("Using [%d, %d](width, height) as network input dimensions.\n", w, h);

    /*
    cv::FileStorage params("../data/params.json",
                            cv::FileStorage::FORMAT_JSON | cv::FileStorage::READ);
    params["rotation_matrix"] >> R;
    params["translation_matrix"] >> T;
    params["essential_matrix"] >> E;
    params["fundamental_matrix"] >> F;
    params["left_matrix"] >> leftCameraMatrix;
    params["left_dist_coeffs"] >> leftDistCoeffs;
    params["right_matrix"] >> rightCameraMatrix;
    params["right_dist_coeffs"] >> rightDistCoeffs;
    */
    GStCamera left(0, 1920, 1080, 30, CAM_APPSINK | CAM_NVIDIA, "BGR", "");
    GStCamera right(1, 1920, 1080, 30, CAM_APPSINK | CAM_NVIDIA, "BGR", "");


    
    cv::Mat img_left;
    cv::Mat img_right;

    for(int i = 0; i < 100; i++)
    {
        left.cap >> img_left;
        right.cap >> img_right;
    }

    std::vector<cv::KeyPoint> keypoints_l, keypoints_r;
    cv::Mat descriptors_l;
    cv::Mat descriptors_r;

    cv::Ptr<cv::ORB> detector = cv::ORB::create();
    detector->detectAndCompute(img_left, cv::Mat(), keypoints_l, descriptors_l);
    detector->detectAndCompute(img_right, cv::Mat(), keypoints_r, descriptors_r);

    int FLANN_INDEX_LSH = 6;

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    //cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector<cv::DMatch> knn_matches;
    matcher.match(descriptors_l, descriptors_r, knn_matches);

    double min_dist = 0, max_dist=0;

    for(int i = 0; i < descriptors_l.rows; i++)
    {
        double dist = knn_matches[i].distance;
        if(dist < min_dist) min_dist = dist;
        if(dist > max_dist) max_dist = dist;
    }

    
    std::vector<cv::DMatch> good_matches;
    for(int i = 0; i < descriptors_l.rows; i++)
    {
        if(knn_matches[i].distance <= cv::max(2*min_dist, 30.0))
        {
            good_matches.push_back(knn_matches[i]);
        }
    }

    cv::Mat img_left_drawn = cv::Mat(img_left);
    cv::Mat img_right_drawn = cv::Mat(img_right);

    cv::Mat img_matches;

    cv::drawMatches(img_left_drawn,
        keypoints_l,
        img_right_drawn,
        keypoints_r,
        knn_matches,
        img_matches);

    cv::Mat smallmatches;
    cv::resize(img_matches, smallmatches, cv::Size(640,240));

    cv::imshow("Good matches", smallmatches);
    cv::waitKey(0);
    
    cv::Mat pts8_l, pts8_r, pts8_l_f32, pts8_r_f32;

    for(int i = 0; i < 8; i++)
    {
        pts8_l.push_back(keypoints_l[good_matches[i].queryIdx].pt);
        pts8_r.push_back(keypoints_r[good_matches[i].queryIdx].pt);
    }

    pts8_l.convertTo(pts8_l_f32, CV_32F);
    pts8_r.convertTo(pts8_r_f32, CV_32F);


    cv::Mat F = cv::findFundamentalMat(pts8_l_f32, pts8_r_f32);
    cout << "F: " << F << endl << endl;
    cv::Mat H1, H2;
    cv::stereoRectifyUncalibrated(pts8_l, pts8_r, F, cv::Size(1920,1080), H1, H2);

    cv::Mat img_left_undistorted, img_right_undistorted;

    H1.convertTo(H1, CV_64F);
    H2.convertTo(H2, CV_64F);

    //cout << H1 << " tyepe "  << H1.cols << " " << H1.rows <<  endl;
    //cout << H2 << endl;

    cv::warpPerspective(img_left, img_left_undistorted, H1,  cv::Size(1920,1080));
    cv::warpPerspective(img_right, img_right_undistorted, H2,  cv::Size(1920,1080));


    img_left = cv::Mat(img_left_undistorted);
    img_right = cv::Mat(img_right_undistorted);

    cv::namedWindow("Depth");
    cv::Mat img_left_int, img_right_int;



    // Read images.
    //auto img_left  = readImgFile(argv[5], w, h);
    //auto img_left  = readBinFile(argv[5]);
    //assert(img_left.size() == (size_t)c * h * w);
    //auto img_right = readImgFile(argv[6], w, h);
    //auto img_right = readBinFile(argv[6]);
    //assert(img_right.size() == (size_t)c * h * w);

    // TensorRT pre-built plan file.
    auto trt_plan_file = weights_file + ".plan";
    std::ifstream trt_plan(trt_plan_file, std::ios::binary);

    // Note: the plugin_container object lifetime must be at least the same as the engine.
    auto plugin_container = IPluginContainer::create(gLogger);
    ICudaEngine* engine   = nullptr;
    // Check if we can load pre-built model from TRT plan file.
    // Currently only ResNet18_2D supports serialization.
    if (model_type == "resnet18_2D" && trt_plan.good())
    {
        printf("Loading TensorRT plan from %s...\n", trt_plan_file.c_str());
        // StereoDnnPluginFactory object is stateless as it adds plugins to corresponding container.
        StereoDnnPluginFactory factory(*plugin_container);
        IRuntime* runtime = createInferRuntime(gLogger);
        // Load the plan.
        std::stringstream model;
        model << trt_plan.rdbuf();
        model.seekg(0, model.beg);
        const auto& model_final = model.str();
        // Deserialize model.
        engine = runtime->deserializeCudaEngine(model_final.c_str(), model_final.size(), &factory);
    }
    else
    {
        // Create builder and network.
        IBuilder* builder = createInferBuilder(gLogger);

        // For now only ResNet18_2D has proper support for FP16.
        INetworkDefinition* network = nullptr;
        if (model_type == "nvsmall")
        {
            if (w == 1025)
                network = createNVSmall1025x321Network(*builder, *plugin_container, DimsCHW { c, h, w }, weights, DataType::kFLOAT, gLogger);
            else if (w == 513)
                network = createNVTiny513x161Network(  *builder, *plugin_container, DimsCHW { c, h, w }, weights, DataType::kFLOAT, gLogger);
            else
                assert(false);
        }
        else if (model_type == "resnet18")
        {
            if (w == 1025)
                network = createResNet18_1025x321Network(*builder, *plugin_container, DimsCHW { c, h, w }, weights, DataType::kFLOAT, gLogger);
            else
            {
                printf("ResNet-18 model supports only 1025x321 input image.\n");
                exit(1);
            }
        }
        else if (model_type == "resnet18_2D")
        {
            //if (w == 513)
                network = createResNet_18_2D513x257Network(*builder, *plugin_container, DimsCHW { c, h, w }, weights, data_type, gLogger);
            //else
            //{
            //    printf("ResNet18_2D model supports only 513x161 input image.\n");
            //    exit(1);
            //}
        }
        else
            assert(false);

        builder->setMaxBatchSize(1);
        size_t workspace_bytes = 1024 * 1024 * 1024;
        builder->setMaxWorkspaceSize(workspace_bytes);

        builder->setHalf2Mode(data_type == DataType::kHALF);
        // Build the network.
        engine = builder->buildCudaEngine(*network);
        network->destroy();

        if (model_type == "resnet18_2D")
        {
            printf("Saving TensorRT plan to %s...\n", trt_plan_file.c_str());
            IHostMemory *model_stream = engine->serialize();
            std::ofstream trt_plan_out(trt_plan_file, std::ios::binary);
            trt_plan_out.write((const char*)model_stream->data(), model_stream->size());
        }
    }



    assert(engine->getNbBindings() == 3);
    void* buffers[3];
    int in_idx_left = engine->getBindingIndex("left");
    assert(in_idx_left == 0);
    int in_idx_right = engine->getBindingIndex("right");
    assert(in_idx_right == 1);
    int out_idx = engine->getBindingIndex("disp");
    assert(out_idx == 2);

    IExecutionContext *context = engine->createExecutionContext();

    bool use_profiler = true;
    context->setProfiler(use_profiler ? &s_profiler : nullptr);

    std::vector<float> output(h * w);

    while(1)
    {

        //Read images.
        left.cap >> img_left;
        right.cap >> img_right;

        //img_left = cv::Mat(img_left_undistorted);
        //img_right = cv::Mat(img_right_undistorted);


        img_left_int = cv::Mat(img_left);
        img_right_int = cv::Mat(img_right);

        //assert(img.data != nullptr);
        // 0. Convert to float.
        img_left.convertTo(img_left, CV_32F);
        img_right.convertTo(img_right, CV_32F);
        // 1. Resize.
        cv::resize(img_left, img_left, cv::Size(286, 160), 0, 0,cv::INTER_AREA);
        cv::resize(img_left_int, img_left_int, cv::Size(286, 160), 0, 0,cv::INTER_AREA);
        cv::resize(img_right, img_right, cv::Size(286, 160), 0, 0,cv::INTER_AREA);
        cv::resize(img_right_int, img_right_int, cv::Size(286, 160), 0, 0,cv::INTER_AREA);

        cv::copyMakeBorder(img_left, img_left, 0, 1, 113, 114, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
        cv::copyMakeBorder(img_left_int, img_left_int, 0, 1, 113, 114, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
        cv::copyMakeBorder(img_right, img_right, 0, 1, 113, 114, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
        cv::copyMakeBorder(img_right_int, img_right_int, 0, 1, 113, 114, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );

        // 2. Convert BGR -> RGB.
        // 2. Convert BGR -> RGB.
        cv::cvtColor(img_left, img_left, cv::COLOR_BGR2RGB);
        cv::cvtColor(img_right, img_right, cv::COLOR_BGR2RGB);
        
        cv::imshow("Left", img_left_int);
        cv::imshow("Right", img_right_int);

        // 3. Convert HWC -> CHW.
        img_left = img_left.reshape(1, w * h).t();
        img_right = img_right.reshape(1, w * h).t();
        // 4. Scale.
        img_left /= 255.0;
        img_right /= 255.0;


        // Allocate GPU memory and copy data.
        CHECK(cudaMalloc(&buffers[in_idx_left],  513 * 161 * 3 * sizeof(float)));
        CHECK(cudaMalloc(&buffers[in_idx_right], 513 * 161 * 3 * sizeof(float)));
        CHECK(cudaMalloc(&buffers[out_idx],      output.size() * sizeof(float)));

        CHECK(cudaMemcpy(buffers[in_idx_left],  img_left.data,  513 * 161 * 3 * sizeof(float),  cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(buffers[in_idx_right], img_right.data, 513 * 161 * 3 * sizeof(float), cudaMemcpyHostToDevice));

        // Do the inference.
        auto host_start = std::chrono::high_resolution_clock::now();
        auto err        = context->execute(1, buffers);
        auto host_end   = std::chrono::high_resolution_clock::now();
        assert(err);
        UNUSED(err);
        auto host_elapsed_ms = std::chrono::duration<float, std::milli>(host_end - host_start).count();
        printf("Host time: %.4fms\n", host_elapsed_ms);

        if (use_profiler)
            s_profiler.printLayerTimes();

        // Copy output back to host.
        CHECK(cudaMemcpy(output.data(), buffers[out_idx], output.size() * sizeof(float), cudaMemcpyDeviceToHost));

        // Write results.
        // 1. As binary file.
        //auto res_file = std::ofstream(argv[7], std::ios::binary);
        //res_file.write((char*)output.data(), output.size() * sizeof(float));
        // 2. As PNG image.F
        auto img_f = cv::Mat(h, w, CV_32F, output.data());
        // Same as in KITTI, reduce quantization effects by storing as 16-bit PNG.
        img_f *= 256;
        // resnet18_2D model normalizes disparity using sigmoid, so bring it back to pixels.
        if (model_type == "resnet18_2D")
            img_f *= w;
        cv::Mat img_u16;
        img_f.convertTo(img_u16, CV_16U);
        cv::imshow("Depth", img_u16);
        //cv::imwrite(std::string(argv[7]) + ".png", img_u16);
        if(cv::waitKey(20) == 'q') break;

    }
    // Cleanup.
    context->destroy();
    engine->destroy();
    for (auto b: buffers)
        CHECK(cudaFree(b));

    printf("Done\n");
    return 0;
}