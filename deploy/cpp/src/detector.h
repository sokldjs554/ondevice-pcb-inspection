// 슬라이딩 윈도우 결함 검출기 (ONNX Runtime C++ API)
//
// deploy/detect_board.py의 SlidingWindowDetector를 C++로 옮긴 것.
// 파이썬 버전과 같은 전처리(-1~1 정규화), 같은 후처리(인접 윈도우 묶기)를 쓴다.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "image_io.h"

constexpr int PATCH = 64;
constexpr int NUM_CLASSES = 7;
extern const char* const CLASS_NAMES[NUM_CLASSES];

struct Detection {
    int cls = 0;
    Box box{0, 0, 0, 0};
    float score = 0.f;
    float scale = 1.f;          // 어느 스케일에서 나온 검출인지 (멀티스케일용)
    float template_diff = -1.f;  // 템플릿 차분 비율 (-1이면 계산 안 함)

    const char* name() const { return CLASS_NAMES[cls]; }
};

// 단계별 소요 시간 (ms). 어디가 병목인지 보려고 나눠서 잰다.
struct Timing {
    double preprocess = 0;
    double inference = 0;
    double postprocess = 0;
    double total() const { return preprocess + inference + postprocess; }
};

class SlidingWindowDetector {
public:
    // threads=0이면 onnxruntime 기본값(코어 수)을 쓴다.
    SlidingWindowDetector(const std::string& model_path, int stride = 32,
                          float thresh = 0.8f, int threads = 0);

    std::vector<Detection> detect(const GrayImage& gray);

    // 이미지를 줄여가며 여러 번 훑어서 64px 윈도우보다 큰 결함까지 잡는다.
    std::vector<Detection> detectMultiscale(const GrayImage& gray,
                                            const std::vector<float>& scales = {1.0f, 0.5f});

    int channels() const { return channels_; }
    int windowCount() const { return last_windows_; }
    const Timing& timing() const { return timing_; }

private:
    Ort::Env env_;
    Ort::SessionOptions opts_;
    Ort::Session session_{nullptr};
    std::string input_name_, output_name_;

    int stride_;
    float thresh_;
    int channels_ = 1;
    int last_windows_ = 0;
    Timing timing_;
};

// 겹치는 검출을 하나로 합친다 (클래스별 IoU NMS).
std::vector<Detection> merge_overlapping(std::vector<Detection> dets, float iou_thresh = 0.3f);

float box_iou(const Box& a, const Box& b);
