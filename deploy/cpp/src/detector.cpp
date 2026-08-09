#include "detector.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <queue>

const char* const CLASS_NAMES[NUM_CLASSES] = {
    "normal", "open", "short", "mousebite", "spur", "copper", "pinhole"};

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

void softmax(const float* logits, float* probs) {
    float max_v = *std::max_element(logits, logits + NUM_CLASSES);
    float sum = 0.f;
    for (int i = 0; i < NUM_CLASSES; i++) {
        probs[i] = std::exp(logits[i] - max_v);
        sum += probs[i];
    }
    for (int i = 0; i < NUM_CLASSES; i++) probs[i] /= sum;
}

// 결함으로 판정된 인접 윈도우를 BFS로 묶어 박스를 만든다.
// (파이썬의 cv2.connectedComponents(connectivity=8)에 대응)
std::vector<Detection> group_windows(const std::vector<int>& cls_map,
                                     const std::vector<float>& conf_map,
                                     int ny, int nx, int stride, float thresh) {
    std::vector<Detection> dets;
    std::vector<char> visited(static_cast<size_t>(ny) * nx, 0);
    const int dy[] = {-1, -1, -1, 0, 0, 1, 1, 1};
    const int dx[] = {-1, 0, 1, -1, 1, -1, 0, 1};

    for (int i = 0; i < ny * nx; i++) {
        if (visited[i] || cls_map[i] == 0 || conf_map[i] < thresh) continue;
        int c = cls_map[i];
        int min_y = ny, min_x = nx, max_y = -1, max_x = -1;
        float best = 0.f;
        std::queue<int> q;
        q.push(i);
        visited[i] = 1;
        while (!q.empty()) {
            int cur = q.front();
            q.pop();
            int cy = cur / nx, cx = cur % nx;
            min_y = std::min(min_y, cy); max_y = std::max(max_y, cy);
            min_x = std::min(min_x, cx); max_x = std::max(max_x, cx);
            best = std::max(best, conf_map[cur]);
            for (int k = 0; k < 8; k++) {
                int py = cy + dy[k], px = cx + dx[k];
                if (py < 0 || py >= ny || px < 0 || px >= nx) continue;
                int ni = py * nx + px;
                if (!visited[ni] && cls_map[ni] == c && conf_map[ni] >= thresh) {
                    visited[ni] = 1;
                    q.push(ni);
                }
            }
        }
        Detection d;
        d.cls = c;
        d.box = {min_x * stride, min_y * stride, max_x * stride + PATCH, max_y * stride + PATCH};
        d.score = best;
        dets.push_back(d);
    }
    return dets;
}

}  // namespace

SlidingWindowDetector::SlidingWindowDetector(const std::string& model_path, int stride,
                                             float thresh, int threads)
    : env_(ORT_LOGGING_LEVEL_WARNING, "pcb"), stride_(stride), thresh_(thresh) {
    if (threads > 0) {
        // 라즈베리파이처럼 코어가 적은 장비에서 스레드 수를 직접 조절해보려고 열어뒀다.
        opts_.SetIntraOpNumThreads(threads);
    }
    opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_ = Ort::Session(env_, model_path.c_str(), opts_);

    Ort::AllocatorWithDefaultOptions alloc;
    input_name_ = session_.GetInputNameAllocated(0, alloc).get();
    output_name_ = session_.GetOutputNameAllocated(0, alloc).get();
    auto shape = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    channels_ = static_cast<int>(shape[1]);  // SimpleCNN=1, MobileNetV2=3
}

std::vector<Detection> SlidingWindowDetector::detect(const GrayImage& gray) {
    timing_ = Timing{};
    if (gray.w < PATCH || gray.h < PATCH) return {};

    const int ny = (gray.h - PATCH) / stride_ + 1;
    const int nx = (gray.w - PATCH) / stride_ + 1;
    const int num = ny * nx;
    last_windows_ = num;
    const size_t patch_floats = static_cast<size_t>(channels_) * PATCH * PATCH;

    // 1. 전처리: 윈도우를 전부 잘라서 NCHW 텐서로 만든다
    auto t0 = Clock::now();
    std::vector<float> input(static_cast<size_t>(num) * patch_floats);
    for (int wy = 0; wy < ny; wy++) {
        for (int wx = 0; wx < nx; wx++) {
            float* dst = input.data() + static_cast<size_t>(wy * nx + wx) * patch_floats;
            for (int py = 0; py < PATCH; py++) {
                const unsigned char* row = &gray.data[static_cast<size_t>(wy * stride_ + py) * gray.w
                                                      + wx * stride_];
                for (int px = 0; px < PATCH; px++) {
                    float f = (row[px] / 255.0f - 0.5f) / 0.5f;  // 학습 때와 같은 정규화
                    for (int c = 0; c < channels_; c++) {
                        dst[c * PATCH * PATCH + py * PATCH + px] = f;
                    }
                }
            }
        }
    }
    timing_.preprocess = ms_since(t0);

    // 2. 추론: 파이썬 버전처럼 256개씩 나눠서 돌린다.
    //    한 번에 다 넣으면 361개 * 3채널 기준으로 임시 버퍼가 커져서, 메모리가 작은
    //    장비를 생각해 쪼갰다. 속도 차이는 거의 없었다.
    const int BATCH = 256;
    std::vector<float> logits(static_cast<size_t>(num) * NUM_CLASSES);
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const char* in_names[] = {input_name_.c_str()};
    const char* out_names[] = {output_name_.c_str()};

    t0 = Clock::now();
    for (int start = 0; start < num; start += BATCH) {
        int n = std::min(BATCH, num - start);
        std::vector<int64_t> dims = {n, channels_, PATCH, PATCH};
        Ort::Value in = Ort::Value::CreateTensor<float>(
            mem, input.data() + static_cast<size_t>(start) * patch_floats,
            static_cast<size_t>(n) * patch_floats, dims.data(), dims.size());
        auto out = session_.Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);
        std::copy_n(out[0].GetTensorData<float>(), static_cast<size_t>(n) * NUM_CLASSES,
                    logits.begin() + static_cast<size_t>(start) * NUM_CLASSES);
    }
    timing_.inference = ms_since(t0);

    // 3. 후처리: 윈도우별 클래스/확신도 -> 인접 윈도우 묶기
    t0 = Clock::now();
    std::vector<int> cls_map(num);
    std::vector<float> conf_map(num);
    for (int i = 0; i < num; i++) {
        float probs[NUM_CLASSES];
        softmax(&logits[static_cast<size_t>(i) * NUM_CLASSES], probs);
        int best = static_cast<int>(std::max_element(probs, probs + NUM_CLASSES) - probs);
        cls_map[i] = best;
        conf_map[i] = probs[best];
    }
    auto dets = group_windows(cls_map, conf_map, ny, nx, stride_, thresh_);
    timing_.postprocess = ms_since(t0);
    return dets;
}

std::vector<Detection> SlidingWindowDetector::detectMultiscale(const GrayImage& gray,
                                                               const std::vector<float>& scales) {
    std::vector<Detection> all;
    Timing sum;
    int windows = 0;
    for (float s : scales) {
        GrayImage img = (s == 1.0f) ? gray : resize_area(gray, s);
        auto dets = detect(img);
        sum.preprocess += timing_.preprocess;
        sum.inference += timing_.inference;
        sum.postprocess += timing_.postprocess;
        windows += last_windows_;
        for (Detection& d : dets) {
            // 축소본에서 찾은 좌표를 원래 크기로 되돌린다
            d.box = {static_cast<int>(std::lround(d.box.x1 / s)),
                     static_cast<int>(std::lround(d.box.y1 / s)),
                     static_cast<int>(std::lround(d.box.x2 / s)),
                     static_cast<int>(std::lround(d.box.y2 / s))};
            d.scale = s;
            all.push_back(d);
        }
    }
    timing_ = sum;
    last_windows_ = windows;
    return merge_overlapping(std::move(all));
}

float box_iou(const Box& a, const Box& b) {
    int ix1 = std::max(a.x1, b.x1), iy1 = std::max(a.y1, b.y1);
    int ix2 = std::min(a.x2, b.x2), iy2 = std::min(a.y2, b.y2);
    long inter = static_cast<long>(std::max(0, ix2 - ix1)) * std::max(0, iy2 - iy1);
    long ua = static_cast<long>(a.x2 - a.x1) * (a.y2 - a.y1);
    long ub = static_cast<long>(b.x2 - b.x1) * (b.y2 - b.y1);
    long uni = ua + ub - inter;
    return uni > 0 ? static_cast<float>(inter) / uni : 0.f;
}

std::vector<Detection> merge_overlapping(std::vector<Detection> dets, float iou_thresh) {
    // 확신도가 높은 것부터 남기고, 같은 종류이면서 많이 겹치는 것은 버린다.
    // 클래스별로 따로 보기 때문에 서로 다른 결함이 하나로 뭉치지는 않는다.
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) { return a.score > b.score; });
    std::vector<Detection> kept;
    for (const Detection& d : dets) {
        bool dup = false;
        for (const Detection& k : kept) {
            if (k.cls == d.cls && box_iou(d.box, k.box) >= iou_thresh) { dup = true; break; }
        }
        if (!dup) kept.push_back(d);
    }
    return kept;
}
