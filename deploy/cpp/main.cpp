// PCB 보드 결함 검출 - C++ 버전 (ONNX Runtime C++ API)
//
// deploy/detect_board.py의 추론 부분을 C++로 옮긴 것.
// 라즈베리파이 같은 장비에서 파이썬 없이 추론을 돌리는 상황을 연습하기 위해 만들었다.
// 이미지 입출력은 stb 싱글헤더 라이브러리를 사용해서 의존성을 onnxruntime 하나로 줄였다.
//
// 사용법: ./detect_board <model.onnx> <board_image.jpg> [threshold=0.8]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <queue>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb_image_write.h"

const int PATCH = 64;
const int STRIDE = 32;
const int NUM_CLASSES = 7;
const char* CLASS_NAMES[NUM_CLASSES] = {"normal",    "open", "short", "mousebite",
                                        "spur", "copper", "pinhole"};

struct Detection {
    int cls;
    int x1, y1, x2, y2;
    float score;
};

// 한 윈도우의 로짓에서 softmax 확률 계산
static void softmax(const float* logits, float* probs) {
    float max_v = *std::max_element(logits, logits + NUM_CLASSES);
    float sum = 0.f;
    for (int i = 0; i < NUM_CLASSES; i++) {
        probs[i] = std::exp(logits[i] - max_v);
        sum += probs[i];
    }
    for (int i = 0; i < NUM_CLASSES; i++) probs[i] /= sum;
}

// 결함으로 판정된 인접 윈도우를 BFS로 묶어서 박스 생성 (파이썬의 connectedComponents 대응)
static std::vector<Detection> group_windows(const std::vector<int>& cls_map,
                                            const std::vector<float>& conf_map,
                                            int ny, int nx, float thresh) {
    std::vector<Detection> dets;
    std::vector<char> visited(ny * nx, 0);
    const int dy[] = {-1, -1, -1, 0, 0, 1, 1, 1};
    const int dx[] = {-1, 0, 1, -1, 1, -1, 0, 1};

    for (int i = 0; i < ny * nx; i++) {
        if (visited[i] || cls_map[i] == 0 || conf_map[i] < thresh) continue;
        int c = cls_map[i];
        // 같은 클래스의 인접 칸들을 한 덩어리로
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
        d.x1 = min_x * STRIDE;
        d.y1 = min_y * STRIDE;
        d.x2 = max_x * STRIDE + PATCH;
        d.y2 = max_y * STRIDE + PATCH;
        d.score = best;
        dets.push_back(d);
    }
    return dets;
}

// 결과 확인용: 회색 이미지를 RGB로 바꾸고 검출 박스를 초록색으로 그려서 저장
static void save_result(const unsigned char* gray, int w, int h,
                        const std::vector<Detection>& dets, const char* path) {
    std::vector<unsigned char> rgb(w * h * 3);
    for (int i = 0; i < w * h; i++) {
        rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = gray[i];
    }
    for (const auto& d : dets) {
        for (int t = 0; t < 2; t++) {  // 두께 2px
            for (int x = d.x1; x < d.x2 && x < w; x++) {
                for (int yy : {d.y1 + t, d.y2 - 1 - t}) {
                    if (yy < 0 || yy >= h || x < 0) continue;
                    int i = (yy * w + x) * 3;
                    rgb[i] = 0; rgb[i + 1] = 255; rgb[i + 2] = 0;
                }
            }
            for (int y = d.y1; y < d.y2 && y < h; y++) {
                for (int xx : {d.x1 + t, d.x2 - 1 - t}) {
                    if (xx < 0 || xx >= w || y < 0) continue;
                    int i = (y * w + xx) * 3;
                    rgb[i] = 0; rgb[i + 1] = 255; rgb[i + 2] = 0;
                }
            }
        }
    }
    stbi_write_png(path, w, h, 3, rgb.data(), w * 3);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("사용법: %s <model.onnx> <board_image.jpg> [threshold=0.8]\n", argv[0]);
        return 1;
    }
    const char* model_path = argv[1];
    const char* image_path = argv[2];
    float thresh = (argc > 3) ? std::stof(argv[3]) : 0.8f;

    // 1. 이미지 로드 (흑백)
    int w, h, ch;
    unsigned char* gray = stbi_load(image_path, &w, &h, &ch, 1);
    if (!gray) {
        std::printf("이미지를 열 수 없습니다: %s\n", image_path);
        return 1;
    }

    // 2. 모델 로드
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "pcb");
    Ort::SessionOptions opts;
    Ort::Session session(env, model_path, opts);

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    // 모델 입력 채널 수 확인 (SimpleCNN=1, MobileNetV2=3)
    auto input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    int channels = static_cast<int>(input_shape[1]);

    // 3. 슬라이딩 윈도우를 전부 잘라서 배치 하나로 만들기
    int ny = (h - PATCH) / STRIDE + 1;
    int nx = (w - PATCH) / STRIDE + 1;
    int num = ny * nx;
    std::vector<float> input(static_cast<size_t>(num) * channels * PATCH * PATCH);
    for (int wy = 0; wy < ny; wy++) {
        for (int wx = 0; wx < nx; wx++) {
            int base = (wy * nx + wx) * channels * PATCH * PATCH;
            for (int py = 0; py < PATCH; py++) {
                for (int px = 0; px < PATCH; px++) {
                    unsigned char v = gray[(wy * STRIDE + py) * w + (wx * STRIDE + px)];
                    float f = (v / 255.0f - 0.5f) / 0.5f;  // 학습 때와 같은 정규화
                    for (int c = 0; c < channels; c++) {
                        input[base + c * PATCH * PATCH + py * PATCH + px] = f;
                    }
                }
            }
        }
    }

    // 4. 추론 (배치 한 번에)
    auto t0 = std::chrono::steady_clock::now();
    std::vector<int64_t> in_dims = {num, channels, PATCH, PATCH};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
        mem, input.data(), input.size(), in_dims.data(), in_dims.size());
    const char* in_names[] = {input_name.get()};
    const char* out_names[] = {output_name.get()};
    auto outputs = session.Run(Ort::RunOptions{nullptr}, in_names, &in_tensor, 1, out_names, 1);
    const float* logits = outputs[0].GetTensorData<float>();
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 5. 후처리: 윈도우별 클래스/확신도 -> 인접 윈도우 묶기
    std::vector<int> cls_map(num);
    std::vector<float> conf_map(num);
    for (int i = 0; i < num; i++) {
        float probs[NUM_CLASSES];
        softmax(logits + i * NUM_CLASSES, probs);
        int best = static_cast<int>(std::max_element(probs, probs + NUM_CLASSES) - probs);
        cls_map[i] = best;
        conf_map[i] = probs[best];
    }
    auto dets = group_windows(cls_map, conf_map, ny, nx, thresh);

    // 6. 결과 출력
    std::printf("검출 완료: %zu건, 추론 %.1f ms (윈도우 %d개, thresh %.2f)\n",
                dets.size(), ms, num, thresh);
    for (const auto& d : dets) {
        std::printf("  %-10s box=[%d, %d, %d, %d] score=%.3f\n",
                    CLASS_NAMES[d.cls], d.x1, d.y1, d.x2, d.y2, d.score);
    }

    save_result(gray, w, h, dets, "result_cpp.png");
    std::printf("결과 이미지 저장: result_cpp.png\n");

    stbi_image_free(gray);
    return 0;
}
