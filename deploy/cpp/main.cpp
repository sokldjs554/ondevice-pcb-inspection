// PCB 보드 결함 검출 - C++ 버전 (ONNX Runtime C++ API)
//
// deploy/detect_board.py를 C++로 옮긴 것. 파이썬을 올리기 어려운 장비에서 추론을
// 돌리는 상황을 연습하려고 만들었고, 파이썬 버전과 같은 옵션(템플릿 비교, 박스 정밀화,
// 멀티스케일, TCP 전송)을 그대로 지원한다.
//
// 사용법: ./detect_board --model <onnx> --image <jpg> [옵션]
//         ./detect_board <onnx> <jpg> [thresh]      (예전 방식도 그대로 동작)
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "src/detector.h"
#include "src/image_io.h"
#include "src/postprocess.h"
#include "src/tcp_sender.h"

namespace {

struct Options {
    std::string model = "models/mobilenet_v2_int8.onnx";
    std::string image;
    std::string tpl;          // "auto"면 _test.jpg -> _temp.jpg
    std::string send;         // "host:port"
    std::string out = "result_cpp.png";
    float thresh = 0.8f;
    int stride = 32;
    int threads = 0;          // 0 = onnxruntime 기본값
    int bench = 0;            // >0이면 벤치마크 모드 반복 횟수
    bool refine = false;
    bool multiscale = false;
};

void print_usage(const char* prog) {
    std::printf(
        "사용법: %s --model <onnx> --image <board.jpg> [옵션]\n"
        "\n"
        "  --model <경로>     ONNX 모델 (기본: models/mobilenet_v2_int8.onnx)\n"
        "  --image <경로>     검사할 보드 이미지\n"
        "  --template <경로>  결함 없는 템플릿 이미지 (auto = _test.jpg -> _temp.jpg)\n"
        "  --refine           템플릿 차분으로 박스를 실제 결함 크기로 좁힘\n"
        "  --multiscale       보드를 축소해가며 여러 번 훑어 큰 결함까지 검출\n"
        "  --thresh <실수>    결함 판정 확신도 (기본 0.8)\n"
        "  --stride <정수>    슬라이딩 윈도우 간격 (기본 32)\n"
        "  --threads <정수>   추론 스레드 수 (기본: 코어 수)\n"
        "  --send host:port   결과를 TCP(JSON)로 모니터링 서버에 전송\n"
        "  --out <경로>       결과 이미지 저장 경로 (기본 result_cpp.png)\n"
        "  --bench <횟수>     검출을 반복해서 스레드 수별 속도를 비교\n",
        prog);
}

bool parse_args(int argc, char** argv, Options* o) {
    // 예전 문서에 적어둔 위치 인자 방식(<model> <image> [thresh])도 계속 받아준다.
    if (argc >= 3 && argv[1][0] != '-') {
        o->model = argv[1];
        o->image = argv[2];
        if (argc > 3 && argv[3][0] != '-') o->thresh = std::stof(argv[3]);
        return true;
    }
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::printf("%s 뒤에 값이 필요합니다\n", name);
                return nullptr;
            }
            return argv[++i];
        };
        if (a == "--model")           { auto v = next("--model");     if (!v) return false; o->model = v; }
        else if (a == "--image")      { auto v = next("--image");     if (!v) return false; o->image = v; }
        else if (a == "--template")   { auto v = next("--template");  if (!v) return false; o->tpl = v; }
        else if (a == "--send")       { auto v = next("--send");      if (!v) return false; o->send = v; }
        else if (a == "--out")        { auto v = next("--out");       if (!v) return false; o->out = v; }
        else if (a == "--thresh")     { auto v = next("--thresh");    if (!v) return false; o->thresh = std::stof(v); }
        else if (a == "--stride")     { auto v = next("--stride");    if (!v) return false; o->stride = std::stoi(v); }
        else if (a == "--threads")    { auto v = next("--threads");   if (!v) return false; o->threads = std::stoi(v); }
        else if (a == "--bench")      { auto v = next("--bench");     if (!v) return false; o->bench = std::stoi(v); }
        else if (a == "--refine")     { o->refine = true; }
        else if (a == "--multiscale") { o->multiscale = true; }
        else if (a == "--help" || a == "-h") { return false; }
        else { std::printf("모르는 옵션: %s\n", a.c_str()); return false; }
    }
    return !o->image.empty();
}

std::string auto_template_path(const std::string& image_path) {
    // DeepPCB는 결함 이미지와 템플릿 이미지가 _test / _temp로 짝지어져 있다.
    const std::string from = "_test.jpg", to = "_temp.jpg";
    size_t pos = image_path.rfind(from);
    if (pos == std::string::npos) return "";
    return image_path.substr(0, pos) + to;
}

std::string base_name(const std::string& path) {
    size_t pos = path.find_last_of('/');
    return pos == std::string::npos ? path : path.substr(pos + 1);
}

// 벤치마크: 스레드 수를 바꿔가며 같은 보드를 여러 번 검출하고 시간을 비교한다.
// 임베디드 장비에서는 코어를 다 쓰는 게 항상 이득이 아니라서 직접 재보고 정해야 한다.
int run_benchmark(const Options& o, const GrayImage& gray) {
    std::vector<int> thread_opts;
    if (o.threads > 0) {
        thread_opts.push_back(o.threads);
    } else {
        int cores = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
        for (int t : {1, 2, 4, 8}) {
            if (t <= cores) thread_opts.push_back(t);
        }
        if (thread_opts.empty()) thread_opts.push_back(1);
    }

    std::printf("벤치마크: %s, %d회 반복 (코어 %u개)\n", base_name(o.model).c_str(), o.bench,
                std::thread::hardware_concurrency());
    // 한글은 터미널 폭 계산이 안 맞아서 표 머리글만 영문으로 뒀다
    std::printf("%-8s %10s %10s %10s %10s %10s %10s\n", "threads", "mean(ms)", "min(ms)",
                "p95(ms)", "pre(ms)", "post(ms)", "speedup");

    double baseline = 0;
    for (int threads : thread_opts) {
        SlidingWindowDetector det(o.model, o.stride, o.thresh, threads);
        det.detect(gray);  // 워밍업 (첫 회는 메모리 할당 때문에 느리다)

        std::vector<double> total, pre, post;
        for (int i = 0; i < o.bench; i++) {
            det.detect(gray);
            total.push_back(det.timing().inference);
            pre.push_back(det.timing().preprocess);
            post.push_back(det.timing().postprocess);
        }
        std::vector<double> sorted = total;
        std::sort(sorted.begin(), sorted.end());
        double sum = 0;
        for (double v : total) sum += v;
        double mean = sum / total.size();
        double p95 = sorted[std::min(sorted.size() - 1,
                                     static_cast<size_t>(sorted.size() * 0.95))];
        double pre_sum = 0, post_sum = 0;
        for (size_t i = 0; i < pre.size(); i++) { pre_sum += pre[i]; post_sum += post[i]; }

        if (baseline == 0) baseline = mean;
        std::printf("%-8d %10.1f %10.1f %10.1f %10.1f %10.1f %9.2fx\n", threads, mean,
                    sorted.front(), p95, pre_sum / pre.size(), post_sum / post.size(),
                    baseline / mean);
    }
    std::printf("\nmean/min/p95는 추론 시간, pre/post는 전처리·후처리 시간이다.\n");
    std::printf("전처리·후처리는 내가 짠 단일 스레드 코드라 스레드를 늘려도 그대로다.\n");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    if (!parse_args(argc, argv, &o)) {
        print_usage(argv[0]);
        return 1;
    }

    GrayImage gray = load_gray(o.image);
    if (gray.empty()) {
        std::printf("이미지를 열 수 없습니다: %s\n", o.image.c_str());
        return 1;
    }

    if (o.bench > 0) return run_benchmark(o, gray);

    SlidingWindowDetector detector(o.model, o.stride, o.thresh, o.threads);
    auto t0 = std::chrono::steady_clock::now();
    std::vector<Detection> dets =
        o.multiscale ? detector.detectMultiscale(gray) : detector.detect(gray);
    double elapsed_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();

    // 템플릿 비교로 정상 구조물 오검출 제거 (+ 옵션에 따라 박스 정밀화)
    if (!o.tpl.empty()) {
        std::string tpl_path = (o.tpl == "auto") ? auto_template_path(o.image) : o.tpl;
        GrayImage tpl = tpl_path.empty() ? GrayImage{} : load_gray(tpl_path);
        if (tpl.empty()) {
            std::printf("템플릿을 열 수 없습니다: %s\n", tpl_path.c_str());
        } else if (tpl.w != gray.w || tpl.h != gray.h) {
            std::printf("템플릿 크기가 다릅니다 (%dx%d vs %dx%d)\n", tpl.w, tpl.h, gray.w, gray.h);
        } else {
            size_t before = dets.size();
            dets = filter_by_template(dets, gray, tpl, 0.08f, o.refine);
            std::printf("템플릿 비교: %zu건 -> %zu건 (%zu건 제거%s)\n", before, dets.size(),
                        before - dets.size(), o.refine ? ", 박스 정밀화 적용" : "");
        }
    }

    const Timing& tm = detector.timing();
    std::printf("검출 완료: %zu건, %.2f초 (윈도우 %d개, thresh %.2f%s)\n", dets.size(),
                elapsed_ms / 1000.0, detector.windowCount(), o.thresh,
                o.multiscale ? ", 멀티스케일" : "");
    std::printf("  단계별: 전처리 %.1f ms / 추론 %.1f ms / 후처리 %.1f ms\n", tm.preprocess,
                tm.inference, tm.postprocess);
    for (const Detection& d : dets) {
        std::printf("  %-10s box=[%d, %d, %d, %d] score=%.3f", d.name(), d.box.x1, d.box.y1,
                    d.box.x2, d.box.y2, d.score);
        if (d.template_diff >= 0.f) std::printf(" diff=%.3f", d.template_diff);
        std::printf("\n");
    }

    std::vector<Box> boxes;
    for (const Detection& d : dets) boxes.push_back(d.box);
    if (save_with_boxes(o.out, gray, boxes)) {
        std::printf("결과 이미지 저장: %s\n", o.out.c_str());
    }

    if (!o.send.empty()) {
        std::string host, err;
        int port = 0;
        if (!parse_host_port(o.send, &host, &port)) {
            std::printf("--send 형식이 잘못됐습니다 (host:port): %s\n", o.send.c_str());
            return 1;
        }
        std::string payload = build_json(base_name(o.image), elapsed_ms / 1000.0, dets);
        if (send_json(host, port, payload, &err)) {
            std::printf("검출 결과 전송 완료 -> %s\n", o.send.c_str());
        } else {
            std::printf("전송 실패: %s\n", err.c_str());
            return 1;
        }
    }
    return 0;
}
