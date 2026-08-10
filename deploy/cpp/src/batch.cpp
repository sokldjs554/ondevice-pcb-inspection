#include "batch.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <queue>
#include <thread>

#include <dirent.h>

#include "tcp_sender.h"

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

std::string base_name(const std::string& path) {
    size_t pos = path.find_last_of('/');
    return pos == std::string::npos ? path : path.substr(pos + 1);
}

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// 전처리까지 끝난 한 장. 파이프라인에서 스레드 사이로 넘어가는 단위다.
struct Item {
    std::string path;
    SlidingWindowDetector::Prepared prep;
    double load_ms = 0;
    double pre_ms = 0;
    bool valid = false;
};

// 스레드 사이에 Item을 넘기는 큐. 크기를 제한해서 메모리가 무한정 늘지 않게 한다.
// (보드 한 장의 텐서가 361 x 3 x 64 x 64 x 4바이트 = 약 17MB라 제한이 필요하다)
class ItemQueue {
public:
    explicit ItemQueue(size_t cap) : cap_(cap) {}

    void push(Item item) {
        std::unique_lock<std::mutex> lock(m_);
        not_full_.wait(lock, [&] { return q_.size() < cap_; });
        q_.push(std::move(item));
        lock.unlock();
        not_empty_.notify_one();
    }

    void close() {
        std::lock_guard<std::mutex> lock(m_);
        closed_ = true;
        not_empty_.notify_all();
    }

    // 큐가 비고 생산자가 끝났으면 false
    bool pop(Item* out) {
        std::unique_lock<std::mutex> lock(m_);
        not_empty_.wait(lock, [&] { return !q_.empty() || closed_; });
        if (q_.empty()) return false;
        *out = std::move(q_.front());
        q_.pop();
        lock.unlock();
        not_full_.notify_one();
        return true;
    }

private:
    std::queue<Item> q_;
    std::mutex m_;
    std::condition_variable not_empty_, not_full_;
    size_t cap_;
    bool closed_ = false;
};

Item load_and_prepare(const SlidingWindowDetector& det, const std::string& path) {
    Item item;
    item.path = path;
    auto t0 = Clock::now();
    GrayImage gray = load_gray(path);
    item.load_ms = ms_since(t0);
    if (gray.empty()) return item;
    t0 = Clock::now();
    item.prep = det.preprocess(gray);
    item.pre_ms = ms_since(t0);
    item.valid = true;
    return item;
}

}  // namespace

std::vector<std::string> list_boards(const std::string& dir, int limit) {
    std::vector<std::string> all, test_only;
    DIR* d = opendir(dir.c_str());
    if (!d) return all;
    while (dirent* e = readdir(d)) {
        std::string name = e->d_name;
        if (!ends_with(name, ".jpg") && !ends_with(name, ".png")) continue;
        std::string full = dir + "/" + name;
        all.push_back(full);
        if (name.find("_test.") != std::string::npos) test_only.push_back(full);
    }
    closedir(d);

    // 템플릿(_temp)까지 같이 검사하면 결과가 이상해지므로 _test가 있으면 그것만 쓴다
    std::vector<std::string>& picked = test_only.empty() ? all : test_only;
    std::sort(picked.begin(), picked.end());
    if (limit > 0 && static_cast<int>(picked.size()) > limit) picked.resize(limit);
    return picked;
}

int run_batch(SlidingWindowDetector& detector, const BatchOptions& opt) {
    std::vector<std::string> boards = list_boards(opt.dir, opt.limit);
    if (boards.empty()) {
        std::printf("검사할 이미지를 찾지 못했습니다: %s\n", opt.dir.c_str());
        return 0;
    }

    std::ofstream jsonl;
    if (!opt.out_jsonl.empty()) jsonl.open(opt.out_jsonl, std::ios::app);

    std::string host;
    int port = 0;
    bool do_send = !opt.send.empty() && parse_host_port(opt.send, &host, &port);

    std::printf("배치 검사: %s (%zu장), %s 모드\n", opt.dir.c_str(), boards.size(),
                opt.pipeline ? "파이프라인" : "순차");

    int n_defect_boards = 0, n_defects = 0, n_done = 0;
    double sum_load = 0, sum_pre = 0, sum_infer = 0;
    auto wall_t0 = Clock::now();

    // 한 장을 처리하고 결과를 출력·전송하는 공통 부분
    auto consume = [&](Item& item) {
        if (!item.valid) {
            std::printf("  [건너뜀] %s (읽기 실패)\n", base_name(item.path).c_str());
            return;
        }
        auto t0 = Clock::now();
        std::vector<Detection> dets = detector.inferPrepared(item.prep);
        double infer_ms = ms_since(t0);

        n_done++;
        n_defects += static_cast<int>(dets.size());
        if (!dets.empty()) n_defect_boards++;
        sum_load += item.load_ms;
        sum_pre += item.pre_ms;
        sum_infer += infer_ms;

        if (!opt.quiet) {
            std::printf("  %-24s %s (결함 %zu건, 읽기 %.0f / 전처리 %.0f / 추론 %.0f ms)\n",
                        base_name(item.path).c_str(), dets.empty() ? "정상" : "불량", dets.size(),
                        item.load_ms, item.pre_ms, infer_ms);
        }
        std::string line = build_json(base_name(item.path), infer_ms / 1000.0, dets);
        if (jsonl.is_open()) jsonl << line;
        if (do_send) {
            std::string err;
            if (!send_json(host, port, line, &err)) {
                std::printf("    전송 실패: %s\n", err.c_str());
            }
        }
    };

    if (!opt.pipeline) {
        for (const std::string& path : boards) {
            Item item = load_and_prepare(detector, path);
            consume(item);
        }
    } else {
        // 생산자 스레드가 읽기+전처리를, 메인 스레드가 추론을 맡는다.
        // 큐 크기를 2로 둬서 앞서가더라도 메모리를 두 장치 이상 쓰지 않게 했다.
        ItemQueue queue(2);
        std::thread producer([&] {
            for (const std::string& path : boards) {
                queue.push(load_and_prepare(detector, path));
            }
            queue.close();
        });

        Item item;
        while (queue.pop(&item)) {
            consume(item);
        }
        producer.join();
    }

    double wall = ms_since(wall_t0);
    std::printf("\n검사 %d장 완료 | 불량 %d장 / 정상 %d장 | 결함 %d건\n", n_done, n_defect_boards,
                n_done - n_defect_boards, n_defects);
    std::printf("총 %.2f초, 장당 %.0f ms, 처리량 %.2f 장/초\n", wall / 1000.0,
                n_done ? wall / n_done : 0.0, n_done ? n_done * 1000.0 / wall : 0.0);
    std::printf("단계별 합계: 읽기 %.0f ms / 전처리 %.0f ms / 추론 %.0f ms\n", sum_load, sum_pre,
                sum_infer);
    if (jsonl.is_open()) std::printf("결과 저장: %s\n", opt.out_jsonl.c_str());
    return n_done;
}
