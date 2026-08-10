// 폴더 안의 보드를 연속으로 검사하는 모드
//
// 실제 검사 장비는 보드 한 장이 아니라 라인 위로 계속 흘러오는 보드를 처리합니다.
// 그 상황을 흉내 내서, 폴더 하나를 통째로 훑으며 처리량(장/초)을 재도록 만들었습니다.
//
// 두 가지 방식을 비교할 수 있습니다.
//   순차 : 한 장을 [읽기 → 전처리 → 추론] 다 끝내고 다음 장으로
//   파이프라인 : 읽기·전처리를 다른 스레드가 맡아, 추론과 겹쳐서 진행
#pragma once

#include <string>
#include <vector>

#include "detector.h"

struct BatchOptions {
    std::string dir;
    std::string out_jsonl;   // 비어 있지 않으면 결과를 한 줄 JSON으로 저장
    std::string send;        // host:port. 지정하면 보드마다 결과를 TCP로 전송
    int limit = 0;           // 0이면 폴더 전체
    bool pipeline = false;   // 전처리를 별도 스레드로 분리
    bool quiet = false;      // 보드별 출력 생략, 요약만
};

// 폴더 안의 *_test.jpg 를 이름순으로 모은다 (없으면 모든 .jpg)
std::vector<std::string> list_boards(const std::string& dir, int limit);

// 배치 검사를 실행하고 요약을 출력한다. 반환값은 처리한 보드 수.
int run_batch(SlidingWindowDetector& detector, const BatchOptions& opt);
