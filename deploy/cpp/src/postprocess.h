// 템플릿(결함 없는 정상 보드) 비교 후처리
//
// deploy/detect_board.py의 local_max_diff / refine_box_by_template / filter_by_template를
// C++로 옮긴 것. OpenCV의 morphologyEx, connectedComponentsWithStats를 직접 구현했다.
#pragma once

#include <vector>

#include "detector.h"
#include "image_io.h"

// 박스 안에서 템플릿과 가장 많이 다른 16x16 블록의 차분 비율.
// 박스 전체 평균을 쓰면 작은 결함이 큰 박스에 희석돼서 정상 구조물과 구분이 안 됐다.
float local_max_diff(const GrayImage& gray, const GrayImage& tpl, const Box& box,
                     int block = 16, int pixel_thresh = 100);

// 템플릿과 다른 영역만 남겨서 검출 박스를 실제 결함 크기로 좁힌다.
// pad는 DeepPCB 정답 박스의 여백에 맞춘 값 (8px일 때 IoU가 가장 높았다).
Box refine_box_by_template(const Box& box, const GrayImage& gray, const GrayImage& tpl,
                           int pad = 8, int pixel_thresh = 100);

// 템플릿과 거의 같은 검출(= 설계상 원래 있는 구조물)을 제거한다.
std::vector<Detection> filter_by_template(std::vector<Detection> dets, const GrayImage& gray,
                                          const GrayImage& tpl, float min_diff = 0.08f,
                                          bool refine = false);
