// 이미지 입출력 / 그리기 (stb 싱글헤더 래퍼)
//
// OpenCV를 쓰지 않는 이유: 라즈베리파이 같은 장비에 OpenCV까지 올리면 용량이 커진다.
// 여기서 필요한 건 "흑백으로 읽기 / 절반으로 줄이기 / 박스 그려서 저장하기" 세 가지뿐이라
// 헤더 파일 두 개짜리 stb로 충분했다.
#pragma once

#include <string>
#include <vector>

struct GrayImage {
    int w = 0;
    int h = 0;
    std::vector<unsigned char> data;  // h*w, 행 우선

    bool empty() const { return data.empty(); }
    unsigned char at(int y, int x) const { return data[static_cast<size_t>(y) * w + x]; }
};

struct Box {
    int x1, y1, x2, y2;
};

// 흑백 1채널로 읽는다. 실패하면 empty() 이미지를 돌려준다.
GrayImage load_gray(const std::string& path);

// 면적 평균으로 축소한다 (OpenCV INTER_AREA와 같은 방식).
// 멀티스케일 검출에서 보드를 절반으로 줄일 때 쓴다.
GrayImage resize_area(const GrayImage& src, float scale);

// 흑백 이미지를 RGB로 바꾸고 박스를 그려서 PNG로 저장한다.
bool save_with_boxes(const std::string& path, const GrayImage& gray,
                     const std::vector<Box>& boxes, int thickness = 2);
