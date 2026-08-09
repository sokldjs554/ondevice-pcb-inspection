#include "image_io.h"

#include <algorithm>
#include <cmath>

// stb는 남이 만든 코드라 경고를 켜두면 수백 줄이 뜬다. 이 파일에서만 잠시 꺼둔다.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#define STB_IMAGE_IMPLEMENTATION
#include "../third_party/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../third_party/stb_image_write.h"
#pragma GCC diagnostic pop

GrayImage load_gray(const std::string& path) {
    GrayImage img;
    int w = 0, h = 0, ch = 0;
    unsigned char* buf = stbi_load(path.c_str(), &w, &h, &ch, 1);  // 마지막 1 = 흑백 강제
    if (!buf) return img;
    img.w = w;
    img.h = h;
    img.data.assign(buf, buf + static_cast<size_t>(w) * h);
    stbi_image_free(buf);
    return img;
}

GrayImage resize_area(const GrayImage& src, float scale) {
    GrayImage dst;
    if (src.empty() || scale <= 0.f) return dst;
    if (scale == 1.0f) return src;

    dst.w = std::max(1, static_cast<int>(std::round(src.w * scale)));
    dst.h = std::max(1, static_cast<int>(std::round(src.h * scale)));
    dst.data.resize(static_cast<size_t>(dst.w) * dst.h);

    // 출력 픽셀 하나가 입력의 어느 사각형에서 왔는지 계산해서 그 영역의 평균을 쓴다.
    // 그냥 최근접 이웃으로 줄이면 얇은 배선이 통째로 사라져서 결함까지 날아간다.
    for (int y = 0; y < dst.h; y++) {
        int sy0 = static_cast<int>(y / scale);
        int sy1 = std::min(src.h, std::max(sy0 + 1, static_cast<int>((y + 1) / scale)));
        for (int x = 0; x < dst.w; x++) {
            int sx0 = static_cast<int>(x / scale);
            int sx1 = std::min(src.w, std::max(sx0 + 1, static_cast<int>((x + 1) / scale)));
            int sum = 0, n = 0;
            for (int yy = sy0; yy < sy1; yy++) {
                for (int xx = sx0; xx < sx1; xx++) {
                    sum += src.at(yy, xx);
                    n++;
                }
            }
            dst.data[static_cast<size_t>(y) * dst.w + x] =
                static_cast<unsigned char>(n ? (sum + n / 2) / n : 0);
        }
    }
    return dst;
}

bool save_with_boxes(const std::string& path, const GrayImage& gray,
                     const std::vector<Box>& boxes, int thickness) {
    if (gray.empty()) return false;
    std::vector<unsigned char> rgb(static_cast<size_t>(gray.w) * gray.h * 3);
    for (size_t i = 0; i < gray.data.size(); i++) {
        rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = gray.data[i];
    }

    auto put = [&](int x, int y) {
        if (x < 0 || x >= gray.w || y < 0 || y >= gray.h) return;
        size_t i = (static_cast<size_t>(y) * gray.w + x) * 3;
        rgb[i] = 0; rgb[i + 1] = 255; rgb[i + 2] = 0;  // 초록
    };
    for (const Box& b : boxes) {
        for (int t = 0; t < thickness; t++) {
            for (int x = b.x1; x < b.x2; x++) { put(x, b.y1 + t); put(x, b.y2 - 1 - t); }
            for (int y = b.y1; y < b.y2; y++) { put(b.x1 + t, y); put(b.x2 - 1 - t, y); }
        }
    }
    return stbi_write_png(path.c_str(), gray.w, gray.h, 3, rgb.data(), gray.w * 3) != 0;
}
