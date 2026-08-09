#include "postprocess.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <queue>

namespace {

// 박스 영역의 "템플릿과 다른 픽셀" 마스크 (1 = 다름)
std::vector<unsigned char> diff_mask(const GrayImage& gray, const GrayImage& tpl,
                                     const Box& box, int pixel_thresh, int& bw, int& bh) {
    int x1 = std::max(0, box.x1), y1 = std::max(0, box.y1);
    int x2 = std::min(gray.w, box.x2), y2 = std::min(gray.h, box.y2);
    bw = std::max(0, x2 - x1);
    bh = std::max(0, y2 - y1);
    std::vector<unsigned char> mask(static_cast<size_t>(bw) * bh, 0);
    for (int y = 0; y < bh; y++) {
        for (int x = 0; x < bw; x++) {
            int a = gray.at(y1 + y, x1 + x);
            int b = tpl.at(y1 + y, x1 + x);
            mask[static_cast<size_t>(y) * bw + x] = (std::abs(a - b) > pixel_thresh) ? 1 : 0;
        }
    }
    return mask;
}

// 사각형 커널 팽창/침식. 둘을 이어서 쓰면 닫힘(closing) 연산이 된다.
std::vector<unsigned char> morph(const std::vector<unsigned char>& src, int w, int h,
                                 int ksize, bool dilate) {
    std::vector<unsigned char> dst(src.size(), dilate ? 0 : 1);
    int r = ksize / 2;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            unsigned char v = dilate ? 0 : 1;
            for (int dy = -r; dy <= r && (dilate ? !v : v); dy++) {
                for (int dx = -r; dx <= r; dx++) {
                    int yy = y + dy, xx = x + dx;
                    // 경계 밖은 팽창에서는 0, 침식에서는 1로 본다 (OpenCV 기본과 동일한 효과)
                    unsigned char s = (yy < 0 || yy >= h || xx < 0 || xx >= w)
                                          ? (dilate ? 0 : 1)
                                          : src[static_cast<size_t>(yy) * w + xx];
                    if (dilate && s) { v = 1; break; }
                    if (!dilate && !s) { v = 0; break; }
                }
            }
            dst[static_cast<size_t>(y) * w + x] = v;
        }
    }
    return dst;
}

}  // namespace

float local_max_diff(const GrayImage& gray, const GrayImage& tpl, const Box& box,
                     int block, int pixel_thresh) {
    if (gray.empty() || tpl.empty()) return 1.f;
    int bw = 0, bh = 0;
    auto mask = diff_mask(gray, tpl, box, pixel_thresh, bw, bh);
    if (bw <= 0 || bh <= 0) return 0.f;

    // 블록 평균을 빨리 구하려고 적분 영상(summed-area table)을 만든다.
    // 640x640 보드에서 박스마다 이중 루프를 도는 것보다 확실히 빨랐다.
    std::vector<int> integral(static_cast<size_t>(bh + 1) * (bw + 1), 0);
    for (int y = 0; y < bh; y++) {
        int row_sum = 0;
        for (int x = 0; x < bw; x++) {
            row_sum += mask[static_cast<size_t>(y) * bw + x];
            integral[static_cast<size_t>(y + 1) * (bw + 1) + x + 1] =
                integral[static_cast<size_t>(y) * (bw + 1) + x + 1] + row_sum;
        }
    }
    auto rect_sum = [&](int x0, int y0, int x1, int y1) {
        return integral[static_cast<size_t>(y1) * (bw + 1) + x1]
             - integral[static_cast<size_t>(y0) * (bw + 1) + x1]
             - integral[static_cast<size_t>(y1) * (bw + 1) + x0]
             + integral[static_cast<size_t>(y0) * (bw + 1) + x0];
    };

    float best = 0.f;
    int step = std::max(1, block / 2);
    for (int y = 0; y < std::max(1, bh - block + 1); y += step) {
        for (int x = 0; x < std::max(1, bw - block + 1); x += step) {
            int x2 = std::min(bw, x + block), y2 = std::min(bh, y + block);
            int area = (x2 - x) * (y2 - y);
            if (area <= 0) continue;
            best = std::max(best, static_cast<float>(rect_sum(x, y, x2, y2)) / area);
        }
    }
    return best;
}

Box refine_box_by_template(const Box& box, const GrayImage& gray, const GrayImage& tpl,
                           int pad, int pixel_thresh) {
    int bw = 0, bh = 0;
    auto mask = diff_mask(gray, tpl, box, pixel_thresh, bw, bh);
    if (bw <= 0 || bh <= 0) return box;

    // 끊어진 차분 영역을 하나로 잇는다 (7x7 닫힘)
    mask = morph(mask, bw, bh, 7, true);
    mask = morph(mask, bw, bh, 7, false);

    // 가장 큰 덩어리를 결함으로 본다 (BFS 라벨링 + 면적/경계 계산)
    std::vector<char> visited(mask.size(), 0);
    int best_area = 0;
    int bx0 = 0, by0 = 0, bx1 = 0, by1 = 0;
    const int dy[] = {-1, -1, -1, 0, 0, 1, 1, 1};
    const int dx[] = {-1, 0, 1, -1, 1, -1, 0, 1};
    for (int i = 0; i < bw * bh; i++) {
        if (visited[i] || !mask[i]) continue;
        int area = 0, x0 = bw, y0 = bh, x1 = -1, y1 = -1;
        std::queue<int> q;
        q.push(i);
        visited[i] = 1;
        while (!q.empty()) {
            int cur = q.front();
            q.pop();
            int cy = cur / bw, cx = cur % bw;
            area++;
            x0 = std::min(x0, cx); x1 = std::max(x1, cx);
            y0 = std::min(y0, cy); y1 = std::max(y1, cy);
            for (int k = 0; k < 8; k++) {
                int ny = cy + dy[k], nx = cx + dx[k];
                if (ny < 0 || ny >= bh || nx < 0 || nx >= bw) continue;
                int ni = ny * bw + nx;
                if (!visited[ni] && mask[ni]) { visited[ni] = 1; q.push(ni); }
            }
        }
        if (area > best_area) {
            best_area = area;
            bx0 = x0; by0 = y0; bx1 = x1; by1 = y1;
        }
    }
    if (best_area == 0) return box;

    int ox = std::max(0, box.x1), oy = std::max(0, box.y1);
    return {std::max(0, ox + bx0 - pad), std::max(0, oy + by0 - pad),
            std::min(gray.w, ox + bx1 + 1 + pad), std::min(gray.h, oy + by1 + 1 + pad)};
}

std::vector<Detection> filter_by_template(std::vector<Detection> dets, const GrayImage& gray,
                                          const GrayImage& tpl, float min_diff, bool refine) {
    std::vector<Detection> kept;
    for (Detection& d : dets) {
        float ratio = local_max_diff(gray, tpl, d.box);
        d.template_diff = ratio;
        if (ratio < min_diff) continue;
        if (refine) d.box = refine_box_by_template(d.box, gray, tpl);
        kept.push_back(d);
    }
    return kept;
}
