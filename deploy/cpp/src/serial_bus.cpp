#include "serial_bus.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>

#include <fcntl.h>
#include <unistd.h>

#if defined(__linux__)
#include <linux/i2c-dev.h>
#include <linux/spi/spidev.h>
#include <sys/ioctl.h>
#endif

namespace {

void put_u16(std::vector<uint8_t>& v, uint16_t x) {
    v.push_back(static_cast<uint8_t>(x & 0xFF));
    v.push_back(static_cast<uint8_t>(x >> 8));
}

uint16_t get_u16(const std::vector<uint8_t>& v, size_t off) {
    return static_cast<uint16_t>(v[off] | (v[off + 1] << 8));
}

// 좌표가 640을 넘을 일은 없지만, 버스로 나가는 값은 범위를 강제해둔다
uint16_t clamp_u16(int x) {
    if (x < 0) return 0;
    if (x > 65535) return 65535;
    return static_cast<uint16_t>(x);
}

}  // namespace

uint16_t crc16_ccitt(const uint8_t* data, size_t len) {
    // CRC-16/CCITT-FALSE. 임베디드에서 흔히 쓰는 방식이고 표를 안 만들어도 될 만큼 짧다.
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < len; i++) {
        crc ^= static_cast<uint16_t>(data[i]) << 8;
        for (int b = 0; b < 8; b++) {
            crc = (crc & 0x8000) ? static_cast<uint16_t>((crc << 1) ^ 0x1021)
                                 : static_cast<uint16_t>(crc << 1);
        }
    }
    return crc;
}

std::vector<uint8_t> build_frame(uint32_t board_id, const std::vector<Detection>& dets) {
    // 결함이 아주 많아도 한 프레임이 커지지 않도록 상한을 둔다 (수신 쪽 버퍼 보호)
    const size_t MAX_DEFECTS = 24;
    size_t n = std::min(dets.size(), MAX_DEFECTS);

    std::vector<uint8_t> payload;
    payload.reserve(5 + n * DEFECT_BYTES);
    payload.push_back(static_cast<uint8_t>(board_id & 0xFF));
    payload.push_back(static_cast<uint8_t>((board_id >> 8) & 0xFF));
    payload.push_back(static_cast<uint8_t>((board_id >> 16) & 0xFF));
    payload.push_back(static_cast<uint8_t>((board_id >> 24) & 0xFF));
    payload.push_back(static_cast<uint8_t>(n));
    for (size_t i = 0; i < n; i++) {
        const Detection& d = dets[i];
        payload.push_back(static_cast<uint8_t>(d.cls));
        put_u16(payload, clamp_u16(d.box.x1));
        put_u16(payload, clamp_u16(d.box.y1));
        put_u16(payload, clamp_u16(d.box.x2));
        put_u16(payload, clamp_u16(d.box.y2));
        // 확신도는 0~1 실수인데 1바이트로 줄인다 (0.4% 해상도면 판정용으로 충분)
        int score = static_cast<int>(d.score * 255.0f + 0.5f);
        payload.push_back(static_cast<uint8_t>(std::min(255, std::max(0, score))));
    }

    std::vector<uint8_t> frame;
    frame.reserve(FRAME_HEADER + payload.size() + 2);
    frame.push_back(FRAME_MARK0);
    frame.push_back(FRAME_MARK1);
    frame.push_back(FRAME_VERSION);
    frame.push_back(dets.empty() ? 0 : 1);  // 보드 판정
    put_u16(frame, static_cast<uint16_t>(payload.size()));
    frame.insert(frame.end(), payload.begin(), payload.end());
    put_u16(frame, crc16_ccitt(frame.data(), frame.size()));
    return frame;
}

bool parse_frame(const std::vector<uint8_t>& frame, uint32_t* board_id,
                 std::vector<Detection>* dets, std::string* err) {
    auto fail = [&](const char* m) {
        if (err) *err = m;
        return false;
    };
    if (frame.size() < FRAME_HEADER + 2) return fail("프레임이 너무 짧습니다");
    if (frame[0] != FRAME_MARK0 || frame[1] != FRAME_MARK1) return fail("시작 마커가 다릅니다");
    if (frame[2] != FRAME_VERSION) return fail("모르는 버전입니다");

    size_t len = get_u16(frame, 4);
    if (frame.size() != FRAME_HEADER + len + 2) return fail("길이가 맞지 않습니다");

    uint16_t want = get_u16(frame, FRAME_HEADER + len);
    uint16_t got = crc16_ccitt(frame.data(), FRAME_HEADER + len);
    if (want != got) return fail("CRC가 맞지 않습니다");

    size_t p = FRAME_HEADER;
    if (len < 5) return fail("payload가 너무 짧습니다");
    *board_id = static_cast<uint32_t>(frame[p]) | (static_cast<uint32_t>(frame[p + 1]) << 8) |
                (static_cast<uint32_t>(frame[p + 2]) << 16) |
                (static_cast<uint32_t>(frame[p + 3]) << 24);
    size_t n = frame[p + 4];
    p += 5;
    if (len != 5 + n * DEFECT_BYTES) return fail("결함 수와 길이가 맞지 않습니다");

    dets->clear();
    for (size_t i = 0; i < n; i++) {
        Detection d;
        d.cls = frame[p];
        d.box = {get_u16(frame, p + 1), get_u16(frame, p + 3), get_u16(frame, p + 5),
                 get_u16(frame, p + 7)};
        d.score = frame[p + 9] / 255.0f;
        dets->push_back(d);
        p += DEFECT_BYTES;
    }
    return true;
}

uint32_t board_id_from_name(const std::string& name) {
    uint32_t id = 0;
    for (char c : name) {
        if (c >= '0' && c <= '9') id = id * 10 + static_cast<uint32_t>(c - '0');
        else if (id) break;  // 숫자 덩어리 하나만 쓴다
    }
    return id;
}

// ── 통로 구현 ─────────────────────────────────────────────────────────────

namespace {

class LoopbackTransport : public Transport {
public:
    bool send(const std::vector<uint8_t>& data, std::string*) override {
        last_ = data;
        return true;
    }
    std::string name() const override { return "loopback"; }
    const std::vector<uint8_t>* lastSent() const override { return &last_; }

private:
    std::vector<uint8_t> last_;
};

#if defined(__linux__)

class SpiTransport : public Transport {
public:
    SpiTransport(int fd, std::string dev, int speed) : fd_(fd), dev_(std::move(dev)), speed_(speed) {}
    ~SpiTransport() override { if (fd_ >= 0) ::close(fd_); }

    bool send(const std::vector<uint8_t>& data, std::string* err) override {
        // SPI는 보내면서 동시에 받는 구조라, 수신 버퍼도 같이 넘긴다.
        // (MCU가 응답을 실어 보내는 경우를 대비해 rx도 받아둔다)
        std::vector<uint8_t> rx(data.size(), 0);
        spi_ioc_transfer tr{};
        tr.tx_buf = reinterpret_cast<unsigned long>(data.data());
        tr.rx_buf = reinterpret_cast<unsigned long>(rx.data());
        tr.len = static_cast<uint32_t>(data.size());
        tr.speed_hz = static_cast<uint32_t>(speed_);
        tr.bits_per_word = 8;
        if (::ioctl(fd_, SPI_IOC_MESSAGE(1), &tr) < 1) {
            if (err) *err = std::string("SPI 전송 실패: ") + std::strerror(errno);
            return false;
        }
        return true;
    }
    std::string name() const override { return "SPI " + dev_; }

private:
    int fd_;
    std::string dev_;
    int speed_;
};

class I2cTransport : public Transport {
public:
    I2cTransport(int fd, std::string dev, int addr) : fd_(fd), dev_(std::move(dev)), addr_(addr) {}
    ~I2cTransport() override { if (fd_ >= 0) ::close(fd_); }

    bool send(const std::vector<uint8_t>& data, std::string* err) override {
        ssize_t n = ::write(fd_, data.data(), data.size());
        if (n != static_cast<ssize_t>(data.size())) {
            if (err) *err = std::string("I2C 전송 실패: ") + std::strerror(errno);
            return false;
        }
        return true;
    }
    std::string name() const override {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "I2C %s addr 0x%02X", dev_.c_str(), addr_);
        return buf;
    }

private:
    int fd_;
    std::string dev_;
    int addr_;
};

#endif  // __linux__

}  // namespace

std::unique_ptr<Transport> make_loopback() {
    return std::unique_ptr<Transport>(new LoopbackTransport());
}

std::unique_ptr<Transport> make_spi(const std::string& device, int speed_hz, std::string* err) {
#if defined(__linux__)
    int fd = ::open(device.c_str(), O_RDWR);
    if (fd < 0) {
        if (err) *err = device + " 열기 실패: " + std::strerror(errno);
        return nullptr;
    }
    uint8_t mode = SPI_MODE_0;   // 대부분의 MCU가 쓰는 기본 모드
    uint8_t bits = 8;
    uint32_t speed = static_cast<uint32_t>(speed_hz);
    if (::ioctl(fd, SPI_IOC_WR_MODE, &mode) < 0 ||
        ::ioctl(fd, SPI_IOC_WR_BITS_PER_WORD, &bits) < 0 ||
        ::ioctl(fd, SPI_IOC_WR_MAX_SPEED_HZ, &speed) < 0) {
        if (err) *err = std::string("SPI 설정 실패: ") + std::strerror(errno);
        ::close(fd);
        return nullptr;
    }
    return std::unique_ptr<Transport>(new SpiTransport(fd, device, speed_hz));
#else
    (void)device; (void)speed_hz;
    if (err) *err = "SPI는 리눅스에서만 지원합니다 (라즈베리파이 등). --bus-selftest로 프레임만 확인할 수 있습니다.";
    return nullptr;
#endif
}

std::unique_ptr<Transport> make_i2c(const std::string& device_and_addr, std::string* err) {
#if defined(__linux__)
    // "/dev/i2c-1:0x42" 형태를 장치와 주소로 나눈다
    size_t pos = device_and_addr.rfind(':');
    if (pos == std::string::npos) {
        if (err) *err = "형식이 잘못됐습니다 (예: /dev/i2c-1:0x42)";
        return nullptr;
    }
    std::string dev = device_and_addr.substr(0, pos);
    int addr = static_cast<int>(std::strtol(device_and_addr.c_str() + pos + 1, nullptr, 0));
    int fd = ::open(dev.c_str(), O_RDWR);
    if (fd < 0) {
        if (err) *err = dev + " 열기 실패: " + std::strerror(errno);
        return nullptr;
    }
    if (::ioctl(fd, I2C_SLAVE, addr) < 0) {
        if (err) *err = std::string("I2C 주소 설정 실패: ") + std::strerror(errno);
        ::close(fd);
        return nullptr;
    }
    return std::unique_ptr<Transport>(new I2cTransport(fd, dev, addr));
#else
    (void)device_and_addr;
    if (err) *err = "I2C는 리눅스에서만 지원합니다. --bus-selftest로 프레임만 확인할 수 있습니다.";
    return nullptr;
#endif
}

bool run_bus_selftest() {
    // 하드웨어 없이 확인할 수 있는 것: 프레임을 만들고 → 통로로 보내고 →
    // 되돌려 받아 해석했을 때 원래 값이 그대로 나오는가, 그리고 망가진 프레임을 걸러내는가.
    std::printf("버스 프레임 자체 점검 (하드웨어 없이 루프백으로 확인)\n\n");

    std::vector<Detection> dets;
    dets.push_back({1, {32, 0, 128, 64}, 0.999f, 1.f, -1.f});
    dets.push_back({5, {160, 128, 224, 224}, 1.000f, 1.f, -1.f});
    dets.push_back({6, {320, 320, 416, 384}, 0.830f, 1.f, -1.f});

    auto bus = make_loopback();
    std::vector<uint8_t> frame = build_frame(41000, dets);
    std::string err;
    if (!bus->send(frame, &err)) {
        std::printf("  [실패] 전송: %s\n", err.c_str());
        return false;
    }
    const std::vector<uint8_t>* rx = bus->lastSent();
    std::printf("  프레임 %zu바이트 (결함 %zu건) — 헤더 6 + payload %zu + CRC 2\n", frame.size(),
                dets.size(), frame.size() - 8);
    std::printf("  앞 12바이트: ");
    for (size_t i = 0; i < 12 && i < frame.size(); i++) std::printf("%02X ", frame[i]);
    std::printf("\n");

    uint32_t id = 0;
    std::vector<Detection> back;
    if (!parse_frame(*rx, &id, &back, &err)) {
        std::printf("  [실패] 해석: %s\n", err.c_str());
        return false;
    }
    bool ok = (id == 41000) && (back.size() == dets.size());
    for (size_t i = 0; ok && i < back.size(); i++) {
        ok = back[i].cls == dets[i].cls && back[i].box.x1 == dets[i].box.x1 &&
             back[i].box.y1 == dets[i].box.y1 && back[i].box.x2 == dets[i].box.x2 &&
             back[i].box.y2 == dets[i].box.y2 &&
             std::abs(back[i].score - dets[i].score) < 0.01f;
    }
    std::printf("  왕복 확인: 보드 번호 %u, 결함 %zu건 → %s\n", id, back.size(),
                ok ? "원본과 일치" : "불일치");
    if (!ok) return false;

    // 일부러 한 바이트를 뒤집어서 CRC가 잡아내는지 본다
    std::vector<uint8_t> broken = frame;
    broken[10] ^= 0x01;
    bool caught = !parse_frame(broken, &id, &back, &err);
    std::printf("  1비트 손상 검출: %s (%s)\n", caught ? "잡아냄" : "못 잡음", err.c_str());
    if (!caught) return false;

    // 길이를 속인 프레임도 거부해야 한다
    std::vector<uint8_t> short_frame(frame.begin(), frame.begin() + frame.size() - 3);
    bool caught2 = !parse_frame(short_frame, &id, &back, &err);
    std::printf("  잘린 프레임 검출: %s (%s)\n", caught2 ? "잡아냄" : "못 잡음", err.c_str());
    if (!caught2) return false;

    std::printf("\n통과. SPI 1MHz 기준 %zu바이트는 약 %.2f ms, 100kHz I2C에서는 약 %.1f ms 걸립니다.\n",
                frame.size(), frame.size() * 8 / 1000.0, frame.size() * 9 / 100.0);
    std::printf("실제 장치에 보내려면 --spi /dev/spidev0.0 또는 --i2c /dev/i2c-1:0x42 를 씁니다.\n");
    return true;
}
