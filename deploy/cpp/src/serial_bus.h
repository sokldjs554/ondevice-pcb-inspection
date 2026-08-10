// SPI / I2C로 검사 결과를 내보내는 코드
//
// TCP(Ethernet)는 관리 PC로 결과를 모을 때 쓰지만, 검사 장비 안에서 MCU나 PLC 같은
// 다른 칩에 판정을 알려야 하는 경우도 있습니다. 그럴 때 쓰는 게 SPI나 I2C 같은
// 보드 내부 버스입니다. 이 버스는 대역폭이 낮아서 JSON을 그대로 보내기 어렵고,
// 길이가 정해진 바이너리 프레임을 쓰는 게 일반적입니다.
//
// 그래서 두 가지를 나눠서 만들었습니다.
//   1) 프레임 만들기/해석하기 (build_frame / parse_frame) — 하드웨어와 무관
//   2) 실제로 내보내는 통로 (Transport) — spidev / i2c-dev / 루프백
//
// SPI·I2C 개발보드가 없어서, 같은 프레임을 메모리로 돌려받는 루프백 통로를 만들어
// 프레이밍과 CRC를 검증했습니다. spidev/i2c-dev 쪽 코드는 리눅스 ioctl API를 그대로
// 쓰기 때문에 라즈베리파이에서는 장치 경로만 지정하면 동작하도록 짰습니다.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "detector.h"

// ── 프레임 형식 ────────────────────────────────────────────────────────────
//  오프셋  크기  내용
//    0      2    시작 마커 0xA5 0x5A (수신 쪽이 프레임 시작을 찾는 기준)
//    2      1    버전 (1)
//    3      1    보드 판정 (0=정상, 1=불량)
//    4      2    payload 길이 (little endian)
//    6      N    payload
//   6+N     2    CRC-16/CCITT-FALSE (헤더+payload 전체)
//
//  payload = [보드 번호 4바이트][결함 수 1바이트] + 결함마다 10바이트
//            결함 = [종류 1][x1 2][y1 2][x2 2][y2 2][확신도 1(0~255)]
//
//  결함 6건이면 총 6 + 5 + 60 + 2 = 73바이트. SPI 1MHz에서 1ms 이내에 나갑니다.
constexpr uint8_t FRAME_MARK0 = 0xA5;
constexpr uint8_t FRAME_MARK1 = 0x5A;
constexpr uint8_t FRAME_VERSION = 1;
constexpr size_t FRAME_HEADER = 6;
constexpr size_t DEFECT_BYTES = 10;

uint16_t crc16_ccitt(const uint8_t* data, size_t len);

std::vector<uint8_t> build_frame(uint32_t board_id, const std::vector<Detection>& dets);

// 프레임을 되돌린다. 마커·길이·CRC가 맞지 않으면 false.
bool parse_frame(const std::vector<uint8_t>& frame, uint32_t* board_id,
                 std::vector<Detection>* dets, std::string* err);

// 파일 이름에서 숫자만 뽑아 보드 번호로 쓴다 (00041000_test.jpg -> 41000)
uint32_t board_id_from_name(const std::string& name);

// ── 통로 ──────────────────────────────────────────────────────────────────
class Transport {
public:
    virtual ~Transport() = default;
    virtual bool send(const std::vector<uint8_t>& data, std::string* err) = 0;
    virtual std::string name() const = 0;
    // 루프백만 구현. 방금 보낸 바이트를 돌려준다.
    virtual const std::vector<uint8_t>* lastSent() const { return nullptr; }
};

// spi: "/dev/spidev0.0", i2c: "/dev/i2c-1:0x42"
std::unique_ptr<Transport> make_spi(const std::string& device, int speed_hz, std::string* err);
std::unique_ptr<Transport> make_i2c(const std::string& device_and_addr, std::string* err);
std::unique_ptr<Transport> make_loopback();

// 하드웨어 없이 프레임 생성 → 전송 → 해석까지 돌려보는 자체 점검. 실패하면 false.
bool run_bus_selftest();
