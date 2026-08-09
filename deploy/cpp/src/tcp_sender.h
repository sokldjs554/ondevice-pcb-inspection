// 검출 결과를 TCP(JSON 한 줄)로 모니터링 서버에 보내는 코드
//
// deploy/monitor_server.py가 받는 형식과 똑같이 맞췄다. 즉 라즈베리파이 쪽 프로그램을
// 파이썬으로 두든 C++로 두든 서버는 바꿀 필요가 없다.
// JSON 라이브러리를 따로 붙이지 않고 문자열로 직접 만든다 (필드가 몇 개 안 된다).
#pragma once

#include <string>
#include <vector>

#include "detector.h"

// monitor_server.py가 파싱하는 형식의 JSON 한 줄을 만든다.
std::string build_json(const std::string& image_name, double elapsed_sec,
                       const std::vector<Detection>& dets);

// host:port로 접속해서 payload를 보내고 끊는다. 실패하면 false와 함께 err에 이유를 담는다.
bool send_json(const std::string& host, int port, const std::string& payload, std::string* err);

// "192.168.0.10:5000" 형태를 host/port로 나눈다.
bool parse_host_port(const std::string& s, std::string* host, int* port);
