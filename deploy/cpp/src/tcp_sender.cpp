#include "tcp_sender.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <ctime>

#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

namespace {

std::string now_string() {
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    return buf;
}

std::string fmt(const char* f, double v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), f, v);
    return buf;
}

}  // namespace

std::string build_json(const std::string& image_name, double elapsed_sec,
                       const std::vector<Detection>& dets) {
    std::string s = "{\"image\": \"" + image_name + "\", \"timestamp\": \"" + now_string() +
                    "\", \"elapsed_sec\": " + fmt("%.2f", elapsed_sec) +
                    ", \"source\": \"cpp\", \"defects\": [";
    for (size_t i = 0; i < dets.size(); i++) {
        const Detection& d = dets[i];
        if (i) s += ", ";
        s += "{\"type\": \"";
        s += d.name();
        s += "\", \"box\": [" + std::to_string(d.box.x1) + ", " + std::to_string(d.box.y1) + ", " +
             std::to_string(d.box.x2) + ", " + std::to_string(d.box.y2) + "]";
        s += ", \"score\": " + fmt("%.3f", d.score);
        if (d.template_diff >= 0.f) s += ", \"template_diff\": " + fmt("%.4f", d.template_diff);
        s += "}";
    }
    s += "]}\n";  // 서버가 줄 단위로 읽으므로 마지막에 개행
    return s;
}

bool parse_host_port(const std::string& s, std::string* host, int* port) {
    size_t pos = s.rfind(':');
    if (pos == std::string::npos || pos == 0 || pos + 1 >= s.size()) return false;
    *host = s.substr(0, pos);
    *port = std::atoi(s.c_str() + pos + 1);
    return *port > 0;
}

bool send_json(const std::string& host, int port, const std::string& payload, std::string* err) {
    // getaddrinfo를 쓰면 IP든 호스트 이름이든 그대로 넘길 수 있다.
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    int rc = getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res);
    if (rc != 0) {
        if (err) *err = std::string("주소 확인 실패: ") + gai_strerror(rc);
        return false;
    }

    int fd = -1;
    for (addrinfo* p = res; p; p = p->ai_next) {
        fd = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (fd < 0) continue;
        if (::connect(fd, p->ai_addr, p->ai_addrlen) == 0) break;
        ::close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    if (fd < 0) {
        if (err) *err = std::string("접속 실패: ") + std::strerror(errno);
        return false;
    }

    // send는 요청한 만큼 다 안 보낼 수 있어서 보낸 바이트만큼 밀면서 반복한다.
    size_t sent = 0;
    while (sent < payload.size()) {
        ssize_t n = ::send(fd, payload.data() + sent, payload.size() - sent, 0);
        if (n <= 0) {
            if (err) *err = std::string("전송 실패: ") + std::strerror(errno);
            ::close(fd);
            return false;
        }
        sent += static_cast<size_t>(n);
    }
    ::close(fd);  // 서버는 연결이 끊길 때까지 읽으므로 여기서 닫아야 한다
    return true;
}
