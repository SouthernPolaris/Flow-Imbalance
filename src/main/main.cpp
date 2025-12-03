#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <signal.h>
#include <iomanip>

#include "OrderBook.h"
#include "OFI.h"
#include "Predictor.h"
#include <numeric>
#include <algorithm>

static bool keep_running = true;
void sigint_handler(int){ keep_running = false; }

using steady_clock = std::chrono::steady_clock;
using ns = std::chrono::nanoseconds;

struct Stats {
    std::vector<double> lat_recv_decision_us;
    std::vector<double> lat_src_recv_us;
    void push(double a, double b) {
        lat_recv_decision_us.push_back(a);
        lat_src_recv_us.push_back(b);
    }
};

int main(int argc, char** argv) {
    signal(SIGINT, sigint_handler);
    // const char* bind_addr = "0.0.0.0";
    int port = 9000;
    if (argc > 1) port = std::atoi(argv[1]);

    // Setup UDP socket
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) { perror("socket"); return 1; }
    sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    if (bind(sock, (sockaddr*)&addr, sizeof(addr)) < 0) { perror("bind"); return 1; }

    std::cout << "Listening UDP on port " << port << "\n";

    OrderBook ob;
    Predictor pred(0.15, 40.0);
    Stats stats;

    // minimal buffer for incoming UDP line
    const int BUF_SZ = 2048;
    char buf[BUF_SZ];

    // For OFI we need the previous tick
    bool have_prev = false;
    Tick prev_tick;

    while (keep_running) {
        sockaddr_in src;
        socklen_t srclen = sizeof(src);
        ssize_t n = recvfrom(sock, buf, BUF_SZ - 1, 0, (sockaddr*)&src, &srclen);
        if (n <= 0) continue;
        buf[n] = '\0';
        // host recv timestamp
        auto recv_tp = std::chrono::system_clock::now();
        double recv_ts = std::chrono::duration_cast<std::chrono::duration<double>>(recv_tp.time_since_epoch()).count();

        // parse CSV: seq,src_ts,price,size
        uint64_t seq = 0;
        double src_ts = 0.0;
        double price = 0.0;
        uint32_t size = 0;
        // use sscanf for speed (robust enough for this format)
        int matched = std::sscanf(buf, "%lu,%lf,%lf,%u", &seq, &src_ts, &price, &size);
        if (matched < 4) {
            // try alternative parse with long long on systems where %llu isn't right
            unsigned long long tmpseq;
            int m2 = std::sscanf(buf, "%llu,%lf,%lf,%u", &tmpseq, &src_ts, &price, &size);
            if (m2 >= 4) seq = tmpseq;
            else continue;
        }

        Tick tick;
        tick.seq = seq;
        tick.src_ts = src_ts;
        tick.recv_ts = recv_ts;
        tick.price = price;
        tick.size = size;

        // apply to simple orderbook (store latest)
        // compute using prev tick
        double ofi = 0.0;
        if (have_prev) {
            ofi = compute_ofi(prev_tick, tick);
        }
        prev_tick = tick;
        have_prev = true;
        ob.apply_tick(tick);

        // predictor timing
        auto dec_start = steady_clock::now();
        int action = pred.process_sample(ofi);
        auto dec_end = steady_clock::now();

        double recv_to_decision_us = std::chrono::duration_cast<ns>(dec_end - dec_start).count() / 1000.0;
        double src_to_recv_us = (tick.recv_ts - tick.src_ts) * 1e6;

        stats.push(recv_to_decision_us, src_to_recv_us);

        // emit signal (print for now)
        if (action != 0) {
            const char* act = action > 0 ? "BUY" : "SELL";
            double ewma = pred.get_ewma();
            std::cout << "[" << seq << "] " << act << " ewma=" << std::fixed << std::setprecision(2) << ewma
                      << " ofi=" << ofi
                      << " recv->dec(us)=" << recv_to_decision_us
                      << " src->recv(us)=" << src_to_recv_us
                      << "\n";
        }

        // optionally: throttle printing to avoid slowing everything; MVP leaves as-is

    }

    // Summary stats
    auto print_stats = [](const std::vector<double>& v, const char* name){
        if (v.empty()) return;
        std::vector<double> copy = v;
        std::sort(copy.begin(), copy.end());
        auto p = [&](double q){
            int idx = int(q * (copy.size()-1));
            return copy[idx];
        };
        std::cout << "STAT " << name << " count=" << copy.size()
                  << " p50=" << p(0.5)
                  << " p90=" << p(0.9)
                  << " p99=" << p(0.99)
                  << " mean=" << (std::accumulate(copy.begin(), copy.end(), 0.0)/copy.size())
                  << "\n";
    };

    print_stats(stats.lat_recv_decision_us, "recv->decision_us");
    print_stats(stats.lat_src_recv_us, "src->recv_us");

    close(sock);
    return 0;
}
