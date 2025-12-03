#pragma once
#include <cstdint>
#include <deque>
#include <mutex>

/*
 Simple low-latency predictor:
 - maintains EWMA of OFI
 - issues BUY when ewma > threshold, SELL when ewma < -threshold
*/

class Predictor {
public:
    Predictor(double alpha = 0.2, double threshold = 50.0);
    // process single OFI sample; returns action: 1=BUY, -1=SELL, 0=HOLD
    int process_sample(double ofi);
    double get_ewma() const;
private:
    double alpha_;
    double threshold_;
    double ewma_;
    mutable std::mutex mtx_;
};
