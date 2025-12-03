#include "Predictor.h"

Predictor::Predictor(double alpha, double threshold)
    : alpha_(alpha), threshold_(threshold), ewma_(0.0) {}

int Predictor::process_sample(double ofi) {
    // update ewma: ewma = alpha * x + (1-alpha) * ewma
    // lightweight lock because we plan single-threaded MVP; can remove for lockfree later
    std::lock_guard<std::mutex> lk(mtx_);
    ewma_ = alpha_ * ofi + (1.0 - alpha_) * ewma_;
    if (ewma_ > threshold_) return 1;
    if (ewma_ < -threshold_) return -1;
    return 0;
}

double Predictor::get_ewma() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return ewma_;
}
