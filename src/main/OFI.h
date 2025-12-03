#pragma once
#include "OrderBook.h"

// Very small OFI helper, compute OFI-like score from two consecutive ticks.
// OFI approximation used here: sign(price_delta) * size
inline double compute_ofi(const Tick& prev, const Tick& cur) {
    double dp = cur.price - prev.price;
    if (dp > 0) return double(cur.size);
    if (dp < 0) return -double(cur.size);
    return 0.0;
}
