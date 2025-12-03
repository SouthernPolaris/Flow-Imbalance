#pragma once
#include <cstdint>
#include <optional>

/*
 Simple L1-ish state to keep last price and last size.
 This is intentionally minimal for now
*/

struct Tick {
    uint64_t seq;
    double src_ts;
    double recv_ts;
    double price;
    uint32_t size;
};

class OrderBook {
public:
    OrderBook();
    // ~OrderBook();

    void apply_tick(const Tick& t);
    // returns price delta (price - last_price) or 0 if no previous
    double last_price_delta() const;
    double last_price() const;
    uint32_t last_size() const;
    bool has_last() const;
private:
    double last_price_;
    uint32_t last_size_;
    bool has_last_;
};
