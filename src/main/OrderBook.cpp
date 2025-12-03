#include "OrderBook.h"

OrderBook::OrderBook() : last_price_(0.0), last_size_(0), has_last_(false) {}

void OrderBook::apply_tick(const Tick& t) {
    last_price_ = t.price;
    last_size_ = t.size;
    has_last_ = true;
}

double OrderBook::last_price_delta() const {
    return has_last_ ? (last_price_ - last_price_) : 0.0; // trivial placeholder
}

double OrderBook::last_price() const { return last_price_; }
uint32_t OrderBook::last_size() const { return last_size_; }
bool OrderBook::has_last() const { return has_last_; }
