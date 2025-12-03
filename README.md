# Flow Imbalance

## Requirements
- Linux (recommended) or WSL / macOS (minor socket adjustments)
- C++20 toolchain (g++ 11+/clang 13+)
- CMake 3.10+
- Python 3

## Build
mkdir build && cd build
cmake ..
make -j

## Run
1. Start the C++ listener:
   ./flow_imbalance 9000

2. In another terminal, start feed:
   python3 feedgen.py 127.0.0.1 9000 2000

This will generate ~2000 ticks/sec. The C++ program will log BUY/SELL events when the EWMA OFI crosses thresholds.
On Ctrl+C the program prints latency summaries.
