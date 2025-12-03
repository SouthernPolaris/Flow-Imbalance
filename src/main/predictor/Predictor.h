#pragma once
#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>

// Optional OpenCL support is enabled via CMake option BUILD_WITH_OPENCL
#ifdef BUILD_WITH_OPENCL
#include <CL/cl.h>
#endif

/*
 Simple low-latency predictor:
 - maintains EWMA of OFI
 - issues BUY when ewma > threshold, SELL when ewma < -threshold
*/

class Predictor {
public:
    enum class Mode { CPU = 0, GPU };

    Predictor(double alpha = 0.2, double threshold = 50.0, Mode mode = Mode::CPU);
    // process single OFI sample; returns action: 1=BUY, -1=SELL, 0=HOLD
    int process_sample(double ofi);
    double get_ewma() const;

    // process a batch of data that contains multiple independent sequences.
    // Input layout: concatenated sequences, each of length `seq_len`.
    // `data.size()` must be `num_seqs * seq_len`.
    // `out` will be filled with num_seqs * seq_len prediction values (1/-1/0).
    // GPU mode will attempt to run using OpenCL if available; otherwise falls back to CPU.
    bool process_batch(const std::vector<double>& data, size_t num_seqs, size_t seq_len, std::vector<int>& out);

    // change runtime mode; if GPU requested but not available, remains CPU
    void set_mode(Mode m);
    // query current configured mode
    Mode get_mode() const;
    // whether GPU (OpenCL) path is available at runtime
    bool gpu_available() const;

private:
    double alpha_;
    double threshold_;
    double ewma_;
    mutable std::mutex mtx_;
    Mode mode_;

#ifdef BUILD_WITH_OPENCL
    // OpenCL runtime handles
    cl_platform_id clPlatform_ = nullptr;
    cl_device_id clDevice_ = nullptr;
    cl_context clContext_ = nullptr;
    cl_program clProgram_ = nullptr;
    cl_command_queue clQueue_ = nullptr;
    bool clInitialized_ = false;

    bool init_opencl();
    void teardown_opencl();
#endif
};
