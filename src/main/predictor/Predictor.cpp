#include "Predictor.h"
#include <iostream>
#include <cstring>

Predictor::Predictor(double alpha, double threshold, Mode mode)
    : alpha_(alpha), threshold_(threshold), ewma_(0.0), mode_(mode) {
#ifdef BUILD_WITH_OPENCL
    if (mode_ == Mode::GPU) {
        if (!init_opencl()) {
            std::cerr << "Predictor: OpenCL init failed; falling back to CPU mode\n";
            mode_ = Mode::CPU;
        }
    }
#endif
}

int Predictor::process_sample(double ofi) {
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

Predictor::Mode Predictor::get_mode() const {
    return mode_;
}

bool Predictor::gpu_available() const {
#ifdef BUILD_WITH_OPENCL
    return clInitialized_;
#else
    (void)0; // silence unused-warning
    return false;
#endif
}

void Predictor::set_mode(Mode m) {
#ifdef BUILD_WITH_OPENCL
    if (m == Mode::GPU && !clInitialized_) {
        if (!init_opencl()) {
            std::cerr << "Predictor: OpenCL init failed; staying in CPU mode\n";
            mode_ = Mode::CPU;
            return;
        }
    }
#endif
    mode_ = m;
}

bool Predictor::process_batch(const std::vector<double>& data, size_t num_seqs, size_t seq_len, std::vector<int>& out) {
    if (num_seqs == 0 || seq_len == 0) return false;
    if (data.size() != num_seqs * seq_len) return false;
    out.assign(num_seqs * seq_len, 0);

    if (mode_ == Mode::GPU) {
#ifdef BUILD_WITH_OPENCL
        if (!clInitialized_) {
            if (!init_opencl()) {
                std::cerr << "Predictor: OpenCL not available; falling back to CPU batch\n";
            } else {
                // proceed with GPU
            }
        }
        if (clInitialized_) {
            // Build kernel source (simple): each work-item processes one independent sequence sequentially
            const char* kernelSrc =
                "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
                "__kernel void ewma_batch(__global const double* data, __global int* out, double alpha, double threshold, uint seq_len) {\n"
                "    uint seq_id = get_global_id(0);\n"
                "    uint base = seq_id * seq_len;\n"
                "    double ewma = 0.0;\n"
                "    for (uint i = 0; i < seq_len; ++i) {\n"
                "        double x = data[base + i];\n                ewma = alpha * x + (1.0 - alpha) * ewma;\n"
                "        int pred = 0;\n"
                "        if (ewma > threshold) pred = 1;\n"
                "        else if (ewma < -threshold) pred = -1;\n"
                "        out[base + i] = pred;\n"
                "    }\n"
                "}\n";

            cl_int err = CL_SUCCESS;
            // build program at runtime (we store program in clProgram_ if not built)
            if (!clProgram_) {
                clProgram_ = clCreateProgramWithSource(clContext_, 1, &kernelSrc, nullptr, &err);
                if (err != CL_SUCCESS) {
                    std::cerr << "Predictor: clCreateProgramWithSource failed: " << err << "\n";
                    return false;
                }
                err = clBuildProgram(clProgram_, 1, &clDevice_, nullptr, nullptr, nullptr);
                if (err != CL_SUCCESS) {
                    // get build log
                    size_t logsz = 0;
                    clGetProgramBuildInfo(clProgram_, clDevice_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logsz);
                    std::string log(logsz, '\0');
                    clGetProgramBuildInfo(clProgram_, clDevice_, CL_PROGRAM_BUILD_LOG, logsz, &log[0], nullptr);
                    std::cerr << "Predictor: clBuildProgram failed:\n" << log << "\n";
                    clReleaseProgram(clProgram_);
                    clProgram_ = nullptr;
                    return false;
                }
            }

            cl_mem inBuf = clCreateBuffer(clContext_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * data.size(), (void*)data.data(), &err);
            if (err != CL_SUCCESS) { std::cerr << "Predictor: clCreateBuffer in failed\n"; return false; }
            cl_mem outBuf = clCreateBuffer(clContext_, CL_MEM_WRITE_ONLY, sizeof(int) * out.size(), nullptr, &err);
            if (err != CL_SUCCESS) { std::cerr << "Predictor: clCreateBuffer out failed\n"; clReleaseMemObject(inBuf); return false; }

            cl_kernel kernel = clCreateKernel(clProgram_, "ewma_batch", &err);
            if (err != CL_SUCCESS) { std::cerr << "Predictor: clCreateKernel failed\n"; clReleaseMemObject(inBuf); clReleaseMemObject(outBuf); return false; }

            err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inBuf);
            err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &outBuf);
            double alpha = alpha_;
            double threshold = threshold_;
            err |= clSetKernelArg(kernel, 2, sizeof(double), &alpha);
            err |= clSetKernelArg(kernel, 3, sizeof(double), &threshold);
            unsigned int s_len = static_cast<unsigned int>(seq_len);
            err |= clSetKernelArg(kernel, 4, sizeof(unsigned int), &s_len);
            if (err != CL_SUCCESS) { std::cerr << "Predictor: clSetKernelArg failed\n"; clReleaseKernel(kernel); clReleaseMemObject(inBuf); clReleaseMemObject(outBuf); return false; }

            size_t global = num_seqs;
            err = clEnqueueNDRangeKernel(clQueue_, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) { std::cerr << "Predictor: clEnqueueNDRangeKernel failed: " << err << "\n"; clReleaseKernel(kernel); clReleaseMemObject(inBuf); clReleaseMemObject(outBuf); return false; }

            err = clEnqueueReadBuffer(clQueue_, outBuf, CL_TRUE, 0, sizeof(int) * out.size(), out.data(), 0, nullptr, nullptr);
            if (err != CL_SUCCESS) { std::cerr << "Predictor: clEnqueueReadBuffer failed\n"; clReleaseKernel(kernel); clReleaseMemObject(inBuf); clReleaseMemObject(outBuf); return false; }

            clReleaseKernel(kernel);
            clReleaseMemObject(inBuf);
            clReleaseMemObject(outBuf);
            return true;
        }
#endif
        // if OpenCL not available, fallthrough to CPU
    }

    // CPU fallback: treat each sequence independently, do sequential EWMA per sequence
    for (size_t s = 0; s < num_seqs; ++s) {
        double local_ewma = 0.0; // independent sequences start from 0 by contract
        size_t base = s * seq_len;
        for (size_t i = 0; i < seq_len; ++i) {
            double x = data[base + i];
            local_ewma = alpha_ * x + (1.0 - alpha_) * local_ewma;
            int pred = 0;
            if (local_ewma > threshold_) pred = 1;
            else if (local_ewma < -threshold_) pred = -1;
            out[base + i] = pred;
        }
    }
    return true;
}

#ifdef BUILD_WITH_OPENCL
bool Predictor::init_opencl() {
    if (clInitialized_) return true;
    cl_int err = CL_SUCCESS;
    cl_uint numPlatforms = 0;
    err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) {
        std::cerr << "Predictor: clGetPlatformIDs found no platforms or returned error: " << err << "\n";
        return false;
    }
    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);

    std::cerr << "Predictor: found " << numPlatforms << " OpenCL platform(s)\n";
    // print platform names for diagnostics
    for (cl_uint i = 0; i < numPlatforms; ++i) {
        size_t n = 0;
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, nullptr, &n);
        if (n > 0) {
            std::string name(n, '\0');
            clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, n, &name[0], nullptr);
            // trim trailing nulls
            if (!name.empty() && name.back() == '\0') name.pop_back();
            std::cerr << "  platform[" << i << "] = '" << name << "'\n";
        }
    }

    // pick first platform with a GPU device, otherwise any device
    for (auto p : platforms) {
        cl_uint numDevices = 0;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices) == CL_SUCCESS && numDevices > 0) {
            std::vector<cl_device_id> devices(numDevices);
            clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);
            clPlatform_ = p; clDevice_ = devices[0];
            std::cerr << "Predictor: selected GPU device on platform\n";
            break;
        }
    }
    if (!clDevice_) {
        // try any device
        for (auto p : platforms) {
            cl_uint numDevices = 0;
            if (clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices) == CL_SUCCESS && numDevices > 0) {
                std::vector<cl_device_id> devices(numDevices);
                clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, numDevices, devices.data(), nullptr);
                clPlatform_ = p; clDevice_ = devices[0];
                std::cerr << "Predictor: selected non-GPU device on platform\n";
                break;
            }
        }
    }
    if (!clDevice_) {
        std::cerr << "Predictor: no OpenCL devices found on any platform\n";
        return false;
    }

    // print selected device info
    {
        size_t n = 0;
        clGetDeviceInfo(clDevice_, CL_DEVICE_NAME, 0, nullptr, &n);
        if (n > 0) {
            std::string dname(n, '\0');
            clGetDeviceInfo(clDevice_, CL_DEVICE_NAME, n, &dname[0], nullptr);
            if (!dname.empty() && dname.back() == '\0') dname.pop_back();
            std::cerr << "Predictor: using device '" << dname << "'\n";
        }
        cl_device_type dtype = 0;
        if (clGetDeviceInfo(clDevice_, CL_DEVICE_TYPE, sizeof(dtype), &dtype, nullptr) == CL_SUCCESS) {
            std::cerr << "Predictor: device type=" << dtype << "\n";
        }
    }

    clContext_ = clCreateContext(nullptr, 1, &clDevice_, nullptr, nullptr, &err);
    if (err != CL_SUCCESS || !clContext_) return false;
#ifdef CL_VERSION_2_0
    const cl_queue_properties props[] = { 0 };
    clQueue_ = clCreateCommandQueueWithProperties(clContext_, clDevice_, props, &err);
#else
    clQueue_ = clCreateCommandQueue(clContext_, clDevice_, 0, &err);
#endif
    if (err != CL_SUCCESS || !clQueue_) { clReleaseContext(clContext_); clContext_ = nullptr; return false; }
    clInitialized_ = true;
    return true;
}

void Predictor::teardown_opencl() {
    if (!clInitialized_) return;
    if (clProgram_) { clReleaseProgram(clProgram_); clProgram_ = nullptr; }
    if (clQueue_) { clReleaseCommandQueue(clQueue_); clQueue_ = nullptr; }
    if (clContext_) { clReleaseContext(clContext_); clContext_ = nullptr; }
    clDevice_ = nullptr;
    clPlatform_ = nullptr;
    clInitialized_ = false;
}
#endif
