#ifndef WW_ENGINE_H
#define WW_ENGINE_H

#include <cstdint>
#include <map>

namespace WW {

#define MAX_DIMS 4

#define MAX_THREAD "8";

enum class StateCode {
    OK,
    FAILED
};

enum class Backend {
    CPU,
    CUDA
};

enum class DataType {
    FLOAT32,
    HALF
};

class Tensor {
public:
    int64_t ElementCount() {
        int64_t count = 1;
        for (int i = 0; i < dims; i++) {
            count *= ne[i];
        }
        return count;
    }

    DataType    type { DataType::FLOAT32 };
    Backend     backend { Backend::CPU };
    int         dims { 0 };
    int64_t     ne[MAX_DIMS] { 0 };
    size_t      nb[MAX_DIMS] { 0 };
    void*       data { nullptr };
};

using InputTensorMap = std::map<std::string, Tensor>;

using OutputTensorMap = std::map<std::string, Tensor>;


}

#endif