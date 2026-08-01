#pragma once

#include "llama.cpp/ggml/include/ggml-cpu.h"
#include "llama.cpp/ggml/include/ggml.h"

// Current ggml separates generic conversion traits from CPU vector-dot
// traits.  KTransformers' bundled llamafile code predates that split and
// expects one value object, so retain that narrow interface locally.
struct llamafile_ggml_type_traits_compat {
    ggml_to_float_t to_float;
    ggml_from_float_t from_float;
    ggml_type vec_dot_type;
};

inline llamafile_ggml_type_traits_compat ggml_internal_get_type_traits(ggml_type type) {
    const ggml_type_traits* generic_traits = ggml_get_type_traits(type);
    const ggml_type_traits_cpu* cpu_traits = ggml_get_type_traits_cpu(type);
    return {generic_traits->to_float, cpu_traits->from_float, cpu_traits->vec_dot_type};
}
