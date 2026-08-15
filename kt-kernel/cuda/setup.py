from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="KTransformersOps",
    ext_modules=[
        CUDAExtension(
            "KTransformersOps",
            [
                "custom_gguf/dequant.cu",
                "binding.cpp",
                "gptq_marlin/gptq_marlin.cu",
                "moe/moe_topk_softmax_kernels.cu",
                # 'gptq_marlin_repack.cu',
            ],
            extra_compile_args={
                # binding.cpp gates the CUDA-only exports on this definition.
                "cxx": ["-O3", "-DKTRANSFORMERS_USE_CUDA=1"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-Xcompiler",
                    "-fPIC",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
