import math
import os.path

import torch
import argparse

import triton
import triton.language as tl
from triton.language.extra.cuda.libdevice import tanh, exp


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip_mi200():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == 'hip' and target.arch == 'gfx90a'


def get_cuda_autotune_config():
    configs = []
    for group_size_m in [8, 16]:
        configs.extend([
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': group_size_m},
                          num_stages=3,
                          num_warps=8),
        ])
    return configs


def get_autotune_config():
    return get_cuda_autotune_config()


@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
        a_ptr, b_ptr, c_ptr, bias_ptr, aux_ptr,

        M, N, K,

        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,

        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,
        USE_BIAS: tl.constexpr,
        ACTIVATION: tl.constexpr,
        ACCUMULATE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    # scale_up = SCALE_UP
    # scale_down = (1 / (scale_up * scale_up))

    if ACCUMULATE:
        initial_acc = tl.load(c_ptrs, mask=c_mask, other=0.0)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=(k * BLOCK_SIZE_K + offs_k[None, :]) < K, other=0.0)
        b = tl.load(b_ptrs, mask=(k * BLOCK_SIZE_K + offs_k[:, None]) < K, other=0.0)

        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float16)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if USE_BIAS:
        offs_bias = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        bias_ptrs = bias_ptr + offs_bias
        bias_mask = offs_bias < N
        bias = tl.load(bias_ptrs, mask=bias_mask, other=0.0)
        bias = tl.expand_dims(bias, axis=0)
        accumulator += bias

    if ACTIVATION == "gelu":
        # in fwd pass, aux_ptr is used for writing out intermediates before act. fn is applied
        preact_ptrs = aux_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        tl.store(preact_ptrs, accumulator, mask=c_mask)
        accumulator = gelu_approx(accumulator)

    if ACTIVATION == "dgelu":
        # in bwd pass, aux_ptr is used for supplying dgelu inputs
        inp_ptrs = aux_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        inp = tl.load(inp_ptrs, c_mask)
        accumulator *= dgelu_approx(inp)

    if ACCUMULATE:
        accumulator += initial_acc

    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def tanh_fp16_approx(x):
    # NOTE: This is a special tanh specifically optimize to aid gelu approximation
    # It approaches tails quicker than normal tanh.
    # Note: Variables are named after their initial value as per well known approximations, but gradient
    # optimized further, hence different values
    twoseven = 24.748026766450316
    nine = 8.519021440881644
    one = 1
    minus_one = -1
    x_sq = x * x
    approx = x * ((twoseven.to(tl.float16) + x_sq) / (twoseven.to(tl.float16) + (nine.to(tl.float16) * x_sq)))
    return tl.clamp(approx, minus_one.to(tl.float16), one.to(tl.float16))


@triton.jit
def gelu_approx(x):
    half = 0.49998889193404505
    sqrt_hlf_pi = 0.8036158252814497
    magic = 0.018446066957599785
    return half.to(tl.float16) * x * (
            1 + tanh_fp16_approx(sqrt_hlf_pi.to(tl.float16) * (x + magic.to(tl.float16) * x * x * x)))


@triton.jit
def gelu_exact(x):
    x = x.to(tl.float32)
    cube = 0.044715 * x * x * x
    result = 0.5 * x * (1.0 + tanh(0.7978845608 * (x + cube)))
    return result.to(tl.float16)


@triton.jit
def tanh_fp16(x):
    return 1 - (2 * (1 / (1 + tl.exp(x * 2))))


@triton.jit
def dgelu_approx(x):
    half = 0.5
    inv_sqrt_two = 0.82177734375
    inv_sqrt_two_pi = 0.393310546875

    return half.to(tl.float16) * (1 + erf_approx(x * inv_sqrt_two.to(tl.float16))) + x * inv_sqrt_two_pi.to(
        tl.float16) * tl.exp((-x * x) / 2).to(tl.float16)


@triton.jit
def erf_approx(x):
    twoseven = 28.921875
    nine = 9.4140625
    appr = x * ((twoseven.to(tl.float16) + (x * x)) / (twoseven.to(tl.float16) + (nine.to(tl.float16) * (x * x))))
    return tl.clamp(appr, -1, 1)


@triton.jit
def approx_gauss(x):
    a1 = 1.78125
    a2 = -0.07061767578125
    one = 1.0
    b1 = -0.08197021484375
    b2 = 0.140625
    x_sq = x * x
    x_pow4 = x_sq * x_sq
    num = a1.to(tl.float16) + a2.to(tl.float16) * x_sq
    den = one.to(tl.float16) + b1.to(tl.float16) * x_sq + b2.to(tl.float16) * x_pow4
    return num / den


@triton.jit
def erf_fp16(x):
    return tl.erf(x.to(tl.float32)).to(tl.float16)


def matmul(a, b, bias, accumulate=False, activation="None"):
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.zeros((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    pre_act = torch.full((M, N),
                         fill_value=1.0,
                         device=a.device,
                         dtype=torch.float16) if activation != "None" else None
    matmul_kernel[grid](
        a, b, c, bias, pre_act,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        USE_BIAS=bias is not None,
        ACTIVATION=activation,
        ACCUMULATE=accumulate,
    )
    return c


def dgelu(x):
    upstream_grad = torch.ones_like(x)
    return torch.ops.aten.gelu_backward(upstream_grad, x)


# %%
parser = argparse.ArgumentParser(description='Matmul Kernel Generator')
parser.add_argument("transpose_a")
parser.add_argument("transpose_b")
parser.add_argument("accumulate")
parser.add_argument("activation")
parser.add_argument("use_bias")
parser.add_argument("act_derivative")

args = vars(parser.parse_args())

transpose_a = args.get("transpose_a") == "True"
transpose_b = args.get("transpose_b") == "True"
accumulate = args.get("accumulate") == "True"
activation = args.get("activation")
use_bias = args.get("use_bias") == "True"
act_derivative = args.get("act_derivative") == "True"

torch.manual_seed(0)
a = torch.randn((4096, 4096), dtype=torch.float16, device='cuda') * 0.2
b = torch.randn((4096, 4096), dtype=torch.float16, device='cuda') * 0.2
bias = torch.randn((1, 4096), dtype=torch.float16, device='cuda') * 0.2 if use_bias else None

if transpose_a:
    a = a.transpose(-1, -2)

if transpose_b:
    b = b.transpose(-1, -2)

torch_output = torch.matmul(a, b)
if use_bias:
    torch_output += bias

if activation == "gelu":
    if act_derivative:
        torch_output *= dgelu(torch.ones_like(torch_output))
    else:
        torch_output = torch.nn.functional.gelu(torch_output)

triton_output = matmul(a, b, bias, accumulate=accumulate, activation=("d" if act_derivative else "") + activation)
if accumulate:
    # recompute because accumulation and autotuning results in garbage values because of state-fullness
    triton_output = matmul(a, b, bias, accumulate=accumulate, activation=("d" if act_derivative else "") + activation)
print("torch_output", torch_output)
print("triton_output", triton_output)

matmul_kernel_fn = matmul_kernel
cache = matmul_kernel_fn.fn.cache[0]
chosen_kernel = None
for config, compiled_kernel in cache.items():
    constants = compiled_kernel.src.constants
    is_match = True
    for k, v in matmul_kernel_fn.best_config.kwargs.items():
        if constants[k] != v:
            is_match = False
            break

    if matmul_kernel_fn.best_config.num_ctas != compiled_kernel.metadata.num_ctas:
        is_match = False
        continue

    if matmul_kernel_fn.best_config.num_stages != compiled_kernel.metadata.num_stages:
        is_match = False
        continue

    if matmul_kernel_fn.best_config.num_warps != compiled_kernel.metadata.num_warps:
        is_match = False
        continue

    if is_match:
        chosen_kernel = compiled_kernel
        break

assert chosen_kernel is not None

if os.path.exists("kernels_out") is False:
    os.mkdir("kernels_out")

kernel_name = f"matmul_kernel{'_bias' if use_bias else ''}{'_accumulate' if accumulate else ''}{'_transpose_a' if transpose_a else ''}{'_transpose_b' if transpose_b else ''}{'_' + ('d' if act_derivative else '') + 'gelu' if activation == 'gelu' else ''}"
KERNEL_NUMBER = (1 * int(accumulate)) + (2 * int(transpose_a)) + (4 * int(transpose_b)) + (
        8 * int(use_bias)) + (16 * int(activation == 'gelu')) + (16 * int(act_derivative))
print("KERNEL_NUMBER:", KERNEL_NUMBER)
print(f"accumulate: {accumulate}, transpose_a: {transpose_a}, transpose_b: {transpose_b}, use_bias: {use_bias}, "
      f"activation: {activation}, act_derivative: {act_derivative}")
with open(f"kernels_out/triton_{kernel_name}.h", "w") as fp:
    fp.write("/*\n"
             " * WARNING: This is an autogenerated file. DO NOT EDIT.\n"
             " * This file was generated by generate_kernels.sh which intern launches compile_matmul_kernel.py"
             " */\n")
    fp.write("#pragma once\n\n#include <string>\n\n")

    fp.write(f"const std::string TRITON_MATMUL_KERNEL_{KERNEL_NUMBER}_SOURCE_PTX = R\"(\n")
    fp.write(chosen_kernel.asm["ptx"].replace(f"matmul_kernel",
                                              kernel_name))  # rename the kernel in PTX
    fp.write(")\";\n\n")

    fp.write(f"#define TRITON_MATMUL_KERNEL_{KERNEL_NUMBER}_SHARED_MEMORY_SIZE {chosen_kernel.metadata.shared}\n")
    fp.write(
        f"#define TRITON_MATMUL_KERNEL_{KERNEL_NUMBER}_BLOCK_SIZE_M {matmul_kernel_fn.best_config.kwargs['BLOCK_SIZE_M']}\n")
    fp.write(
        f"#define TRITON_MATMUL_KERNEL_{KERNEL_NUMBER}_BLOCK_SIZE_N {matmul_kernel_fn.best_config.kwargs['BLOCK_SIZE_N']}\n")
    fp.write(f"#define TRITON_MATMUL_KERNEL_{KERNEL_NUMBER}_NUM_WARPS {matmul_kernel_fn.best_config.num_warps}\n")
    fp.write(f"#define TRITON_MATMUL_KERNEL_{KERNEL_NUMBER}_FUNCTION_NAME \"{kernel_name}\"\n")

rtol = 1e-2
if torch.allclose(triton_output, torch_output, atol=1e-1, rtol=rtol):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")
exit(0)

# %%
# Benchmark
# ---------
#
# Square Matrix Performance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We can now compare the performance of our kernel against that of cuBLAS or rocBLAS. Here we focus on square matrices,
# but feel free to arrange this script as you wish to benchmark any other matrix shape.

ref_lib = 'cuBLAS' if is_cuda() else 'rocBLAS'

configs = [triton.testing.Benchmark(
    x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
    x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
    line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
    # Possible values for `line_arg`
    # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
    line_vals=[ref_lib.lower(), "triton"],  # Label name for the lines
    line_names=[ref_lib, "Triton"],  # Line styles
    styles=[("green", "-"), ("blue", "-")],
    ylabel="TFLOPS",  # Label name for the y-axis
    plot_name="matmul-performance-bf16",
    args={}
)]


@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    bias = torch.randn((N,), device='cuda', dtype=torch.float16)

    if transpose_b:
        a = a.transpose(-1, -2)

    if transpose_b:
        b = b.transpose(-1, -2)

    quantiles = [0.5, 0.2, 0.8]

    activation_fn_flops = 10 * M * N if activation == "gelu" else 0

    def act(x):
        if activation == "gelu":
            if act_derivative:
                return dgelu(x)
            else:
                return torch.nn.functional.gelu(x)
        else:
            return x

    if provider == ref_lib.lower():
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: act(torch.matmul(a, b) + bias), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b, bias,
                                                                    accumulate=accumulate,
                                                                    activation=(
                                                                                   "d" if act_derivative else "") + activation),
                                                     quantiles=quantiles)
    perf = lambda ms: (2 * M * N * K + activation_fn_flops) * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=True, print_data=True)
