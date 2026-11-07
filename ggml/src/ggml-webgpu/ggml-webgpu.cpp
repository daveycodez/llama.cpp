/*
    WebGPU backend implementation.
    Note: Use ClangFormat to format this file.
*/

#include "ggml-webgpu.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-webgpu-shader-lib.hpp"
#include "ggml.h"

#ifdef __EMSCRIPTEN__
#    include <emscripten/emscripten.h>
#endif

#include <webgpu/webgpu_cpp.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#ifdef GGML_WEBGPU_GPU_PROFILE
#    include <iomanip>
#endif
#if defined(GGML_WEBGPU_DEBUG) || defined(GGML_WEBGPU_CPU_PROFILE) || defined(GGML_WEBGPU_GPU_PROFILE)
#    include <iostream>
#endif
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#define ROUNDUP_POW2(x, pow2) (((x) + ((pow2) - 1)) & ~((pow2) - 1))
#define CEIL_DIV(M, N)        (((M) + (N) - 1) / (N))

// Return a rectangular grid of workgroups with minimal over-provisioned workgroups.
// Assumes that the total number of workgroups does not exceed max_per_dim^2.
static inline void compute_2d_workgroups(uint32_t total_wg, uint32_t max_per_dim, uint32_t & wg_x, uint32_t & wg_y) {
    wg_y = std::max(1u, CEIL_DIV(total_wg, max_per_dim));
    wg_x = CEIL_DIV(total_wg, wg_y);
}

static inline uint32_t ggml_webgpu_u32_from_f32(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

#ifdef GGML_WEBGPU_DEBUG
#    define WEBGPU_LOG_DEBUG(msg)  std::cout << msg << std::endl
#    define WEBGPU_DEBUG_BUF_ELEMS 512
#else
#    define WEBGPU_LOG_DEBUG(msg) ((void) 0)
#endif  // GGML_WEBGPU_DEBUG

#ifdef GGML_WEBGPU_CPU_PROFILE
// total timing (aggregated)
#    define WEBGPU_CPU_PROFILE_TOTAL_START(id) auto cpu_total_start_##id = std::chrono::high_resolution_clock::now();

#    define WEBGPU_CPU_PROFILE_TOTAL_END(id, ctx)                                                         \
        auto   cpu_total_end_##id = std::chrono::high_resolution_clock::now();                            \
        double cpu_total_time_##id =                                                                      \
            std::chrono::duration<double, std::milli>(cpu_total_end_##id - cpu_total_start_##id).count(); \
        (ctx)->cpu_time_ms[#id] += cpu_total_time_##id;
// fine-grained timing (not included in totals)
#    define WEBGPU_CPU_PROFILE_DETAIL_START(id) auto cpu_detail_start_##id = std::chrono::high_resolution_clock::now();

#    define WEBGPU_CPU_PROFILE_DETAIL_END(id, ctx)                                                          \
        auto   cpu_detail_end_##id = std::chrono::high_resolution_clock::now();                             \
        double cpu_detail_time_##id =                                                                       \
            std::chrono::duration<double, std::milli>(cpu_detail_end_##id - cpu_detail_start_##id).count(); \
        (ctx)->cpu_detail_ms[#id] += cpu_detail_time_##id;
#else
#    define WEBGPU_CPU_PROFILE_TOTAL_START(id)
#    define WEBGPU_CPU_PROFILE_TOTAL_END(id, ctx)
#    define WEBGPU_CPU_PROFILE_DETAIL_START(id)
#    define WEBGPU_CPU_PROFILE_DETAIL_END(id, ctx)
#endif  // GGML_WEBGPU_CPU_PROFILE

#ifdef GGML_WEBGPU_GPU_PROFILE
#    define WEBGPU_MAX_PROFILE_QUERY_COUNT        4096u
#    define WEBGPU_TIMESTAMP_QUERY_BUF_SIZE_BYTES (WEBGPU_MAX_PROFILE_QUERY_COUNT * sizeof(uint64_t))
#endif

/* Constants */

#define WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_SIZE 64u
#define WEBGPU_NUM_PARAM_SLOT_SAFETY_MARGIN      10u
#define WEBGPU_RUNTIME_WAIT_TIMEOUT_MS           30000u
#define WEBGPU_RUNTIME_WAIT_TIMEOUT_NS           (WEBGPU_RUNTIME_WAIT_TIMEOUT_MS * 1e6)
#define WEBGPU_PARAMS_BUF_SIZE_BYTES             128  // enough for 32 parameters
#define WEBGPU_SET_ROWS_ERROR_BUF_SIZE_BYTES     4
#define WEBGPU_STORAGE_BUF_BINDING_MULT          4    // a storage buffer binding size must be a multiple of 4
#define WEBGPU_PVR_COMMAND_SUBMIT_BATCH_SIZE     16u
#define WEBGPU_PVR_FIRST_SUBMIT_BATCH_SIZE       4u
#define WEBGPU_PVR_RMS_NORM_MUL_WG_SIZE          128u
#define WEBGPU_PVR_READBACK_MAX_BYTES            (32u << 20)  // state buffers read back whole on PowerVR (the recurrent state with its snapshots)

/* End Constants */

// This is a "fake" base pointer, since WebGPU buffers do not have pointers to
// their locations.
static void * const webgpu_ptr_base = (void *) (uintptr_t) 0x1000;  // NOLINT

static size_t ggml_webgpu_tensor_offset(const ggml_tensor * tensor) {
    const ggml_tensor * base_tensor = tensor->view_src ? tensor->view_src : tensor;
    return (size_t) ((uintptr_t) base_tensor->data - (uintptr_t) webgpu_ptr_base) + tensor->view_offs;
}

/* Struct definitions */

// Forward reference
static void ggml_webgpu_create_buffer(wgpu::Device &    device,
                                      wgpu::Buffer &    buffer,
                                      size_t            size,
                                      wgpu::BufferUsage usage,
                                      const char *      label);

// Slot-based parameter arena for compute graph encoding. Each encoded kernel
// gets a unique uniform-buffer slice within the current batch, and the slot
// cursor is reset immediately after that batch is submitted.
struct webgpu_param_arena {
    wgpu::Buffer buffer;
    size_t       slot_stride = 0;
    size_t       slot_size   = 0;
    uint32_t     slot_count  = 0;
    uint32_t     next_slot   = 0;

    // PowerVR: the batch's parameters are written with one WriteBuffer at submit
    std::vector<uint8_t> shadow;
    size_t               shadow_end = 0;

    void init(wgpu::Device device, size_t slot_size, uint32_t slot_count, size_t alignment) {
        this->slot_stride = ROUNDUP_POW2(slot_size, alignment);
        this->slot_size   = slot_size;
        this->slot_count  = slot_count;
        this->next_slot   = 0;

        ggml_webgpu_create_buffer(device, buffer, this->slot_stride * slot_count,
                                  wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Uniform, "ggml_webgpu_param_arena");
    }

    size_t alloc_slot(size_t size) {
        GGML_ASSERT(size <= slot_size);
        if (next_slot >= slot_count) {
            GGML_ABORT("ggml_webgpu: parameter arena exhausted while encoding a batch");
        }

        return slot_stride * next_slot++;
    }

    void reset() { next_slot = 0; }

    void write_deferred(size_t offset, const void * data, size_t size) {
        if (shadow.size() < slot_stride * slot_count) {
            shadow.resize(slot_stride * slot_count);
        }
        memcpy(shadow.data() + offset, data, size);
        shadow_end = std::max(shadow_end, offset + size);
    }

    void flush(wgpu::Queue & queue) {
        if (shadow_end > 0) {
            queue.WriteBuffer(buffer, 0, shadow.data(), ROUNDUP_POW2(shadow_end, 4));
            shadow_end = 0;
        }
    }

    void cleanup() {
        if (buffer) {
            buffer.Destroy();
            buffer = nullptr;
        }
    }

    ~webgpu_param_arena() { this->cleanup(); }
};

struct webgpu_encoded_op {
    uint32_t num_kernels = 0;
#ifdef GGML_WEBGPU_GPU_PROFILE
    std::vector<std::string> pipeline_names;
#endif
};

struct webgpu_dispatch_desc {
    webgpu_pipeline                   pipeline;
    std::vector<uint32_t>             params;
    std::vector<wgpu::BindGroupEntry> bind_group_entries;
    std::pair<uint32_t, uint32_t>     workgroups = { 1, 1 };
};

struct webgpu_capabilities {
    wgpu::Limits limits;
    bool         supports_subgroups       = false;
    bool         supports_subgroup_matrix = false;
    bool         supports_dot_product     = false;

    uint32_t sg_mat_m = 0;
    uint32_t sg_mat_n = 0;
    uint32_t sg_mat_k = 0;

    uint32_t subgroup_size     = 0;
    uint32_t min_subgroup_size = 0;
    uint32_t max_subgroup_size = 0;
};

// Stores global webgpu members
struct webgpu_global_context_struct {
    wgpu::Instance instance;
    wgpu::Adapter  adapter;
    wgpu::Device   device;
    wgpu::Queue    queue;
    uint32_t       command_submit_batch_size = WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_SIZE;
    uint32_t       max_inflight_batches      = UINT32_MAX;

    webgpu_capabilities  capabilities;
    // Shared buffer to move data from device to host
    wgpu::Buffer         get_tensor_staging_buf;
    // Global mutex for get_tensor
    std::recursive_mutex mutex;

    wgpu::Buffer    memset_params_buf;
    webgpu_pipeline memset_pipeline;

    // Packed source data and write descriptors for set_tensor_n
    wgpu::Buffer    set_tensor_n_src_buf;
    wgpu::Buffer    set_tensor_n_descs_buf;
    webgpu_pipeline set_tensor_n_pipeline;

    std::string vendor;

    // No work was submitted since the queue was last seen idle. Only used on
    // PowerVR (see ggml_backend_webgpu_synchronize).
    bool queue_idle = false;

    // PowerVR: bumped by everything that may write a buffer (submissions,
    // WriteBuffer), and a copy of a small buffer read back at that count
    // (see ggml_backend_webgpu_buffer_get_tensor)
    uint64_t             pvr_write_epoch    = 0;
    WGPUBuffer           pvr_readback_src   = nullptr;
    uint64_t             pvr_readback_epoch = 0;
    std::vector<uint8_t> pvr_readback_data;

    // TODO: We should rework the CPU profiling time handling to make it more useful. ref: https://github.com/ggml-org/llama.cpp/pull/22050
#ifdef GGML_WEBGPU_CPU_PROFILE
    // Profiling: labeled CPU time in ms (total)
    std::unordered_map<std::string, double> cpu_time_ms;
    // Profiling: detailed CPU time in ms
    std::unordered_map<std::string, double> cpu_detail_ms;
#endif

#ifdef GGML_WEBGPU_DEBUG
    wgpu::Buffer debug_host_buf;
    wgpu::Buffer debug_dev_buf;
#endif

    ~webgpu_global_context_struct() {
        if (this->get_tensor_staging_buf) {
            this->get_tensor_staging_buf.Destroy();
            this->get_tensor_staging_buf = nullptr;
        }
        if (this->memset_params_buf) {
            this->memset_params_buf.Destroy();
            this->memset_params_buf = nullptr;
        }
        if (this->set_tensor_n_src_buf) {
            this->set_tensor_n_src_buf.Destroy();
            this->set_tensor_n_src_buf = nullptr;
        }
        if (this->set_tensor_n_descs_buf) {
            this->set_tensor_n_descs_buf.Destroy();
            this->set_tensor_n_descs_buf = nullptr;
        }
#ifdef GGML_WEBGPU_DEBUG
        if (this->debug_host_buf) {
            this->debug_host_buf.Destroy();
            this->debug_host_buf = nullptr;
        }
        if (this->debug_dev_buf) {
            this->debug_dev_buf.Destroy();
            this->debug_dev_buf = nullptr;
        }
#endif
    }
};

typedef std::shared_ptr<webgpu_global_context_struct> webgpu_global_context;

// Imagination PowerVR GPUs (vendor "img-tec"), on which several paths below
// differ: see ggml_webgpu_mul_mat_pvr, ggml_webgpu_flash_attn_pvr and
// ggml_backend_webgpu_synchronize.
static bool ggml_webgpu_is_pvr(const webgpu_global_context & global_ctx) {
    return global_ctx->vendor == "img-tec";
}

// On PowerVR, Q4_K and Q6_K matrices are repacked as they are uploaded, in
// place (the same bytes, moved), into the layout that mul_mat_vec_pvr.wgsl
// reads fastest ("layout A"): block b of row r becomes block b * rows + r, so
// neighbouring rows' blocks are neighbours, and each block's parts sit in
// planes of their own, every load aligned. Q4_K: headers (d, dmin, scales, 16
// bytes), then quants (128 bytes). Q6_K: ql (128 bytes, piece p's 4 words
// at 64 n + 8 (p % 4) and + 32, n = p / 4, together), qh (64 bytes, piece
// p's 2 words at 32 n + 8 (p % 4) together), scales (16), d (2). The values
// read, and so every product, are those of the original layout. Repacked
// tensors are marked by tensor->extra; only the PowerVR matrix products read
// them, and reading one back gives the original layout.
static char ggml_webgpu_pvr_repacked_marker;

static bool ggml_webgpu_pvr_is_repacked(const ggml_tensor * t) {
    return t->extra == &ggml_webgpu_pvr_repacked_marker;
}

// Whether a tensor is repacked when it is uploaded whole.
static bool ggml_webgpu_pvr_repackable(const webgpu_global_context & global_ctx, const ggml_tensor * t) {
    return ggml_webgpu_is_pvr(global_ctx) && (t->type == GGML_TYPE_Q4_K || t->type == GGML_TYPE_Q6_K) &&
           t->view_src == nullptr && t->ne[0] % 256 == 0 && t->ne[1] % 4 == 0 && t->ne[2] == 1 && t->ne[3] == 1 &&
           ggml_is_contiguous(t) && t->buffer != nullptr &&
           ggml_webgpu_tensor_offset(t) % global_ctx->capabilities.limits.minStorageBufferOffsetAlignment == 0;
}

// Whether an op reads a tensor that is, or may become, repacked, or a view of one.
static bool ggml_webgpu_pvr_reads_repackable(const webgpu_global_context & global_ctx, const ggml_tensor * t) {
    if (t == nullptr || !ggml_webgpu_is_pvr(global_ctx) || (t->type != GGML_TYPE_Q4_K && t->type != GGML_TYPE_Q6_K)) {
        return false;
    }
    const ggml_tensor * base = t->view_src ? t->view_src : t;
    return base->ne[0] % 256 == 0 && base->ne[1] % 4 == 0 && base->ne[2] == 1 && base->ne[3] == 1 &&
           ggml_is_contiguous(base);
}

// Moves the bytes of block column b (rows m) of a whole tensor between the
// original layout (orig) and layout A (a): to_a copies orig into a, else a
// into orig. nblk is the tensor's block count; a holds whole planes.
static void ggml_webgpu_pvr_repack_column(ggml_type type, int64_t m, int64_t nb, int64_t b, bool to_a, uint8_t * orig,
                                          uint8_t * a) {
    const int64_t nblk = m * nb;
    for (int64_t r = 0; r < m; r++) {
        const int64_t blk = b * m + r;
        if (type == GGML_TYPE_Q4_K) {
            uint8_t * o = orig + (r * nb + b) * 144;
            // header, then the 128 bytes of quants
            uint8_t * h = a + blk * 16;
            uint8_t * q = a + nblk * 16 + blk * 128;
            if (to_a) {
                memcpy(h, o, 16);
                memcpy(q, o + 16, 128);
            } else {
                memcpy(o, h, 16);
                memcpy(o + 16, q, 128);
            }
        } else {
            uint8_t * o = orig + (r * nb + b) * 210;
            for (int p = 0; p < 8; p++) {
                const int n  = p >> 2;
                const int l0 = 8 * (p & 3);
                uint8_t * ql = a + blk * 128 + 16 * p;
                uint8_t * qh = a + nblk * 128 + blk * 64 + 8 * p;
                uint8_t * oa = o + 64 * n + l0;
                uint8_t * oh = o + 128 + 32 * n + l0;
                if (to_a) {
                    memcpy(ql, oa, 8);
                    memcpy(ql + 8, oa + 32, 8);
                    memcpy(qh, oh, 8);
                } else {
                    memcpy(oa, ql, 8);
                    memcpy(oa + 32, ql + 8, 8);
                    memcpy(oh, qh, 8);
                }
            }
            uint8_t * sc = a + nblk * 192 + blk * 16;
            uint8_t * d  = a + nblk * 208 + blk * 2;
            if (to_a) {
                memcpy(sc, o + 192, 16);
                memcpy(d, o + 208, 2);
            } else {
                memcpy(o + 192, sc, 16);
                memcpy(o + 208, d, 2);
            }
        }
    }
}

// PowerVR: where a fused short convolution's nodes sit in the graph (see
// ggml_webgpu_pvr_shortconv_nodes): the gather, B * X, the CONCAT, SSM_CONV,
// the MUL by C (where the fused op runs) and the snapshots' CPYs, slot 0 first
struct ggml_webgpu_pvr_shortconv_at {
    int              gi = -1, bi = -1, ci = -1, si = -1, yi = -1;
    std::vector<int> snaps;
};

// All the base objects needed to run operations on a WebGPU device
struct webgpu_context_struct {
    // Points to global instances owned by ggml_backend_webgpu_reg_context
    webgpu_global_context global_ctx;

    std::unique_ptr<ggml_webgpu_shader_lib> shader_lib;

    webgpu_param_arena       param_arena;
    wgpu::Buffer             set_rows_dev_error_buf;
    wgpu::Buffer             set_rows_host_error_buf;
    wgpu::CommandEncoder     active_command_encoder;
    wgpu::ComputePassEncoder active_compute_pass;
    bool                     batch_compute_passes = true;

    // PowerVR matrix products: a buffer of its own for src1, so that it can be
    // bound read-only (see ggml_webgpu_mul_mat_pvr)
    // (two, so that an op can write the next product's src1 while it reads its own)
    wgpu::Buffer pvr_staging_buf[2];
    size_t       pvr_staging_size[2] = { 0, 0 };
    int          pvr_staged_idx      = 0;  // the one that holds pvr_staged_src1
    // the src1 the staging buffer holds, while no other op ran since: products
    // with the same src1 back to back (FFN gate and up) skip the copy
    const ggml_tensor * pvr_staged_src1 = nullptr;
    bool                pvr_staged_f16  = false;  // ... packed as f16 (for mul_mat_pvr_gemm.wgsl)
    // the op just encoded wrote pvr_staged_src1 itself (its output, also into the staging buffer)
    bool                pvr_staged_by_op = false;
    // PowerVR prompt batches: the weights dequantized to f16, in a texture
    // (see mul_mat_pvr_dequant.wgsl)
    wgpu::Texture     pvr_dequant_tex;
    wgpu::TextureView pvr_dequant_view;
    uint32_t          pvr_dequant_rows = 0;
    // PowerVR logits top-k: its scratch space (see ggml_webgpu_topk_pvr), a buffer
    // of its own so that the logits and the scratch are each bound read-only
    // where they are only read
    wgpu::Buffer pvr_topk_buf;
    size_t       pvr_topk_size = 0;
    // PowerVR: the short convolutions fused in the last graph (see
    // ggml_webgpu_pvr_shortconv_nodes), found again when a graph differs from
    // it: its nodes, their outputs, shapes and sources (sig), the nodes the
    // fused ops replace (skip), and the fused ops by the index where each runs
    struct {
        const ggml_cgraph *                                   graph = nullptr;
        std::vector<uintptr_t>                                sig;
        std::vector<uint8_t>                                  skip;
        std::unordered_map<int, ggml_webgpu_pvr_shortconv_at> fused;
    } pvr_sc;
    // PowerVR: a product of a few columns was split per column (logged once)
    bool pvr_split_logged = false;

#ifdef GGML_WEBGPU_GPU_PROFILE
    // Profiling: per-shader GPU time in ms
    std::unordered_map<std::string, double> shader_gpu_time_ms;
    wgpu::Buffer                            profile_timestamp_dev_buf;
    wgpu::Buffer                            profile_timestamp_host_buf;
    wgpu::QuerySet                          profile_timestamp_query_set;
    uint32_t                                profile_timestamp_query_count = 0;
#endif

    ~webgpu_context_struct() {
#ifdef GGML_WEBGPU_GPU_PROFILE
        if (this->profile_timestamp_host_buf) {
            this->profile_timestamp_host_buf.Destroy();
            this->profile_timestamp_host_buf = nullptr;
        }
        if (this->profile_timestamp_dev_buf) {
            this->profile_timestamp_dev_buf.Destroy();
            this->profile_timestamp_dev_buf = nullptr;
        }
        if (this->profile_timestamp_query_set) {
            this->profile_timestamp_query_set.Destroy();
            this->profile_timestamp_query_set = nullptr;
        }
#endif
        if (this->set_rows_host_error_buf) {
            this->set_rows_host_error_buf.Destroy();
            this->set_rows_host_error_buf = nullptr;
        }
        if (this->pvr_topk_buf) {
            this->pvr_topk_buf.Destroy();
            this->pvr_topk_buf = nullptr;
        }
        for (wgpu::Buffer & buf : this->pvr_staging_buf) {
            if (buf) {
                buf.Destroy();
                buf = nullptr;
            }
        }
        if (this->pvr_dequant_tex) {
            this->pvr_dequant_view = nullptr;
            this->pvr_dequant_tex.Destroy();
            this->pvr_dequant_tex = nullptr;
        }
        if (this->set_rows_dev_error_buf) {
            this->set_rows_dev_error_buf.Destroy();
            this->set_rows_dev_error_buf = nullptr;
        }
    }
};

typedef std::shared_ptr<webgpu_context_struct> webgpu_context;

// Metadata required for the ggml backend registration/discovery interface
struct ggml_backend_webgpu_reg_context {
    // Since the Instance is a global entrypoint into the WebGPU API, it lives here
    webgpu_global_context webgpu_global_ctx;
    size_t                device_count;
    const char *          name;
};

// Per-device struct for the global logical device interface
struct ggml_backend_webgpu_device_context {
    webgpu_global_context webgpu_global_ctx;
    std::string           device_name;
    std::string           device_desc;
};

// Per-thread data required to actually run WebGPU operations in a backend instance
struct ggml_backend_webgpu_context {
    webgpu_context webgpu_ctx;
    std::string    name;
};

// Per-thread data related to buffers
struct ggml_backend_webgpu_buffer_context {
    wgpu::Buffer          buffer;
    std::string           label;
    webgpu_global_context global_ctx;

    ggml_backend_webgpu_buffer_context(wgpu::Buffer buf, std::string lbl, webgpu_global_context global_ctx_) :
        buffer(std::move(buf)),
        label(std::move(lbl)),
        global_ctx(std::move(global_ctx_)) {}
};

/* WebGPU object initializations */

static webgpu_pipeline ggml_webgpu_create_pipeline(wgpu::Device &                           device,
                                                   const char *                             shader_code,
                                                   const char *                             label,
                                                   const std::vector<wgpu::ConstantEntry> & constants = {}) {
    wgpu::ShaderSourceWGSL shader_source;
    shader_source.code = shader_code;

    wgpu::ShaderModuleDescriptor shader_desc;
    shader_desc.nextInChain = &shader_source;

    wgpu::ShaderModule shader_module = device.CreateShaderModule(&shader_desc);

    wgpu::ComputePipelineDescriptor pipeline_desc;
    pipeline_desc.label              = label;
    pipeline_desc.compute.module     = shader_module;
    pipeline_desc.compute.entryPoint = "main";   // Entry point in the WGSL code
    pipeline_desc.layout             = nullptr;  // nullptr means auto layout
    if (constants.size() > 0) {
        pipeline_desc.compute.constants     = constants.data();
        pipeline_desc.compute.constantCount = constants.size();
    }
    return { device.CreateComputePipeline(&pipeline_desc), label };
}

static void ggml_webgpu_create_buffer(wgpu::Device &    device,
                                      wgpu::Buffer &    buffer,
                                      size_t            size,
                                      wgpu::BufferUsage usage,
                                      const char *      label) {
    wgpu::BufferDescriptor buffer_desc;
    buffer_desc.size             = size;
    buffer_desc.usage            = usage;
    buffer_desc.label            = label;
    buffer_desc.mappedAtCreation = false;

    // TODO: error handling
    buffer = device.CreateBuffer(&buffer_desc);
}

static wgpu::Buffer ggml_webgpu_tensor_buf(const ggml_tensor * tensor) {
    ggml_backend_webgpu_buffer_context * ctx = (ggml_backend_webgpu_buffer_context *) tensor->buffer->context;
    return ctx->buffer;
}

// Binding offset for a tensor: the largest aligned offset at or before the tensor whose
// distance to the tensor is a whole number of type blocks, so shaders can index the
// misalignment in elements even for block quantized types.
static size_t ggml_webgpu_tensor_align_offset(const ggml_tensor * t, size_t alignment) {
    const size_t offset    = ggml_webgpu_tensor_offset(t);
    const size_t type_size = ggml_type_size(t->type);
    size_t       aligned   = offset & ~(alignment - 1);
    while ((offset - aligned) % type_size != 0) {
        GGML_ASSERT(aligned >= alignment);
        aligned -= alignment;
    }
    return aligned;
}

static size_t ggml_webgpu_tensor_misalignment(const ggml_tensor * t, size_t alignment) {
    return ggml_webgpu_tensor_offset(t) - ggml_webgpu_tensor_align_offset(t, alignment);
}

static size_t ggml_webgpu_tensor_misalignment(webgpu_context & ctx, const ggml_tensor * t) {
    return ggml_webgpu_tensor_misalignment(t, ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment);
}

static size_t ggml_webgpu_tensor_align_offset(webgpu_context & ctx, const ggml_tensor * t) {
    return ggml_webgpu_tensor_align_offset(t, ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment);
}

static size_t ggml_webgpu_tensor_binding_size(const ggml_tensor * t, size_t alignment) {
    return ROUNDUP_POW2(ggml_nbytes(t) + ggml_webgpu_tensor_misalignment(t, alignment),
                        WEBGPU_STORAGE_BUF_BINDING_MULT);
}

static size_t ggml_webgpu_tensor_binding_size(webgpu_context & ctx, const ggml_tensor * t) {
    return ggml_webgpu_tensor_binding_size(t, ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment);
}

static bool ggml_webgpu_tensor_binding_overlap(const webgpu_global_context & global_ctx,
                                               const ggml_tensor *           a,
                                               const ggml_tensor *           b) {
    if (a->buffer != b->buffer) {
        return false;
    }

    const size_t alignment = global_ctx->capabilities.limits.minStorageBufferOffsetAlignment;
    const size_t a_offset  = ggml_webgpu_tensor_align_offset(a, alignment);
    const size_t b_offset  = ggml_webgpu_tensor_align_offset(b, alignment);
    return a_offset < b_offset + ggml_webgpu_tensor_binding_size(b, alignment) &&
           b_offset < a_offset + ggml_webgpu_tensor_binding_size(a, alignment);
}

static bool ggml_webgpu_tensor_binding_overlap_range(const webgpu_global_context & global_ctx,
                                                     ggml_tensor *                 tensor,
                                                     ggml_backend_buffer_t         buffer,
                                                     size_t                        offset,
                                                     size_t                        size) {
    if (tensor->buffer != buffer) {
        return false;
    }

    const size_t alignment     = global_ctx->capabilities.limits.minStorageBufferOffsetAlignment;
    const size_t tensor_offset = ggml_webgpu_tensor_align_offset(tensor, alignment);
    return tensor_offset < offset + size && offset < tensor_offset + ggml_webgpu_tensor_binding_size(tensor, alignment);
}

struct ggml_webgpu_merged_binding_range {
    size_t offset;
    size_t size;
};

static ggml_webgpu_merged_binding_range ggml_webgpu_tensor_merged_binding_range(
    webgpu_context &                     ctx,
    std::initializer_list<ggml_tensor *> tensors) {
    size_t merged_offset = SIZE_MAX;
    size_t merged_end    = 0;

    for (ggml_tensor * tensor : tensors) {
        const size_t bind_offset = ggml_webgpu_tensor_align_offset(ctx, tensor);
        const size_t bind_end    = bind_offset + ggml_webgpu_tensor_binding_size(ctx, tensor);

        merged_offset = std::min(merged_offset, bind_offset);
        merged_end    = std::max(merged_end, bind_end);
    }

    return { merged_offset, merged_end - merged_offset };
}

static uint32_t ggml_webgpu_tensor_merged_element_offset(const ggml_tensor *                      tensor,
                                                         const ggml_webgpu_merged_binding_range & merged_range) {
    return (uint32_t) ((ggml_webgpu_tensor_offset(tensor) - merged_range.offset) / ggml_type_size(tensor->type));
}

static wgpu::BindGroupEntry ggml_webgpu_make_bind_group_entry(uint32_t     binding,
                                                              wgpu::Buffer buffer,
                                                              uint64_t     offset,
                                                              uint64_t     size) {
    wgpu::BindGroupEntry entry = {};
    entry.binding              = binding;
    entry.buffer               = std::move(buffer);
    entry.offset               = offset;
    entry.size                 = size;
    return entry;
}

static wgpu::BindGroupEntry ggml_webgpu_make_tensor_bind_group_entry(webgpu_context & ctx,
                                                                     uint32_t         binding,
                                                                     ggml_tensor *    tensor) {
    return ggml_webgpu_make_bind_group_entry(binding, ggml_webgpu_tensor_buf(tensor),
                                             ggml_webgpu_tensor_align_offset(ctx, tensor),
                                             ggml_webgpu_tensor_binding_size(ctx, tensor));
}

/** End WebGPU object initializations */

/** WebGPU Actions */

template <typename T>
static void ggml_backend_webgpu_check_wait_status(wgpu::WaitStatus wait_status,
                                                  T                callback_status,
                                                  T                success_status,
                                                  const char *     wait_name,
                                                  const char *     failure_name,
                                                  const char *     callback_message) {
    if (wait_status == wgpu::WaitStatus::TimedOut) {
        GGML_ABORT("ggml_webgpu: %s timed out after %u ms\n", wait_name, WEBGPU_RUNTIME_WAIT_TIMEOUT_MS);
    }
    if (wait_status == wgpu::WaitStatus::Error) {
        GGML_ABORT("ggml_webgpu: %s failed\n", wait_name);
    }
    if (callback_status != success_status) {
        GGML_ABORT("ggml_webgpu: %s failed with status %d: %s\n", failure_name, static_cast<int>(callback_status),
                   callback_message);
    }
}

// TODO: these next two functions may want tuning across different platforms and workloads,
static uint32_t ggml_backend_webgpu_get_max_inflight_batches() {
    return UINT32_MAX;
}

static uint32_t ggml_backend_webgpu_get_command_submit_batch_size() {
    return WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_SIZE;
}

static void ggml_backend_webgpu_wait_queue(webgpu_global_context & ctx) {
    wgpu::QueueWorkDoneStatus callback_status = wgpu::QueueWorkDoneStatus::Error;
    std::string               callback_message;

    const wgpu::WaitStatus wait_status = ctx->instance.WaitAny(
        ctx->queue.OnSubmittedWorkDone(
            wgpu::CallbackMode::AllowSpontaneous,
            [&callback_status, &callback_message](wgpu::QueueWorkDoneStatus status, wgpu::StringView message) {
                callback_status  = status;
                callback_message = std::string(message);
            }),
        WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);

    ggml_backend_webgpu_check_wait_status(wait_status, callback_status, wgpu::QueueWorkDoneStatus::Success,
                                          "Queue wait", "Queue work", callback_message.c_str());
    ctx->queue_idle = true;
}

static void ggml_backend_webgpu_map_buffer(webgpu_global_context & ctx,
                                           wgpu::Buffer &          buffer,
                                           wgpu::MapMode           mode,
                                           size_t                  offset,
                                           size_t                  size) {
    wgpu::MapAsyncStatus callback_status = wgpu::MapAsyncStatus::Error;
    std::string          callback_message;

    const wgpu::WaitStatus wait_status = ctx->instance.WaitAny(
        buffer.MapAsync(mode, offset, size, wgpu::CallbackMode::AllowSpontaneous,
                        [&callback_status, &callback_message](wgpu::MapAsyncStatus status, wgpu::StringView message) {
                            callback_status  = status;
                            callback_message = std::string(message);
                        }),
        WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);

    ggml_backend_webgpu_check_wait_status(wait_status, callback_status, wgpu::MapAsyncStatus::Success,
                                          "Buffer map wait", "Buffer map", callback_message.c_str());
}

static void ggml_backend_webgpu_submit_commands(webgpu_context &          ctx,
                                                const wgpu::CommandBuffer commands,
                                                uint32_t &                num_inflight_batches) {
    if (num_inflight_batches >= ctx->global_ctx->max_inflight_batches) {
        ggml_backend_webgpu_wait_queue(ctx->global_ctx);
        num_inflight_batches = 0;
    }

    ctx->param_arena.flush(ctx->global_ctx->queue);
    ctx->global_ctx->queue_idle = false;
    ctx->global_ctx->pvr_write_epoch++;
    ctx->global_ctx->queue.Submit(1, &commands);
    num_inflight_batches++;
}

#ifdef GGML_WEBGPU_DEBUG
// This function adds debugging information to shaders, as WebGPU does not support printing directly.
// To use, add a bind group entry to the setup for the shader you are debugging, add the buffer and
// debug statements in the shader, and then call this function after encoding the commands and submitting them.
static void ggml_backend_webgpu_debug(webgpu_global_context & ctx) {
    wgpu::CommandEncoder encoder = ctx->device.CreateCommandEncoder();
    encoder.CopyBufferToBuffer(ctx->debug_dev_buf, 0, ctx->debug_host_buf, 0, ctx->debug_host_buf.GetSize());
    wgpu::CommandBuffer commands = encoder.Finish();
    ctx->queue.Submit(1, &commands);
    ggml_backend_webgpu_map_buffer(ctx, ctx->debug_host_buf, wgpu::MapMode::Read, 0, ctx->debug_host_buf.GetSize());
    const float * debug_data = (const float *) ctx->debug_host_buf.GetConstMappedRange();
    std::cout << "debug[0]: " << debug_data[0] << "\n";
    ctx->debug_host_buf.Unmap();
}
#endif

static webgpu_encoded_op ggml_backend_webgpu_build_multi(webgpu_context &                          ctx,
                                                         const std::vector<webgpu_dispatch_desc> & dispatches) {
    webgpu_encoded_op            result = {};
    std::vector<wgpu::BindGroup> bind_groups;
    std::vector<size_t>          param_offsets;
    result.num_kernels = dispatches.size();

    for (size_t i = 0; i < dispatches.size(); i++) {
        const webgpu_dispatch_desc & dispatch     = dispatches[i];
        const size_t                 param_size   = dispatch.params.size() * sizeof(uint32_t);
        const size_t                 param_offset = ctx->param_arena.alloc_slot(param_size);

        std::vector<wgpu::BindGroupEntry> entries            = dispatch.bind_group_entries;
        uint32_t                          params_binding_num = entries.size();
        entries.push_back(ggml_webgpu_make_bind_group_entry(params_binding_num, ctx->param_arena.buffer, param_offset,
                                                            ctx->param_arena.slot_size));

        wgpu::BindGroupDescriptor bind_group_desc;
        bind_group_desc.layout     = dispatch.pipeline.pipeline.GetBindGroupLayout(0);
        bind_group_desc.entryCount = entries.size();
        bind_group_desc.entries    = entries.data();
        bind_group_desc.label      = dispatch.pipeline.name.c_str();
        bind_groups.push_back(ctx->global_ctx->device.CreateBindGroup(&bind_group_desc));
        param_offsets.push_back(param_offset);
    }

    const bool defer_params = ggml_webgpu_is_pvr(ctx->global_ctx);
    for (size_t i = 0; i < param_offsets.size(); i++) {
        if (defer_params) {
            // one WriteBuffer per batch instead of one per dispatch
            ctx->param_arena.write_deferred(param_offsets[i], dispatches[i].params.data(),
                                            dispatches[i].params.size() * sizeof(uint32_t));
        } else {
            ctx->global_ctx->queue.WriteBuffer(ctx->param_arena.buffer, param_offsets[i], dispatches[i].params.data(),
                                               dispatches[i].params.size() * sizeof(uint32_t));
        }
    }

#ifdef GGML_WEBGPU_GPU_PROFILE
    for (size_t i = 0; i < dispatches.size(); i++) {
        GGML_ASSERT(ctx->profile_timestamp_query_count + 2 <= WEBGPU_MAX_PROFILE_QUERY_COUNT);
        const uint32_t query_begin = ctx->profile_timestamp_query_count++;
        const uint32_t query_end   = ctx->profile_timestamp_query_count++;

        wgpu::PassTimestampWrites ts_writes   = {};
        ts_writes.querySet                    = ctx->profile_timestamp_query_set;
        ts_writes.beginningOfPassWriteIndex   = query_begin;
        ts_writes.endOfPassWriteIndex         = query_end;
        wgpu::ComputePassDescriptor pass_desc = {};
        pass_desc.timestampWrites             = &ts_writes;

        wgpu::ComputePassEncoder pass = ctx->active_command_encoder.BeginComputePass(&pass_desc);

        pass.SetPipeline(dispatches[i].pipeline.pipeline);
        pass.SetBindGroup(0, bind_groups[i]);
        pass.DispatchWorkgroups(dispatches[i].workgroups.first, dispatches[i].workgroups.second, 1);
        pass.End();
        result.pipeline_names.push_back(dispatches[i].pipeline.name);
    }
#else
    for (size_t i = 0; i < dispatches.size(); i++) {
        if (ctx->batch_compute_passes) {
            ctx->active_compute_pass.SetPipeline(dispatches[i].pipeline.pipeline);
            ctx->active_compute_pass.SetBindGroup(0, bind_groups[i]);
            ctx->active_compute_pass.DispatchWorkgroups(dispatches[i].workgroups.first, dispatches[i].workgroups.second,
                                                        1);
        } else {
            wgpu::ComputePassEncoder pass = ctx->active_command_encoder.BeginComputePass();
            pass.SetPipeline(dispatches[i].pipeline.pipeline);
            pass.SetBindGroup(0, bind_groups[i]);
            pass.DispatchWorkgroups(dispatches[i].workgroups.first, dispatches[i].workgroups.second, 1);
            pass.End();
        }
    }
#endif

    return result;
}

static webgpu_encoded_op ggml_backend_webgpu_build(webgpu_context &                  ctx,
                                                   webgpu_pipeline &                 pipeline,
                                                   std::vector<uint32_t>             params,
                                                   std::vector<wgpu::BindGroupEntry> bind_group_entries,
                                                   uint32_t                          wg_x,
                                                   uint32_t                          wg_y = 1) {
    return ggml_backend_webgpu_build_multi(
        ctx, {
                 { pipeline, std::move(params), std::move(bind_group_entries), { wg_x, wg_y } },
    });
}

static void ggml_backend_webgpu_buffer_memset(webgpu_global_context & ctx,
                                              wgpu::Buffer &          buf,
                                              uint32_t                value,
                                              size_t                  offset,
                                              size_t                  size) {
    std::lock_guard<std::recursive_mutex> lock(ctx->mutex);

    const size_t max_bind_size = ctx->capabilities.limits.maxStorageBufferBindingSize;
    const size_t max_wg_size   = ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    while (size > 0) {
        const size_t align_offset = offset & ~(ctx->capabilities.limits.minStorageBufferOffsetAlignment - 1);
        const size_t lead         = offset - align_offset;
        const size_t chunk_size   = std::min(size, max_bind_size - lead);
        const size_t bind_size    = ROUNDUP_POW2(chunk_size + lead, WEBGPU_STORAGE_BUF_BINDING_MULT);

        std::vector<uint32_t>             params  = { (uint32_t) lead, (uint32_t) chunk_size, value };
        std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_bind_group_entry(0, buf, align_offset,
                                                                                        bind_size) };
        uint32_t                          wg_x;
        uint32_t                          wg_y;
        uint32_t                          n_values = CEIL_DIV(chunk_size + lead, sizeof(uint32_t));
        uint32_t                          total_wg = CEIL_DIV(n_values, max_wg_size);
        compute_2d_workgroups(total_wg, ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);

        ctx->queue.WriteBuffer(ctx->memset_params_buf, 0, params.data(), params.size() * sizeof(uint32_t));

        wgpu::BindGroupEntry params_entry = {};
        params_entry.binding              = 1;
        params_entry.buffer               = ctx->memset_params_buf;
        params_entry.offset               = 0;
        params_entry.size                 = WEBGPU_PARAMS_BUF_SIZE_BYTES;
        entries.push_back(params_entry);

        wgpu::BindGroupDescriptor bind_group_desc;
        bind_group_desc.layout     = ctx->memset_pipeline.pipeline.GetBindGroupLayout(0);
        bind_group_desc.entryCount = entries.size();
        bind_group_desc.entries    = entries.data();
        bind_group_desc.label      = ctx->memset_pipeline.name.c_str();
        wgpu::BindGroup bind_group = ctx->device.CreateBindGroup(&bind_group_desc);

        wgpu::CommandEncoder     encoder = ctx->device.CreateCommandEncoder();
        wgpu::ComputePassEncoder pass    = encoder.BeginComputePass();
        pass.SetPipeline(ctx->memset_pipeline.pipeline);
        pass.SetBindGroup(0, bind_group);
        pass.DispatchWorkgroups(wg_x, wg_y, 1);
        pass.End();

        wgpu::CommandBuffer              command  = encoder.Finish();
        std::vector<wgpu::CommandBuffer> commands = { command };
        ctx->queue_idle                           = false;
        ctx->pvr_write_epoch++;
        ctx->queue.Submit(commands.size(), commands.data());

        offset += chunk_size;
        size -= chunk_size;
    }
}

/** End WebGPU Actions */

/** GGML Backend Interface */

static const char * ggml_backend_webgpu_name(ggml_backend_t backend) {
    ggml_backend_webgpu_context * ctx = (ggml_backend_webgpu_context *) backend->context;
    return ctx->name.c_str();
}

static void ggml_backend_webgpu_free(ggml_backend_t backend) {
    ggml_backend_webgpu_context * ctx = (ggml_backend_webgpu_context *) backend->context;
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_free(" << ctx->name << ")");

#ifdef GGML_WEBGPU_CPU_PROFILE
    std::cout << "\n[ggml_webgpu cpu profiling summary]\n";
    double total_cpu = 0.0;
    for (const auto & kv : ctx->webgpu_ctx->global_ctx->cpu_time_ms) {
        total_cpu += kv.second;
    }
    std::cout << "ggml_webgpu: total cpu time: " << total_cpu << " ms\n";
    std::cout << "ggml_webgpu: cpu breakdown:\n";
    for (const auto & kv : ctx->webgpu_ctx->global_ctx->cpu_time_ms) {
        double pct = (total_cpu > 0.0) ? (kv.second / total_cpu * 100.0) : 0.0;
        std::cout << "ggml_webgpu:  " << kv.first << ": " << kv.second << " ms (" << pct << "%)\n";
    }
    if (ctx->webgpu_ctx->global_ctx->cpu_detail_ms.size() > 0) {
        std::cout << "ggml_webgpu: cpu detailed breakdown:\n";
    }
    for (const auto & kv : ctx->webgpu_ctx->global_ctx->cpu_detail_ms) {
        double pct = (total_cpu > 0.0) ? (kv.second / total_cpu * 100.0) : 0.0;
        std::cout << "ggml_webgpu:  " << kv.first << ": " << kv.second << " ms (" << pct << "%)\n";
    }
#endif

#ifdef GGML_WEBGPU_GPU_PROFILE
    std::cout << "\n[ggml_webgpu gpu profiling summary]\n";
    double total_gpu = 0.0;
    for (const auto & kv : ctx->webgpu_ctx->shader_gpu_time_ms) {
        total_gpu += kv.second;
    }
    std::cout << "ggml_webgpu: total gpu time (all shaders): " << total_gpu << " ms\n";
    std::cout << "\nggml_webgpu: gpu breakdown:\n";
    for (const auto & kv : ctx->webgpu_ctx->shader_gpu_time_ms) {
        double pct = (total_gpu > 0.0) ? (kv.second / total_gpu * 100.0) : 0.0;
        std::cout << "ggml_webgpu:  " << kv.first << ": " << kv.second << " ms (" << std::fixed << std::setprecision(2)
                  << pct << "%)\n";
    }
#endif

#if defined(GGML_WEBGPU_CPU_PROFILE) && defined(GGML_WEBGPU_GPU_PROFILE)
    std::cout << "ggml_webgpu: gpu/cpu ratio: " << (total_cpu > 0.0 ? total_gpu / total_cpu : 0.0) << "\n";
#endif

    delete ctx;
    delete backend;
}

static webgpu_encoded_op ggml_webgpu_cpy(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_cpy_pipeline(shader_lib_ctx);

    auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = {
        ne, (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        // Convert byte-strides to element-strides
        (uint32_t) (src->nb[0] / ggml_type_size(src->type)), (uint32_t) (src->nb[1] / ggml_type_size(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_size(src->type)), (uint32_t) (src->nb[3] / ggml_type_size(src->type)),
        (uint32_t) (dst->nb[0] / ggml_type_size(dst->type)), (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)), (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
        // Logical shapes
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2], (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1], (uint32_t) dst->ne[2]
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst),
    };

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = CEIL_DIV(ne, decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

static webgpu_encoded_op ggml_webgpu_set(webgpu_context & ctx,
                                         ggml_tensor *    src0,
                                         ggml_tensor *    src1,
                                         ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_set_pipeline(shader_lib_ctx);

    auto *     decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
    const bool inplace   = decisions->inplace;

    const uint32_t ne            = inplace ? (uint32_t) ggml_nelements(src1) : (uint32_t) ggml_nelements(dst);
    const uint32_t dst_type_size = (uint32_t) ggml_type_size(dst->type);

    std::vector<uint32_t> params = {
        ne,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        (uint32_t) (((const int32_t *) dst->op_params)[3] / dst_type_size),

        (uint32_t) (src1->nb[0] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)),

        1u,
        (uint32_t) (((const int32_t *) dst->op_params)[0] / dst_type_size),
        (uint32_t) (((const int32_t *) dst->op_params)[1] / dst_type_size),
        (uint32_t) (((const int32_t *) dst->op_params)[2] / dst_type_size),

        (uint32_t) src1->ne[0],
        (uint32_t) src1->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) src1->ne[3],
    };

    std::vector<wgpu::BindGroupEntry> entries;
    uint32_t                          binding_index = 0;
    if (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0));
        binding_index++;
    }
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_index, src1));
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_index + 1, dst));

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = CEIL_DIV(ne, decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

static webgpu_encoded_op ggml_webgpu_arange(webgpu_context & ctx, ggml_tensor * dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_arange_pipeline(shader_lib_ctx);
    auto *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    const uint32_t        ne     = (uint32_t) ggml_nelements(dst);
    std::vector<uint32_t> params = {
        ne,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 0)),
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 2)),
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, dst),
    };

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = CEIL_DIV(ne, decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

static webgpu_encoded_op ggml_webgpu_pad(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_pad_pipeline(shader_lib_ctx);

    auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    const uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = {
        ne,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        // Strides (in elements)
        (uint32_t) (src->nb[0] / ggml_type_size(src->type)),
        (uint32_t) (src->nb[1] / ggml_type_size(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_size(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_size(src->type)),
        // Shapes
        (uint32_t) src->ne[0],
        (uint32_t) src->ne[1],
        (uint32_t) src->ne[2],
        (uint32_t) src->ne[3],
        (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2],
        (uint32_t) dst->ne[3],
        // Pad sizes
        (uint32_t) ggml_get_op_params_i32(dst, 0),
        (uint32_t) ggml_get_op_params_i32(dst, 1),
        (uint32_t) ggml_get_op_params_i32(dst, 2),
        (uint32_t) ggml_get_op_params_i32(dst, 3),
        (uint32_t) ggml_get_op_params_i32(dst, 4),
        (uint32_t) ggml_get_op_params_i32(dst, 5),
        (uint32_t) ggml_get_op_params_i32(dst, 6),
        (uint32_t) ggml_get_op_params_i32(dst, 7),
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst),
    };

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = CEIL_DIV(ne, decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

static webgpu_encoded_op ggml_webgpu_solve_tri(webgpu_context & ctx,
                                               ggml_tensor *    src0,
                                               ggml_tensor *    src1,
                                               ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size        = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    shader_lib_ctx.wg_mem_limit_bytes = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupStorageSize;

    webgpu_pipeline pipeline = ctx->shader_lib->get_solve_tri_pipeline(shader_lib_ctx);

    auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),

        (uint32_t) (src0->nb[0] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),

        (uint32_t) (src1->nb[0] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)),

        (uint32_t) (dst->nb[0] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),

        (uint32_t) src1->ne[0],
        (uint32_t) dst->ne[2],
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst),
    };

    const uint32_t wg_x = CEIL_DIV((uint32_t) src1->ne[0], decisions->wg_size);
    const uint32_t wg_y = (uint32_t) (dst->ne[2] * dst->ne[3]);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

static webgpu_encoded_op ggml_webgpu_conv_2d(webgpu_context & ctx,
                                             ggml_tensor *    src0,
                                             ggml_tensor *    src1,
                                             ggml_tensor *    dst) {
    const int32_t s0 = ggml_get_op_params_i32(dst, 0);
    const int32_t s1 = ggml_get_op_params_i32(dst, 1);
    const int32_t p0 = ggml_get_op_params_i32(dst, 2);
    const int32_t p1 = ggml_get_op_params_i32(dst, 3);
    const int32_t d0 = ggml_get_op_params_i32(dst, 4);
    const int32_t d1 = ggml_get_op_params_i32(dst, 5);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),

        (uint32_t) (src0->nb[0] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),

        (uint32_t) (src1->nb[0] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)),

        (uint32_t) (dst->nb[0] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),

        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],

        (uint32_t) src1->ne[0],
        (uint32_t) src1->ne[1],

        (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2],
        (uint32_t) dst->ne[3],

        (uint32_t) s0,
        (uint32_t) s1,
        (uint32_t) p0,
        (uint32_t) p1,
        (uint32_t) d0,
        (uint32_t) d1,
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_conv2d_pipeline(shader_lib_ctx);

    auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);

    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// Same param/binding layout as conv_2d; the shader differs
static webgpu_encoded_op ggml_webgpu_conv_2d_dw(webgpu_context & ctx,
                                                ggml_tensor *    src0,
                                                ggml_tensor *    src1,
                                                ggml_tensor *    dst) {
    const int32_t s0 = ggml_get_op_params_i32(dst, 0);
    const int32_t s1 = ggml_get_op_params_i32(dst, 1);
    const int32_t p0 = ggml_get_op_params_i32(dst, 2);
    const int32_t p1 = ggml_get_op_params_i32(dst, 3);
    const int32_t d0 = ggml_get_op_params_i32(dst, 4);
    const int32_t d1 = ggml_get_op_params_i32(dst, 5);

    // Scalar params matching conv2d_dw.wgsl (weight src0 [KW,KH,1,C], input src1, output dst).
    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),

        (uint32_t) ggml_nelements(dst),
        (uint32_t) dst->ne[2],
        (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1],
        (uint32_t) src1->ne[0],
        (uint32_t) src1->ne[1],
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],

        (uint32_t) s0,
        (uint32_t) s1,
        (uint32_t) p0,
        (uint32_t) p1,
        (uint32_t) d0,
        (uint32_t) d1,
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    // Input layout: contiguous -> WHCN, contiguous-channels -> CWHN
    const bool      whcn      = ggml_is_contiguous(src1);
    webgpu_pipeline pipeline  = ctx->shader_lib->get_conv2d_dw_pipeline(shader_lib_ctx, whcn);
    auto *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);

    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

static webgpu_encoded_op ggml_webgpu_im2col(webgpu_context & ctx,
                                            ggml_tensor *    src0,
                                            ggml_tensor *    src1,
                                            ggml_tensor *    dst) {
    const int32_t s0    = ggml_get_op_params_i32(dst, 0);
    const int32_t s1    = ggml_get_op_params_i32(dst, 1);
    const int32_t p0    = ggml_get_op_params_i32(dst, 2);
    const int32_t p1    = ggml_get_op_params_i32(dst, 3);
    const int32_t d0    = ggml_get_op_params_i32(dst, 4);
    const int32_t d1    = ggml_get_op_params_i32(dst, 5);
    const bool    is_2D = ggml_get_op_params_i32(dst, 6) == 1;

    const uint32_t KW = src0->ne[0];
    const uint32_t KH = is_2D ? src0->ne[1] : 1;
    const uint32_t IC = is_2D ? src0->ne[2] : src0->ne[1];

    const uint32_t IW = src1->ne[0];
    const uint32_t IH = is_2D ? src1->ne[1] : 1;
    const uint32_t N  = is_2D ? src1->ne[3] : src1->ne[2];

    const uint32_t OW = dst->ne[1];
    const uint32_t OH = is_2D ? dst->ne[2] : 1;

    const uint32_t si0 = (uint32_t) (src1->nb[0] / ggml_type_size(src1->type));
    const uint32_t si1 = is_2D ? (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)) : 0;
    const uint32_t si2 = is_2D ? (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)) :
                                 (uint32_t) (src1->nb[1] / ggml_type_size(src1->type));
    const uint32_t si3 = is_2D ? (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)) :
                                 (uint32_t) (src1->nb[2] / ggml_type_size(src1->type));

    const uint32_t so0 = (uint32_t) (dst->nb[0] / ggml_type_size(dst->type));
    const uint32_t so1 = (uint32_t) (dst->nb[1] / ggml_type_size(dst->type));
    const uint32_t so2 = is_2D ? (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)) : 0;
    const uint32_t so3 = is_2D ? (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)) :
                                 (uint32_t) (dst->nb[2] / ggml_type_size(dst->type));

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),

        si0,
        si1,
        si2,
        si3,
        so0,
        so1,
        so2,
        so3,

        KW,
        KH,
        IC,

        IW,
        IH,
        N,

        OW,
        OH,

        (uint32_t) s0,
        (uint32_t) s1,
        (uint32_t) p0,
        (uint32_t) p1,
        (uint32_t) d0,
        (uint32_t) d1,
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src1),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_im2col_pipeline(shader_lib_ctx);

    auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);

    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

static webgpu_encoded_op ggml_webgpu_ssm_conv(webgpu_context & ctx,
                                              ggml_tensor *    src0,
                                              ggml_tensor *    src1,
                                              ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_ssm_conv_pipeline(shader_lib_ctx);
    auto *          decisions = static_cast<ggml_webgpu_ssm_conv_shader_decisions *>(pipeline.context.get());

    const uint32_t token_tiles = CEIL_DIV((uint32_t) dst->ne[1], decisions->tokens_per_wg);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),

        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),

        (uint32_t) (dst->nb[0] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),

        (uint32_t) src1->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2],
        token_tiles,
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst),
    };

    const uint32_t wg_x = CEIL_DIV((uint32_t) src0->ne[1], decisions->block_size);
    const uint32_t wg_y = token_tiles * (uint32_t) dst->ne[2];
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

static webgpu_encoded_op ggml_webgpu_ssm_scan(webgpu_context & ctx,
                                              ggml_tensor *    src0,
                                              ggml_tensor *    src1,
                                              ggml_tensor *    src2,
                                              ggml_tensor *    src3,
                                              ggml_tensor *    src4,
                                              ggml_tensor *    src5,
                                              ggml_tensor *    src6,
                                              ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.src3                           = src3;
    shader_lib_ctx.src4                           = src4;
    shader_lib_ctx.src5                           = src5;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size        = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    shader_lib_ctx.supports_subgroups = ctx->global_ctx->capabilities.supports_subgroups;

    uint8_t xbc_overlap = 0;
    if (ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src1, src4)) {
        xbc_overlap |= 0b110;  // x/B
    }
    if (ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src1, src5)) {
        xbc_overlap |= 0b101;  // x/C
    }
    if (ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src4, src5)) {
        xbc_overlap |= 0b011;  // B/C
    }

    webgpu_pipeline pipeline  = ctx->shader_lib->get_ssm_scan_pipeline(shader_lib_ctx, xbc_overlap);
    auto *          decisions = static_cast<ggml_webgpu_ssm_scan_shader_decisions *>(pipeline.context.get());
    xbc_overlap               = decisions->xbc_overlap;

    uint32_t offset_x   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type));
    uint32_t offset_dt  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_size(src2->type));
    uint32_t offset_A   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src3) / ggml_type_size(src3->type));
    uint32_t offset_B   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src4) / ggml_type_size(src4->type));
    uint32_t offset_C   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src5) / ggml_type_size(src5->type));
    uint32_t offset_ids = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src6) / ggml_type_size(src6->type));

    ggml_webgpu_merged_binding_range xbc_merged_range = {};

    if (xbc_overlap == 0b110) {  // x/B
        xbc_merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { src1, src4 });
        offset_x         = ggml_webgpu_tensor_merged_element_offset(src1, xbc_merged_range);
        offset_B         = ggml_webgpu_tensor_merged_element_offset(src4, xbc_merged_range);
    } else if (xbc_overlap == 0b011) {  // B/C
        xbc_merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { src4, src5 });
        offset_B         = ggml_webgpu_tensor_merged_element_offset(src4, xbc_merged_range);
        offset_C         = ggml_webgpu_tensor_merged_element_offset(src5, xbc_merged_range);
    } else if (xbc_overlap == 0b111) {  // x/B/C
        xbc_merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { src1, src4, src5 });
        offset_x         = ggml_webgpu_tensor_merged_element_offset(src1, xbc_merged_range);
        offset_B         = ggml_webgpu_tensor_merged_element_offset(src4, xbc_merged_range);
        offset_C         = ggml_webgpu_tensor_merged_element_offset(src5, xbc_merged_range);
    }

    GGML_ASSERT(xbc_overlap == 0 || xbc_overlap == 0b110 || xbc_overlap == 0b011 || xbc_overlap == 0b111);

    size_t xbc_bind_offset = 0;
    size_t xbc_bind_size   = 0;
    if (xbc_overlap > 0) {
        xbc_bind_offset = xbc_merged_range.offset;
        xbc_bind_size   = xbc_merged_range.size;
    }

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        offset_x,
        offset_dt,
        offset_A,
        offset_B,
        offset_C,
        offset_ids,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),

        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),

        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)),

        (uint32_t) (src2->nb[1] / ggml_type_size(src2->type)),
        (uint32_t) (src2->nb[2] / ggml_type_size(src2->type)),

        (uint32_t) src3->ne[0],
        (uint32_t) (src3->nb[1] / ggml_type_size(src3->type)),

        (uint32_t) (src4->nb[1] / ggml_type_size(src4->type)),
        (uint32_t) (src4->nb[2] / ggml_type_size(src4->type)),
        (uint32_t) (src4->nb[3] / ggml_type_size(src4->type)),

        (uint32_t) (src5->nb[1] / ggml_type_size(src5->type)),
        (uint32_t) (src5->nb[2] / ggml_type_size(src5->type)),
        (uint32_t) (src5->nb[3] / ggml_type_size(src5->type)),

        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        (uint32_t) src4->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) ggml_nelements(src1),
        (uint32_t) ggml_get_op_params_i32(dst, 0),
    };

    uint32_t binding_num = 0;

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_num++, src0),
    };
    // xbc_merged binding
    if (xbc_overlap > 0) {
        entries.push_back(ggml_webgpu_make_bind_group_entry(binding_num++, ggml_webgpu_tensor_buf(src1),
                                                            xbc_bind_offset, xbc_bind_size));
    }
    // x
    if (!(xbc_overlap & 0b100)) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_num++, src1));
    }
    // B
    if (!(xbc_overlap & 0b010)) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_num++, src4));
    }
    // C
    if (!(xbc_overlap & 0b001)) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_num++, src5));
    }

    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_num++, src2));
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_num++, src3));
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_num++, src6));
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_num++, dst));

    const uint32_t total_wg       = (uint32_t) (src0->ne[1] * src0->ne[2] * src1->ne[3]);
    const uint32_t max_wg_per_dim = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;
    uint32_t       wg_x;
    uint32_t       wg_y;
    compute_2d_workgroups(total_wg, max_wg_per_dim, wg_x, wg_y);

    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

static webgpu_encoded_op ggml_webgpu_gated_delta_net(webgpu_context & ctx,
                                                     ggml_tensor *    src0,
                                                     ggml_tensor *    src1,
                                                     ggml_tensor *    src2,
                                                     ggml_tensor *    src3,
                                                     ggml_tensor *    src4,
                                                     ggml_tensor *    src5,
                                                     ggml_tensor *    dst,
                                                     ggml_tensor *    dst_fuse) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.src3                           = src3;
    shader_lib_ctx.src4                           = src4;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.dst_fuse                       = dst_fuse;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_gated_delta_net_pipeline(shader_lib_ctx);

    const uint32_t s_v      = (uint32_t) src2->ne[0];
    const uint32_t h        = (uint32_t) src2->ne[1];
    const uint32_t n_tokens = (uint32_t) src2->ne[2];
    const uint32_t n_seqs   = (uint32_t) src2->ne[3];
    const uint32_t K        = (uint32_t) ggml_get_op_params_i32(dst, 0);
    const float    scale    = 1.0f / sqrtf((float) s_v);
    uint32_t       scale_u32;
    memcpy(&scale_u32, &scale, sizeof(scale_u32));

    std::vector<uint32_t> params = {
        h,
        n_tokens,
        n_seqs,
        s_v * h * n_tokens * n_seqs,

        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),

        (uint32_t) (src2->nb[1] / ggml_type_size(src2->type)),
        (uint32_t) (src2->nb[2] / ggml_type_size(src2->type)),
        (uint32_t) (src2->nb[3] / ggml_type_size(src2->type)),

        (uint32_t) (src4->nb[1] / ggml_type_size(src4->type)),
        (uint32_t) (src4->nb[2] / ggml_type_size(src4->type)),
        (uint32_t) (src4->nb[3] / ggml_type_size(src4->type)),

        (uint32_t) src0->ne[1],
        (uint32_t) (src2->ne[3] / src0->ne[3]),
        K,
        scale_u32,
        dst_fuse ? (uint32_t) (dst_fuse->nb[2] / ggml_type_size(dst_fuse->type)) : 0,
        dst_fuse ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst_fuse) / ggml_type_size(dst_fuse->type)) : 0,
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0), ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, src2), ggml_webgpu_make_tensor_bind_group_entry(ctx, 3, src3),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 4, src4), ggml_webgpu_make_tensor_bind_group_entry(ctx, 5, src5),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 6, dst),
    };

    if (dst_fuse) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 7, dst_fuse));
    }

    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, h, n_seqs);
}

static std::optional<webgpu_encoded_op> ggml_webgpu_set_rows(webgpu_context & ctx,
                                                             ggml_tensor *    src,
                                                             ggml_tensor *    idx,
                                                             ggml_tensor *    dst) {
    // For set rows specifically, we need to check if src and idx are empty
    // tensors.
    if (ggml_is_empty(src) || ggml_is_empty(idx)) {
        return std::nullopt;
    }

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = idx;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_set_rows_pipeline(shader_lib_ctx);

    auto * decisions = static_cast<ggml_webgpu_set_rows_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, idx) / ggml_type_size(idx->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        // Convert byte-strides to element-strides
        (uint32_t) (src->nb[1] / ggml_type_size(src->type)), (uint32_t) (src->nb[2] / ggml_type_size(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_size(src->type)), (uint32_t) (idx->nb[0] / ggml_type_size(idx->type)),
        (uint32_t) (idx->nb[1] / ggml_type_size(idx->type)), (uint32_t) (idx->nb[2] / ggml_type_size(idx->type)),
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)), (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
        // Shape of src
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2], (uint32_t) src->ne[3],
        // Shape of idx
        (uint32_t) (idx->ne[1]), (uint32_t) (idx->ne[2])
    };

    std::vector<wgpu::BindGroupEntry> entries;
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src));
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, idx));
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst));

    if (decisions->i64_idx) {
        entries.push_back(ggml_webgpu_make_bind_group_entry(3, ctx->set_rows_dev_error_buf, 0,
                                                            ctx->set_rows_dev_error_buf.GetSize()));
    }

    uint32_t threads;
    if (ggml_is_quantized(dst->type)) {
        const uint32_t blocks_per_row = src->ne[0] / ggml_blck_size(dst->type);
        threads =
            (src->ne[1] * src->ne[2] * src->ne[3]) * (decisions->pair_blocks ? (blocks_per_row / 2) : blocks_per_row);
    } else if (decisions->vec4) {
        threads = (src->ne[1] * src->ne[2] * src->ne[3]) * (src->ne[0] / 4);
    } else {
        threads = src->ne[0] * src->ne[1] * src->ne[2] * src->ne[3];
    }
    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = CEIL_DIV(threads, decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// Workgroup size is a common constant
static std::vector<wgpu::ConstantEntry> ggml_webgpu_wg_size_entry(uint32_t wg_size) {
    std::vector<wgpu::ConstantEntry> constants(1);
    constants[0].key   = "wg_size";
    constants[0].value = wg_size;
    return constants;
}

static webgpu_encoded_op ggml_webgpu_get_rows(webgpu_context & ctx,
                                              ggml_tensor *    src,
                                              ggml_tensor *    idx,
                                              ggml_tensor *    dst) {
    const bool float_parallel = src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16 || src->type == GGML_TYPE_I32;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = nullptr;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    const uint32_t offset_src  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type));
    const uint32_t offset_dst  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type));
    const uint32_t stride_src1 = (uint32_t) (src->nb[1] / ggml_type_size(src->type));
    const uint32_t stride_src2 = (uint32_t) (src->nb[2] / ggml_type_size(src->type));
    const uint32_t stride_src3 = (uint32_t) (src->nb[3] / ggml_type_size(src->type));

    const bool vec4_aligned = offset_src % 4 == 0 && offset_dst % 4 == 0 && stride_src1 % 4 == 0 &&
                              stride_src2 % 4 == 0 && stride_src3 % 4 == 0;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_get_rows_pipeline(shader_lib_ctx, vec4_aligned);
    auto *          decisions = static_cast<ggml_webgpu_get_rows_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = { offset_src,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, idx) / ggml_type_size(idx->type)),
                                     offset_dst,
                                     stride_src1,
                                     stride_src2,
                                     stride_src3,
                                     (uint32_t) (idx->nb[0] / ggml_type_size(idx->type)),
                                     (uint32_t) (idx->nb[1] / ggml_type_size(idx->type)),
                                     (uint32_t) (idx->nb[2] / ggml_type_size(idx->type)),
                                     (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
                                     (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
                                     (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
                                     (uint32_t) dst->ne[0],
                                     (uint32_t) dst->ne[1],
                                     (uint32_t) dst->ne[2],
                                     (uint32_t) dst->ne[3],
                                     (uint32_t) (idx->ne[1]),
                                     (uint32_t) (idx->ne[2]) };

    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, idx),
                                                  ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst) };

    uint32_t blocks_per_row = (uint32_t) (dst->ne[0] / (decisions->vectorized ? 4 : 1));
    uint32_t total_rows     = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    uint32_t total_threads  = float_parallel ? blocks_per_row * total_rows : total_rows;
    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = CEIL_DIV(total_threads, decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

static void ggml_webgpu_quantize_q8_dispatch(webgpu_context &                    ctx,
                                             ggml_tensor *                       src0,
                                             ggml_tensor *                       src1,
                                             ggml_tensor *                       dst,
                                             std::vector<webgpu_dispatch_desc> & dispatches) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};

    shader_lib_ctx.src0               = src0;
    shader_lib_ctx.src1               = src1;
    shader_lib_ctx.dst                = dst;
    shader_lib_ctx.max_wg_size        = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    shader_lib_ctx.supports_subgroups = ctx->global_ctx->capabilities.supports_subgroups;

    webgpu_pipeline qq8_pipeline = ctx->shader_lib->get_quantize_q8_pipeline(shader_lib_ctx);

    // quantize_q8 pipeline
    const size_t dst_offset           = ggml_webgpu_tensor_offset(dst);
    const size_t q8_src1_align_offset = ROUNDUP_POW2(
        dst_offset + ggml_nbytes(dst), ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment);
    const size_t q8_src1_binding_size = ROUNDUP_POW2(
        src1->ne[3] * src1->ne[2] * src1->ne[1] * (36 /* sizeof(q8_1) */ * (src1->ne[0] / /* block_size */ 32)),
        WEBGPU_STORAGE_BUF_BINDING_MULT);

    std::vector<uint32_t> q8_params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)),
        (uint32_t) src1->ne[0],
        (uint32_t) src1->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) src1->ne[3],
    };

    std::vector<wgpu::BindGroupEntry> q8_entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src1),
        ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(dst), q8_src1_align_offset, q8_src1_binding_size)
    };

    auto q8_decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(qq8_pipeline.context.get());

    uint32_t       q8_wg_size     = q8_decisions->wg_size;
    uint32_t       q8_wg_x        = 1;
    uint32_t       q8_wg_y        = 1;
    const uint32_t wg_per_vec     = (src0->ne[0] / 4 + (q8_wg_size - 1)) / q8_wg_size;
    const uint32_t q8_total_wg    = src1->ne[1] * src1->ne[2] * src1->ne[3] * wg_per_vec;
    const uint32_t max_wg_per_dim = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;
    compute_2d_workgroups(q8_total_wg, max_wg_per_dim, q8_wg_x, q8_wg_y);

    dispatches.push_back({
        qq8_pipeline, std::move(q8_params), std::move(q8_entries), { q8_wg_x, q8_wg_y }
    });
}

// Imagination PowerVR GPUs read storage bindings declared read_write several
// times slower than read-only ones, which all the generic shaders use, and
// run the tiled mat-mat shaders slowly. Matrix products of the K-quants go
// through shaders of their own there, which read every operand through
// read-only bindings: mul_mat_vec_pvr.wgsl for one column, mul_mat_pvr.wgsl
// for a few, and for prompt batches mul_mat_pvr_dequant.wgsl, which writes
// the weights out as f16, then mul_mat_pvr_gemm.wgsl.

// Columns from which prompt batches dequantize the weights first
#define WEBGPU_MUL_MAT_PVR_GEMM_MIN_N 32u
// Columns up to which mul_mat_vec_pvr.wgsl takes them all at once
#define WEBGPU_MUL_MAT_VEC_PVR_MAX_COLS 8u

// texture rows holding m x k dequantized weights (8 per texel)
static uint32_t ggml_webgpu_pvr_dequant_rows(uint64_t m, uint64_t k) {
    return (uint32_t) CEIL_DIV(m * (k / 8), WEBGPU_MUL_MAT_PVR_TEX_W);
}

// Whether shaders of wg_size threads can place threads by subgroup and lane
// (as mul_mat_vec_pvr.wgsl and flash_attn_pvr_mq.wgsl do, to pair lanes in
// shuffles) with groups of up to min_lanes neighbouring lanes: every
// subgroup size the device may pick divides wg_size and holds min_lanes.
static bool ggml_webgpu_pvr_subgroups_tile(const webgpu_global_context & global_ctx, uint32_t wg_size, uint32_t min_lanes) {
    const webgpu_capabilities & caps = global_ctx->capabilities;
    return caps.supports_subgroups && caps.min_subgroup_size >= min_lanes && caps.max_subgroup_size > 0 &&
           (caps.min_subgroup_size & (caps.min_subgroup_size - 1)) == 0 && wg_size % caps.max_subgroup_size == 0;
}

// mul_mat_vec_pvr.wgsl adds up a row's lanes (up to 8, neighbours) with subgroup shuffles
static bool ggml_webgpu_mul_mat_vec_pvr_subgroups(const webgpu_global_context & global_ctx) {
    return ggml_webgpu_pvr_subgroups_tile(global_ctx, WEBGPU_MUL_MAT_VEC_PVR_WG_SIZE, 8);
}

static bool ggml_webgpu_mul_mat_pvr_use_gemm(webgpu_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1) {
    return src1->ne[1] >= WEBGPU_MUL_MAT_PVR_GEMM_MIN_N &&
           ggml_webgpu_pvr_dequant_rows(src0->ne[1], src0->ne[0]) <=
               ctx->global_ctx->capabilities.limits.maxTextureDimension2D &&
           WEBGPU_MUL_MAT_PVR_TEX_W <= ctx->global_ctx->capabilities.limits.maxTextureDimension2D;
}

static bool ggml_webgpu_can_use_mul_mat_pvr(webgpu_context &    ctx,
                                            const ggml_tensor * src0,
                                            const ggml_tensor * src1,
                                            const ggml_tensor * dst) {
    if (!ggml_webgpu_is_pvr(ctx->global_ctx)) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q6_K) {
        return false;
    }
    // mul_mat_pvr.wgsl reads each weight once per 16 columns, which is slow
    // through a read_write binding: leave weights sharing dst's buffer to the
    // generic shaders, unless they are dequantized first (only the dequantize
    // shader reads them, and it writes elsewhere)
    if (src1->ne[1] > 1 && ggml_webgpu_tensor_buf(src0).Get() == ggml_webgpu_tensor_buf(dst).Get() &&
        !ggml_webgpu_mul_mat_pvr_use_gemm(ctx, src0, src1) && !ggml_webgpu_pvr_is_repacked(src0)) {
        return false;
    }
    const size_t staged_bytes = (size_t) src1->ne[0] * src1->ne[1] * sizeof(float);
    return src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && src0->ne[2] == 1 && src0->ne[3] == 1 &&
           src1->ne[2] == 1 && src1->ne[3] == 1 && src0->ne[0] % 256 == 0 &&
           src1->nb[0] == ggml_type_size(src1->type) && src0->nb[0] == ggml_type_size(src0->type) &&
           dst->nb[0] == ggml_type_size(dst->type) && dst->nb[1] == dst->ne[0] * ggml_type_size(dst->type) &&
           src1->ne[1] <= ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension &&
           staged_bytes <= ctx->global_ctx->capabilities.limits.maxStorageBufferBindingSize;
}

// The weights' offset (blocks from the binding) and row stride as the PowerVR
// product shaders take them: for repacked weights (layout A, bound at their
// start), 0 and the number of rows.
static uint32_t ggml_webgpu_pvr_src0_offset(webgpu_context & ctx, const ggml_tensor * src0) {
    if (ggml_webgpu_pvr_is_repacked(src0)) {
        GGML_ASSERT(ggml_webgpu_tensor_misalignment(ctx, src0) == 0);
        return 0;
    }
    return (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type));
}

static uint32_t ggml_webgpu_pvr_src0_stride(const ggml_tensor * src0) {
    return ggml_webgpu_pvr_is_repacked(src0) ? (uint32_t) src0->ne[1] : (uint32_t) (src0->nb[1] / ggml_type_size(src0->type));
}

// Grows one of the PowerVR scratch buffers to at least size bytes. Commands
// already encoded keep the old buffer alive.
static void ggml_webgpu_pvr_scratch(webgpu_context & ctx, wgpu::Buffer & buf, size_t & buf_size, size_t size,
                                    const char * label) {
    if (!buf || buf_size < size) {
        buf_size = std::max<size_t>(size, 64 * 1024);
        ggml_webgpu_create_buffer(ctx->global_ctx->device, buf, buf_size, wgpu::BufferUsage::Storage, label);
    }
}

// Whether the staging buffer holds src1 (f32, or f16 for the GEMM) already.
static bool ggml_webgpu_pvr_is_staged(webgpu_context & ctx, const ggml_tensor * src1, bool f16) {
    const ggml_tensor * staged = ctx->pvr_staged_src1;
    return staged != nullptr && ctx->pvr_staged_f16 == f16 &&
           ggml_webgpu_tensor_buf(staged).Get() == ggml_webgpu_tensor_buf(src1).Get() &&
           ggml_webgpu_tensor_offset(staged) == ggml_webgpu_tensor_offset(src1) && staged->ne[0] == src1->ne[0] &&
           staged->ne[1] == src1->ne[1] && (src1->ne[1] == 1 || staged->nb[1] == src1->nb[1]);
}

// Switches to the other staging buffer, of at least size bytes, for writing
// the next src1 (the current one may still be read by the op writing it).
static void ggml_webgpu_pvr_stage_next(webgpu_context & ctx, size_t size) {
    const int next = ctx->pvr_staged_idx ^ 1;
    ggml_webgpu_pvr_scratch(ctx, ctx->pvr_staging_buf[next], ctx->pvr_staging_size[next], size, "pvr_staging_buf");
    ctx->pvr_staged_idx  = next;
    ctx->pvr_staged_src1 = nullptr;
}

// Matrix products on PowerVR: copy src1 into the staging buffer (f32, or f16
// for mul_mat_pvr_gemm.wgsl), then multiply from read-only bindings.
// glu_other: SwiGLU fused in (mul_mat_pvr_gemm.wgsl, GLU), dst then being
// the GLU's output and glu_other its other operand; glu_mm_is_gate says
// which operand the product is.
static webgpu_encoded_op ggml_webgpu_mul_mat_pvr(webgpu_context & ctx,
                                                 ggml_tensor *    src0,
                                                 ggml_tensor *    src1,
                                                 ggml_tensor *    dst,
                                                 ggml_tensor *    glu_other      = nullptr,
                                                 bool             glu_mm_is_gate = false) {
    const uint32_t k            = (uint32_t) src0->ne[0];
    const uint32_t m            = (uint32_t) dst->ne[0];
    const uint32_t n            = (uint32_t) src1->ne[1];
    const bool     gemm         = ggml_webgpu_mul_mat_pvr_use_gemm(ctx, src0, src1);
    const size_t   staged_bytes = ROUNDUP_POW2((size_t) k * n * (gemm ? sizeof(ggml_fp16_t) : sizeof(float)), 16);

    std::vector<webgpu_dispatch_desc> dispatches;

    const bool reuse_staged = ggml_webgpu_pvr_is_staged(ctx, src1, gemm);
    if (!reuse_staged) {
        ggml_webgpu_pvr_stage_next(ctx, staged_bytes);
    }
    ctx->pvr_staged_src1 = src1;
    ctx->pvr_staged_f16  = gemm;
    wgpu::Buffer & staging_buf = ctx->pvr_staging_buf[ctx->pvr_staged_idx];

    // 1. copy src1 into the staging buffer
    if (!reuse_staged) {
        webgpu_pipeline       stage_pipeline = ctx->shader_lib->get_mul_mat_pvr_stage_pipeline(gemm);
        std::vector<uint32_t> stage_params   = {
            (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
            k,
            (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
            n,
        };
        std::vector<wgpu::BindGroupEntry> stage_entries = {
            ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src1),
            ggml_webgpu_make_bind_group_entry(1, staging_buf, 0, staged_bytes),
        };
        // the f16 copy converts a pair of values per thread
        const uint32_t per_column = gemm ? k / 2 : k;
        dispatches.push_back({
            stage_pipeline, std::move(stage_params), std::move(stage_entries),
            { CEIL_DIV(per_column, WEBGPU_MUL_MAT_PVR_STAGE_WG_SIZE), n }
        });
    }

    if (gemm) {
        // 2. the weights to f16, read-only as only this shader reads them
        const uint32_t rows = ggml_webgpu_pvr_dequant_rows(m, k);
        if (!ctx->pvr_dequant_tex || ctx->pvr_dequant_rows < rows) {
            // commands already encoded keep the old texture alive
            ctx->pvr_dequant_rows = rows;
            wgpu::TextureDescriptor desc;
            desc.label         = "pvr_dequant_tex";
            desc.usage         = wgpu::TextureUsage::StorageBinding | wgpu::TextureUsage::TextureBinding;
            desc.dimension     = wgpu::TextureDimension::e2D;
            desc.size          = { WEBGPU_MUL_MAT_PVR_TEX_W, rows, 1 };
            desc.format        = wgpu::TextureFormat::RGBA32Uint;
            desc.mipLevelCount = 1;
            desc.sampleCount   = 1;
            ctx->pvr_dequant_tex  = ctx->global_ctx->device.CreateTexture(&desc);
            ctx->pvr_dequant_view = ctx->pvr_dequant_tex.CreateView();
        }
        wgpu::BindGroupEntry tex_entry = {};
        tex_entry.binding              = 1;
        tex_entry.textureView          = ctx->pvr_dequant_view;
        uint32_t wg_x = 1;
        uint32_t wg_y = 1;
        // (repacked weights: a workgroup per block column of WG_SIZE / 32 rows)
        const uint32_t dequant_wg =
            ggml_webgpu_pvr_is_repacked(src0) ?
                CEIL_DIV(m, WEBGPU_MUL_MAT_PVR_DEQUANT_WG_SIZE / 32) * (k / 256) :
                CEIL_DIV(m * (k / 8), WEBGPU_MUL_MAT_PVR_DEQUANT_WG_SIZE);
        compute_2d_workgroups(dequant_wg,
                              ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
        dispatches.push_back({
            ctx->shader_lib->get_mul_mat_pvr_dequant_pipeline(src0->type, ggml_webgpu_pvr_is_repacked(src0)),
            { ggml_webgpu_pvr_src0_offset(ctx, src0), ggml_webgpu_pvr_src0_stride(src0), m, k },
            { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0), tex_entry },
            { wg_x, wg_y }
        });

        // 3. the product
        tex_entry.binding = 0;
        compute_2d_workgroups(CEIL_DIV(m, WEBGPU_MUL_MAT_PVR_GEMM_ROWS) * CEIL_DIV(n, WEBGPU_MUL_MAT_PVR_GEMM_COLS),
                              ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
        std::vector<wgpu::BindGroupEntry> gemm_entries = {
            tex_entry, ggml_webgpu_make_bind_group_entry(1, staging_buf, 0, staged_bytes),
            ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst)
        };
        uint32_t offset_glu = 0;
        if (glu_other != nullptr) {
            gemm_entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 3, glu_other));
            offset_glu = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, glu_other) / ggml_type_size(glu_other->type));
        }
        dispatches.push_back({
            ctx->shader_lib->get_mul_mat_pvr_gemm_pipeline(glu_other == nullptr ? 0 : (glu_mm_is_gate ? 2 : 1)),
            { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)), m, n, k, offset_glu },
            std::move(gemm_entries), { wg_x, wg_y }
        });
        return ggml_backend_webgpu_build_multi(ctx, dispatches);
    }

    // 2. the product, with the weights read-only unless they share dst's buffer
    const bool      src0_ro = ggml_webgpu_tensor_buf(src0).Get() != ggml_webgpu_tensor_buf(dst).Get();
    webgpu_pipeline pipeline;
    uint32_t        total_wg;
    if (n <= WEBGPU_MUL_MAT_VEC_PVR_MAX_COLS) {
        // one column, or a few (a speculative batch, the end of a prompt) read against the weights once
        const uint32_t ncol = n;
        const bool     sg   = ggml_webgpu_mul_mat_vec_pvr_subgroups(ctx->global_ctx);
        pipeline            = ctx->shader_lib->get_mul_mat_vec_pvr_pipeline(
            src0->type, src0_ro, ncol, sg, {},
            ggml_webgpu_shader_lib::ggml_webgpu_mul_mat_vec_pvr_choose_shape(src0->type, ggml_webgpu_pvr_is_repacked(src0),
                                                                             ncol, m, sg, {}));
        auto * decisions = static_cast<ggml_webgpu_mul_mat_vec_shader_decisions *>(pipeline.context.get());
        total_wg         = CEIL_DIV(m, decisions->outputs_per_wg);
    } else {
        pipeline = ctx->shader_lib->get_mul_mat_pvr_pipeline(src0->type, src0_ro, ggml_webgpu_pvr_is_repacked(src0));
        total_wg = CEIL_DIV(m, WEBGPU_MUL_MAT_PVR_TILE) * CEIL_DIV(n, WEBGPU_MUL_MAT_PVR_TILE);
    }

    std::vector<uint32_t> params = { ggml_webgpu_pvr_src0_offset(ctx, src0),
                                     0u,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
                                     m,
                                     n,
                                     k,
                                     ggml_webgpu_pvr_src0_stride(src0),
                                     k,
                                     0u,
                                     0u,
                                     0u,
                                     0u,
                                     1u,
                                     1u,
                                     1u,
                                     1u };
    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0),
        ggml_webgpu_make_bind_group_entry(1, staging_buf, 0, staged_bytes),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst),
    };
    uint32_t wg_x = 1;
    uint32_t wg_y = 1;
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    dispatches.push_back({
        pipeline, std::move(params), std::move(entries), { wg_x, wg_y }
    });

    return ggml_backend_webgpu_build_multi(ctx, dispatches);
}

// Whether the PowerVR mat-vec (mul_mat_vec_pvr.wgsl) takes this product.
static bool ggml_webgpu_mul_mat_vec_pvr_applies(webgpu_context &    ctx,
                                                const ggml_tensor * src0,
                                                const ggml_tensor * src1,
                                                const ggml_tensor * dst) {
    return ggml_webgpu_can_use_mul_mat_pvr(ctx, src0, src1, dst) && src1->ne[1] <= WEBGPU_MUL_MAT_VEC_PVR_MAX_COLS &&
           !ggml_webgpu_mul_mat_pvr_use_gemm(ctx, src0, src1) && ggml_webgpu_tensor_buf(src0).Get() != ggml_webgpu_tensor_buf(dst).Get();
}

// The PowerVR mat-vec with the next op(s) fused in, to save dispatches (each
// costs tens of microseconds there): fuse.add, dst = product + add (the
// residual, after the attention, conv or FFN output); fuse.glu, dst =
// SwiGLU(product with src0, product with src0b), the FFN's gate and up,
// with fuse.stage_out also writing dst into the other staging buffer for
// the down product. Each column's arithmetic is the separate ops'.
static webgpu_encoded_op ggml_webgpu_mul_mat_vec_pvr_fused(webgpu_context &                                      ctx,
                                                           ggml_tensor *                                         src0,
                                                           ggml_tensor *                                         src0b,
                                                           ggml_tensor *                                         src1,
                                                           ggml_tensor *                                         add,
                                                           ggml_tensor *                                         dst,
                                                           const ggml_webgpu_mul_mat_vec_pvr_fusion & fuse) {
    const uint32_t k            = (uint32_t) src0->ne[0];
    const uint32_t m            = (uint32_t) dst->ne[0];
    const uint32_t n            = (uint32_t) src1->ne[1];
    const size_t   staged_bytes = ROUNDUP_POW2((size_t) k * n * sizeof(float), 16);

    std::vector<webgpu_dispatch_desc> dispatches;
    if (!ggml_webgpu_pvr_is_staged(ctx, src1, false)) {
        ggml_webgpu_pvr_stage_next(ctx, staged_bytes);
        wgpu::Buffer & staging_buf = ctx->pvr_staging_buf[ctx->pvr_staged_idx];
        dispatches.push_back({
            ctx->shader_lib->get_mul_mat_pvr_stage_pipeline(false),
            { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)), k,
              (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)), n },
            { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src1),
              ggml_webgpu_make_bind_group_entry(1, staging_buf, 0, staged_bytes) },
            { CEIL_DIV(k, WEBGPU_MUL_MAT_PVR_STAGE_WG_SIZE), n }
        });
    }
    ctx->pvr_staged_src1 = src1;
    ctx->pvr_staged_f16  = false;
    const int src1_idx   = ctx->pvr_staged_idx;

    const bool      sg       = ggml_webgpu_mul_mat_vec_pvr_subgroups(ctx->global_ctx);
    webgpu_pipeline pipeline = ctx->shader_lib->get_mul_mat_vec_pvr_pipeline(
        src0->type, true, n, sg, fuse,
        ggml_webgpu_shader_lib::ggml_webgpu_mul_mat_vec_pvr_choose_shape(src0->type, ggml_webgpu_pvr_is_repacked(src0), n,
                                                                         m, sg, fuse));
    auto * decisions = static_cast<ggml_webgpu_mul_mat_vec_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = {
        ggml_webgpu_pvr_src0_offset(ctx, src0),
        0u,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        m,
        n,
        k,
        ggml_webgpu_pvr_src0_stride(src0),
        k,
        // stride_02: src0b's offset (GLU)
        src0b ? ggml_webgpu_pvr_src0_offset(ctx, src0b) : 0u,
        // stride_12: add's offset (ADD)
        add && !fuse.add_in_dst ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, add) / ggml_type_size(add->type)) : 0u,
        0u,
        0u,
        1u,
        1u,
        1u,
        1u
    };
    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0),
        ggml_webgpu_make_bind_group_entry(1, ctx->pvr_staging_buf[src1_idx], 0, staged_bytes),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst),
    };
    // the fused-in bindings follow dst in this order (see mul_mat_vec_pvr.wgsl)
    if (fuse.add && !fuse.add_in_dst) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, entries.size(), add));
    }
    if (fuse.glu) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, entries.size(), src0b));
    }
    if (fuse.stage_out) {
        // dst, packed, becomes the staged src1 of the next product
        const size_t out_bytes = ROUNDUP_POW2((size_t) m * n * sizeof(float), 16);
        ggml_webgpu_pvr_stage_next(ctx, out_bytes);
        entries.push_back(ggml_webgpu_make_bind_group_entry(entries.size(), ctx->pvr_staging_buf[ctx->pvr_staged_idx], 0, out_bytes));
        ctx->pvr_staged_src1  = dst;
        ctx->pvr_staged_f16   = false;
        ctx->pvr_staged_by_op = true;
    }
    uint32_t wg_x = 1;
    uint32_t wg_y = 1;
    compute_2d_workgroups(CEIL_DIV(m, decisions->outputs_per_wg),
                          ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    dispatches.push_back({ pipeline, std::move(params), std::move(entries), { wg_x, wg_y } });
    return ggml_backend_webgpu_build_multi(ctx, dispatches);
}

// A product (PowerVR mat-vec) followed by an ADD of it and another tensor
// of its shape: returns the other tensor, or nullptr when they can't fuse.
// in_dst: the other tensor is the ADD's output itself (computed in place).
static ggml_tensor * ggml_webgpu_mul_mat_pvr_add_operand(webgpu_context &           ctx,
                                                         const struct ggml_cgraph * cgraph,
                                                         int                        node_idx,
                                                         bool &                     in_dst) {
    if (!ggml_webgpu_is_pvr(ctx->global_ctx) || !ggml_can_fuse(cgraph, node_idx, { GGML_OP_MUL_MAT, GGML_OP_ADD })) {
        return nullptr;
    }
    ggml_tensor * mm  = cgraph->nodes[node_idx];
    ggml_tensor * add = cgraph->nodes[node_idx + 1];
    if (!ggml_webgpu_mul_mat_vec_pvr_applies(ctx, mm->src[0], mm->src[1], mm)) {
        return nullptr;
    }
    ggml_tensor * other = add->src[0] == mm ? add->src[1] : add->src[0];
    if ((add->src[0] != mm && add->src[1] != mm) || other == mm) {
        return nullptr;
    }
    if (other->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || !ggml_are_same_shape(other, mm) ||
        !ggml_are_same_shape(add, mm) || !ggml_is_contiguous(other) || !ggml_is_contiguous(add) ||
        ggml_webgpu_tensor_buf(mm->src[0]).Get() == ggml_webgpu_tensor_buf(add).Get()) {
        return nullptr;
    }
    in_dst = false;
    if (ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, other, add)) {
        // computed in place: read the operand through dst's binding
        if (ggml_webgpu_tensor_buf(other).Get() != ggml_webgpu_tensor_buf(add).Get() ||
            ggml_webgpu_tensor_offset(other) != ggml_webgpu_tensor_offset(add)) {
            return nullptr;
        }
        in_dst = true;
    }
    return other;
}

// The FFN's gate and up products (PowerVR mat-vecs of one src1) followed by
// the SwiGLU of the two: returns the GLU node's index offset (2), and sets
// gate and up, or returns 0 when they can't fuse.
static int ggml_webgpu_mul_mat_pvr_glu_nodes(webgpu_context &           ctx,
                                             const struct ggml_cgraph * cgraph,
                                             int                        node_idx,
                                             ggml_tensor *&             gate,
                                             ggml_tensor *&             up) {
    if (!ggml_webgpu_is_pvr(ctx->global_ctx)) {
        return 0;
    }
    const enum ggml_op ops[3]  = { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU };
    const int          out_idx = node_idx + 2;
    // the two products are only read by the GLU
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, 3, ops, &out_idx, 1)) {
        return 0;
    }
    ggml_tensor * mm0 = cgraph->nodes[node_idx];
    ggml_tensor * mm1 = cgraph->nodes[node_idx + 1];
    ggml_tensor * glu = cgraph->nodes[node_idx + 2];
    if (ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || glu->src[1] == nullptr || ggml_get_op_params_i32(glu, 1) != 0) {
        return 0;
    }
    if (!((glu->src[0] == mm0 && glu->src[1] == mm1) || (glu->src[0] == mm1 && glu->src[1] == mm0))) {
        return 0;
    }
    gate = glu->src[0];
    up   = glu->src[1];
    if (mm0->src[1] != mm1->src[1] && !(ggml_webgpu_tensor_buf(mm0->src[1]).Get() == ggml_webgpu_tensor_buf(mm1->src[1]).Get() &&
                                        ggml_webgpu_tensor_offset(mm0->src[1]) == ggml_webgpu_tensor_offset(mm1->src[1]) &&
                                        ggml_are_same_shape(mm0->src[1], mm1->src[1]) && mm0->src[1]->nb[1] == mm1->src[1]->nb[1])) {
        return 0;
    }
    if (gate->src[0]->type != up->src[0]->type || !ggml_are_same_shape(gate->src[0], up->src[0]) ||
        gate->src[0]->nb[1] != up->src[0]->nb[1] ||
        ggml_webgpu_pvr_is_repacked(gate->src[0]) != ggml_webgpu_pvr_is_repacked(up->src[0]) ||
        !ggml_webgpu_mul_mat_vec_pvr_applies(ctx, gate->src[0], gate->src[1], gate) ||
        !ggml_webgpu_mul_mat_vec_pvr_applies(ctx, up->src[0], up->src[1], up)) {
        return 0;
    }
    if (glu->type != GGML_TYPE_F32 || !ggml_are_same_shape(glu, gate) || !ggml_is_contiguous(glu) ||
        ggml_webgpu_tensor_buf(gate->src[0]).Get() == ggml_webgpu_tensor_buf(glu).Get()) {
        return 0;
    }
    return 2;
}

// A prompt batch's FFN on PowerVR: the gate or up product followed by the
// SwiGLU that takes it and the other one (already computed). The product's
// shader applies the SwiGLU instead of writing its result, which saves
// writing and reading back two activations as large as the FFN, and buffer
// writes are slow there.
static bool ggml_webgpu_can_fuse_mul_mat_pvr_swiglu(webgpu_context &           ctx,
                                                    const struct ggml_cgraph * cgraph,
                                                    int                        node_idx) {
    if (!ggml_webgpu_is_pvr(ctx->global_ctx) || !ggml_can_fuse(cgraph, node_idx, { GGML_OP_MUL_MAT, GGML_OP_GLU })) {
        return false;
    }
    ggml_tensor * mm  = cgraph->nodes[node_idx];
    ggml_tensor * glu = cgraph->nodes[node_idx + 1];
    if (ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || glu->src[1] == nullptr || ggml_get_op_params_i32(glu, 1) != 0) {
        return false;  // not split, or swapped
    }
    ggml_tensor * other = glu->src[0] == mm ? glu->src[1] : glu->src[0];
    if (glu->src[0] != mm && glu->src[1] != mm) {
        return false;
    }
    if (!ggml_webgpu_can_use_mul_mat_pvr(ctx, mm->src[0], mm->src[1], mm) ||
        !ggml_webgpu_mul_mat_pvr_use_gemm(ctx, mm->src[0], mm->src[1])) {
        return false;
    }
    // the other operand and the output laid out as the product, the output
    // bound apart from the other operand
    return other->type == GGML_TYPE_F32 && glu->type == GGML_TYPE_F32 && ggml_are_same_shape(other, mm) &&
           ggml_are_same_shape(glu, mm) && ggml_is_contiguous(other) && ggml_is_contiguous(glu) &&
           !ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, other, glu);
}

static webgpu_encoded_op ggml_webgpu_mul_mat(webgpu_context & ctx,
                                             ggml_tensor *    src0,
                                             ggml_tensor *    src1,
                                             ggml_tensor *    dst) {
    if (ggml_webgpu_can_use_mul_mat_pvr(ctx, src0, src1, dst)) {
        return ggml_webgpu_mul_mat_pvr(ctx, src0, src1, dst);
    }
    // only the PowerVR shaders read repacked weights (supports_op keeps the others away)
    GGML_ASSERT(!ggml_webgpu_pvr_is_repacked(src0));
    ctx->pvr_staged_src1 = nullptr;

    // PowerVR: a product of 2 to 8 columns (a speculative batch) that the PowerVR shaders don't take
    // (weights of other types) runs as one-column products, so that each column's arithmetic is the
    // one-token decode's: the generic shaders' batched paths sum in other orders, and speculative
    // decoding would then write other text than decoding one token at a time
    if (ggml_webgpu_is_pvr(ctx->global_ctx) && dst->ne[1] >= 2 && dst->ne[1] <= WEBGPU_MUL_MAT_VEC_PVR_MAX_COLS &&
        dst->ne[2] == 1 && dst->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 && src0->ne[2] == 1 &&
        src0->ne[3] == 1) {
        if (!ctx->pvr_split_logged) {
            ctx->pvr_split_logged = true;
            GGML_LOG_INFO("ggml_webgpu: PowerVR: %s products of 2 to %u columns run a column at a time\n",
                          ggml_type_name(src0->type), WEBGPU_MUL_MAT_VEC_PVR_MAX_COLS);
        }
        webgpu_encoded_op result = {};
        for (int64_t c = 0; c < dst->ne[1]; c++) {
            // column c of src1 and dst, as views
            ggml_tensor src1_c = *src1;
            ggml_tensor dst_c  = *dst;
            for (auto [view, t] : { std::pair<ggml_tensor *, ggml_tensor *>{ &src1_c, src1 }, { &dst_c, dst } }) {
                view->view_src  = t->view_src ? t->view_src : t;
                view->view_offs = t->view_offs + c * t->nb[1];
                view->ne[1]     = 1;
            }
            webgpu_encoded_op op = ggml_webgpu_mul_mat(ctx, src0, &src1_c, &dst_c);
            // (the PowerVR shaders, which take a column of weights that share dst's buffer, note the
            // staged src1 by pointer: the next column's view is another tensor at that address)
            ctx->pvr_staged_src1 = nullptr;
            result.num_kernels += op.num_kernels;
#ifdef GGML_WEBGPU_GPU_PROFILE
            result.pipeline_names.insert(result.pipeline_names.end(), op.pipeline_names.begin(),
                                         op.pipeline_names.end());
#endif
        }
        return result;
    }

    // Determine if this is a mat-vec operation
    bool use_mat_vec = (dst->ne[1] <= GGML_WEBGPU_MUL_MAT_VEC_MAX_ROWS);

    // use MMVQ path for mat-vec
    bool use_mmvq = ggml_webgpu_can_use_mmvq(
        src0, src1, src1->ne[1], ctx->global_ctx->capabilities.supports_dot_product, ctx->global_ctx->vendor);

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};

    shader_lib_ctx.src0                     = src0;
    shader_lib_ctx.src1                     = src1;
    shader_lib_ctx.dst                      = dst;
    shader_lib_ctx.max_wg_size              = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    shader_lib_ctx.supports_subgroups       = ctx->global_ctx->capabilities.supports_subgroups;
    shader_lib_ctx.supports_subgroup_matrix = ctx->global_ctx->capabilities.supports_subgroup_matrix;
    shader_lib_ctx.sg_mat_m                 = ctx->global_ctx->capabilities.sg_mat_m;
    shader_lib_ctx.sg_mat_n                 = ctx->global_ctx->capabilities.sg_mat_n;
    shader_lib_ctx.sg_mat_k                 = ctx->global_ctx->capabilities.sg_mat_k;
    shader_lib_ctx.min_subgroup_size        = ctx->global_ctx->capabilities.min_subgroup_size;
    shader_lib_ctx.max_subgroup_size        = ctx->global_ctx->capabilities.max_subgroup_size;
    shader_lib_ctx.supports_dot_product     = ctx->global_ctx->capabilities.supports_dot_product;
    shader_lib_ctx.vendor                   = ctx->global_ctx->vendor;

    // Get or create pipeline
    webgpu_pipeline                   pipeline;
    std::vector<webgpu_dispatch_desc> dispatches;
    const bool src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1) && !use_mmvq;

    if (use_mat_vec) {
        if (use_mmvq) {
            ggml_webgpu_quantize_q8_dispatch(ctx, src0, src1, dst, dispatches);
        }
        pipeline = ctx->shader_lib->get_mul_mat_vec_pipeline(shader_lib_ctx, src_overlap);
    } else {
        pipeline = ctx->shader_lib->get_mul_mat_fast_pipeline(shader_lib_ctx, src_overlap);
    }

    uint32_t offset_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type));
    uint32_t offset_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type));
    size_t   merged_offset = 0;
    size_t   merged_size   = 0;
    if (src_overlap) {
        const ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
        merged_offset = merged_range.offset;
        merged_size   = merged_range.size;
        offset_src0   = ggml_webgpu_tensor_merged_element_offset(src0, merged_range);
        offset_src1   = ggml_webgpu_tensor_merged_element_offset(src1, merged_range);
    }

    // Build params
    std::vector<uint32_t> params = { offset_src0,
                                     offset_src1,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
                                     (uint32_t) dst->ne[0],
                                     (uint32_t) dst->ne[1],
                                     (uint32_t) src0->ne[0],
                                     (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
                                     (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
                                     (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
                                     (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
                                     (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),
                                     (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)),
                                     (uint32_t) src0->ne[2],
                                     (uint32_t) src0->ne[3],
                                     (uint32_t) (src1->ne[2] / src0->ne[2]),
                                     (uint32_t) (src1->ne[3] / src0->ne[3]) };

    // Build bind group entries
    std::vector<wgpu::BindGroupEntry> entries = {};
    if (use_mmvq) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0));
        auto & mmvq_qq8_entry = dispatches[0].bind_group_entries[1];
        entries.push_back(ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(dst), mmvq_qq8_entry.offset,
                                                            mmvq_qq8_entry.size));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst));
    } else if (src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0), merged_offset, merged_size));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    } else {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst));
    }

    // Calculate workgroup dimensions
    uint32_t       wg_x           = 1;
    uint32_t       wg_y           = 1;
    const uint32_t max_wg_per_dim = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;

    if (use_mat_vec) {
        auto * decisions = static_cast<ggml_webgpu_mul_mat_vec_shader_decisions *>(pipeline.context.get());

        uint32_t batches       = dst->ne[2] * dst->ne[3];
        uint32_t output_groups = CEIL_DIV(dst->ne[0], decisions->outputs_per_wg);
        uint32_t total_wg      = output_groups * batches;
        compute_2d_workgroups(total_wg, max_wg_per_dim, wg_x, wg_y);
    } else {
        auto * decisions = static_cast<ggml_webgpu_mul_mat_shader_decisions *>(pipeline.context.get());

        // Fast-path tiled/subgroup calculations
        uint32_t wg_m;
        uint32_t wg_n;
        if (decisions->use_subgroup_matrix) {
            uint32_t wg_m_sg_tile =
                decisions->subgroup_m * decisions->subgroup_matrix_m * ctx->global_ctx->capabilities.sg_mat_m;
            wg_m = CEIL_DIV(dst->ne[0], wg_m_sg_tile);
            uint32_t wg_n_sg_tile =
                decisions->subgroup_n * decisions->subgroup_matrix_n * ctx->global_ctx->capabilities.sg_mat_n;
            wg_n = CEIL_DIV(dst->ne[1], wg_n_sg_tile);
        } else {
            uint32_t tile_m_s = decisions->tile_m * decisions->wg_size_m;
            uint32_t tile_n_s = decisions->tile_n * decisions->wg_size_n;
            wg_m              = CEIL_DIV(dst->ne[0], tile_m_s);
            wg_n              = CEIL_DIV(dst->ne[1], tile_n_s);
        }
        uint32_t total_wg = wg_m * wg_n * dst->ne[2] * dst->ne[3];
        compute_2d_workgroups(total_wg, max_wg_per_dim, wg_x, wg_y);
    }

    dispatches.push_back({
        pipeline, std::move(params), std::move(entries), { wg_x, wg_y }
    });

    return ggml_backend_webgpu_build_multi(ctx, dispatches);
}

static webgpu_encoded_op ggml_webgpu_mul_mat_id_vec(webgpu_context & ctx,
                                                    ggml_tensor *    src0,
                                                    ggml_tensor *    src1,
                                                    ggml_tensor *    src2,
                                                    ggml_tensor *    dst) {
    const uint32_t param_n_expert      = (uint32_t) src0->ne[2];
    const uint32_t param_n_expert_used = (uint32_t) dst->ne[1];

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.supports_subgroups             = ctx->global_ctx->capabilities.supports_subgroups;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_mul_mat_id_vec_pipeline(shader_lib_ctx);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_size(src2->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        param_n_expert,
        param_n_expert_used,
        (uint32_t) src1->ne[1],
        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0), ggml_webgpu_tensor_align_offset(ctx, src0),
                                          ggml_webgpu_tensor_binding_size(ctx, src0)),
        ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(src1), ggml_webgpu_tensor_align_offset(ctx, src1),
                                          ggml_webgpu_tensor_binding_size(ctx, src1)),
        ggml_webgpu_make_bind_group_entry(2, ggml_webgpu_tensor_buf(src2), ggml_webgpu_tensor_align_offset(ctx, src2),
                                          ggml_webgpu_tensor_binding_size(ctx, src2)),
        ggml_webgpu_make_bind_group_entry(3, ggml_webgpu_tensor_buf(dst), ggml_webgpu_tensor_align_offset(ctx, dst),
                                          ggml_webgpu_tensor_binding_size(ctx, dst)),
    };

    uint32_t wg_x = 1;
    uint32_t wg_y = 1;

    auto * decisions = static_cast<ggml_webgpu_mul_mat_vec_shader_decisions *>(pipeline.context.get());

    const uint32_t max_wg_per_dim = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;
    uint32_t       output_groups  = CEIL_DIV(dst->ne[0], decisions->outputs_per_wg);
    uint32_t       total_wg       = output_groups * param_n_expert_used;
    compute_2d_workgroups(total_wg, max_wg_per_dim, wg_x, wg_y);

    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// upper bound on the number of slot tiles emitted by mul_mat_id_gather.wgsl
static uint32_t ggml_webgpu_mul_mat_id_max_tiles(const ggml_tensor * dst) {
    const uint32_t n_expert = (uint32_t) dst->src[0]->ne[2];
    const uint32_t n_slots  = (uint32_t) (dst->ne[1] * dst->ne[2]);  // n_expert_used * n_tokens

    return CEIL_DIV(n_slots, WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_SIZE_N) + std::min(n_expert, n_slots);
}

// tile count followed by (expert, first slot, n slots) per tile
static size_t ggml_webgpu_mul_mat_id_tiles_nbytes(const ggml_tensor * dst) {
    return (1 + 3 * (size_t) ggml_webgpu_mul_mat_id_max_tiles(dst)) * sizeof(uint32_t);
}

static webgpu_encoded_op ggml_webgpu_mul_mat_id(webgpu_context & ctx,
                                                ggml_tensor *    src0,
                                                ggml_tensor *    src1,
                                                ggml_tensor *    src2,
                                                ggml_tensor *    dst) {
    // we can use mat-vec fast path
    if (dst->ne[2] == 1) {
        return ggml_webgpu_mul_mat_id_vec(ctx, src0, src1, src2, dst);
    }

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    // Get or create pipeline
    webgpu_pipeline gather_pipeline;
    webgpu_pipeline main_pipeline;

    std::vector<webgpu_dispatch_desc> dispatches;

    gather_pipeline = ctx->shader_lib->get_mul_mat_id_gather_pipeline(shader_lib_ctx);
    main_pipeline   = ctx->shader_lib->get_mul_mat_id_pipeline(shader_lib_ctx);

    const uint32_t param_n_expert      = (uint32_t) src0->ne[2];
    const uint32_t param_n_expert_used = (uint32_t) dst->ne[1];
    const uint32_t param_n_tokens      = (uint32_t) dst->ne[2];

    // params for mul_mat_id_gather.wgsl
    std::vector<uint32_t> gather_params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_size(src2->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        param_n_expert,
        param_n_expert_used,
        param_n_tokens,
        (uint32_t) (src2->nb[1] / ggml_type_size(src2->type)),
        (uint32_t) dst->ne[0],
    };

    const size_t dst_offset       = ggml_webgpu_tensor_offset(dst);
    const size_t ids_cache_nbytes = src2->ne[0] * src1->ne[2] * sizeof(uint32_t);
    const size_t ids_tiles_nbytes = ggml_webgpu_mul_mat_id_tiles_nbytes(dst);

    const size_t ids_cache_align_offset = ROUNDUP_POW2(
        dst_offset + ggml_nbytes(dst), ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment);
    const size_t ids_tiles_align_offset =
        ROUNDUP_POW2(ids_cache_align_offset + ids_cache_nbytes,
                     ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment);

    const size_t ids_cache_bind_size = ROUNDUP_POW2(ids_cache_nbytes, WEBGPU_STORAGE_BUF_BINDING_MULT);
    const size_t ids_tiles_bind_size = ROUNDUP_POW2(ids_tiles_nbytes, WEBGPU_STORAGE_BUF_BINDING_MULT);

    // bind group entries for mul_mat_id_gather.wgsl
    std::vector<wgpu::BindGroupEntry> gather_entries = {
        ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src2), ggml_webgpu_tensor_align_offset(ctx, src2),
                                          ggml_webgpu_tensor_binding_size(ctx, src2)),
        ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(dst), ids_cache_align_offset, ids_cache_bind_size),
        ggml_webgpu_make_bind_group_entry(2, ggml_webgpu_tensor_buf(dst), ids_tiles_align_offset, ids_tiles_bind_size),
        ggml_webgpu_make_bind_group_entry(3, ggml_webgpu_tensor_buf(dst), ggml_webgpu_tensor_align_offset(ctx, dst),
                                          ggml_webgpu_tensor_binding_size(ctx, dst)),
    };

    dispatches.push_back({
        gather_pipeline, std::move(gather_params), std::move(gather_entries), { 1, 1 }
    });

    // params for mul_mat_id.wgsl
    std::vector<uint32_t> main_params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        param_n_expert,
        param_n_expert_used,
        param_n_tokens,
        (uint32_t) src1->ne[1],
        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
    };

    // bind group entries for mul_mat_id.wgsl
    std::vector<wgpu::BindGroupEntry> main_entries = {
        ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0), ggml_webgpu_tensor_align_offset(ctx, src0),
                                          ggml_webgpu_tensor_binding_size(ctx, src0)),
        ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(src1), ggml_webgpu_tensor_align_offset(ctx, src1),
                                          ggml_webgpu_tensor_binding_size(ctx, src1)),
        ggml_webgpu_make_bind_group_entry(2, ggml_webgpu_tensor_buf(dst), ggml_webgpu_tensor_align_offset(ctx, dst),
                                          ggml_webgpu_tensor_binding_size(ctx, dst)),
        ggml_webgpu_make_bind_group_entry(3, ggml_webgpu_tensor_buf(dst), ids_cache_align_offset, ids_cache_bind_size),
        ggml_webgpu_make_bind_group_entry(4, ggml_webgpu_tensor_buf(dst), ids_tiles_align_offset, ids_tiles_bind_size),
    };

    // Calculate workgroup dimensions
    uint32_t wg_x = 1;
    uint32_t wg_y = 1;

    auto * main_decisions = static_cast<ggml_webgpu_mul_mat_shader_decisions *>(main_pipeline.context.get());

    // must match `TILE_N_SLOTS` of mul_mat_id_gather.wgsl
    GGML_ASSERT(main_decisions->tile_n * main_decisions->wg_size_n == WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_SIZE_N);

    const uint32_t tile_m_s = main_decisions->tile_m * main_decisions->wg_size_m;
    const uint32_t wg_m     = CEIL_DIV(dst->ne[0], tile_m_s);
    const uint32_t total_wg = wg_m * ggml_webgpu_mul_mat_id_max_tiles(dst);

    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);

    dispatches.push_back({
        main_pipeline, std::move(main_params), std::move(main_entries), { wg_x, wg_y }
    });

    return ggml_backend_webgpu_build_multi(ctx, dispatches);
}

struct ggml_webgpu_flash_attn_op {
    ggml_webgpu_shader_lib_context    shader_lib_ctx = {};
    std::vector<uint32_t>             params;
    std::vector<wgpu::BindGroupEntry> entries;
    size_t                            kv_bind_offset = 0;
    size_t                            kv_bind_size   = 0;
    bool                              has_mask       = false;
    bool                              has_sinks      = false;
    bool                              kv_overlap     = false;
};

static bool ggml_webgpu_flash_attn_use_vec_path(const webgpu_global_context & global_ctx,
                                                const ggml_tensor *           Q,
                                                const ggml_tensor *           K,
                                                const ggml_tensor *           V,
                                                int64_t                       n_q) {
    const size_t storage_offset_alignment = global_ctx->capabilities.limits.minStorageBufferOffsetAlignment;

    const bool k_float_vec4_aligned = (K->type != GGML_TYPE_F16 && K->type != GGML_TYPE_F32) ||
                                      ggml_webgpu_flash_attn_float_vec4_aligned(K, storage_offset_alignment);
    const bool v_float_vec4_aligned = (V->type != GGML_TYPE_F16 && V->type != GGML_TYPE_F32) ||
                                      ggml_webgpu_flash_attn_float_vec4_aligned(V, storage_offset_alignment);

    const uint32_t k_vec_head_align =
        ggml_is_quantized(K->type) ? ggml_blck_size(K->type) : GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH;
    const uint32_t v_vec_head_align =
        ggml_is_quantized(V->type) ? ggml_blck_size(V->type) : GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH;
    const bool kv_vec_head_dims_aligned = Q->ne[0] % k_vec_head_align == 0 && V->ne[0] % v_vec_head_align == 0;

    return global_ctx->capabilities.supports_subgroups && (n_q < GGML_WEBGPU_FLASH_ATTN_VEC_MAX_SEQ_LEN) &&
           kv_vec_head_dims_aligned && k_float_vec4_aligned && v_float_vec4_aligned;
}

static ggml_webgpu_flash_attn_op ggml_webgpu_flash_attn_prepare(webgpu_context & ctx,
                                                                ggml_tensor *    Q,
                                                                ggml_tensor *    K,
                                                                ggml_tensor *    V,
                                                                ggml_tensor *    mask,
                                                                ggml_tensor *    sinks,
                                                                ggml_tensor *    dst) {
    float scale         = ggml_get_op_params_f32(dst, 0);
    float max_bias      = ggml_get_op_params_f32(dst, 1);
    float logit_softcap = ggml_get_op_params_f32(dst, 2);
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }
    float n_head_log2 = float(1u << (uint32_t) floor(log2(Q->ne[2])));
    float m0          = powf(2.0f, -(max_bias) / n_head_log2);
    float m1          = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    ggml_webgpu_flash_attn_op op               = {};
    op.shader_lib_ctx.src0                     = Q;
    op.shader_lib_ctx.src1                     = K;
    op.shader_lib_ctx.src2                     = V;
    op.shader_lib_ctx.src3                     = mask;
    op.shader_lib_ctx.src4                     = sinks;
    op.shader_lib_ctx.dst                      = dst;
    op.shader_lib_ctx.supports_subgroups       = ctx->global_ctx->capabilities.supports_subgroups;
    op.shader_lib_ctx.supports_subgroup_matrix = ctx->global_ctx->capabilities.supports_subgroup_matrix;
    op.shader_lib_ctx.max_wg_size              = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    op.shader_lib_ctx.wg_mem_limit_bytes       = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupStorageSize;
    op.shader_lib_ctx.sg_mat_m                 = ctx->global_ctx->capabilities.sg_mat_m;
    op.shader_lib_ctx.sg_mat_n                 = ctx->global_ctx->capabilities.sg_mat_n;
    op.shader_lib_ctx.sg_mat_k                 = ctx->global_ctx->capabilities.sg_mat_k;
    op.shader_lib_ctx.min_subgroup_size        = ctx->global_ctx->capabilities.min_subgroup_size;
    op.shader_lib_ctx.max_subgroup_size        = ctx->global_ctx->capabilities.max_subgroup_size;

    op.has_mask   = mask != nullptr;
    op.has_sinks  = sinks != nullptr;
    op.kv_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, K, V);

    uint32_t offset_k = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, K) / ggml_type_size(K->type));
    uint32_t offset_v = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, V) / ggml_type_size(V->type));
    if (op.kv_overlap) {
        const ggml_webgpu_merged_binding_range merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { K, V });
        op.kv_bind_offset                                   = merged_range.offset;
        op.kv_bind_size                                     = merged_range.size;
        offset_k                                            = ggml_webgpu_tensor_merged_element_offset(K, merged_range);
        offset_v                                            = ggml_webgpu_tensor_merged_element_offset(V, merged_range);
    }

    op.params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, Q) / ggml_type_size(Q->type)),
        offset_k,
        offset_v,
        op.has_mask ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, mask) / ggml_type_size(mask->type)) : 0,
        op.has_sinks ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, sinks) / ggml_type_size(sinks->type)) : 0,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) Q->ne[2],                              // number of heads
        (uint32_t) Q->ne[1],                              // sequence length (Q)
        (uint32_t) K->ne[1],                              // sequence length (K/V)
        (uint32_t) (Q->nb[1] / ggml_type_size(Q->type)),  // stride (elements/blocks) of Q in dimension 1
        (uint32_t) (Q->nb[2] / ggml_type_size(Q->type)),  // stride (elements/blocks) of Q in dimension 2
        (uint32_t) (Q->nb[3] / ggml_type_size(Q->type)),  // stride (elements/blocks) of Q in dimension 3
        (uint32_t) (K->nb[1] / ggml_type_size(K->type)),  // stride (elements/blocks) of K in dimension 1
        (uint32_t) (K->nb[2] / ggml_type_size(K->type)),  // stride (elements/blocks) of K in dimension 2
        (uint32_t) (K->nb[3] / ggml_type_size(K->type)),  // stride (elements/blocks) of K in dimension 3
        (uint32_t) (V->nb[1] / ggml_type_size(V->type)),  // stride (elements/blocks) of V in dimension 1
        (uint32_t) (V->nb[2] / ggml_type_size(V->type)),  // stride (elements/blocks) of V in dimension 2
        (uint32_t) (V->nb[3] / ggml_type_size(V->type)),  // stride (elements/blocks) of V in dimension 3
        op.has_mask ? (uint32_t) (mask->nb[3] / ggml_type_size(mask->type)) : 0,  // stride of mask dim 3
        (uint32_t) (Q->ne[2] / K->ne[2]),  // repeat factor for K/V in dim 2 (MHA/MQA/GQA)
        ggml_webgpu_u32_from_f32(scale),   // scale (possibly adjusted for logit softcap)
        ggml_webgpu_u32_from_f32(max_bias),
        ggml_webgpu_u32_from_f32(logit_softcap),
        ggml_webgpu_u32_from_f32(n_head_log2),
        ggml_webgpu_u32_from_f32(m0),
        ggml_webgpu_u32_from_f32(m1)
    };
    op.entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, Q),
    };
    if (op.kv_overlap) {
        op.entries.push_back(
            ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(K), op.kv_bind_offset, op.kv_bind_size));
    } else {
        op.entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, K));
        op.entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, V));
    }
    uint32_t binding_index = op.kv_overlap ? 2u : 3u;
    if (op.has_mask) {
        op.entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_index++, mask));
    }
    if (op.has_sinks) {
        op.entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_index++, sinks));
    }
    op.entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_index++, dst));

    return op;
}

static uint32_t ggml_webgpu_flash_attn_vec_nwg(uint32_t vec_nwg_cap, uint32_t kv_tile, uint32_t seq_len_kv) {
    uint32_t       nwg     = 1u;
    const uint64_t kv_span = (uint64_t) kv_tile;
    while ((2u * nwg * kv_span) < (uint64_t) seq_len_kv && nwg < vec_nwg_cap) {
        nwg <<= 1;
    }
    return std::min(nwg, vec_nwg_cap);
}

static webgpu_encoded_op ggml_webgpu_flash_attn_direct(webgpu_context & ctx, const ggml_webgpu_flash_attn_op & op) {
    webgpu_pipeline pipeline    = ctx->shader_lib->get_flash_attn_pipeline(op.shader_lib_ctx, op.kv_overlap);
    auto *          decisions   = static_cast<ggml_webgpu_flash_attn_decisions *>(pipeline.context.get());
    uint32_t        wg_per_head = CEIL_DIV(op.shader_lib_ctx.src0->ne[1], decisions->q_tile);
    uint32_t        wg_x        = wg_per_head * op.shader_lib_ctx.src0->ne[2] * op.shader_lib_ctx.src0->ne[3];
    return ggml_backend_webgpu_build(ctx, pipeline, op.params, op.entries, wg_x);
}

static webgpu_encoded_op ggml_webgpu_flash_attn_vec(webgpu_context &          ctx,
                                                    ggml_tensor *             Q,
                                                    ggml_tensor *             K,
                                                    ggml_tensor *             V,
                                                    ggml_tensor *             mask,
                                                    ggml_tensor *             sinks,
                                                    ggml_tensor *             dst,
                                                    ggml_webgpu_flash_attn_op op) {
    webgpu_pipeline pipeline  = ctx->shader_lib->get_flash_attn_vec_pipeline(op.shader_lib_ctx, op.kv_overlap);
    auto *          decisions = static_cast<ggml_webgpu_flash_attn_vec_decisions *>(pipeline.context.get());

    wgpu::Buffer blk_buf         = {};
    uint64_t     blk_size_bytes  = 0;
    uint32_t     blk_nblk0       = 0;
    uint32_t     blk_nblk1       = 0;
    uint32_t     blk_batch_count = 0;

    const uint32_t vec_nwg_cap = ctx->global_ctx->capabilities.min_subgroup_size;
    uint32_t       nwg         = ggml_webgpu_flash_attn_vec_nwg(vec_nwg_cap, decisions->kv_tile, (uint32_t) K->ne[1]);
    const uint64_t nrows       = (uint64_t) Q->ne[1] * Q->ne[2] * Q->ne[3];
    const bool     use_vec_reduce = nwg > 1u;
    GGML_ASSERT(nrows <= UINT32_MAX);

    uint64_t     tmp_stats_base  = 0;
    uint64_t     tmp_size_bytes  = 0;
    wgpu::Buffer tmp_buf         = {};
    uint64_t     tmp_bind_offset = 0;
    uint64_t     tmp_bind_size   = 0;
    const size_t align_bytes     = ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment;
    const size_t dst_offset      = ggml_webgpu_tensor_offset(dst);
    size_t       scratch_offset  = ROUNDUP_POW2(dst_offset + ggml_nbytes(dst), align_bytes);

    if (use_vec_reduce) {
        const uint64_t tmp_data_elems  = nrows * (uint64_t) V->ne[0] * nwg;
        const uint64_t tmp_stats_elems = nrows * 2u * nwg;
        tmp_stats_base                 = tmp_data_elems;
        tmp_size_bytes =
            ROUNDUP_POW2((tmp_data_elems + tmp_stats_elems) * sizeof(float), WEBGPU_STORAGE_BUF_BINDING_MULT);
        GGML_ASSERT(tmp_stats_base <= UINT32_MAX);
        tmp_buf         = ggml_webgpu_tensor_buf(dst);
        tmp_bind_offset = scratch_offset;
        tmp_bind_size   = tmp_size_bytes;
        scratch_offset  = ROUNDUP_POW2(scratch_offset + tmp_size_bytes, align_bytes);
    } else {
        // nwg==1 writes final dst directly in vec-split; bind tmp to a tiny non-overlapping scratch region.
        tmp_size_bytes  = WEBGPU_STORAGE_BUF_BINDING_MULT;
        tmp_buf         = ggml_webgpu_tensor_buf(dst);
        tmp_bind_offset = scratch_offset;
        tmp_bind_size   = tmp_size_bytes;
        scratch_offset  = ROUNDUP_POW2(scratch_offset + tmp_size_bytes, align_bytes);
    }

    webgpu_pipeline                   blk_pipeline;
    std::vector<uint32_t>             blk_params;
    std::vector<wgpu::BindGroupEntry> blk_entries;
    if (op.has_mask) {
        blk_nblk0                   = CEIL_DIV((uint32_t) K->ne[1], decisions->kv_tile);
        blk_nblk1                   = (uint32_t) Q->ne[1];
        blk_buf                     = ggml_webgpu_tensor_buf(dst);
        const uint32_t stride_mask3 = (uint32_t) (mask->nb[3] / ggml_type_size(mask->type));
        blk_batch_count             = stride_mask3 > 0 ? (uint32_t) Q->ne[3] : 1u;
        const uint64_t blk_elems    = (uint64_t) blk_nblk0 * blk_nblk1 * blk_batch_count;
        blk_size_bytes              = ROUNDUP_POW2(blk_elems * sizeof(uint32_t), WEBGPU_STORAGE_BUF_BINDING_MULT);
        const ggml_webgpu_shader_lib_context blk_shader_ctx = op.shader_lib_ctx;
        blk_pipeline = ctx->shader_lib->get_flash_attn_blk_pipeline(blk_shader_ctx, decisions->kv_tile);

        blk_params = {
            (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, mask) / ggml_type_size(mask->type)),  // offset_mask
            (uint32_t) Q->ne[1],                                                                   // seq_len_q
            (uint32_t) K->ne[1],                                                                   // seq_len_kv
            stride_mask3,                                                                          // stride_mask3
            blk_nblk0,                                                                             // nblk0
            blk_nblk1,                                                                             // nblk1
        };
        blk_entries = {
            ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(mask),
                                              ggml_webgpu_tensor_align_offset(ctx, mask),
                                              ggml_webgpu_tensor_binding_size(ctx, mask)),
            ggml_webgpu_make_bind_group_entry(1, blk_buf, scratch_offset, blk_size_bytes),
        };
        scratch_offset = ROUNDUP_POW2(scratch_offset + blk_size_bytes, align_bytes);
    }

    std::vector<uint32_t> split_params = op.params;
    if (op.has_mask) {
        split_params.push_back(0u);                     // blk_base
        split_params.push_back(blk_nblk0);              // blk_nblk0
        split_params.push_back(blk_nblk1);              // blk_nblk1
    }
    split_params.push_back(0u);                         // tmp_data_base
    split_params.push_back((uint32_t) tmp_stats_base);  // tmp_stats_base
    split_params.push_back(nwg);                        // nwg

    std::vector<wgpu::BindGroupEntry> split_entries = {
        ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(Q), ggml_webgpu_tensor_align_offset(ctx, Q),
                                          ggml_webgpu_tensor_binding_size(ctx, Q)),
    };
    if (op.kv_overlap) {
        split_entries.push_back(
            ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(K), op.kv_bind_offset, op.kv_bind_size));
    } else {
        split_entries.push_back(ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(K),
                                                                  ggml_webgpu_tensor_align_offset(ctx, K),
                                                                  ggml_webgpu_tensor_binding_size(ctx, K)));
        split_entries.push_back(ggml_webgpu_make_bind_group_entry(2, ggml_webgpu_tensor_buf(V),
                                                                  ggml_webgpu_tensor_align_offset(ctx, V),
                                                                  ggml_webgpu_tensor_binding_size(ctx, V)));
    }
    uint32_t split_binding_index = op.kv_overlap ? 2u : 3u;
    if (op.has_mask) {
        split_entries.push_back(ggml_webgpu_make_bind_group_entry(split_binding_index++, ggml_webgpu_tensor_buf(mask),
                                                                  ggml_webgpu_tensor_align_offset(ctx, mask),
                                                                  ggml_webgpu_tensor_binding_size(ctx, mask)));
    }
    if (op.has_sinks) {
        split_entries.push_back(ggml_webgpu_make_bind_group_entry(split_binding_index++, ggml_webgpu_tensor_buf(sinks),
                                                                  ggml_webgpu_tensor_align_offset(ctx, sinks),
                                                                  ggml_webgpu_tensor_binding_size(ctx, sinks)));
    }
    if (op.has_mask) {
        split_entries.push_back(
            ggml_webgpu_make_bind_group_entry(split_binding_index++, blk_buf, blk_entries[1].offset, blk_size_bytes));
    }
    split_entries.push_back(
        ggml_webgpu_make_bind_group_entry(split_binding_index++, tmp_buf, tmp_bind_offset, tmp_bind_size));
    split_entries.push_back(ggml_webgpu_make_bind_group_entry(split_binding_index++, ggml_webgpu_tensor_buf(dst),
                                                              ggml_webgpu_tensor_align_offset(ctx, dst),
                                                              ggml_webgpu_tensor_binding_size(ctx, dst)));

    webgpu_pipeline                   reduce_pipeline;
    std::vector<uint32_t>             reduce_params;
    std::vector<wgpu::BindGroupEntry> reduce_entries;
    if (use_vec_reduce) {
        const uint32_t reduce_sg_size = ctx->global_ctx->capabilities.max_subgroup_size;
        const uint32_t reduce_wg_size = std::max(
            reduce_sg_size,
            (uint32_t) std::min<uint64_t>((uint64_t) nwg * reduce_sg_size,
                                          ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup));
        ggml_webgpu_shader_lib_context reduce_shader_ctx = op.shader_lib_ctx;
        reduce_shader_ctx.max_wg_size                    = reduce_wg_size;
        reduce_pipeline = ctx->shader_lib->get_flash_attn_vec_reduce_pipeline(reduce_shader_ctx);

        reduce_params = {
            (uint32_t) nrows,                                                                    // nrows
            (uint32_t) Q->ne[1],                                                                 // seq_len_q
            (uint32_t) Q->ne[2],                                                                 // n_heads
            (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),  // offset_dst
            nwg,                                                                                 // nwg
            0u,                                                                                  // tmp_data_base
            (uint32_t) tmp_stats_base,                                                           // tmp_stats_base
        };

        reduce_entries = {
            ggml_webgpu_make_bind_group_entry(0, tmp_buf, tmp_bind_offset, tmp_size_bytes),
            ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(dst), ggml_webgpu_tensor_align_offset(ctx, dst),
                                              ggml_webgpu_tensor_binding_size(ctx, dst)),
        };
    }

    uint32_t       wg_x           = Q->ne[1] * Q->ne[2] * Q->ne[3];
    const uint64_t split_wg_total = (uint64_t) wg_x * nwg;
    GGML_ASSERT(split_wg_total <= UINT32_MAX);

    std::vector<webgpu_dispatch_desc> dispatches;

    if (op.has_mask) {
        dispatches.push_back({
            blk_pipeline, std::move(blk_params), std::move(blk_entries), { blk_nblk0, blk_nblk1 * blk_batch_count }
        });
    }
    dispatches.push_back({
        pipeline, std::move(split_params), std::move(split_entries), { (uint32_t) split_wg_total, 1u }
    });
    if (use_vec_reduce) {
        dispatches.push_back({
            reduce_pipeline, std::move(reduce_params), std::move(reduce_entries), { (uint32_t) nrows, 1u }
        });
    }

    return ggml_backend_webgpu_build_multi(ctx, dispatches);
}

// One query token on PowerVR, or a few: flash_attn_pvr.wgsl, as the vector
// path takes milliseconds per call there.
#define WEBGPU_FLASH_ATTN_PVR_VEC_MAX_ROWS 8
static bool ggml_webgpu_can_use_flash_attn_pvr(webgpu_context &    ctx,
                                               const ggml_tensor * Q,
                                               const ggml_tensor * K,
                                               const ggml_tensor * V,
                                               const ggml_tensor * mask,
                                               const ggml_tensor * sinks,
                                               const ggml_tensor * dst) {
    if (!ggml_webgpu_is_pvr(ctx->global_ctx)) {
        return false;
    }
    const int64_t head_dim = Q->ne[0];
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 ||
        dst->type != GGML_TYPE_F32 || sinks != nullptr || Q->ne[3] != 1 || K->ne[3] != 1 ||
        V->ne[3] != 1 || V->ne[0] != head_dim || K->ne[0] != head_dim || head_dim % 4 != 0 || head_dim > 256 ||
        Q->ne[2] % K->ne[2] != 0 || K->ne[2] != V->ne[2] || K->ne[1] != V->ne[1] || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (ggml_get_op_params_f32(dst, 1) != 0.0f || ggml_get_op_params_f32(dst, 2) != 0.0f) {
        return false;  // ALiBi, logit softcap
    }
    if (mask != nullptr && (mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->ne[3] != 1 ||
                            mask->nb[0] != 2 || mask->ne[0] < K->ne[1] || mask->ne[1] < Q->ne[1])) {
        return false;
    }
    if (Q->nb[0] != sizeof(float)) {
        return false;
    }
    // its workgroup memory must fit (see ggml_webgpu_flash_attn_pvr_wg_size)
    if (ggml_webgpu_flash_attn_pvr_wg_size((uint32_t) head_dim,
                                           ctx->global_ctx->capabilities.limits.maxComputeWorkgroupStorageSize) == 0) {
        return false;
    }
    // K and V bound read_write (when one shares dst's buffer) must not overlap
    const WGPUBuffer dst_buf = ggml_webgpu_tensor_buf(dst).Get();
    if ((ggml_webgpu_tensor_buf(K).Get() == dst_buf || ggml_webgpu_tensor_buf(V).Get() == dst_buf) &&
        ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, K, V)) {
        return false;
    }
    // K and V are read through vec4<f16>
    const size_t k_off = ggml_webgpu_tensor_misalignment(ctx, K) / ggml_type_size(K->type);
    const size_t v_off = ggml_webgpu_tensor_misalignment(ctx, V) / ggml_type_size(V->type);
    if (K->nb[0] != 2 || V->nb[0] != 2 || k_off % 4 != 0 || v_off % 4 != 0 || (K->nb[1] / 2) % 4 != 0 ||
        (K->nb[2] / 2) % 4 != 0 || (V->nb[1] / 2) % 4 != 0 || (V->nb[2] / 2) % 4 != 0) {
        return false;
    }
    return true;
}

static webgpu_encoded_op ggml_webgpu_flash_attn_pvr(webgpu_context & ctx,
                                                    ggml_tensor *    Q,
                                                    ggml_tensor *    K,
                                                    ggml_tensor *    V,
                                                    ggml_tensor *    mask,
                                                    ggml_tensor *    dst) {
    const uint32_t   head_dim = (uint32_t) Q->ne[0];
    const WGPUBuffer dst_buf  = ggml_webgpu_tensor_buf(dst).Get();
    const bool kv_ro = ggml_webgpu_tensor_buf(K).Get() != dst_buf && ggml_webgpu_tensor_buf(V).Get() != dst_buf;
    webgpu_pipeline pipeline = ctx->shader_lib->get_flash_attn_pvr_pipeline(
        head_dim, mask != nullptr, kv_ro, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupStorageSize);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, Q) / ggml_type_size(Q->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, K) / ggml_type_size(K->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, V) / ggml_type_size(V->type)),
        mask != nullptr ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, mask) / ggml_type_size(mask->type)) : 0u,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) K->ne[1],
        (uint32_t) (Q->nb[2] / ggml_type_size(Q->type)),
        (uint32_t) (K->nb[1] / ggml_type_size(K->type)),
        (uint32_t) (K->nb[2] / ggml_type_size(K->type)),
        (uint32_t) (V->nb[1] / ggml_type_size(V->type)),
        (uint32_t) (V->nb[2] / ggml_type_size(V->type)),
        (uint32_t) (Q->ne[2] / K->ne[2]),
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 0)),
        (uint32_t) (Q->nb[1] / ggml_type_size(Q->type)),
        mask != nullptr ? (uint32_t) (mask->nb[1] / ggml_type_size(mask->type)) : 0u,
    };
    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, Q),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, K),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, V),
    };
    if (mask != nullptr) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 3, mask));
    }
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, mask != nullptr ? 4 : 3, dst));
    // a workgroup per query token and head
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, (uint32_t) Q->ne[1], (uint32_t) Q->ne[2]);
}

// Several query rows (prompt batches) on PowerVR: flash_attn_pvr_mq.wgsl,
// which reads K and V 8 values at a time. Returns the threads per row it
// splits a row's head_dim over (subgroup shuffles add up their scores), or 0
// when it can't take the op. Each thread keeps head_dim / split values of Q
// and of the output in registers.
static uint32_t ggml_webgpu_flash_attn_pvr_mq_split(webgpu_context &    ctx,
                                                    const ggml_tensor * Q,
                                                    const ggml_tensor * K,
                                                    const ggml_tensor * V) {
    const uint32_t head_dim = (uint32_t) Q->ne[0];
    const size_t   k_off    = ggml_webgpu_tensor_misalignment(ctx, K) / ggml_type_size(K->type);
    const size_t   v_off    = ggml_webgpu_tensor_misalignment(ctx, V) / ggml_type_size(V->type);
    if (head_dim % 8 != 0 || k_off % 8 != 0 || v_off % 8 != 0 || (K->nb[1] / 2) % 8 != 0 || (K->nb[2] / 2) % 8 != 0 ||
        (V->nb[1] / 2) % 8 != 0 || (V->nb[2] / 2) % 8 != 0) {
        return 0;
    }
    if (head_dim % 32 == 0 && head_dim >= 128 && head_dim <= 256 &&
        ggml_webgpu_pvr_subgroups_tile(ctx->global_ctx, WEBGPU_FLASH_ATTN_PVR_MQ_ROWS * 4, 4)) {
        return 4;
    }
    if (head_dim % 16 == 0 && head_dim <= 128 &&
        ggml_webgpu_pvr_subgroups_tile(ctx->global_ctx, WEBGPU_FLASH_ATTN_PVR_MQ_ROWS * 2, 2)) {
        return 2;
    }
    return head_dim <= 64 ? 1 : 0;
}

static webgpu_encoded_op ggml_webgpu_flash_attn_pvr_mq(webgpu_context & ctx,
                                                       ggml_tensor *    Q,
                                                       ggml_tensor *    K,
                                                       ggml_tensor *    V,
                                                       ggml_tensor *    mask,
                                                       ggml_tensor *    dst,
                                                       uint32_t         split) {
    const uint32_t   head_dim = (uint32_t) Q->ne[0];
    const WGPUBuffer dst_buf  = ggml_webgpu_tensor_buf(dst).Get();
    const bool kv_ro = ggml_webgpu_tensor_buf(K).Get() != dst_buf && ggml_webgpu_tensor_buf(V).Get() != dst_buf;
    webgpu_pipeline pipeline =
        ctx->shader_lib->get_flash_attn_pvr_mq_pipeline(head_dim, split, mask != nullptr, kv_ro);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, Q) / ggml_type_size(Q->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, K) / ggml_type_size(K->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, V) / ggml_type_size(V->type)),
        mask != nullptr ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, mask) / ggml_type_size(mask->type)) : 0u,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) K->ne[1],
        (uint32_t) Q->ne[1],
        (uint32_t) Q->ne[2],
        (uint32_t) (Q->nb[1] / ggml_type_size(Q->type)),
        (uint32_t) (Q->nb[2] / ggml_type_size(Q->type)),
        (uint32_t) (K->nb[1] / ggml_type_size(K->type)),
        (uint32_t) (K->nb[2] / ggml_type_size(K->type)),
        (uint32_t) (V->nb[1] / ggml_type_size(V->type)),
        (uint32_t) (V->nb[2] / ggml_type_size(V->type)),
        mask != nullptr ? (uint32_t) (mask->nb[1] / ggml_type_size(mask->type)) : 0u,
        (uint32_t) (Q->ne[2] / K->ne[2]),
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 0)),
    };
    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, Q),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, K),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, V),
    };
    if (mask != nullptr) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 3, mask));
    }
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, mask != nullptr ? 4 : 3, dst));
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries,
                                     CEIL_DIV((uint32_t) Q->ne[1], WEBGPU_FLASH_ATTN_PVR_MQ_ROWS), (uint32_t) Q->ne[2]);
}

static webgpu_encoded_op ggml_webgpu_flash_attn(webgpu_context & ctx,
                                                ggml_tensor *    Q,
                                                ggml_tensor *    K,
                                                ggml_tensor *    V,
                                                ggml_tensor *    mask,
                                                ggml_tensor *    sinks,
                                                ggml_tensor *    dst) {
    if (ggml_webgpu_can_use_flash_attn_pvr(ctx, Q, K, V, mask, sinks, dst)) {
        // a few query tokens (a speculative batch, the end of a prompt) run faster as one-token problems
        if (Q->ne[1] <= WEBGPU_FLASH_ATTN_PVR_VEC_MAX_ROWS) {
            return ggml_webgpu_flash_attn_pvr(ctx, Q, K, V, mask, dst);
        }
        if (const uint32_t split = ggml_webgpu_flash_attn_pvr_mq_split(ctx, Q, K, V)) {
            return ggml_webgpu_flash_attn_pvr_mq(ctx, Q, K, V, mask, dst, split);
        }
    }
    ggml_webgpu_flash_attn_op op = ggml_webgpu_flash_attn_prepare(ctx, Q, K, V, mask, sinks, dst);
    if (ggml_webgpu_flash_attn_use_vec_path(ctx->global_ctx, Q, K, V, Q->ne[1])) {
        return ggml_webgpu_flash_attn_vec(ctx, Q, K, V, mask, sinks, dst, std::move(op));
    }
    return ggml_webgpu_flash_attn_direct(ctx, op);
}

static webgpu_encoded_op ggml_webgpu_unary_op(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    bool is_unary = dst->op == GGML_OP_UNARY;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = nullptr;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_unary_pipeline(shader_lib_ctx);

    auto *     decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
    const bool inplace   = decisions->inplace;

    uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = { ne,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
                                     (uint32_t) (src->nb[0] / ggml_type_size(src->type)),
                                     (uint32_t) (src->nb[1] / ggml_type_size(src->type)),
                                     (uint32_t) (src->nb[2] / ggml_type_size(src->type)),
                                     (uint32_t) (src->nb[3] / ggml_type_size(src->type)),
                                     (uint32_t) src->ne[0],
                                     (uint32_t) src->ne[1],
                                     (uint32_t) src->ne[2] };

    ggml_tensor * effective_src = src;
    if (is_unary) {
        ggml_unary_op unary_op = ggml_get_unary_op(dst);
        switch (unary_op) {
            case GGML_UNARY_OP_XIELU:
                {
                    // Get float parameters and reinterpret their bit patterns as uint32_t
                    // for passing through the params buffer
                    float alpha_n = ggml_get_op_params_f32(dst, 1);
                    float alpha_p = ggml_get_op_params_f32(dst, 2);
                    float beta    = ggml_get_op_params_f32(dst, 3);
                    float eps     = ggml_get_op_params_f32(dst, 4);
                    params.push_back(ggml_webgpu_u32_from_f32(alpha_n));
                    params.push_back(ggml_webgpu_u32_from_f32(alpha_p));
                    params.push_back(ggml_webgpu_u32_from_f32(beta));
                    params.push_back(ggml_webgpu_u32_from_f32(eps));
                    break;
                }
            default:
                break;
        }
    } else if (dst->op == GGML_OP_CLAMP) {
        float clamp_min = ggml_get_op_params_f32(dst, 0);
        float clamp_max = ggml_get_op_params_f32(dst, 1);
        params.push_back(ggml_webgpu_u32_from_f32(clamp_min));
        params.push_back(ggml_webgpu_u32_from_f32(clamp_max));
    } else if (dst->op == GGML_OP_FILL) {
        float fill_val = ggml_get_op_params_f32(dst, 0);
        params.push_back(ggml_webgpu_u32_from_f32(fill_val));
        effective_src = dst;  // fill simply fills dst
    }

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, effective_src),
    };
    if (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

static webgpu_encoded_op ggml_webgpu_binary_op(webgpu_context & ctx,
                                               ggml_tensor *    src0,
                                               ggml_tensor *    src1,
                                               ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    const bool      src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1);
    webgpu_pipeline pipeline    = ctx->shader_lib->get_binary_pipeline(shader_lib_ctx, src_overlap);
    auto *          decisions   = static_cast<ggml_webgpu_binary_shader_decisions *>(pipeline.context.get());

    uint32_t ne = (uint32_t) ggml_nelements(dst);

    size_t src0_webgpu_tensor_align_offset = ggml_webgpu_tensor_align_offset(ctx, src0);
    size_t src1_webgpu_tensor_align_offset = ggml_webgpu_tensor_align_offset(ctx, src1);

    uint32_t offset_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type));
    uint32_t offset_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type));
    size_t   merged_offset = 0;
    size_t   merged_size   = 0;
    if (decisions->src_overlap) {
        const ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
        merged_offset = merged_range.offset;
        merged_size   = merged_range.size;
        offset_src0   = ggml_webgpu_tensor_merged_element_offset(src0, merged_range);
        offset_src1   = ggml_webgpu_tensor_merged_element_offset(src1, merged_range);
    }

    std::vector<uint32_t> params = {
        ne,
        offset_src0,
        offset_src1,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) (src0->nb[0] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),
        (uint32_t) (src1->nb[0] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        (uint32_t) src1->ne[0],
        (uint32_t) src1->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) src1->ne[3],
    };

    std::vector<wgpu::BindGroupEntry> entries;

    if (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0), merged_offset, merged_size));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    } else {
        entries.push_back(ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0),
                                                            src0_webgpu_tensor_align_offset,
                                                            ggml_webgpu_tensor_binding_size(ctx, src0)));
        entries.push_back(ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(src1),
                                                            src1_webgpu_tensor_align_offset,
                                                            ggml_webgpu_tensor_binding_size(ctx, src1)));
        if (!decisions->inplace && !decisions->overlap) {
            entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst));
        }
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

static webgpu_encoded_op ggml_webgpu_add_id(webgpu_context & ctx,
                                            ggml_tensor *    src0,
                                            ggml_tensor *    src1,
                                            ggml_tensor *    src2,
                                            ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_add_id_pipeline(shader_lib_ctx);

    auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_size(src2->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
        (uint32_t) (src2->nb[0] / ggml_type_size(src2->type)),
        (uint32_t) (src2->nb[1] / ggml_type_size(src2->type)),
        (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2],
    };

    std::vector<wgpu::BindGroupEntry> entries;

    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0));
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1));
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, src2));

    if (!decisions->inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 3, dst));
    }

    uint32_t       wg_x           = 1;
    uint32_t       wg_y           = 1;
    uint32_t       total_wg       = ggml_nrows(dst);
    const uint32_t max_wg_per_dim = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;
    compute_2d_workgroups(total_wg, max_wg_per_dim, wg_x, wg_y);

    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

static webgpu_encoded_op ggml_webgpu_concat(webgpu_context & ctx,
                                            ggml_tensor *    src0,
                                            ggml_tensor *    src1,
                                            ggml_tensor *    dst) {
    uint32_t ne  = (uint32_t) ggml_nelements(dst);
    uint32_t dim = (uint32_t) dst->op_params[0];
    if (ggml_nbytes(src0) == 0 && ggml_nbytes(src1) == 0) {
        return {};
    }

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    const bool      src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1) ||
                                  ggml_nbytes(src0) == 0 || ggml_nbytes(src1) == 0;
    webgpu_pipeline pipeline    = ctx->shader_lib->get_concat_pipeline(shader_lib_ctx, src_overlap);
    auto *          decisions   = static_cast<ggml_webgpu_binary_shader_decisions *>(pipeline.context.get());

    uint32_t offset_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type));
    uint32_t offset_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type));
    size_t   merged_offset = 0;
    size_t   merged_size   = 0;
    if (decisions->src_overlap) {
        if (ggml_nbytes(src0) == 0) {
            merged_offset = ggml_webgpu_tensor_align_offset(ctx, src1);
            merged_size   = ggml_webgpu_tensor_binding_size(ctx, src1);
            offset_src0   = 0;
            offset_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type));
        } else if (ggml_nbytes(src1) == 0) {
            merged_offset = ggml_webgpu_tensor_align_offset(ctx, src0);
            merged_size   = ggml_webgpu_tensor_binding_size(ctx, src0);
            offset_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type));
            offset_src1   = 0;
        } else {
            const ggml_webgpu_merged_binding_range merged_range =
                ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
            merged_offset = merged_range.offset;
            merged_size   = merged_range.size;
            offset_src0   = ggml_webgpu_tensor_merged_element_offset(src0, merged_range);
            offset_src1   = ggml_webgpu_tensor_merged_element_offset(src1, merged_range);
        }
    }

    std::vector<uint32_t> params = { ne,
                                     offset_src0,
                                     offset_src1,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
                                     (uint32_t) (src0->nb[0] / ggml_type_size(src0->type)),
                                     (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
                                     (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
                                     (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),
                                     (uint32_t) (src1->nb[0] / ggml_type_size(src1->type)),
                                     (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
                                     (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
                                     (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)),
                                     (uint32_t) dst->ne[0],
                                     (uint32_t) dst->ne[1],
                                     (uint32_t) dst->ne[2],
                                     dim,
                                     (uint32_t) src0->ne[dim] };

    std::vector<wgpu::BindGroupEntry> entries = {};
    if (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0), merged_offset, merged_size));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    } else {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = CEIL_DIV(ne, decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

static webgpu_encoded_op ggml_webgpu_repeat(webgpu_context & ctx, ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = { ne,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) /
                                                 ggml_type_size(src0->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
                                     (uint32_t) (src0->nb[0] / ggml_type_size(src0->type)),
                                     (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
                                     (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
                                     (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),
                                     (uint32_t) (src0->ne[0]),
                                     (uint32_t) (src0->ne[1]),
                                     (uint32_t) (src0->ne[2]),
                                     (uint32_t) (src0->ne[3]),
                                     (uint32_t) (dst->ne[0]),
                                     (uint32_t) (dst->ne[1]),
                                     (uint32_t) (dst->ne[2]) };

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_repeat_pipeline(shader_lib_ctx);
    auto *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = CEIL_DIV(ne, decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// PowerVR: a block's post-norm (RMS_NORM, MUL), the residual ADD and the next
// norm (RMS_NORM, MUL) as one dispatch (see rms_norm_add_pvr.wgsl): models
// with post-norms (LFM3) run this twice per layer, and every dispatch costs
// tens of microseconds there. Few rows only (the next norm's output is staged
// for a mat-vec). Fills the five nodes' tensors, or returns false.
struct ggml_webgpu_pvr_post_norm {
    ggml_tensor * o, * w1, * x, * a, * w2, * out;
    float         eps1, eps2;
};

// Whether two tensors are the same bytes or apart (or in different buffers). A
// fused op that writes one where a thread of another workgroup (row) may
// still read the other is only race-free then: gallocr may place a tensor
// over part of another, at a shift.
static bool ggml_webgpu_pvr_same_or_apart(const ggml_tensor * a, const ggml_tensor * b) {
    if (ggml_webgpu_tensor_buf(a).Get() != ggml_webgpu_tensor_buf(b).Get()) {
        return true;
    }
    const size_t oa = ggml_webgpu_tensor_offset(a);
    const size_t ob = ggml_webgpu_tensor_offset(b);
    return (oa == ob && ggml_nbytes(a) == ggml_nbytes(b)) || oa + ggml_nbytes(a) <= ob || ob + ggml_nbytes(b) <= oa;
}

static bool ggml_webgpu_pvr_post_norm_nodes(webgpu_context &            ctx,
                                            const struct ggml_cgraph * cgraph,
                                            int                        node_idx,
                                            ggml_webgpu_pvr_post_norm & f) {
    if (!ggml_webgpu_is_pvr(ctx->global_ctx) ||
        !ggml_can_fuse_subgraph(cgraph, node_idx,
                                { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD, GGML_OP_RMS_NORM, GGML_OP_MUL },
                                { node_idx + 2, node_idx + 4 })) {
        return false;
    }
    ggml_tensor * n1  = cgraph->nodes[node_idx];
    ggml_tensor * m1  = cgraph->nodes[node_idx + 1];
    ggml_tensor * add = cgraph->nodes[node_idx + 2];
    ggml_tensor * n2  = cgraph->nodes[node_idx + 3];
    ggml_tensor * m2  = cgraph->nodes[node_idx + 4];
    // the other operand of a binary op whose one operand is t
    auto other = [](const ggml_tensor * op, const ggml_tensor * t) -> ggml_tensor * {
        return op->src[0] == t ? op->src[1] : op->src[1] == t ? op->src[0] : nullptr;
    };
    f.o   = n1->src[0];
    f.w1  = other(m1, n1);
    f.x   = other(add, m1);
    f.a   = add;
    f.w2  = other(m2, n2);
    f.out = m2;
    if (!f.w1 || !f.x || !f.w2 || n2->src[0] != add) {
        return false;
    }
    // rows of ne0 values, contiguous; the norms' weights one row each
    const int64_t ne0 = f.o->ne[0];
    for (const ggml_tensor * t : { f.o, f.x, f.a, f.out }) {
        if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || !ggml_are_same_shape(t, f.o) ||
            ggml_webgpu_tensor_buf(t).Get() != ggml_webgpu_tensor_buf(f.o).Get()) {
            return false;
        }
    }
    for (const ggml_tensor * t : { n1, m1, n2 }) {
        if (t->type != GGML_TYPE_F32 || !ggml_are_same_shape(t, f.o)) {
            return false;
        }
    }
    // (the weights bound read-only: not in the buffer bound read-write)
    for (const ggml_tensor * t : { f.w1, f.w2 }) {
        if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || t->ne[0] != ne0 || ggml_nrows(t) != 1 ||
            ggml_webgpu_tensor_buf(t).Get() == ggml_webgpu_tensor_buf(f.o).Get()) {
            return false;
        }
    }
    if (f.o->ne[2] != 1 || f.o->ne[3] != 1 || f.o->ne[1] > WEBGPU_MUL_MAT_VEC_PVR_MAX_COLS) {
        return false;
    }
    // a column belongs to one thread throughout, so what it writes (a, out) may be what it reads (o, x)
    // or apart from it, but not over another row's
    for (const ggml_tensor * wr : { f.a, f.out }) {
        for (const ggml_tensor * rd : { f.o, f.x, f.a }) {
            if (wr != rd && !ggml_webgpu_pvr_same_or_apart(wr, rd)) {
                return false;
            }
        }
    }
    f.eps1 = ggml_get_op_params_f32(n1, 0);
    f.eps2 = ggml_get_op_params_f32(n2, 0);
    return true;
}

static webgpu_encoded_op ggml_webgpu_pvr_rms_norm_add(webgpu_context & ctx, const ggml_webgpu_pvr_post_norm & f) {
    const ggml_webgpu_merged_binding_range act = ggml_webgpu_tensor_merged_binding_range(ctx, { f.o, f.x, f.a, f.out });
    std::vector<uint32_t> params = {
        ggml_webgpu_tensor_merged_element_offset(f.o, act),
        ggml_webgpu_tensor_merged_element_offset(f.x, act),
        ggml_webgpu_tensor_merged_element_offset(f.a, act),
        ggml_webgpu_tensor_merged_element_offset(f.out, act),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, f.w1) / sizeof(float)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, f.w2) / sizeof(float)),
        (uint32_t) f.o->ne[0],
        (uint32_t) ggml_nrows(f.o),
        ggml_webgpu_u32_from_f32(f.eps1),
        ggml_webgpu_u32_from_f32(f.eps2),
    };
    const size_t bytes = ROUNDUP_POW2(ggml_nbytes(f.out), 16);
    ggml_webgpu_pvr_stage_next(ctx, bytes);
    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(f.o), act.offset, act.size),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, f.w1),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, f.w2),
        ggml_webgpu_make_bind_group_entry(3, ctx->pvr_staging_buf[ctx->pvr_staged_idx], 0, bytes),
    };
    ctx->pvr_staged_src1  = f.out;
    ctx->pvr_staged_f16   = false;
    ctx->pvr_staged_by_op = true;
    webgpu_pipeline pipeline = ctx->shader_lib->get_rms_norm_add_pvr_pipeline(WEBGPU_PVR_RMS_NORM_MUL_WG_SIZE);
    uint32_t        wg_x;
    uint32_t        wg_y;
    compute_2d_workgroups((uint32_t) ggml_nrows(f.o), ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension,
                          wg_x, wg_y);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

static std::optional<webgpu_encoded_op> ggml_webgpu_rms_norm_mul(webgpu_context & ctx,
                                                                 ggml_tensor *    rn_src,
                                                                 ggml_tensor *    rn_dst,
                                                                 ggml_tensor *    mul_src0,
                                                                 ggml_tensor *    mul_src1,
                                                                 ggml_tensor *    dst) {
    ggml_tensor * mul_src;

    if (ggml_webgpu_tensor_equal(rn_dst, mul_src0)) {
        mul_src = mul_src1;
    } else if (ggml_webgpu_tensor_equal(rn_dst, mul_src1)) {
        mul_src = mul_src0;
    } else {
        GGML_ABORT("rms_norm must be equal to the one of mul_src0 and mul_src1");
    }

    uint32_t offset_rn_src = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, rn_src) / ggml_type_size(rn_src->type));
    uint32_t offset_mul_src =
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, mul_src) / ggml_type_size(mul_src->type));
    size_t merged_offset = 0;
    size_t merged_size   = 0;

    std::vector<uint32_t> params = {
        offset_rn_src,
        offset_mul_src,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) (rn_src->nb[1] / ggml_type_size(rn_src->type)),
        (uint32_t) (rn_src->nb[2] / ggml_type_size(rn_src->type)),
        (uint32_t) (rn_src->nb[3] / ggml_type_size(rn_src->type)),
        (uint32_t) (mul_src->nb[1] / ggml_type_size(mul_src->type)),
        (uint32_t) (mul_src->nb[2] / ggml_type_size(mul_src->type)),
        (uint32_t) (mul_src->nb[3] / ggml_type_size(mul_src->type)),
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
        (uint32_t) mul_src->ne[0],
        (uint32_t) mul_src->ne[1],
        (uint32_t) mul_src->ne[2],
        (uint32_t) mul_src->ne[3],
        (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2],
        (uint32_t) dst->ne[3],
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(rn_dst, 0))  // epsilon, treated as f32 in the shader
    };

    std::vector<wgpu::BindGroupEntry> entries;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = rn_src;
    shader_lib_ctx.src1                           = mul_src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    if (ggml_webgpu_is_pvr(ctx->global_ctx)) {
        // a workgroup per row: PowerVR runs rows of a model's widths faster with fewer threads
        shader_lib_ctx.max_wg_size = std::min(shader_lib_ctx.max_wg_size, WEBGPU_PVR_RMS_NORM_MUL_WG_SIZE);
    }

    const bool src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, rn_src, mul_src);
    // PowerVR: a norm's output on a few tokens is a mat-vec's src1, so write
    // it into the staging buffer as well, which saves that copy's dispatch
    const bool staged = ggml_webgpu_is_pvr(ctx->global_ctx) && !src_overlap && dst->type == GGML_TYPE_F32 &&
                        dst->ne[1] <= WEBGPU_MUL_MAT_VEC_PVR_MAX_COLS && dst->ne[2] == 1 && dst->ne[3] == 1 &&
                        ggml_is_contiguous(dst) && !ggml_webgpu_tensor_equal(rn_src, dst) &&
                        !ggml_webgpu_tensor_equal(mul_src, dst);
    webgpu_pipeline pipeline = ctx->shader_lib->get_rms_norm_mul_pipeline(shader_lib_ctx, src_overlap, staged);
    auto *          decisions   = static_cast<ggml_webgpu_rms_norm_mul_shader_decisions *>(pipeline.context.get());

    if (decisions->src_overlap) {
        const ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { rn_src, mul_src });
        merged_offset  = merged_range.offset;
        merged_size    = merged_range.size;
        offset_rn_src  = ggml_webgpu_tensor_merged_element_offset(rn_src, merged_range);
        offset_mul_src = ggml_webgpu_tensor_merged_element_offset(mul_src, merged_range);
        params[0]      = offset_rn_src;
        params[1]      = offset_mul_src;
    }

    if (decisions->inplace || decisions->overlap) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, rn_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, mul_src));
    } else if (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(rn_src), merged_offset, merged_size));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    } else {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, rn_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, mul_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst));
    }
    if (staged) {
        const size_t bytes = ROUNDUP_POW2(ggml_nbytes(dst), 16);
        ggml_webgpu_pvr_stage_next(ctx, bytes);
        entries.push_back(ggml_webgpu_make_bind_group_entry(3, ctx->pvr_staging_buf[ctx->pvr_staged_idx], 0, bytes));
        ctx->pvr_staged_src1  = dst;
        ctx->pvr_staged_f16   = false;
        ctx->pvr_staged_by_op = true;
    }

    uint32_t wg_x;
    uint32_t wg_y;
    compute_2d_workgroups(ggml_nrows(dst), ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x,
                          wg_y);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

static webgpu_encoded_op ggml_webgpu_group_norm(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    const size_t          src_type_size = ggml_type_size(src->type);
    const size_t          dst_type_size = ggml_type_size(dst->type);
    const uint32_t        num_groups    = (uint32_t) ggml_get_op_params_i32(dst, 0);
    std::vector<uint32_t> params        = {
        (uint32_t) src->ne[0],
        (uint32_t) src->ne[1],
        (uint32_t) src->ne[2],
        (uint32_t) src->ne[3],
        num_groups,
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 1)),
        (uint32_t) (src->nb[1] / src_type_size),
        (uint32_t) (src->nb[0] / src_type_size),
        (uint32_t) (src->nb[2] / src_type_size),
        (uint32_t) (src->nb[3] / src_type_size),
        (uint32_t) (dst->nb[1] / dst_type_size),
        (uint32_t) (dst->nb[0] / dst_type_size),
        (uint32_t) (dst->nb[2] / dst_type_size),
        (uint32_t) (dst->nb[3] / dst_type_size),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / src_type_size),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / dst_type_size),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    webgpu_pipeline pipeline   = ctx->shader_lib->get_group_norm_pipeline(shader_lib_ctx);
    auto *          decisions  = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src) };
    if (!decisions->inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    }
    uint32_t wg_x;
    uint32_t wg_y;
    compute_2d_workgroups(num_groups * (uint32_t) src->ne[3],
                          ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

static webgpu_encoded_op ggml_webgpu_row_norm(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) (src->nb[1] / ggml_type_size(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_size(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_size(src->type)),
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
        (uint32_t) src->ne[0],
        (uint32_t) src->ne[1],
        (uint32_t) src->ne[2],
        (uint32_t) src->ne[3],
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 0))  // epsilon, treated as f32 in the shader
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_row_norm_pipeline(shader_lib_ctx);
    auto *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src) };
    if (!decisions->inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    compute_2d_workgroups(ggml_nrows(src), ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x,
                          wg_y);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

static webgpu_encoded_op ggml_webgpu_rope(webgpu_context & ctx,
                                          ggml_tensor *    src0,
                                          ggml_tensor *    src1,
                                          ggml_tensor *    src2,
                                          ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_rope_pipeline(shader_lib_ctx);

    auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    const bool inplace         = decisions->inplace;
    const int  has_freq_factor = (src2 != nullptr);

    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];
    const int n_offs     = ((int32_t *) dst->op_params)[15];

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    memcpy(&freq_base, (int32_t *) dst->op_params + 5, sizeof(float));
    memcpy(&freq_scale, (int32_t *) dst->op_params + 6, sizeof(float));
    memcpy(&ext_factor, (int32_t *) dst->op_params + 7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params + 8, sizeof(float));
    memcpy(&beta_fast, (int32_t *) dst->op_params + 9, sizeof(float));
    memcpy(&beta_slow, (int32_t *) dst->op_params + 10, sizeof(float));

    int sections[4];
    memcpy(sections, (int32_t *) dst->op_params + 11, 4 * sizeof(int));

    float theta_scale = powf(freq_base, -2.0f / n_dims);

    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, true, corr_dims);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        src2 != nullptr ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_size(src2->type)) : 0,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
        (uint32_t) ggml_nelements(src0) / 2,
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        (uint32_t) n_dims,
        (uint32_t) mode,
        ggml_webgpu_u32_from_f32(theta_scale),
        ggml_webgpu_u32_from_f32(attn_factor),
        ggml_webgpu_u32_from_f32(freq_scale),
        ggml_webgpu_u32_from_f32(ext_factor),
        ggml_webgpu_u32_from_f32(corr_dims[0]),
        ggml_webgpu_u32_from_f32(corr_dims[1]),
        (uint32_t) sections[0],
        (uint32_t) sections[1],
        (uint32_t) sections[2],
        (uint32_t) sections[3],
        (uint32_t) n_offs
    };

    std::vector<wgpu::BindGroupEntry> entries     = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0),
                                                      ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1) };
    uint32_t                          dst_binding = 2;
    if (has_freq_factor) {
        dst_binding = 3;
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, src2));
    }
    if (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, dst_binding, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

static webgpu_encoded_op ggml_webgpu_glu(webgpu_context & ctx,
                                         ggml_tensor *    src0,
                                         ggml_tensor *    src1,
                                         ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    const bool      src_overlap = src1 != nullptr && ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1);
    webgpu_pipeline pipeline    = ctx->shader_lib->get_glu_pipeline(shader_lib_ctx, src_overlap);

    auto * decisions = static_cast<ggml_webgpu_glu_shader_decisions *>(pipeline.context.get());

    const int split = (src1 != nullptr);

    uint32_t offset_src0 = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type));
    uint32_t offset_src1 =
        src1 != nullptr ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)) : 0;
    size_t merged_offset = 0;
    size_t merged_size   = 0;
    if (decisions->src_overlap) {
        const ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
        merged_offset = merged_range.offset;
        merged_size   = merged_range.size;
        offset_src0   = ggml_webgpu_tensor_merged_element_offset(src0, merged_range);
        offset_src1   = ggml_webgpu_tensor_merged_element_offset(src1, merged_range);
    }

    std::vector<uint32_t> params = {
        offset_src0,
        offset_src1,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),
        src1 != nullptr ? (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)) :
                          (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        src1 != nullptr ? (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)) :
                          (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        src1 != nullptr ? (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)) :
                          (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
        (uint32_t) ggml_nelements(dst),
        (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2],
        (uint32_t) ((int32_t *) dst->op_params)[1],                // swapped
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 2)),  // alpha, for swiglu_oai
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 3)),  // limit
    };

    std::vector<wgpu::BindGroupEntry> entries;
    uint32_t                          dst_binding = 1;
    if (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0), merged_offset, merged_size));
    } else {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0));
    }
    if (split && !decisions->src_overlap) {
        dst_binding = 2;
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1));
    }
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, dst_binding, dst));

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

static webgpu_encoded_op ggml_webgpu_scale(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = nullptr;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_scale_pipeline(shader_lib_ctx);
    auto *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    // params unchanged
    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) (src->nb[1] / ggml_type_size(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_size(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_size(src->type)),
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
        (uint32_t) ggml_nelements(dst),
        (uint32_t) src->ne[0],
        (uint32_t) src->ne[1],
        (uint32_t) src->ne[2],
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 0)),  // scale
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 1))   // bias
    };

    // bindgroups unchanged
    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src) };

    if (!decisions->inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

static webgpu_encoded_op ggml_webgpu_soft_max(webgpu_context & ctx,
                                              ggml_tensor *    src0,
                                              ggml_tensor *    src1,
                                              ggml_tensor *    src2,
                                              ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_soft_max_pipeline(shader_lib_ctx);
    auto *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    const bool inplace     = decisions->inplace;
    const int  has_mask    = (src1 != nullptr);
    const int  has_sink    = (src2 != nullptr);
    float      max_bias    = ggml_get_op_params_f32(dst, 1);
    float      n_head_log2 = float(1u << (uint32_t) floor(log2(src0->ne[2])));
    float      m0          = powf(2.0f, -(max_bias) / n_head_log2);
    float      m1          = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        has_mask ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)) : 0,
        has_sink ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_size(src2->type)) : 0,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),
        has_mask ? (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)) : 0,
        has_mask ? (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)) : 0,
        has_mask ? (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)) : 0,
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        has_mask ? (uint32_t) src1->ne[2] : 0,
        has_mask ? (uint32_t) src1->ne[3] : 0,
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 0)),  // scale
        ggml_webgpu_u32_from_f32(max_bias),
        ggml_webgpu_u32_from_f32(n_head_log2),
        ggml_webgpu_u32_from_f32(m0),
        ggml_webgpu_u32_from_f32(m1)
    };

    std::vector<wgpu::BindGroupEntry> entries     = { ggml_webgpu_make_bind_group_entry(
        0, ggml_webgpu_tensor_buf(src0), ggml_webgpu_tensor_align_offset(ctx, src0),
        ggml_webgpu_tensor_binding_size(ctx, src0)) };
    uint32_t                          binding_num = 1;
    if (has_mask) {
        entries.push_back(ggml_webgpu_make_bind_group_entry(binding_num, ggml_webgpu_tensor_buf(src1),
                                                            ggml_webgpu_tensor_align_offset(ctx, src1),
                                                            ggml_webgpu_tensor_binding_size(ctx, src1)));
        binding_num++;
    }
    if (has_sink) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_num, src2));
        binding_num++;
    }
    if (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_num, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = ggml_nrows(dst);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

static webgpu_encoded_op ggml_webgpu_argmax(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
                                     (uint32_t) src->ne[0] };

    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_argmax_pipeline(shader_lib_ctx);
    uint32_t        wg_x     = ggml_nelements(dst);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

// PowerVR: the logits' top-k for the sampler (topk_pvr.wgsl), as llama.cpp
// builds it: TOP_K, GET_ROWS of the logits at those indices, the indices
// CPY'd to f32, and the two CONCAT'ed, in three dispatches that write the
// CONCAT's output (each row: the k values largest first, then the indices).
// The scratch space (group maxima, a count and the candidates of each row) is
// a buffer of its own, so that the logits and then the scratch are bound
// read-only.
#define WEBGPU_TOPK_PVR_GPW   16u    // groups per stage 1 and 2 workgroup
#define WEBGPU_TOPK_PVR_CAP   1792u  // candidates per row at most (the workgroup memory: 16 KB)
#define WEBGPU_TOPK_PVR_G_MAX 2048u  // groups per row at most (stage 2 holds their maxima in workgroup memory)

static uint32_t ggml_webgpu_topk_pvr_g_max(uint32_t n, uint32_t group) {
    return ROUNDUP_POW2(CEIL_DIV(n, group), 8u);
}

// Values per group for rows of n values and the top k: 256, or fewer where
// that leaves fewer than k whole groups (small vocabularies); 0 when none fits.
static uint32_t ggml_webgpu_topk_pvr_group(uint64_t n, uint64_t k) {
    for (uint32_t group : { 256u, 128u, 64u, 32u, 16u }) {
        if (n / group >= k) {
            return CEIL_DIV(n, (uint64_t) group) <= WEBGPU_TOPK_PVR_G_MAX ? group : 0;
        }
    }
    return 0;
}

static webgpu_encoded_op ggml_webgpu_topk_pvr(webgpu_context & ctx,
                                              ggml_tensor *    logits,
                                              ggml_tensor *    topk,
                                              ggml_tensor *    out) {
    const uint32_t n     = (uint32_t) logits->ne[0];
    const uint32_t k     = (uint32_t) topk->ne[0];
    const uint32_t rows  = (uint32_t) logits->ne[1];
    const uint32_t group = ggml_webgpu_topk_pvr_group(n, k);
    GGML_ASSERT(group != 0);
    const uint32_t n_wg  = CEIL_DIV(n, group * WEBGPU_TOPK_PVR_GPW);
    const uint32_t g_max = ggml_webgpu_topk_pvr_g_max(n, group);
    const size_t   scratch_size =
        ROUNDUP_POW2((size_t) (g_max + 1 + 2 * WEBGPU_TOPK_PVR_CAP) * rows * sizeof(uint32_t), 16);
    ggml_webgpu_pvr_scratch(ctx, ctx->pvr_topk_buf, ctx->pvr_topk_size, scratch_size, "pvr_topk_buf");
    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, logits) / sizeof(float)),
        (uint32_t) (logits->nb[1] / sizeof(float)),
        n,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, out) / sizeof(float)),
        (uint32_t) (out->nb[1] / sizeof(float)),
    };
    std::vector<webgpu_dispatch_desc> dispatches;
    for (uint32_t stage = 1; stage <= 3; stage++) {
        std::vector<wgpu::BindGroupEntry> entries;
        if (stage < 3) {
            entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, logits),
                        ggml_webgpu_make_bind_group_entry(1, ctx->pvr_topk_buf, 0, scratch_size) };
        } else {
            entries = { ggml_webgpu_make_bind_group_entry(0, ctx->pvr_topk_buf, 0, scratch_size),
                        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, out) };
        }
        dispatches.push_back({ ctx->shader_lib->get_topk_pvr_pipeline(stage, k, group, g_max, WEBGPU_TOPK_PVR_CAP),
                               params, std::move(entries), stage < 3 ? std::make_pair(n_wg, rows) : std::make_pair(rows, 1u) });
    }
    return ggml_backend_webgpu_build_multi(ctx, dispatches);
}

// The CONCAT's index after the TOP_K at node_idx, if the nodes are the
// logits' top-k as above (see ggml_webgpu_topk_pvr), or 0.
static int ggml_webgpu_topk_pvr_fused_nodes(webgpu_context & ctx, ggml_cgraph * cgraph, int node_idx) {
    ggml_tensor * topk = cgraph->nodes[node_idx];
    const ggml_tensor * logits = topk->src[0];
    if (!ggml_webgpu_is_pvr(ctx->global_ctx) || logits->type != GGML_TYPE_F32 || topk->type != GGML_TYPE_I32 ||
        !ggml_is_contiguous(logits) || logits->ne[2] != 1 || logits->ne[3] != 1 || topk->ne[0] > 256 ||
        logits->ne[0] >= (1 << 24) || ggml_webgpu_topk_pvr_group(logits->ne[0], topk->ne[0]) == 0) {
        return 0;
    }
    ggml_tensor * gr   = nullptr;
    ggml_tensor * cast = nullptr;
    for (int i = node_idx + 1; i < cgraph->n_nodes && i < node_idx + 8; i++) {
        ggml_tensor * t = cgraph->nodes[i];
        if (ggml_op_is_empty(t->op)) {
            continue;
        }
        if (t->op == GGML_OP_GET_ROWS && gr == nullptr && t->src[1] == topk) {
            // a [1, n, rows] view of the logits, read row by row
            const ggml_tensor * v = t->src[0];
            if ((v->view_src != logits && v != logits) || v->ne[0] != 1 || v->ne[1] != logits->ne[0] ||
                v->ne[2] != logits->ne[1] || v->nb[1] != sizeof(float) || v->nb[2] != logits->nb[1] ||
                v->type != GGML_TYPE_F32) {
                return 0;
            }
            gr = t;
        } else if (t->op == GGML_OP_CPY && cast == nullptr && t->src[0] == topk && t->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(t)) {
            cast = t;
        } else if (t->op == GGML_OP_CONCAT && gr != nullptr && cast != nullptr && t->type == GGML_TYPE_F32 &&
                   ggml_get_op_params_i32(t, 0) == 0 && t->src[1] == cast && t->src[0]->view_src == gr &&
                   ggml_is_contiguous(t) && t->ne[0] == 2 * topk->ne[0] && t->ne[1] == topk->ne[1] &&
                   t->src[0]->ne[0] == topk->ne[0] && ggml_nelements(t->src[0]) == ggml_nelements(gr)) {
            return i - node_idx;
        } else {
            return 0;
        }
    }
    return 0;
}

static webgpu_encoded_op ggml_webgpu_argsort(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    bool is_top_k = dst->op == GGML_OP_TOP_K;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = nullptr;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size        = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    shader_lib_ctx.wg_mem_limit_bytes = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupStorageSize;

    webgpu_pipeline argsort_pipeline = ctx->shader_lib->get_argsort_pipeline(shader_lib_ctx);
    auto * argsort_decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(argsort_pipeline.context.get());

    webgpu_pipeline argsort_merge_pipeline = ctx->shader_lib->get_argsort_merge_pipeline(shader_lib_ctx);

    const uint32_t src_ne0 = (uint32_t) src->ne[0];
    const uint32_t nrows   = (uint32_t) ggml_nrows(src);
    const uint32_t npr     = CEIL_DIV(src_ne0, argsort_decisions->wg_size);
    const uint32_t block_size =
        is_top_k ? std::min(argsort_decisions->wg_size, (uint32_t) dst->ne[0]) : argsort_decisions->wg_size;
    uint32_t out_ne0 = src_ne0;
    if (is_top_k) {
        if (npr > 1) {
            const uint32_t last_tile = src_ne0 - (npr - 1) * argsort_decisions->wg_size;
            out_ne0                  = (npr - 1) * block_size + std::min(last_tile, block_size);
        } else {
            out_ne0 = block_size;
        }
    }

    uint32_t merge_len    = block_size;
    uint32_t merge_passes = 0;
    while (merge_len < out_ne0) {
        merge_len <<= 1;
        merge_passes++;
    }

    const bool start_in_tmp = (merge_passes % 2) == 1;

    const size_t dst_offset = ggml_webgpu_tensor_offset(dst);
    const size_t idx_nbytes = out_ne0 * ggml_nrows(dst) * sizeof(int32_t);
    const size_t tmp_offset =
        ROUNDUP_POW2(dst_offset + idx_nbytes, ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment);
    const size_t tmp_binding_size = ROUNDUP_POW2(idx_nbytes, WEBGPU_STORAGE_BUF_BINDING_MULT);
    const size_t dst_binding_size =
        ROUNDUP_POW2(idx_nbytes + ggml_webgpu_tensor_misalignment(ctx, dst), WEBGPU_STORAGE_BUF_BINDING_MULT);

    const uint32_t offset_src  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type));
    const uint32_t offset_dst  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type));
    const uint32_t offset_tmp  = 0;
    const uint32_t stride_src1 = (uint32_t) (src->nb[1] / ggml_type_size(src->type));
    const uint32_t stride_src2 = (uint32_t) (src->nb[2] / ggml_type_size(src->type));
    const uint32_t stride_src3 = (uint32_t) (src->nb[3] / ggml_type_size(src->type));
    const uint32_t stride_idx1 = out_ne0;
    const uint32_t stride_idx2 = out_ne0 * (uint32_t) dst->ne[1];
    const uint32_t stride_idx3 = stride_idx2 * (uint32_t) dst->ne[2];

    std::vector<webgpu_dispatch_desc> dispatches;

    const uint32_t init_offset       = start_in_tmp ? offset_tmp : offset_dst;
    const size_t   init_align_offset = start_in_tmp ? tmp_offset : ggml_webgpu_tensor_align_offset(ctx, dst);
    const size_t   init_binding_size = start_in_tmp ? tmp_binding_size : dst_binding_size;

    std::vector<uint32_t> init_params = {
        offset_src,  init_offset, stride_src1, stride_src2,           stride_src3,           stride_idx1,
        stride_idx2, stride_idx3, src_ne0,     (uint32_t) src->ne[1], (uint32_t) src->ne[2], out_ne0,
        block_size,  npr,         nrows
    };

    uint32_t       wg_x_init;
    uint32_t       wg_y_init;
    const uint32_t total_wg_init  = npr * nrows;
    const uint32_t max_wg_per_dim = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;
    compute_2d_workgroups(total_wg_init, max_wg_per_dim, wg_x_init, wg_y_init);

    std::vector<wgpu::BindGroupEntry> init_entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
        ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(dst), init_align_offset, init_binding_size)
    };

    dispatches.push_back({
        argsort_pipeline, std::move(init_params), std::move(init_entries), { wg_x_init, wg_y_init }
    });

    if (merge_passes == 0) {
        return ggml_backend_webgpu_build_multi(ctx, dispatches);
    }

    bool     in_is_tmp = start_in_tmp;
    uint32_t len       = block_size;
    while (len < out_ne0) {
        const uint32_t nm = CEIL_DIV(out_ne0, 2 * len);

        const bool     out_is_tmp  = !in_is_tmp;
        const uint32_t offset_in   = in_is_tmp ? offset_tmp : offset_dst;
        const uint32_t offset_out  = out_is_tmp ? offset_tmp : offset_dst;
        const size_t   align_in    = in_is_tmp ? tmp_offset : ggml_webgpu_tensor_align_offset(ctx, dst);
        const size_t   align_out   = out_is_tmp ? tmp_offset : ggml_webgpu_tensor_align_offset(ctx, dst);
        const size_t   size_in     = in_is_tmp ? tmp_binding_size : dst_binding_size;
        const size_t   size_out    = out_is_tmp ? tmp_binding_size : dst_binding_size;
        const uint32_t top_k_out   = (is_top_k && nm == 1) ? (uint32_t) dst->ne[0] : out_ne0;
        const uint32_t stride_out1 = top_k_out;
        const uint32_t stride_out2 = top_k_out * (uint32_t) dst->ne[1];
        const uint32_t stride_out3 = stride_out2 * (uint32_t) dst->ne[2];

        std::vector<uint32_t> merge_params = { offset_src,
                                               offset_in,
                                               offset_out,
                                               stride_src1,
                                               stride_src2,
                                               stride_src3,
                                               stride_idx1,
                                               stride_idx2,
                                               stride_idx3,
                                               stride_out1,
                                               stride_out2,
                                               stride_out3,
                                               out_ne0,
                                               (uint32_t) src->ne[1],
                                               (uint32_t) src->ne[2],
                                               top_k_out,
                                               len,
                                               nm,
                                               nrows };

        std::vector<wgpu::BindGroupEntry> merge_entries = {
            ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
            ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(dst), align_in, size_in),
            ggml_webgpu_make_bind_group_entry(2, ggml_webgpu_tensor_buf(dst), align_out, size_out)
        };

        uint32_t       wg_x_merge;
        uint32_t       wg_y_merge;
        const uint32_t total_wg_merge = nm * nrows;
        compute_2d_workgroups(total_wg_merge, max_wg_per_dim, wg_x_merge, wg_y_merge);

        dispatches.push_back({
            argsort_merge_pipeline, std::move(merge_params), std::move(merge_entries), { wg_x_merge, wg_y_merge }
        });

        len <<= 1;
        in_is_tmp = !in_is_tmp;
    }

    return ggml_backend_webgpu_build_multi(ctx, dispatches);
}

static webgpu_encoded_op ggml_webgpu_cumsum(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
                                     (uint32_t) src->ne[0] };

    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = nullptr;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_cumsum_pipeline(shader_lib_ctx);
    uint32_t        wg_x;
    uint32_t        wg_y;
    uint32_t        total_wg = ggml_nrows(dst);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

static webgpu_encoded_op ggml_webgpu_sum_rows(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    bool                  total_sum = dst->op == GGML_OP_SUM;
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
                                     total_sum ? 0 : (uint32_t) (src->nb[1] / ggml_type_size(src->type)),
                                     total_sum ? 0 : (uint32_t) (src->nb[2] / ggml_type_size(src->type)),
                                     total_sum ? 0 : (uint32_t) (src->nb[3] / ggml_type_size(src->type)),
                                     total_sum ? static_cast<uint32_t>(ggml_nelements(src)) : (uint32_t) src->ne[0],
                                     total_sum ? 1 : (uint32_t) src->ne[1],
                                     total_sum ? 1 : (uint32_t) src->ne[2],
                                     total_sum ? 1 : (uint32_t) ggml_nrows(dst) };

    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_sum_rows_pipeline(shader_lib_ctx);

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = total_sum ? 1 : ggml_nrows(dst);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

static bool ggml_webgpu_can_fuse_rms_norm_mul(const struct ggml_cgraph * cgraph, int node_idx) {
    if (!ggml_can_fuse(cgraph, node_idx, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
        return false;
    }

    // additional constraints specific to this fusion
    const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
    const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];

    GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
    // rms_norm only supports f32
    if (mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
        return false;
    }
    // if rms_norm is the B operand, then we don't handle broadcast
    if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
        return false;
    }
    // rms_norm shader assumes contiguous rows
    if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
        return false;
    }

    return true;
}

static bool ggml_webgpu_can_fuse_gdn_cache(const struct ggml_cgraph * cgraph, int node_idx, int & num_encoded_ops) {
    const ggml_tensor * gdn = cgraph->nodes[node_idx];

    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    if (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }

    const ggml_tensor * src_v     = gdn->src[2];
    const int64_t       S_v       = src_v->ne[0];
    const int64_t       H         = src_v->ne[1];
    const int64_t       n_tokens  = src_v->ne[2];
    const int64_t       n_seqs    = src_v->ne[3];
    const int64_t       D         = S_v * S_v * H;
    const int64_t       K         = ggml_get_op_params_i32(gdn, 0);  // snapshot slot count
    const int64_t       n_written = std::min<int64_t>(n_tokens, K);  // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    const size_t tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    // snapshot cpy is the first real node after the gdn (skip views/no-ops)
    const ggml_tensor * cpy     = nullptr;
    int                 cpy_idx = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_op_is_empty(n->op) || ggml_is_empty(n)) {
            continue;
        }
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return false;
        }
        cpy     = n;
        cpy_idx = j;
    }
    if (cpy == nullptr) {
        return false;
    }

    const ggml_tensor * cpy_src = cpy->src[0];  // view of the gdn snapshot tail
    const ggml_tensor * cpy_dst = cpy->src[1];  // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    if (cpy_src->op != GGML_OP_VIEW || cpy_src->view_src != gdn || cpy_src->view_offs != tail_off ||
        !ggml_is_contiguous(cpy_src)) {
        return false;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D (the per-seq stride the kernel
    // assumes). ggml_cpy pins src to the same element count.
    const std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    if (cpy_dst->op != GGML_OP_VIEW || cpy_dst->type != GGML_TYPE_F32 || cpy_dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), cpy_dst->ne) ||
        cpy_dst->nb[0] != ggml_type_size(GGML_TYPE_F32) || cpy_dst->nb[1] != (size_t) ggml_row_size(GGML_TYPE_F32, D)) {
        return false;
    }

    num_encoded_ops = cpy_idx - node_idx + 1;

    return true;
}

static webgpu_encoded_op ggml_webgpu_upscale(webgpu_context ctx, ggml_tensor * src, ggml_tensor * dst) {
    const uint32_t        mode_flags = (uint32_t) ggml_get_op_params_i32(dst, 0);
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),

                                     (uint32_t) (src->nb[0] / ggml_type_size(src->type)),
                                     (uint32_t) (src->nb[1] / ggml_type_size(src->type)),
                                     (uint32_t) (src->nb[2] / ggml_type_size(src->type)),
                                     (uint32_t) (src->nb[3] / ggml_type_size(src->type)),

                                     (uint32_t) (dst->nb[0] / ggml_type_size(dst->type)),
                                     (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
                                     (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
                                     (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),

                                     (uint32_t) src->ne[0],
                                     (uint32_t) src->ne[1],
                                     (uint32_t) src->ne[2],
                                     (uint32_t) src->ne[3],

                                     (uint32_t) dst->ne[0],
                                     (uint32_t) dst->ne[1],
                                     (uint32_t) dst->ne[2],
                                     (uint32_t) dst->ne[3],

                                     mode_flags };

    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_upscale_pipeline(shader_lib_ctx);
    auto *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);

    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// Returns the encoded command, or std::nullopt if the operation is a no-op
static std::optional<webgpu_encoded_op> ggml_webgpu_encode(webgpu_context ctx,
                                                           ggml_cgraph *  cgraph,
                                                           int            node_idx,
                                                           int &          num_encoded_ops) {
    ggml_tensor ** nodes = cgraph->nodes;
    ggml_tensor *  node  = nodes[node_idx];

    if (ggml_is_empty(node)) {
        return std::nullopt;
    }
    if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        return std::nullopt;
    }
    WEBGPU_LOG_DEBUG("ggml_webgpu_encode(" << node << ", " << ggml_op_name(node->op) << ")");

    ggml_tensor * src0 = node->src[0];
    ggml_tensor * src1 = node->src[1];
    ggml_tensor * src2 = node->src[2];

    switch (node->op) {
            // no-ops
        case GGML_OP_NONE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_RESHAPE:
            return std::nullopt;
        case GGML_OP_DUP:
        case GGML_OP_CPY:
        case GGML_OP_CONT:
            return ggml_webgpu_cpy(ctx, src0, node);
        case GGML_OP_SET:
            return ggml_webgpu_set(ctx, src0, src1, node);
        case GGML_OP_SET_ROWS:
            return ggml_webgpu_set_rows(ctx, src0, src1, node);
        case GGML_OP_GET_ROWS:
            return ggml_webgpu_get_rows(ctx, src0, src1, node);
        case GGML_OP_MUL_MAT:
            {
                ggml_tensor * gate = nullptr;
                ggml_tensor * up   = nullptr;
                if (const int glu_at = ggml_webgpu_mul_mat_pvr_glu_nodes(ctx, cgraph, node_idx, gate, up)) {
                    num_encoded_ops = glu_at + 1;
                    ggml_webgpu_mul_mat_vec_pvr_fusion fuse;
                    fuse.glu       = true;
                    fuse.stage_out = true;
                    return ggml_webgpu_mul_mat_vec_pvr_fused(ctx, gate->src[0], up->src[0], gate->src[1], nullptr,
                                                             nodes[node_idx + glu_at], fuse);
                }
                bool in_dst = false;
                if (ggml_tensor * other = ggml_webgpu_mul_mat_pvr_add_operand(ctx, cgraph, node_idx, in_dst)) {
                    num_encoded_ops = 2;
                    ggml_webgpu_mul_mat_vec_pvr_fusion fuse;
                    fuse.add        = true;
                    fuse.add_in_dst = in_dst;
                    return ggml_webgpu_mul_mat_vec_pvr_fused(ctx, src0, nullptr, src1, other, nodes[node_idx + 1], fuse);
                }
            }
            if (ggml_webgpu_can_fuse_mul_mat_pvr_swiglu(ctx, cgraph, node_idx)) {
                num_encoded_ops   = 2;
                ggml_tensor * glu = nodes[node_idx + 1];
                const bool    mm_is_gate = glu->src[0] == node;
                webgpu_encoded_op op     = ggml_webgpu_mul_mat_pvr(ctx, src0, src1, glu,
                                                                   mm_is_gate ? glu->src[1] : glu->src[0], mm_is_gate);
                ctx->pvr_staged_src1 = nullptr;  // the SwiGLU's output may sit where a staged src1 was
                return op;
            }
            return ggml_webgpu_mul_mat(ctx, src0, src1, node);
        case GGML_OP_MUL_MAT_ID:
            return ggml_webgpu_mul_mat_id(ctx, src0, src1, src2, node);
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_webgpu_flash_attn(ctx, src0, src1, src2, node->src[3], node->src[4], node);
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            return ggml_webgpu_binary_op(ctx, src0, src1, node);
        case GGML_OP_ADD_ID:
            return ggml_webgpu_add_id(ctx, src0, src1, src2, node);
        case GGML_OP_CONCAT:
            return ggml_webgpu_concat(ctx, src0, src1, node);
        case GGML_OP_REPEAT:
            return ggml_webgpu_repeat(ctx, src0, node);
        case GGML_OP_GROUP_NORM:
            return ggml_webgpu_group_norm(ctx, src0, node);
        case GGML_OP_RMS_NORM:
            {
                ggml_webgpu_pvr_post_norm post_norm;
                if (ggml_webgpu_pvr_post_norm_nodes(ctx, cgraph, node_idx, post_norm)) {
                    num_encoded_ops = 5;
                    return ggml_webgpu_pvr_rms_norm_add(ctx, post_norm);
                }
            }
            if (ggml_webgpu_can_fuse_rms_norm_mul(cgraph, node_idx)) {
                num_encoded_ops        = 2;
                ggml_tensor * mul_node = nodes[node_idx + 1];
                return ggml_webgpu_rms_norm_mul(ctx, src0, node, mul_node->src[0], mul_node->src[1], mul_node);
            } else {
                return ggml_webgpu_row_norm(ctx, src0, node);
            }
        case GGML_OP_NORM:
        case GGML_OP_L2_NORM:
            return ggml_webgpu_row_norm(ctx, src0, node);
        case GGML_OP_ROPE:
            return ggml_webgpu_rope(ctx, src0, src1, src2, node);
        case GGML_OP_GLU:
            return ggml_webgpu_glu(ctx, src0, src1, node);
        case GGML_OP_SCALE:
            return ggml_webgpu_scale(ctx, src0, node);
        case GGML_OP_SOFT_MAX:
            return ggml_webgpu_soft_max(ctx, src0, src1, src2, node);
        case GGML_OP_UNARY:
        case GGML_OP_CLAMP:
        case GGML_OP_FILL:
        case GGML_OP_LOG:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_DIAG:
        case GGML_OP_TRI:
            return ggml_webgpu_unary_op(ctx, src0, node);
        case GGML_OP_SOLVE_TRI:
            return ggml_webgpu_solve_tri(ctx, src0, src1, node);
        case GGML_OP_SSM_CONV:
            return ggml_webgpu_ssm_conv(ctx, src0, src1, node);
        case GGML_OP_SSM_SCAN:
            return ggml_webgpu_ssm_scan(ctx, src0, src1, src2, node->src[3], node->src[4], node->src[5], node->src[6],
                                        node);
        case GGML_OP_GATED_DELTA_NET:
            if (ggml_webgpu_can_fuse_gdn_cache(cgraph, node_idx, num_encoded_ops)) {
                ggml_tensor * dst_fuse = cgraph->nodes[node_idx + num_encoded_ops - 1]->src[1];
                return ggml_webgpu_gated_delta_net(ctx, src0, src1, src2, node->src[3], node->src[4], node->src[5],
                                                   node, dst_fuse);
            } else {
                return ggml_webgpu_gated_delta_net(ctx, src0, src1, src2, node->src[3], node->src[4], node->src[5],
                                                   node, nullptr);
            }
        case GGML_OP_ARANGE:
            return ggml_webgpu_arange(ctx, node);
        case GGML_OP_PAD:
            return ggml_webgpu_pad(ctx, src0, node);
        case GGML_OP_ARGMAX:
            return ggml_webgpu_argmax(ctx, src0, node);
        case GGML_OP_TOP_K:
            if (const int concat_at = ggml_webgpu_topk_pvr_fused_nodes(ctx, cgraph, node_idx)) {
                num_encoded_ops = concat_at + 1;
                return ggml_webgpu_topk_pvr(ctx, src0, node, cgraph->nodes[node_idx + concat_at]);
            }
            // we reuse the same argsort implementation for top_k
            return ggml_webgpu_argsort(ctx, src0, node);
        case GGML_OP_ARGSORT:
            return ggml_webgpu_argsort(ctx, src0, node);
        case GGML_OP_CUMSUM:
            return ggml_webgpu_cumsum(ctx, src0, node);
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
            return ggml_webgpu_sum_rows(ctx, src0, node);
        case GGML_OP_CONV_2D:
            return ggml_webgpu_conv_2d(ctx, src0, src1, node);
        case GGML_OP_CONV_2D_DW:
            return ggml_webgpu_conv_2d_dw(ctx, src0, src1, node);
        case GGML_OP_IM2COL:
            return ggml_webgpu_im2col(ctx, src0, src1, node);
        case GGML_OP_UPSCALE:
            return ggml_webgpu_upscale(ctx, src0, node);
        default:
            return std::nullopt;
    }
}

#ifdef GGML_WEBGPU_GPU_PROFILE
static void ggml_backend_webgpu_collect_profile_results(webgpu_context &                 ctx,
                                                        const std::vector<std::string> & pipeline_names,
                                                        uint32_t &                       num_inflight_batches) {
    if (pipeline_names.empty()) {
        return;
    }

    wgpu::CommandEncoder encoder = ctx->global_ctx->device.CreateCommandEncoder();
    encoder.ResolveQuerySet(ctx->profile_timestamp_query_set, 0, ctx->profile_timestamp_query_count,
                            ctx->profile_timestamp_dev_buf, 0);
    encoder.CopyBufferToBuffer(ctx->profile_timestamp_dev_buf, 0, ctx->profile_timestamp_host_buf, 0,
                               ctx->profile_timestamp_query_count * sizeof(uint64_t));

    wgpu::CommandBuffer profile_commands = encoder.Finish();
    ggml_backend_webgpu_submit_commands(ctx, profile_commands, num_inflight_batches);

    const size_t mapped_size = ctx->profile_timestamp_query_count * sizeof(uint64_t);
    GGML_ASSERT(ctx->profile_timestamp_query_count == 2 * pipeline_names.size());

    ggml_backend_webgpu_map_buffer(ctx->global_ctx, ctx->profile_timestamp_host_buf, wgpu::MapMode::Read, 0,
                                   mapped_size);
    const uint64_t * ts_data = (const uint64_t *) ctx->profile_timestamp_host_buf.GetConstMappedRange(0, mapped_size);

    for (size_t i = 0; i < pipeline_names.size(); ++i) {
        // WebGPU timestamps are in ns; convert to ms.
        const double elapsed_ms = double(ts_data[2 * i + 1] - ts_data[2 * i]) * 1e-6;
        ctx->shader_gpu_time_ms[pipeline_names[i]] += elapsed_ms;
    }

    ctx->profile_timestamp_host_buf.Unmap();
}
#endif

// Don't bother checking set_rows index overflow for now, since practically the WebGPU doesn't need to support
// models that would require it right now.
static void ggml_backend_webgpu_check_set_rows(webgpu_context & ctx, uint32_t & num_inflight_batches) {
#ifdef GGML_WEBGPU_CHECK_SET_ROWS
    wgpu::CommandEncoder encoder = ctx->global_ctx->device.CreateCommandEncoder();
    encoder.CopyBufferToBuffer(ctx->set_rows_dev_error_buf, 0, ctx->set_rows_host_error_buf, 0,
                               ctx->set_rows_host_error_buf.GetSize());
    wgpu::CommandBuffer commands = encoder.Finish();
    ggml_backend_webgpu_submit_commands(ctx, commands, num_inflight_batches);
    ggml_backend_webgpu_map_buffer(ctx->global_ctx, ctx->set_rows_host_error_buf, wgpu::MapMode::Read, 0,
                                   ctx->set_rows_host_error_buf.GetSize());
    const uint32_t * error_data = (const uint32_t *) ctx->set_rows_host_error_buf.GetConstMappedRange();
    if (*error_data) {
        GGML_ABORT("ggml_webgpu: SET_ROWS index > 2^32, unsupported.");
    }
    ctx->set_rows_host_error_buf.Unmap();
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(num_inflight_batches);
#endif
}

// PowerVR: LFM2/LFM3's short convolution on 1 to 8 tokens of one sequence
// (llm_build_lfm_base::build_lfm_shortconv) in one dispatch, shortconv_pvr.wgsl.
// The graph has, per conv layer: the old state gathered from the recurrent
// cache (GET_ROWS), B * X (MUL), the state and B * X concatenated (CONCAT),
// a CPY of each rollback snapshot of the state into the cache, the conv
// (SSM_CONV) and C times it (MUL), the next product's src1. The next layer's
// gather sits among them, so these nodes aren't consecutive: a pass over the
// graph finds them, and the fused op runs at the last (the MUL), the others
// being skipped. It reads the old state from the cache itself, as the
// gather's output may be reused by then, and writes the snapshots. The conv
// has `width` taps (LFM2: 3, LFM3: 7) and the state width - 1 values per
// channel.
struct ggml_webgpu_pvr_shortconv {
    ggml_tensor *              bcx   = nullptr;  // the in-projection's output (B, C and X are views of it)
    ggml_tensor *              b     = nullptr;
    ggml_tensor *              c     = nullptr;
    ggml_tensor *              x     = nullptr;
    ggml_tensor *              cache = nullptr;  // the recurrent cache (conv states)
    ggml_tensor *              rows  = nullptr;  // the gather's view of it (a state per row)
    ggml_tensor *              idx   = nullptr;  // the row to gather
    ggml_tensor *              w     = nullptr;  // the conv weights
    ggml_tensor *              y     = nullptr;  // C times the conv
    std::vector<ggml_tensor *> snaps;            // the snapshots' rows in the cache, slot 0 first
    uint32_t                   n     = 0;        // tokens
    uint32_t                   width = 0;        // the conv's taps
};

static const ggml_tensor * ggml_webgpu_base(const ggml_tensor * t) {
    return t->view_src ? t->view_src : t;
}

static ggml_tensor * ggml_webgpu_base(ggml_tensor * t) {
    return t->view_src ? t->view_src : t;
}

// the tensor under a chain of views (RESHAPE, VIEW, TRANSPOSE: what's left is a computed one)
static const ggml_tensor * ggml_webgpu_unview(const ggml_tensor * t) {
    while (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW) {
        t = t->src[0];
    }
    return t;
}

static ggml_tensor * ggml_webgpu_unview(ggml_tensor * t) {
    while (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW) {
        t = t->src[0];
    }
    return t;
}

// a [ch, n] view of bcx with contiguous rows
static bool ggml_webgpu_pvr_shortconv_part(const ggml_tensor * v, const ggml_tensor * bcx, int64_t ch, int64_t n) {
    return v->view_src == bcx && v->type == GGML_TYPE_F32 && v->ne[0] == ch && v->ne[1] == n && v->ne[2] == 1 &&
           v->ne[3] == 1 && v->nb[0] == sizeof(float) && v->nb[1] == bcx->nb[1] && bcx->type == GGML_TYPE_F32 &&
           ggml_webgpu_tensor_offset(v) % sizeof(float) == 0;
}

// The fused op's tensors, from the nodes where at says they are.
static ggml_webgpu_pvr_shortconv ggml_webgpu_pvr_shortconv_get(const ggml_cgraph *                 cgraph,
                                                               const ggml_webgpu_pvr_shortconv_at & at) {
    ggml_tensor * gr = cgraph->nodes[at.gi];
    ggml_tensor * bx = cgraph->nodes[at.bi];
    ggml_tensor * sc = cgraph->nodes[at.si];
    ggml_tensor * y  = cgraph->nodes[at.yi];
    ggml_webgpu_pvr_shortconv f;
    f.rows  = gr->src[0];
    f.idx   = gr->src[1];
    f.cache = ggml_webgpu_base(f.rows);
    f.b     = bx->src[0];
    f.x     = bx->src[1];
    f.bcx   = ggml_webgpu_base(f.b);
    f.w     = sc->src[1];
    f.y     = y;
    f.c     = y->src[0] == sc ? y->src[1] : y->src[0];
    for (int i : at.snaps) {
        f.snaps.push_back(cgraph->nodes[i]->src[1]);
    }
    f.n     = (uint32_t) sc->ne[1];
    f.width = (uint32_t) f.w->ne[0];
    return f;
}

// Finds the fused short convolutions of cgraph: marks the nodes they replace
// in skip, and records each by the index of the node where it runs.
static void ggml_webgpu_pvr_shortconv_nodes(webgpu_context &                                        ctx,
                                            ggml_cgraph *                                           cgraph,
                                            std::vector<uint8_t> &                                  skip,
                                            std::unordered_map<int, ggml_webgpu_pvr_shortconv_at> & fused) {
    const int n_nodes = cgraph->n_nodes;
    bool      any     = false;
    for (int i = 0; i < n_nodes && !any; i++) {
        any = cgraph->nodes[i]->op == GGML_OP_SSM_CONV;
    }
    if (!any) {
        return;
    }
    // the nodes that read each tensor (through views): (tensor, node) pairs, sorted
    std::vector<std::pair<const ggml_tensor *, int>> reads;
    reads.reserve(4 * n_nodes);
    for (int i = 0; i < n_nodes; i++) {
        const ggml_tensor * t = cgraph->nodes[i];
        if (ggml_op_is_empty(t->op) || ggml_is_empty(t)) {
            continue;
        }
        for (int j = 0; j < GGML_MAX_SRC && t->src[j]; j++) {
            reads.push_back({ ggml_webgpu_base(t->src[j]), i });
        }
    }
    std::sort(reads.begin(), reads.end());
    // whether every node reading t is in allowed or more
    auto users_in = [&](const ggml_tensor * t, std::initializer_list<int> allowed, const std::vector<int> & more) {
        auto it = std::lower_bound(reads.begin(), reads.end(), std::pair<const ggml_tensor *, int>{ t, INT_MIN });
        for (; it != reads.end() && it->first == t; ++it) {
            const int u = it->second;
            if (std::find(allowed.begin(), allowed.end(), u) == allowed.end() &&
                std::find(more.begin(), more.end(), u) == more.end()) {
                return false;
            }
        }
        return true;
    };
    // the index of node t among the 64 before at, or -1
    auto index_before = [&](const ggml_tensor * t, int at) {
        for (int i = at - 1; i >= 0 && i >= at - 64; i--) {
            if (cgraph->nodes[i] == t) {
                return i;
            }
        }
        return -1;
    };
    for (int si = 0; si < n_nodes; si++) {
        ggml_tensor * sc = cgraph->nodes[si];
        if (sc->op != GGML_OP_SSM_CONV || (sc->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        ggml_tensor * cat = sc->src[0];
        ggml_tensor * w   = sc->src[1];
        const int64_t ch  = sc->ne[0];
        const int64_t n   = sc->ne[1];
        // the conv's taps, and the state's values per channel (width - 1); not 4 taps, which
        // ssm_conv.wgsl sums as one expression rather than as fused multiply-adds
        const int64_t width = w->ne[0];
        const int64_t ds    = width - 1;
        if (sc->type != GGML_TYPE_F32 || n < 1 || n > 8 || sc->ne[2] != 1 || sc->ne[3] != 1 || !ggml_is_contiguous(sc) ||
            w->type != GGML_TYPE_F32 || width < 2 || width > 8 || width == 4 || w->ne[1] != ch ||
            !ggml_is_contiguous(w) || cat->op != GGML_OP_CONCAT || ggml_get_op_params_i32(cat, 0) != 0 ||
            cat->type != GGML_TYPE_F32 || cat->ne[0] != n + ds || cat->ne[1] != ch || cat->ne[2] != 1 ||
            !ggml_is_contiguous(cat)) {
            continue;
        }
        const int ci = index_before(cat, si);
        if (ci < 0) {
            continue;
        }
        // the old state: a [width - 1, ch] reshape of a one-row gather from the cache
        ggml_tensor *       st = cat->src[0];
        ggml_tensor *       gr = ggml_webgpu_unview(st);
        if (st->ne[0] != ds || st->ne[1] != ch || !ggml_is_contiguous(st) || st->type != GGML_TYPE_F32 ||
            gr->op != GGML_OP_GET_ROWS || gr->type != GGML_TYPE_F32 || gr->ne[0] != ds * ch || gr->ne[1] != 1 ||
            gr->ne[2] != 1 || ggml_webgpu_tensor_offset(st) != ggml_webgpu_tensor_offset(gr)) {
            continue;
        }
        const int gi = index_before(gr, ci);
        if (gi < 0) {
            continue;
        }
        ggml_tensor * rows = gr->src[0];
        ggml_tensor * idx  = gr->src[1];
        ggml_tensor * cache = ggml_webgpu_base(rows);
        if (rows->type != GGML_TYPE_F32 || rows->nb[0] != sizeof(float) || rows->ne[0] != ds * ch ||
            idx->type != GGML_TYPE_I32 || idx->ne[0] < 1 || cache->type != GGML_TYPE_F32 ||
            ggml_webgpu_tensor_offset(rows) % sizeof(float) != 0 || rows->nb[1] % sizeof(float) != 0) {
            continue;
        }
        // B * X, transposed into the CONCAT
        const ggml_tensor * tr = cat->src[1];
        if (tr->op != GGML_OP_TRANSPOSE) {
            continue;
        }
        ggml_tensor * bx = tr->src[0];
        if (bx->op != GGML_OP_MUL || bx->type != GGML_TYPE_F32 || bx->ne[0] != ch || bx->ne[1] != n ||
            bx->ne[2] != 1) {
            continue;
        }
        const int bi = index_before(bx, ci);
        if (bi < 0) {
            continue;
        }
        ggml_tensor * bcx = ggml_webgpu_base(bx->src[0]);
        if (!ggml_webgpu_pvr_shortconv_part(bx->src[0], bcx, ch, n) ||
            !ggml_webgpu_pvr_shortconv_part(bx->src[1], bcx, ch, n)) {
            continue;
        }
        // C times the conv: the next node that reads the conv
        int yi = -1;
        for (int i = si + 1; i < n_nodes && i < si + 8; i++) {
            const ggml_tensor * t = cgraph->nodes[i];
            if (t->op == GGML_OP_MUL && (t->src[0] == sc || t->src[1] == sc)) {
                yi = i;
                break;
            }
        }
        if (yi < 0) {
            continue;
        }
        ggml_tensor * y    = cgraph->nodes[yi];
        ggml_tensor * cpar = y->src[0] == sc ? y->src[1] : y->src[0];
        if (y->type != GGML_TYPE_F32 || y->ne[0] != ch || y->ne[1] != n || y->ne[2] != 1 || !ggml_is_contiguous(y) ||
            !ggml_webgpu_pvr_shortconv_part(cpar, bcx, ch, n)) {
            continue;
        }
        // the snapshots: CPYs of [width - 1, ch] views of the CONCAT into rows of the cache, slot s at token n - s
        std::vector<int> snap_nodes;
        bool             ok = true;
        for (int i = ci + 1; i < si && ok; i++) {
            ggml_tensor * t = cgraph->nodes[i];
            if (t->op != GGML_OP_CPY || ggml_is_empty(t)) {
                continue;
            }
            const ggml_tensor * src = t->src[0];
            ggml_tensor *       dst = t->src[1];
            if (src->view_src != cat) {
                continue;
            }
            const int64_t s = (int64_t) snap_nodes.size();
            ok = src->ne[0] == ds && src->ne[1] == ch && src->nb[1] == cat->nb[1] &&
                 src->view_offs == (size_t) (n - s) * sizeof(float) && ggml_webgpu_base(dst) == cache &&
                 ggml_nelements(dst) == ds * ch && ggml_is_contiguous(dst) && t->type == GGML_TYPE_F32 &&
                 dst->type == GGML_TYPE_F32 && s < n;
            snap_nodes.push_back(i);
        }
        if (!ok || snap_nodes.empty()) {
            continue;
        }
        // nothing else reads what the skipped nodes write, and they come before the MUL
        if (!(gi < yi && bi < yi && ci < yi && si < yi) || !users_in(cat, { si }, snap_nodes) ||
            !users_in(bx, { ci }, {}) || !users_in(gr, { ci }, {}) || !users_in(sc, { yi }, {})) {
            continue;
        }
        // nor touches the cache between the gather and the MUL (the gather's reads and the CPYs' writes
        // happen at the MUL), nor writes where the row index is: the gather was its last reader, so its
        // memory may be another node's by the MUL (the last conv layer's)
        {
            bool touched = false;
            for (int i = gi + 1; i < yi && !touched; i++) {
                const ggml_tensor * t = cgraph->nodes[i];
                if (ggml_op_is_empty(t->op) || ggml_is_empty(t) || i == bi || i == ci || i == si ||
                    std::find(snap_nodes.begin(), snap_nodes.end(), i) != snap_nodes.end()) {
                    continue;
                }
                touched = ggml_webgpu_base(t) == cache || ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, t, idx);
                for (int j = 0; j < GGML_MAX_SRC && t->src[j] && !touched; j++) {
                    touched = ggml_webgpu_base(t->src[j]) == cache;
                }
            }
            if (touched) {
                continue;
            }
        }
        // y overlaps none of what's read
        if (ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, y, bcx) ||
            ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, y, idx) ||
            ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, y, cache) ||
            ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, y, w) ||
            ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, cache, bcx) ||
            ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, cache, idx) ||
            ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, cache, w)) {
            continue;
        }
        ggml_webgpu_pvr_shortconv_at at;
        at.gi    = gi;
        at.bi    = bi;
        at.ci    = ci;
        at.si    = si;
        at.yi    = yi;
        at.snaps = snap_nodes;
        fused[yi] = at;
        for (int i : { gi, bi, ci, si }) {
            skip[i] = 1;
        }
        for (int i : snap_nodes) {
            skip[i] = 1;
        }
    }
}

static webgpu_encoded_op ggml_webgpu_pvr_shortconv_op(webgpu_context & ctx, const ggml_webgpu_pvr_shortconv & f) {
    const size_t   bcx_at   = ggml_webgpu_tensor_align_offset(ctx, f.bcx);
    const size_t   cache_at = ggml_webgpu_tensor_align_offset(ctx, f.cache);
    const uint32_t ch       = (uint32_t) f.y->ne[0];
    auto           el       = [](size_t bytes) { return (uint32_t) (bytes / sizeof(float)); };
    std::vector<uint32_t> params = {
        el(ggml_webgpu_tensor_offset(f.b) - bcx_at),
        el(ggml_webgpu_tensor_offset(f.c) - bcx_at),
        el(ggml_webgpu_tensor_offset(f.x) - bcx_at),
        el(f.b->nb[1]),
        el(ggml_webgpu_tensor_offset(f.rows) - cache_at),
        el(f.rows->nb[1]),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, f.idx) / sizeof(int32_t)),
        el(ggml_webgpu_tensor_misalignment(ctx, f.w)),
        el(ggml_webgpu_tensor_misalignment(ctx, f.y)),
        el(f.y->nb[1]),
        ch,
        (uint32_t) f.snaps.size(),
    };
    for (size_t s = 0; s < 8; s++) {
        params.push_back(s < f.snaps.size() ? el(ggml_webgpu_tensor_offset(f.snaps[s]) - cache_at) : 0u);
    }
    const size_t bytes = ROUNDUP_POW2(ggml_nbytes(f.y), 16);
    ggml_webgpu_pvr_stage_next(ctx, bytes);
    // the weights bound read-only when they sit in another buffer than the outputs (y, the cache)
    const WGPUBuffer w_buf = ggml_webgpu_tensor_buf(f.w).Get();
    const bool w_ro = w_buf != ggml_webgpu_tensor_buf(f.y).Get() && w_buf != ggml_webgpu_tensor_buf(f.cache).Get();
    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, f.bcx),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, f.cache),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, f.idx),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 3, f.w),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 4, f.y),
        ggml_webgpu_make_bind_group_entry(5, ctx->pvr_staging_buf[ctx->pvr_staged_idx], 0, bytes),
    };
    ctx->pvr_staged_src1  = f.y;
    ctx->pvr_staged_f16   = false;
    ctx->pvr_staged_by_op = true;
    webgpu_pipeline pipeline = ctx->shader_lib->get_shortconv_pvr_pipeline(f.n, f.width, w_ro);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, CEIL_DIV(ch, WEBGPU_SHORTCONV_PVR_WG_SIZE));
}

// The graph's signature for the short-conv pass (see webgpu_context_struct::pvr_sc): per node, the
// node, its output, op, shape, strides and sources (and theirs).
static void ggml_webgpu_pvr_graph_sig(const ggml_cgraph * cgraph, std::vector<uintptr_t> & sig) {
    sig.clear();
    sig.reserve((size_t) cgraph->n_nodes * 12);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * t = cgraph->nodes[i];
        sig.push_back((uintptr_t) t);
        sig.push_back((uintptr_t) t->data);
        sig.push_back((uintptr_t) t->op);
        for (int d = 0; d < GGML_MAX_DIMS; d++) {
            sig.push_back((uintptr_t) t->ne[d]);
        }
        sig.push_back((uintptr_t) t->nb[1]);
        for (int j = 0; j < GGML_MAX_SRC && t->src[j]; j++) {
            sig.push_back((uintptr_t) t->src[j]);
            sig.push_back((uintptr_t) t->src[j]->data);
        }
        sig.push_back(0);
    }
}

static ggml_status ggml_backend_webgpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_graph_compute(" << cgraph->n_nodes << " nodes)");

    ggml_backend_webgpu_context * backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    webgpu_context                ctx         = backend_ctx->webgpu_ctx;

    WEBGPU_CPU_PROFILE_TOTAL_START(graph_compute);

    std::vector<webgpu_encoded_op> commands;

    uint32_t num_batched_kernels  = 0;
    uint32_t num_inflight_batches = 0;
    bool     contains_set_rows    = false;
    int      num_encoded_ops      = 1;
    int      node_idx             = 0;

#ifdef GGML_WEBGPU_GPU_PROFILE
    ctx->profile_timestamp_query_count = 0;
    std::vector<std::string> profile_pipeline_names;
#endif

    ctx->active_command_encoder = ctx->global_ctx->device.CreateCommandEncoder();
    if (ctx->batch_compute_passes) {
        ctx->active_compute_pass = ctx->active_command_encoder.BeginComputePass();
    }
    ctx->pvr_staged_src1 = nullptr;

    // PowerVR: the short convolutions run as one op each (see ggml_webgpu_pvr_shortconv); a graph run
    // again (the same nodes, outputs and sources) reuses what the pass over it found
    std::vector<uint8_t> *                                 pvr_skip      = nullptr;
    std::unordered_map<int, ggml_webgpu_pvr_shortconv_at> * pvr_shortconv = nullptr;
    if (ggml_webgpu_is_pvr(ctx->global_ctx)) {
        auto &                 sc = ctx->pvr_sc;
        std::vector<uintptr_t> sig;
        ggml_webgpu_pvr_graph_sig(cgraph, sig);
        if (sc.graph != cgraph || sc.sig != sig) {
            sc.graph = cgraph;
            sc.sig   = std::move(sig);
            sc.skip.assign(cgraph->n_nodes, 0);
            sc.fused.clear();
            ggml_webgpu_pvr_shortconv_nodes(ctx, cgraph, sc.skip, sc.fused);
        }
        pvr_skip      = &sc.skip;
        pvr_shortconv = &sc.fused;
    }

    while (node_idx < cgraph->n_nodes) {
        if (cgraph->nodes[node_idx]->op == GGML_OP_SET_ROWS) {
            contains_set_rows = true;
        }
        ctx->pvr_staged_by_op = false;
        if (pvr_skip && (*pvr_skip)[node_idx]) {
            node_idx++;
            continue;
        }
        std::optional<webgpu_encoded_op> fused_cmd;
        if (pvr_shortconv && !pvr_shortconv->empty()) {
            auto it = pvr_shortconv->find(node_idx);
            if (it != pvr_shortconv->end()) {
                fused_cmd = ggml_webgpu_pvr_shortconv_op(ctx, ggml_webgpu_pvr_shortconv_get(cgraph, it->second));
            }
        }
        if (auto cmd = fused_cmd ? fused_cmd : ggml_webgpu_encode(ctx, cgraph, node_idx, num_encoded_ops)) {
            commands.push_back(*cmd);
            num_batched_kernels += cmd.value().num_kernels;
#ifdef GGML_WEBGPU_GPU_PROFILE
            profile_pipeline_names.insert(profile_pipeline_names.end(), cmd->pipeline_names.begin(),
                                          cmd->pipeline_names.end());
#endif
        }

        // PowerVR: the GPU waits for the first submit of each graph, so submit
        // the first batch early (after a few dispatches), the rest as usual
        const uint32_t batch_size = ggml_webgpu_is_pvr(ctx->global_ctx) && num_inflight_batches == 0 ?
                                        WEBGPU_PVR_FIRST_SUBMIT_BATCH_SIZE :
                                        ctx->global_ctx->command_submit_batch_size;
        if (num_batched_kernels >= batch_size) {
            if (ctx->active_compute_pass) {
                ctx->active_compute_pass.End();
            }
            num_batched_kernels                = 0;
            wgpu::CommandBuffer batch_commands = ctx->active_command_encoder.Finish();
            ggml_backend_webgpu_submit_commands(ctx, batch_commands, num_inflight_batches);

            // reset state for next batch
            ctx->active_command_encoder = ctx->global_ctx->device.CreateCommandEncoder();
            if (ctx->batch_compute_passes) {
                ctx->active_compute_pass = ctx->active_command_encoder.BeginComputePass();
            }
            ctx->param_arena.reset();
            commands.clear();
#ifdef GGML_WEBGPU_GPU_PROFILE
            // flush before the next batch can overflow the QuerySet
            if (ctx->profile_timestamp_query_count + 2 * ctx->global_ctx->command_submit_batch_size >=
                WEBGPU_MAX_PROFILE_QUERY_COUNT) {
                ggml_backend_webgpu_collect_profile_results(ctx, profile_pipeline_names, num_inflight_batches);
                // reset profile timestamp state
                ctx->profile_timestamp_query_count = 0;
                profile_pipeline_names.clear();
            }
#endif
        }

        // an op that wrote where the staged src1 sat (that tensor's memory
        // now holds another) makes the staged copy stale
        if (ctx->pvr_staged_src1 != nullptr && !ctx->pvr_staged_by_op) {
            for (int i = node_idx; i < node_idx + num_encoded_ops && i < cgraph->n_nodes; i++) {
                const ggml_tensor * t = cgraph->nodes[i];
                if (!ggml_op_is_empty(t->op) && t != ctx->pvr_staged_src1 &&
                    ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, t, ctx->pvr_staged_src1)) {
                    ctx->pvr_staged_src1 = nullptr;
                    break;
                }
            }
        }

        node_idx += num_encoded_ops;
        num_encoded_ops = 1;
    }

    if (ctx->active_compute_pass) {
        ctx->active_compute_pass.End();
        ctx->active_compute_pass = nullptr;
    }

    if (num_batched_kernels > 0) {
        wgpu::CommandBuffer batch_commands = ctx->active_command_encoder.Finish();
        ggml_backend_webgpu_submit_commands(ctx, batch_commands, num_inflight_batches);
        ctx->param_arena.reset();
        commands.clear();
    }
    ctx->active_command_encoder = nullptr;

#ifdef GGML_WEBGPU_GPU_PROFILE
    ggml_backend_webgpu_collect_profile_results(ctx, profile_pipeline_names, num_inflight_batches);
#endif

    if (contains_set_rows) {
        ggml_backend_webgpu_check_set_rows(ctx, num_inflight_batches);
    }

    WEBGPU_CPU_PROFILE_TOTAL_END(graph_compute, ctx->global_ctx);
    return GGML_STATUS_SUCCESS;
}

struct ggml_backend_webgpu_event_context {
    webgpu_global_context global_ctx;
    wgpu::Future          future;
    bool                  recorded = false;
};

static ggml_backend_event_t ggml_backend_webgpu_device_event_new(ggml_backend_dev_t device) {
    ggml_backend_webgpu_device_context * dev_ctx = (ggml_backend_webgpu_device_context *) device->context;

    auto * event_ctx      = new ggml_backend_webgpu_event_context();
    event_ctx->global_ctx = dev_ctx->webgpu_global_ctx;

    auto * event   = new ggml_backend_event;
    event->device  = device;
    event->context = event_ctx;
    return event;
}

static void ggml_backend_webgpu_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    delete static_cast<ggml_backend_webgpu_event_context *>(event->context);
    delete event;
}

static void ggml_backend_webgpu_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    ggml_backend_webgpu_event_context * event_ctx = (ggml_backend_webgpu_event_context *) event->context;
    if (!event_ctx->recorded) {
        return;
    }
    wgpu::WaitStatus status =
        event_ctx->global_ctx->instance.WaitAny(event_ctx->future, WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);
    if (status == wgpu::WaitStatus::TimedOut) {
        GGML_ABORT("ggml_webgpu: event_synchronize timed out after %u ms\n", WEBGPU_RUNTIME_WAIT_TIMEOUT_MS);
    }
    event_ctx->recorded = false;
}

static void ggml_backend_webgpu_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_webgpu_context *       backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    ggml_backend_webgpu_event_context * event_ctx   = (ggml_backend_webgpu_event_context *) event->context;

    event_ctx->future = backend_ctx->webgpu_ctx->global_ctx->queue.OnSubmittedWorkDone(
        wgpu::CallbackMode::AllowSpontaneous, [](wgpu::QueueWorkDoneStatus, wgpu::StringView) {});
    event_ctx->recorded = true;
}

static void ggml_backend_webgpu_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_UNUSED(backend);
    ggml_backend_webgpu_device_event_synchronize(nullptr, event);
}

static void ggml_backend_webgpu_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                  ggml_tensor *         tensor,
                                                  const void *          data,
                                                  size_t                offset,
                                                  size_t                size);

static void ggml_backend_webgpu_set_tensor_async(ggml_backend_t backend,
                                                 ggml_tensor *  tensor,
                                                 const void *   data,
                                                 size_t         offset,
                                                 size_t         size) {
    GGML_UNUSED(backend);
    auto * buf_ctx      = (ggml_backend_webgpu_buffer_context *) tensor->buffer->context;
    // PowerVR: a tensor that is or becomes repacked (see ggml_webgpu_pvr_repacked_marker) is written by
    // set_tensor, which moves the bytes into its layout
    if (ggml_webgpu_pvr_is_repacked(tensor) ||
        (offset == 0 && size == ggml_nbytes(tensor) && ggml_webgpu_pvr_repackable(buf_ctx->global_ctx, tensor))) {
        ggml_backend_webgpu_buffer_set_tensor(tensor->buffer, tensor, data, offset, size);
        return;
    }
    size_t total_offset = ggml_webgpu_tensor_offset(tensor) + offset;

    // Write aligned portion
    buf_ctx->global_ctx->pvr_write_epoch++;
    buf_ctx->global_ctx->queue.WriteBuffer(buf_ctx->buffer, total_offset, data, (size / 4) * 4);

    if (size % 4 != 0) {
        // If size is not a multiple of 4, we need to memset the remaining bytes
        size_t remaining_size = size % 4;

        // pack the remaining bytes into a uint32_t
        uint32_t val32 = 0;

        for (size_t i = 0; i < remaining_size; i++) {
            ((uint8_t *) &val32)[i] = ((const uint8_t *) data)[size - remaining_size + i];
        }
        // memset the remaining bytes
        ggml_backend_webgpu_buffer_memset(buf_ctx->global_ctx, buf_ctx->buffer, val32,
                                          total_offset + (size - remaining_size), remaining_size);
    }
}

static void ggml_backend_webgpu_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                  const ggml_tensor *   tensor,
                                                  void *                data,
                                                  size_t                offset,
                                                  size_t                size);

// On PowerVR a read back (copy, then map) already waits for all the work
// submitted before it, and each wait costs a few ms on Android: read without
// waiting for the queue first, as the generic fallback (synchronize, then
// read) would. This saves a wait per token, before the logits are read.
static void ggml_backend_webgpu_get_tensor_async(ggml_backend_t      backend,
                                                 const ggml_tensor * tensor,
                                                 void *              data,
                                                 size_t              offset,
                                                 size_t              size) {
    ggml_backend_webgpu_context * backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    if (!ggml_webgpu_is_pvr(backend_ctx->webgpu_ctx->global_ctx)) {
        ggml_backend_synchronize(backend);
    }
    ggml_backend_tensor_get(tensor, data, offset, size);
}

static void ggml_backend_webgpu_synchronize(ggml_backend_t backend) {
    ggml_backend_webgpu_context * backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    // The scheduler synchronizes about 20 times per token, and on PowerVR
    // (Android) each wait costs a few ms even with nothing in flight: skip the
    // ones with no work submitted since the last.
    if (ggml_webgpu_is_pvr(backend_ctx->webgpu_ctx->global_ctx) && backend_ctx->webgpu_ctx->global_ctx->queue_idle) {
        return;
    }
    ggml_backend_webgpu_wait_queue(backend_ctx->webgpu_ctx->global_ctx);
}

static ggml_backend_i ggml_backend_webgpu_i = {
    /* .get_name                = */ ggml_backend_webgpu_name,
    /* .free                    = */ ggml_backend_webgpu_free,
    /* .set_tensor_async        = */ ggml_backend_webgpu_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_webgpu_get_tensor_async,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_webgpu_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_webgpu_graph_compute,
    /* .event_record            = */ ggml_backend_webgpu_event_record,
    /* .event_wait              = */ ggml_backend_webgpu_event_wait,
    /* .graph_optimize          = */ NULL,
};

/* End GGML Backend Interface */

/* GGML Backend Buffer Interface */

static void ggml_backend_webgpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_webgpu_buffer_context * ctx = static_cast<ggml_backend_webgpu_buffer_context *>(buffer->context);
    if (ctx != nullptr && ctx->buffer != nullptr) {
        if (ctx->global_ctx->pvr_readback_src == ctx->buffer.Get()) {
            ctx->global_ctx->pvr_readback_src = nullptr;  // a new buffer may get the same handle
        }
        ctx->buffer.Destroy();
        delete ctx;
    }
}

// Returns the "fake" base pointer.
static void * ggml_backend_webgpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return webgpu_ptr_base;
}

static void ggml_backend_webgpu_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                  ggml_tensor *         tensor,
                                                  const void *          data,
                                                  size_t                offset,
                                                  size_t                size);

static void ggml_backend_webgpu_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                     ggml_tensor *         tensor,
                                                     uint8_t               value,
                                                     size_t                offset,
                                                     size_t                size) {
    if (size == 0) {
        WEBGPU_LOG_DEBUG(
            "ggml_backend_webgpu_buffer_memset_tensor: size is zero, "
            "nothing to do.");
        return;
    }

    WEBGPU_CPU_PROFILE_TOTAL_START(memset_tensor);

    ggml_backend_webgpu_buffer_context * buf_ctx = (ggml_backend_webgpu_buffer_context *) buffer->context;

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_memset_tensor(" << buf_ctx->label << ", " << tensor << ", " << value
                                                                 << ", " << offset << ", " << size << ")");

    if (ggml_webgpu_pvr_is_repacked(tensor) && (offset != 0 || size != ggml_nbytes(tensor))) {
        // repacking moves bytes, so only setting the whole tensor is the same in both layouts
        std::vector<uint8_t> part(size, value);
        ggml_backend_webgpu_buffer_set_tensor(buffer, tensor, part.data(), offset, size);
        WEBGPU_CPU_PROFILE_TOTAL_END(memset_tensor, buf_ctx->global_ctx);
        return;
    }

    size_t total_offset = ggml_webgpu_tensor_offset(tensor) + offset;

    // This is a trick to set all bytes of a u32 to the same 1 byte value.
    uint32_t val32 = (uint32_t) value * 0x01010101;
    ggml_backend_webgpu_buffer_memset(buf_ctx->global_ctx, buf_ctx->buffer, val32, total_offset, size);
    WEBGPU_CPU_PROFILE_TOTAL_END(memset_tensor, buf_ctx->global_ctx);
}

static void ggml_backend_webgpu_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                  const ggml_tensor *   tensor,
                                                  void *                data,
                                                  size_t                offset,
                                                  size_t                size);

// Uploads a whole Q4_K or Q6_K tensor in layout A (see
// ggml_webgpu_pvr_repacked_marker), a few thousand rows of a block column at
// a time: in layout A each part of those rows' blocks is contiguous.
static void ggml_webgpu_pvr_upload_repacked(ggml_backend_webgpu_buffer_context * buf_ctx,
                                            ggml_tensor *                        tensor,
                                            const void *                         data) {
    const int64_t m    = tensor->ne[1];
    const int64_t nb   = tensor->ne[0] / 256;
    const int64_t nblk = m * nb;
    const size_t  base = ggml_webgpu_tensor_offset(tensor);
    const size_t  bs   = ggml_type_size(tensor->type);
    // the parts of a block: (offset of its plane, bytes per block)
    std::vector<std::pair<size_t, size_t>> parts;
    if (tensor->type == GGML_TYPE_Q4_K) {
        parts = { { 0, 16 }, { (size_t) nblk * 16, 128 } };
    } else {
        parts = { { 0, 128 }, { (size_t) nblk * 128, 64 }, { (size_t) nblk * 192, 16 }, { (size_t) nblk * 208, 2 } };
    }
    // rows per chunk: a multiple of 4, so that every write is 4-byte aligned
    const int64_t        chunk = std::min<int64_t>(m, 4096);
    std::vector<uint8_t> col((size_t) chunk * bs);
    std::vector<uint8_t> a((size_t) chunk * bs);
    for (int64_t b = 0; b < nb; b++) {
        for (int64_t r0 = 0; r0 < m; r0 += chunk) {
            const int64_t rows = std::min(chunk, m - r0);
            // the chunk's blocks as a tensor of their own (one block per row), repacked
            for (int64_t r = 0; r < rows; r++) {
                memcpy(col.data() + r * bs, (const uint8_t *) data + ((r0 + r) * nb + b) * bs, bs);
            }
            ggml_webgpu_pvr_repack_column(tensor->type, rows, 1, 0, true, col.data(), a.data());
            // its planes are the chunk's parts of the whole tensor's planes
            size_t src_off = 0;
            for (const auto & [plane, per_blk] : parts) {
                const size_t bytes = (size_t) rows * per_blk;
                buf_ctx->global_ctx->pvr_write_epoch++;
                buf_ctx->global_ctx->queue.WriteBuffer(buf_ctx->buffer, base + plane + (size_t) (b * m + r0) * per_blk,
                                                       a.data() + src_off, bytes);
                src_off += bytes;
            }
        }
    }
    tensor->extra = &ggml_webgpu_pvr_repacked_marker;
}

static void ggml_backend_webgpu_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                  ggml_tensor *         tensor,
                                                  const void *          data,
                                                  size_t                offset,
                                                  size_t                size) {
    WEBGPU_CPU_PROFILE_TOTAL_START(set_tensor);
    ggml_backend_webgpu_buffer_context * buf_ctx = (ggml_backend_webgpu_buffer_context *) buffer->context;

    if (offset == 0 && size == ggml_nbytes(tensor) && ggml_webgpu_pvr_repackable(buf_ctx->global_ctx, tensor)) {
        ggml_webgpu_pvr_upload_repacked(buf_ctx, tensor, data);
        WEBGPU_CPU_PROFILE_TOTAL_END(set_tensor, buf_ctx->global_ctx);
        return;
    }
    if (ggml_webgpu_pvr_is_repacked(tensor)) {
        // part of a repacked tensor: read it back whole, write the part, and upload it again
        std::vector<uint8_t> whole(ggml_nbytes(tensor));
        ggml_backend_webgpu_buffer_get_tensor(buffer, tensor, whole.data(), 0, whole.size());
        memcpy(whole.data() + offset, data, size);
        ggml_webgpu_pvr_upload_repacked(buf_ctx, tensor, whole.data());
        WEBGPU_CPU_PROFILE_TOTAL_END(set_tensor, buf_ctx->global_ctx);
        return;
    }

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_set_tensor(" << buf_ctx->label << ", " << tensor << ", " << data
                                                              << ", " << offset << ", " << size << ")");

    size_t total_offset = ggml_webgpu_tensor_offset(tensor) + offset;

    // WriteBuffer needs the offset and the size to be multiples of 4.
    // Write the misaligned head bytes using compute memset, then increment total_offset
    // and data pointer so that they are 4-aligned.
    if (total_offset % 4 != 0) {
        size_t lane = total_offset % 4;  // in-word lane the head starts at (the tail below always starts at 0)
        size_t head = std::min<size_t>(4 - lane, size);

        // Pack head bytes into a uint32_t
        uint32_t head_val = 0;
        for (size_t i = 0; i < head; i++) {
            ((uint8_t *) &head_val)[lane + i] = ((const uint8_t *) data)[i];
        }
        ggml_backend_webgpu_buffer_memset(buf_ctx->global_ctx, buf_ctx->buffer, head_val, total_offset, head);

        total_offset += head;
        size -= head;
        data = (const uint8_t *) data + head;
    }

    if (size > 0) {
        buf_ctx->global_ctx->pvr_write_epoch++;
        buf_ctx->global_ctx->queue.WriteBuffer(buf_ctx->buffer, total_offset, data, (size / 4) * 4);
    }

    if (size % 4 != 0) {
        // If size is not a multiple of 4, we need to memset the remaining bytes
        size_t remaining_size = size % 4;

        // pack the remaining bytes into a uint32_t
        uint32_t val32 = 0;

        for (size_t i = 0; i < remaining_size; i++) {
            ((uint8_t *) &val32)[i] = ((const uint8_t *) data)[size - remaining_size + i];
        }
        // memset the remaining bytes
        ggml_backend_webgpu_buffer_memset(buf_ctx->global_ctx, buf_ctx->buffer, val32,
                                          total_offset + (size - remaining_size), remaining_size);
    }
    WEBGPU_CPU_PROFILE_TOTAL_END(set_tensor, buf_ctx->global_ctx);
}

static void ggml_backend_webgpu_buffer_set_tensor_n(ggml_backend_buffer_t        buffer,
                                                    struct ggml_tensor * const * tensors,
                                                    const void * const *         data,
                                                    const size_t *               offsets,
                                                    const size_t *               sizes,
                                                    size_t                       n_tensors) {
    WEBGPU_CPU_PROFILE_TOTAL_START(set_tensor_n);
    ggml_backend_webgpu_buffer_context * buf_ctx = (ggml_backend_webgpu_buffer_context *) buffer->context;
    webgpu_global_context &              ctx     = buf_ctx->global_ctx;

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_set_tensor_n(" << buf_ctx->label << ", " << (const void *) tensors
                                                                << ", " << (const void *) data << ", " << offsets
                                                                << ", " << sizes << ", " << n_tensors << ")");

    const size_t max_binding_size = ctx->capabilities.limits.maxStorageBufferBindingSize;
    const size_t max_wg_per_dim   = ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;

    std::lock_guard<std::recursive_mutex> lock(ctx->mutex);

    std::vector<uint8_t>  src;    // we use src to copy the data of many tensors
    std::vector<uint32_t> descs;  // [dst offset in bytes, src offset in words, size in bytes, unused]
    wgpu::Buffer          dst_buf;

    auto grow_buffer = [&ctx](wgpu::Buffer & buf, size_t size, const char * label) {
        if (buf == nullptr || buf.GetSize() < size) {
            if (buf) {
                buf.Destroy();
            }
            ggml_webgpu_create_buffer(ctx->device, buf, size, wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Storage,
                                      label);
        }
    };

    auto flush_tensors = [&]() {
        if (descs.empty()) {
            return;
        }

        src.resize(ROUNDUP_POW2(src.size(), 4));
        const size_t descs_size = descs.size() * sizeof(uint32_t);
        const size_t n_dsts     = descs.size() / 4;

        grow_buffer(ctx->set_tensor_n_src_buf, src.size(), "set_tensor_n_src_buf");
        grow_buffer(ctx->set_tensor_n_descs_buf, descs_size, "set_tensor_n_descs_buf");
        ctx->queue.WriteBuffer(ctx->set_tensor_n_src_buf, 0, src.data(), src.size());
        ctx->queue.WriteBuffer(ctx->set_tensor_n_descs_buf, 0, descs.data(), descs_size);

        std::vector<wgpu::BindGroupEntry> entries = {
            ggml_webgpu_make_bind_group_entry(0, dst_buf, 0, dst_buf.GetSize()),
            ggml_webgpu_make_bind_group_entry(1, ctx->set_tensor_n_src_buf, 0, src.size()),
            ggml_webgpu_make_bind_group_entry(2, ctx->set_tensor_n_descs_buf, 0, descs_size),
        };

        wgpu::BindGroupDescriptor bind_group_desc;
        bind_group_desc.layout     = ctx->set_tensor_n_pipeline.pipeline.GetBindGroupLayout(0);
        bind_group_desc.entryCount = entries.size();
        bind_group_desc.entries    = entries.data();
        bind_group_desc.label      = ctx->set_tensor_n_pipeline.name.c_str();
        wgpu::BindGroup bind_group = ctx->device.CreateBindGroup(&bind_group_desc);

        uint32_t wg_x;
        uint32_t wg_y;
        uint32_t total_wg = n_dsts;
        compute_2d_workgroups(total_wg, max_wg_per_dim, wg_x, wg_y);

        wgpu::CommandEncoder     encoder = ctx->device.CreateCommandEncoder();
        wgpu::ComputePassEncoder pass    = encoder.BeginComputePass();
        pass.SetPipeline(ctx->set_tensor_n_pipeline.pipeline);
        pass.SetBindGroup(0, bind_group);
        pass.DispatchWorkgroups(wg_x, wg_y, 1);
        pass.End();

        wgpu::CommandBuffer commands = encoder.Finish();
        ctx->queue_idle              = false;
        ctx->pvr_write_epoch++;
        ctx->queue.Submit(1, &commands);

        src.clear();
        descs.clear();
    };

    if (n_tensors > 0) {
        dst_buf = ggml_webgpu_tensor_buf(tensors[0]);
    }

    for (size_t i = 0; i < n_tensors; i++) {
        if (sizes[i] == 0) {
            continue;
        }
        // PowerVR: a tensor that is or becomes repacked goes through set_tensor, which moves the bytes into its layout
        if (ggml_webgpu_pvr_is_repacked(tensors[i]) || ggml_webgpu_pvr_repackable(ctx, tensors[i])) {
            flush_tensors();
            ggml_backend_webgpu_buffer_set_tensor(buffer, tensors[i], data[i], offsets[i], sizes[i]);
            continue;
        }
        wgpu::Buffer buf    = ggml_webgpu_tensor_buf(tensors[i]);
        size_t       offset = ggml_webgpu_tensor_offset(tensors[i]) + offsets[i];
        if (!descs.empty() && (dst_buf.Get() != buf.Get() || src.size() + 4 + sizes[i] > max_binding_size ||
                               (descs.size() + 4) * sizeof(uint32_t) > max_binding_size)) {
            flush_tensors();
        }
        dst_buf = buf;

        // Place the data at the same in-word byte lane as its destination
        size_t src_offset = ROUNDUP_POW2(src.size(), 4) + offset % 4;
        src.resize(src_offset + sizes[i]);
        memcpy(src.data() + src_offset, data[i], sizes[i]);
        descs.insert(descs.end(), { (uint32_t) offset, (uint32_t) (src_offset / 4), (uint32_t) sizes[i], 0 });
    }
    flush_tensors();

    WEBGPU_CPU_PROFILE_TOTAL_END(set_tensor_n, ctx);
}

static void ggml_backend_webgpu_buffer_get_tensor_raw(ggml_backend_buffer_t buffer,
                                                      const ggml_tensor *   tensor,
                                                      void *                data,
                                                      size_t                offset,
                                                      size_t                size) {
    WEBGPU_CPU_PROFILE_TOTAL_START(get_tensor);
    ggml_backend_webgpu_buffer_context * buf_ctx = (ggml_backend_webgpu_buffer_context *) buffer->context;
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_get_tensor(" << buf_ctx->label << ", " << tensor << ", " << data
                                                              << ", " << offset << ", " << size << ")");
    wgpu::Device device = buf_ctx->global_ctx->device;

    size_t total_offset = ggml_webgpu_tensor_offset(tensor) + offset;

    size_t local_offset = total_offset % 4;
    if (local_offset != 0) {
        // If offset is not a multiple of 4, we need to round it down to the previous
        // multiple of 4
        total_offset -= local_offset;
    }

    size_t final_size = size + local_offset;
    if (final_size % 4 != 0) {
        // If size is not a multiple of 4, we need to round it up to the next
        // multiple of 4
        final_size += 4 - (final_size % 4);
    }

    std::lock_guard<std::recursive_mutex> lock(buf_ctx->global_ctx->mutex);

    // On PowerVR each read back waits several ms, and saving the model's state
    // reads the recurrent state one layer at a time: read a small state buffer
    // back whole (not a compute buffer, which is read every token, a little
    // of it: the logits), and serve further reads from that copy while
    // nothing has been written since.
    const size_t buf_size = buf_ctx->buffer.GetSize();
    const bool   whole    = ggml_webgpu_is_pvr(buf_ctx->global_ctx) && buf_size <= WEBGPU_PVR_READBACK_MAX_BYTES &&
                       buf_size % 4 == 0 && ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_ANY;
    if (ggml_webgpu_is_pvr(buf_ctx->global_ctx) && buf_ctx->global_ctx->pvr_readback_src != nullptr &&
        buf_ctx->global_ctx->pvr_readback_epoch != buf_ctx->global_ctx->pvr_write_epoch) {
        // the copy is stale (written since): free it (it may be tens of MB)
        webgpu_global_context & g = buf_ctx->global_ctx;
        std::vector<uint8_t>().swap(g->pvr_readback_data);
        g->pvr_readback_src = nullptr;
    }
    if (whole) {
        webgpu_global_context & g = buf_ctx->global_ctx;
        if (g->pvr_readback_src != buf_ctx->buffer.Get() || g->pvr_readback_epoch != g->pvr_write_epoch) {
            total_offset = 0;
            final_size   = buf_size;
            local_offset = ggml_webgpu_tensor_offset(tensor) + offset;
        } else {
            std::memcpy(data, g->pvr_readback_data.data() + ggml_webgpu_tensor_offset(tensor) + offset, size);
            WEBGPU_CPU_PROFILE_TOTAL_END(get_tensor, buf_ctx->global_ctx);
            return;
        }
    }

    if (buf_ctx->global_ctx->get_tensor_staging_buf == nullptr ||
        buf_ctx->global_ctx->get_tensor_staging_buf.GetSize() < final_size) {
        // Create a new staging buffer if it doesn't exist or is too small
        if (buf_ctx->global_ctx->get_tensor_staging_buf) {
            buf_ctx->global_ctx->get_tensor_staging_buf.Destroy();
        }
        ggml_webgpu_create_buffer(device, buf_ctx->global_ctx->get_tensor_staging_buf, final_size,
                                  wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead, "get_tensor_staging_buf");
    }

    // Copy the data from the buffer to the staging buffer
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    encoder.CopyBufferToBuffer(buf_ctx->buffer, total_offset, buf_ctx->global_ctx->get_tensor_staging_buf, 0,
                               final_size);
    wgpu::CommandBuffer commands = encoder.Finish();

    // Submit the command buffer to the queue
    buf_ctx->global_ctx->queue_idle = false;
    buf_ctx->global_ctx->queue.Submit(1, &commands);

    // Map the staging buffer to read the data
    ggml_backend_webgpu_map_buffer(buf_ctx->global_ctx, buf_ctx->global_ctx->get_tensor_staging_buf,
                                   wgpu::MapMode::Read, 0, final_size);
    // The copy was the last submission and the queue runs in order, so it is
    // idle now.
    buf_ctx->global_ctx->queue_idle = true;
    // Must specify size here since the staging buffer might be larger than the tensor size
    const void * mapped_range = buf_ctx->global_ctx->get_tensor_staging_buf.GetConstMappedRange(0, final_size);

    // Copy the data from the mapped range to the output buffer
    std::memcpy(data, (const void *) ((const char *) mapped_range + local_offset), size);
    if (whole && total_offset == 0 && final_size == buf_size) {
        webgpu_global_context & g = buf_ctx->global_ctx;
        g->pvr_readback_data.assign((const uint8_t *) mapped_range, (const uint8_t *) mapped_range + buf_size);
        g->pvr_readback_src   = buf_ctx->buffer.Get();
        g->pvr_readback_epoch = g->pvr_write_epoch;
    }
    buf_ctx->global_ctx->get_tensor_staging_buf.Unmap();
    WEBGPU_CPU_PROFILE_TOTAL_END(get_tensor, buf_ctx->global_ctx);
}

static void ggml_backend_webgpu_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                  const ggml_tensor *   tensor,
                                                  void *                data,
                                                  size_t                offset,
                                                  size_t                size) {
    if (!ggml_webgpu_pvr_is_repacked(tensor)) {
        ggml_backend_webgpu_buffer_get_tensor_raw(buffer, tensor, data, offset, size);
        return;
    }
    // a repacked tensor: read it whole and give back the original layout
    const size_t         nbytes = ggml_nbytes(tensor);
    std::vector<uint8_t> a(nbytes);
    std::vector<uint8_t> orig(nbytes);
    ggml_backend_webgpu_buffer_get_tensor_raw(buffer, tensor, a.data(), 0, nbytes);
    const int64_t nb = tensor->ne[0] / 256;
    for (int64_t b = 0; b < nb; b++) {
        ggml_webgpu_pvr_repack_column(tensor->type, tensor->ne[1], nb, b, false, orig.data(), a.data());
    }
    memcpy(data, orig.data() + offset, size);
}

static void ggml_backend_webgpu_buffer_get_tensor_n(ggml_backend_buffer_t              buffer,
                                                    const struct ggml_tensor * const * tensors,
                                                    void * const *                     data,
                                                    const size_t *                     offsets,
                                                    const size_t *                     sizes,
                                                    size_t                             n_tensors) {
    WEBGPU_CPU_PROFILE_TOTAL_START(get_tensor_n);
    ggml_backend_webgpu_buffer_context * buf_ctx = (ggml_backend_webgpu_buffer_context *) buffer->context;
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_get_tensor_n(" << buf_ctx->label << ", " << (const void *) tensors
                                                                << ", " << (const void *) data << ", " << offsets
                                                                << ", " << sizes << ", " << n_tensors << ")");
    webgpu_global_context global_ctx = buf_ctx->global_ctx;
    wgpu::Device          device     = global_ctx->device;

    // PowerVR: repacked tensors are read back through get_tensor, which restores the original layout
    for (size_t i = 0; i < n_tensors; i++) {
        if (ggml_webgpu_pvr_is_repacked(tensors[i])) {
            for (size_t j = 0; j < n_tensors; j++) {
                ggml_backend_webgpu_buffer_get_tensor(buffer, tensors[j], data[j], offsets[j], sizes[j]);
            }
            WEBGPU_CPU_PROFILE_TOTAL_END(get_tensor_n, buf_ctx->global_ctx);
            return;
        }
    }

    typedef struct {
        size_t staging_size;
        size_t align_offset;
        size_t local_offset;
    } tensor_desc;

    std::vector<tensor_desc> t_descs;
    std::vector<std::pair<size_t, size_t>>
        staging_chunk_descs;  // pair (boundary id, sum of the tensor size to the bounary id)
    t_descs.reserve(n_tensors);

    size_t sum_tensor_size = 0;
    for (size_t i = 0; i < n_tensors; i++) {
        const ggml_tensor * tensor = tensors[i];
        tensor_desc         td;

        td.align_offset = ggml_webgpu_tensor_offset(tensor) + offsets[i];

        td.local_offset = td.align_offset % 4;
        if (td.local_offset != 0) {
            // If offset is not a multiple of 4, we need to round it down to the previous
            // multiple of 4
            td.align_offset -= td.local_offset;
        }

        td.staging_size = sizes[i] + td.local_offset;
        if (td.staging_size % 4 != 0) {
            // If size is not a multiple of 4, we need to round it up to the next
            // multiple of 4
            td.staging_size += 4 - (td.staging_size % 4);
        }

        t_descs.push_back(td);

        if (sum_tensor_size + td.staging_size > global_ctx->capabilities.limits.maxBufferSize) {
            if (i == 0) {
                GGML_ABORT("it is not expected that one tensor size exceed the one wgpu::Buffer limit size.");
            }
            staging_chunk_descs.push_back({ i - 1, sum_tensor_size });
            sum_tensor_size = td.staging_size;
        } else {
            sum_tensor_size += td.staging_size;
        }
    }
    staging_chunk_descs.push_back({ n_tensors - 1, sum_tensor_size });

    std::lock_guard<std::recursive_mutex> lock(global_ctx->mutex);

    size_t tsi = 0;
    for (auto & [tei, sum_chunk_size] : staging_chunk_descs) {
        if (global_ctx->get_tensor_staging_buf == nullptr ||
            global_ctx->get_tensor_staging_buf.GetSize() < sum_chunk_size) {
            // Create a new staging buffer if it doesn't exist or is too small
            if (global_ctx->get_tensor_staging_buf) {
                global_ctx->get_tensor_staging_buf.Destroy();
            }
            ggml_webgpu_create_buffer(device, global_ctx->get_tensor_staging_buf, sum_chunk_size,
                                      wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead,
                                      "get_tensor_staging_buf");
        }

        wgpu::CommandEncoder encoder = device.CreateCommandEncoder();

        // Copy the data of the each tensor from the buffer to the staging buffer
        size_t i_offset = 0;
        for (size_t i = tsi; i <= tei; i++) {
            encoder.CopyBufferToBuffer(ggml_webgpu_tensor_buf(tensors[i]), t_descs[i].align_offset,
                                       global_ctx->get_tensor_staging_buf, i_offset, t_descs[i].staging_size);
            i_offset += t_descs[i].staging_size;
        }

        wgpu::CommandBuffer commands = encoder.Finish();

        // Submit the command buffer to the queue
        global_ctx->queue_idle = false;
        global_ctx->queue.Submit(1, &commands);

        // Map the staging buffer to read the data
        ggml_backend_webgpu_map_buffer(global_ctx, global_ctx->get_tensor_staging_buf, wgpu::MapMode::Read, 0,
                                       sum_chunk_size);
        // Must specify size here since the staging buffer might be larger than the tensor size
        const void * mapped_range = global_ctx->get_tensor_staging_buf.GetConstMappedRange(0, sum_chunk_size);

        i_offset = 0;
        for (size_t i = tsi; i <= tei; i++) {
            // Copy the data from the mapped range to the output buffer
            std::memcpy(data[i], (const void *) ((const char *) mapped_range + i_offset + t_descs[i].local_offset),
                        sizes[i]);
            i_offset += t_descs[i].staging_size;
        }

        tsi = tei + 1;
        global_ctx->get_tensor_staging_buf.Unmap();
    }

    WEBGPU_CPU_PROFILE_TOTAL_END(get_tensor_n, buf_ctx->global_ctx);
}

static void ggml_backend_webgpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_clear(" << buffer << ", " << (uint32_t) value << ")");
    WEBGPU_CPU_PROFILE_TOTAL_START(clear);
    ggml_backend_webgpu_buffer_context * buf_ctx = (ggml_backend_webgpu_buffer_context *) buffer->context;
    ggml_backend_webgpu_buffer_memset(buf_ctx->global_ctx, buf_ctx->buffer, value, 0, buffer->size);
    WEBGPU_CPU_PROFILE_TOTAL_END(clear, buf_ctx->global_ctx);
}

static ggml_backend_buffer_i ggml_backend_webgpu_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_webgpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_webgpu_buffer_get_base,
    /* .init_tensor     = */ NULL,  // TODO: optional, needed?
    /* .memset_tensor   = */ ggml_backend_webgpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_webgpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_webgpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,  // TODO: optional, implement this
    /* .clear           = */ ggml_backend_webgpu_buffer_clear,
    /* .reset           = */ NULL,  // TODO: optional, think it coordinates with
    /* .set_tensor_n    = */ ggml_backend_webgpu_buffer_set_tensor_n,
    /* .get_tensor_n    = */ ggml_backend_webgpu_buffer_get_tensor_n,
    /* .cpy_tensor_n    = */ NULL,
};

/* End GGML Backend Buffer Interface */

/* GGML Backend Buffer Type Interface */

static const char * ggml_backend_webgpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    return ctx->device_name.c_str();
}

static ggml_backend_buffer_t ggml_backend_webgpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                          size_t                     size) {
    static std::atomic<int> buffer_count;
    int                     buffer_id = buffer_count++;
    std::string             buf_name  = "tensor_buf" + std::to_string(buffer_id);
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_type_alloc_buffer_" << buffer_id << ": " << size << " bytes");

    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    wgpu::Buffer                         buf;
    ggml_webgpu_create_buffer(ctx->webgpu_global_ctx->device, buf, ROUNDUP_POW2(size, WEBGPU_STORAGE_BUF_BINDING_MULT),
                              wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
                              buf_name.c_str());

    ggml_backend_webgpu_buffer_context * buf_ctx =
        new ggml_backend_webgpu_buffer_context(buf, buf_name, ctx->webgpu_global_ctx);

    return ggml_backend_buffer_init(buft, ggml_backend_webgpu_buffer_interface, buf_ctx, size);
}

static size_t ggml_backend_webgpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_webgpu_device_context * dev_ctx =
        static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    return dev_ctx->webgpu_global_ctx->capabilities.limits.minStorageBufferOffsetAlignment;
}

static size_t ggml_backend_webgpu_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_webgpu_device_context * dev_ctx =
        static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    return dev_ctx->webgpu_global_ctx->capabilities.limits.maxBufferSize;
}

static size_t ggml_backend_webgpu_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft,
                                                             const ggml_tensor *        tensor) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    size_t                               res = ggml_nbytes(tensor);
    switch (tensor->op) {
        case GGML_OP_ARGSORT:
            res = ROUNDUP_POW2(res * 2 + ctx->webgpu_global_ctx->capabilities.limits.minStorageBufferOffsetAlignment,
                               WEBGPU_STORAGE_BUF_BINDING_MULT);
            break;
        case GGML_OP_TOP_K:
            {
                const ggml_tensor * src0 = tensor->src[0];
                if (src0) {
                    const size_t full = sizeof(int32_t) * ggml_nelements(src0);
                    res               = ROUNDUP_POW2(
                        full * 2 + ctx->webgpu_global_ctx->capabilities.limits.minStorageBufferOffsetAlignment,
                        WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            {
                const ggml_tensor * Q            = tensor->src[0];
                const ggml_tensor * K            = tensor->src[1];
                const ggml_tensor * V            = tensor->src[2];
                const ggml_tensor * mask         = tensor->src[3];
                const auto &        capabilities = ctx->webgpu_global_ctx->capabilities;
                // the vec path scratch grows with the query rows, so a longer batch reserves the scratch of the
                // longest vec batch and the size stays monotonic in the batch length
                const int64_t       n_q = std::min<int64_t>(Q->ne[1], GGML_WEBGPU_FLASH_ATTN_VEC_MAX_SEQ_LEN - 1);
                if (ggml_webgpu_flash_attn_use_vec_path(ctx->webgpu_global_ctx, Q, K, V, n_q)) {
                    const bool kv_direct =
                        ggml_webgpu_flash_attn_k_direct(Q, K, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH) ||
                        ggml_webgpu_flash_attn_v_direct(Q, V, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH);
                    const uint32_t kv_tile = ggml_webgpu_flash_attn_get_vec_kv_tile(
                        capabilities.limits.maxComputeWorkgroupStorageSize, (uint32_t) Q->ne[0], (uint32_t) V->ne[0],
                        mask != nullptr, kv_direct);

                    const uint32_t vec_nwg_cap = capabilities.min_subgroup_size;
                    uint32_t       nwg = ggml_webgpu_flash_attn_vec_nwg(vec_nwg_cap, kv_tile, (uint32_t) K->ne[1]);

                    const size_t   align = capabilities.limits.minStorageBufferOffsetAlignment;
                    const uint64_t nrows = (uint64_t) n_q * Q->ne[2] * Q->ne[3];
                    if (nwg > 1u) {
                        const uint64_t tmp_data_elems  = nrows * (uint64_t) V->ne[0] * nwg;
                        const uint64_t tmp_stats_elems = nrows * 2u * nwg;
                        const size_t   tmp_size_bytes = ROUNDUP_POW2((tmp_data_elems + tmp_stats_elems) * sizeof(float),
                                                                     WEBGPU_STORAGE_BUF_BINDING_MULT);
                        res += tmp_size_bytes + align;
                    } else {
                        res += WEBGPU_STORAGE_BUF_BINDING_MULT + align;
                    }
                    if (mask != nullptr) {
                        const uint32_t blk_nblk0       = CEIL_DIV((uint32_t) K->ne[1], kv_tile);
                        const uint32_t blk_nblk1       = (uint32_t) n_q;
                        const uint32_t stride_mask3    = (uint32_t) (mask->nb[3] / ggml_type_size(mask->type));
                        const uint32_t blk_batch_count = stride_mask3 > 0 ? (uint32_t) Q->ne[3] : 1u;
                        const uint64_t blk_elems       = (uint64_t) blk_nblk0 * blk_nblk1 * blk_batch_count;
                        const size_t   blk_size_bytes =
                            ROUNDUP_POW2(blk_elems * sizeof(uint32_t), WEBGPU_STORAGE_BUF_BINDING_MULT);
                        res += blk_size_bytes + align;
                    }
                    res = ROUNDUP_POW2(res, WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            break;
        case GGML_OP_MUL_MAT:
            {
                const ggml_tensor * src0 = tensor->src[0];
                const ggml_tensor * src1 = tensor->src[1];

                // the q8 copy of src1 grows with its rows, so a longer batch reserves the copy of the longest
                // mmvq batch and the size stays monotonic in the batch length
                const int64_t n_rows   = std::min<int64_t>(src1->ne[1], GGML_WEBGPU_MUL_MAT_VEC_MAX_ROWS);
                const bool    use_mmvq = ggml_webgpu_can_use_mmvq(
                    src0, src1, n_rows, ctx->webgpu_global_ctx->capabilities.supports_dot_product,
                    ctx->webgpu_global_ctx->vendor);
                if (use_mmvq) {
                    const size_t q8_src1_size = src1->ne[3] * src1->ne[2] * n_rows *
                                                (36 /* sizeof(q8_1) */ * (src1->ne[0] / /* block_size */ 32));
                    res = ROUNDUP_POW2(res + q8_src1_size +
                                           ctx->webgpu_global_ctx->capabilities.limits.minStorageBufferOffsetAlignment,
                                       WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            break;
        case GGML_OP_MUL_MAT_ID:
            {
                const ggml_tensor * src0 = tensor->src[0];
                const ggml_tensor * src1 = tensor->src[1];
                if (src0 && src1) {
                    const size_t ids_cache_size = sizeof(uint32_t) * tensor->ne[1] * tensor->ne[2];
                    const size_t ids_tiles_size = ggml_webgpu_mul_mat_id_tiles_nbytes(tensor);
                    res                         = ROUNDUP_POW2(
                        res + ids_cache_size + ids_tiles_size +
                            ctx->webgpu_global_ctx->capabilities.limits.minStorageBufferOffsetAlignment * 2,
                        WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            break;
        default:
            break;
    }
    return res;
}

/* End GGML Backend Buffer Type Interface */

/* GGML Backend Device Interface */

static const char * ggml_backend_webgpu_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    return ctx->device_name.c_str();
}

static const char * ggml_backend_webgpu_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    return ctx->device_desc.c_str();
}

static void ggml_backend_webgpu_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    // TODO: for now, return maxBufferSize as both free and total memory
    // Track https://github.com/gpuweb/gpuweb/issues/5505 for updates.
    uint64_t                             max_buffer_size = ctx->webgpu_global_ctx->capabilities.limits.maxBufferSize;
    // If we're on a 32-bit system, clamp to UINTPTR_MAX
#if UINTPTR_MAX < UINT64_MAX
    uint64_t max_ptr_size = static_cast<uint64_t>(UINTPTR_MAX);
    if (max_buffer_size > max_ptr_size) {
        max_buffer_size = max_ptr_size;
    }
#endif
    *free  = static_cast<size_t>(max_buffer_size);
    *total = static_cast<size_t>(max_buffer_size);
}

static enum ggml_backend_dev_type ggml_backend_webgpu_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_webgpu_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_webgpu_device_get_name(dev);
    props->description = ggml_backend_webgpu_device_get_description(dev);
    props->type        = ggml_backend_webgpu_device_get_type(dev);
    ggml_backend_webgpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
        /* .mmap_support          = */ true,
    };
}

static ggml_guid_t ggml_backend_webgpu_guid(void) {
    static ggml_guid guid = { 0x67, 0xc7, 0xa4, 0xb1, 0x78, 0x74, 0x4f, 0x51,
                              0x9d, 0x65, 0x44, 0x6d, 0xe4, 0x1b, 0x82, 0x9a };
    return &guid;
}

static void ggml_webgpu_init_memset_pipeline(webgpu_global_context & ctx) {
    std::vector<wgpu::ConstantEntry> constants(1);
    constants[0].key     = "wg_size";
    constants[0].value   = ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    ctx->memset_pipeline = ggml_webgpu_create_pipeline(ctx->device, wgsl_memset, "memset", constants);
}

static void ggml_webgpu_init_scatter_write_pipeline(webgpu_global_context & ctx) {
    std::vector<wgpu::ConstantEntry> constants(1);
    constants[0].key           = "wg_size";
    constants[0].value         = ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    ctx->set_tensor_n_pipeline = ggml_webgpu_create_pipeline(ctx->device, wgsl_set_tensor_n, "set_tensor_n", constants);
}

static void ggml_backend_webgpu_request_adapter(wgpu::Instance & instance, wgpu::Adapter & adapter) {
    wgpu::RequestAdapterOptions options = {};

#if !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    // TODO: track need for these toggles: https://issues.chromium.org/issues/42251215
    const char * const          adapterEnabledToggles[] = { "vulkan_enable_f16_on_nvidia", "use_vulkan_memory_model" };
    wgpu::DawnTogglesDescriptor adapterTogglesDesc;
    adapterTogglesDesc.enabledToggles     = adapterEnabledToggles;
    adapterTogglesDesc.enabledToggleCount = 2;
    options.nextInChain                   = &adapterTogglesDesc;
#endif

    instance.WaitAny(
        instance.RequestAdapter(
            &options, wgpu::CallbackMode::AllowSpontaneous,
            [&adapter](wgpu::RequestAdapterStatus status, wgpu::Adapter _adapter, wgpu::StringView message) {
                if (status != wgpu::RequestAdapterStatus::Success) {
                    GGML_LOG_ERROR("ggml_webgpu: Failed to get an adapter: %s\n", std::string(message).c_str());
                    return;
                }
                adapter = std::move(_adapter);
            }),
        UINT64_MAX);
}

static void create_webgpu_device(ggml_backend_webgpu_reg_context * ctx) {
    ggml_backend_webgpu_request_adapter(ctx->webgpu_global_ctx->instance, ctx->webgpu_global_ctx->adapter);
    GGML_ASSERT(ctx->webgpu_global_ctx->adapter != nullptr);

    ctx->webgpu_global_ctx->adapter.GetLimits(&ctx->webgpu_global_ctx->capabilities.limits);

    wgpu::AdapterInfo info{};
#if !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    wgpu::AdapterPropertiesSubgroupMatrixConfigs subgroup_matrix_configs{};
    if (ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix)) {
        info.nextInChain = &subgroup_matrix_configs;
    }
#endif
    ctx->webgpu_global_ctx->adapter.GetInfo(&info);
    ctx->webgpu_global_ctx->command_submit_batch_size = ggml_backend_webgpu_get_command_submit_batch_size();
    ctx->webgpu_global_ctx->max_inflight_batches      = ggml_backend_webgpu_get_max_inflight_batches();
    ctx->webgpu_global_ctx->vendor                    = info.vendor;
    if (ggml_webgpu_is_pvr(ctx->webgpu_global_ctx)) {
        // PowerVR runs a token's ~450 dispatches faster in smaller command buffers
        ctx->webgpu_global_ctx->command_submit_batch_size = WEBGPU_PVR_COMMAND_SUBMIT_BATCH_SIZE;
    }
    ctx->webgpu_global_ctx->capabilities.supports_subgroups =
        ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::Subgroups);
    // for dot4I8packed
    ctx->webgpu_global_ctx->capabilities.supports_dot_product = ctx->webgpu_global_ctx->instance.HasWGSLLanguageFeature(
        wgpu::WGSLLanguageFeatureName::Packed4x8IntegerDotProduct);

    bool valid_subgroup_matrix_config = false;
#if !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    // Accept f16 subgroup matrix configurations (square or non-square).
    // NVIDIA GPUs typically report square configs (e.g. 16x16x16),
    // while Intel Xe2 GPUs report non-square configs (e.g. 8x16x16).
    // The shaders are already parameterized to handle any M/N/K dimensions.
    if (ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix)) {
        for (size_t i = 0; i < subgroup_matrix_configs.configCount; i++) {
            const wgpu::SubgroupMatrixConfig config = subgroup_matrix_configs.configs[i];
            if (config.componentType == wgpu::SubgroupMatrixComponentType::F16 &&
                config.resultComponentType == wgpu::SubgroupMatrixComponentType::F16) {
                ctx->webgpu_global_ctx->capabilities.sg_mat_m = config.M;
                ctx->webgpu_global_ctx->capabilities.sg_mat_n = config.N;
                ctx->webgpu_global_ctx->capabilities.sg_mat_k = config.K;
                valid_subgroup_matrix_config                  = true;
                break;
            }
        }
    }
#endif
    ctx->webgpu_global_ctx->capabilities.supports_subgroup_matrix = valid_subgroup_matrix_config;

    // Runtime subgroup size can be any supported size in this range. Shaders
    // that allocate per-lane register arrays must size them for the minimum.
    ctx->webgpu_global_ctx->capabilities.min_subgroup_size = info.subgroupMinSize;
    ctx->webgpu_global_ctx->capabilities.max_subgroup_size = info.subgroupMaxSize;
    // Initialize device
    std::vector<wgpu::FeatureName> required_features       = { wgpu::FeatureName::ShaderF16 };

#if !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    required_features.push_back(wgpu::FeatureName::ImplicitDeviceSynchronization);
    if (ctx->webgpu_global_ctx->capabilities.supports_subgroup_matrix) {
        required_features.push_back(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix);
    }
#endif

    if (ctx->webgpu_global_ctx->capabilities.supports_subgroups) {
        required_features.push_back(wgpu::FeatureName::Subgroups);
    }

#ifdef GGML_WEBGPU_GPU_PROFILE
    required_features.push_back(wgpu::FeatureName::TimestampQuery);
#endif

    wgpu::DeviceDescriptor dev_desc;
    dev_desc.requiredLimits       = &ctx->webgpu_global_ctx->capabilities.limits;
    dev_desc.requiredFeatures     = required_features.data();
    dev_desc.requiredFeatureCount = required_features.size();
    dev_desc.SetDeviceLostCallback(
        wgpu::CallbackMode::AllowSpontaneous,
        [](const wgpu::Device & device, wgpu::DeviceLostReason reason, wgpu::StringView message) {
            if (reason == wgpu::DeviceLostReason::Destroyed) {
                return;
            }
            GGML_UNUSED(device);
            GGML_LOG_ERROR("ggml_webgpu: Device lost! Reason: %d, Message: %s\n", static_cast<int>(reason),
                           std::string(message).c_str());
        });
    dev_desc.SetUncapturedErrorCallback(
        [](const wgpu::Device & device, wgpu::ErrorType reason, wgpu::StringView message) {
            GGML_UNUSED(device);
            GGML_ABORT("ggml_webgpu: Device error! Reason: %d, Message: %s\n", static_cast<int>(reason),
                       std::string(message).c_str());
        });

#if !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    // Enable Dawn-specific toggles to increase native performance
    // TODO: Maybe WebGPU needs a "fast" mode where you can request compilers skip adding checks like these,
    //       only for native performance?
    const char * const          deviceEnabledToggles[]  = { "disable_robustness", "disable_workgroup_init",
                                                            "disable_polyfills_on_integer_div_and_mod" };
    const char * const          deviceDisabledToggles[] = { "timestamp_quantization" };
    wgpu::DawnTogglesDescriptor deviceTogglesDesc;
    deviceTogglesDesc.enabledToggles      = deviceEnabledToggles;
    deviceTogglesDesc.enabledToggleCount  = 3;
    deviceTogglesDesc.disabledToggles     = deviceDisabledToggles;
    deviceTogglesDesc.disabledToggleCount = 1;

    dev_desc.nextInChain = &deviceTogglesDesc;
#endif

    ctx->webgpu_global_ctx->instance.WaitAny(
        ctx->webgpu_global_ctx->adapter.RequestDevice(
            &dev_desc, wgpu::CallbackMode::AllowSpontaneous,
            [ctx](wgpu::RequestDeviceStatus status, wgpu::Device device, wgpu::StringView message) {
                if (status != wgpu::RequestDeviceStatus::Success) {
                    GGML_LOG_ERROR("ggml_webgpu: Failed to get a device: %s\n", std::string(message).c_str());
                    return;
                }
                ctx->webgpu_global_ctx->device = std::move(device);
            }),
        UINT64_MAX);
    GGML_ASSERT(ctx->webgpu_global_ctx->device != nullptr);

    ggml_webgpu_init_memset_pipeline(ctx->webgpu_global_ctx);
    ggml_webgpu_init_scatter_write_pipeline(ctx->webgpu_global_ctx);
    ggml_webgpu_create_buffer(ctx->webgpu_global_ctx->device, ctx->webgpu_global_ctx->memset_params_buf,
                              WEBGPU_PARAMS_BUF_SIZE_BYTES, wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Uniform,
                              "memset_params_buf");
    ctx->webgpu_global_ctx->queue = ctx->webgpu_global_ctx->device.GetQueue();

    GGML_LOG_INFO(
        "ggml_webgpu: adapter_info: vendor_id: %u | vendor: %s | architecture: %s | device_id: %u | name: %s | "
        "device_desc: %s\n",
        info.vendorID, std::string(info.vendor).c_str(), std::string(info.architecture).c_str(), info.deviceID,
        std::string(info.device).c_str(), std::string(info.description).c_str());
}

static webgpu_context initialize_webgpu_context(ggml_backend_dev_t dev) {
    ggml_backend_webgpu_device_context * dev_ctx    = (ggml_backend_webgpu_device_context *) dev->context;
    webgpu_context                       webgpu_ctx = std::make_shared<webgpu_context_struct>();
    webgpu_ctx->global_ctx                          = dev_ctx->webgpu_global_ctx;
    webgpu_ctx->shader_lib = std::make_unique<ggml_webgpu_shader_lib>(dev_ctx->webgpu_global_ctx->device);
    webgpu_ctx->param_arena.init(
        webgpu_ctx->global_ctx->device, WEBGPU_PARAMS_BUF_SIZE_BYTES,
        webgpu_ctx->global_ctx->command_submit_batch_size + WEBGPU_NUM_PARAM_SLOT_SAFETY_MARGIN,
        webgpu_ctx->global_ctx->capabilities.limits.minUniformBufferOffsetAlignment);
    ggml_webgpu_create_buffer(webgpu_ctx->global_ctx->device, webgpu_ctx->set_rows_dev_error_buf,
                              WEBGPU_SET_ROWS_ERROR_BUF_SIZE_BYTES,
                              wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc, "set_rows_dev_error_buf");
    ggml_webgpu_create_buffer(webgpu_ctx->global_ctx->device, webgpu_ctx->set_rows_host_error_buf,
                              WEBGPU_SET_ROWS_ERROR_BUF_SIZE_BYTES,
                              wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead, "set_rows_host_error_buf");
    if (ggml_webgpu_is_pvr(webgpu_ctx->global_ctx)) {
        // Prompt batches on PowerVR use pipelines that the warmup's two
        // tokens don't: create them now, so that the browser compiles them in
        // the background instead of during the first prompt.
        webgpu_ctx->shader_lib->get_mul_mat_pvr_stage_pipeline(true);
        // (the weights are repacked: see ggml_webgpu_pvr_repacked_marker)
        webgpu_ctx->shader_lib->get_mul_mat_pvr_dequant_pipeline(GGML_TYPE_Q4_K, true);
        webgpu_ctx->shader_lib->get_mul_mat_pvr_dequant_pipeline(GGML_TYPE_Q6_K, true);
        for (int glu = 0; glu < 3; glu++) {
            webgpu_ctx->shader_lib->get_mul_mat_pvr_gemm_pipeline(glu);
        }
        // The same for speculative batches (2 to 8 tokens) and the fused products.
        const bool sg = ggml_webgpu_mul_mat_vec_pvr_subgroups(webgpu_ctx->global_ctx);
        ggml_webgpu_mul_mat_vec_pvr_fusion add;
        add.add = true;
        ggml_webgpu_mul_mat_vec_pvr_fusion add_in_dst;
        add_in_dst.add        = true;
        add_in_dst.add_in_dst = true;
        ggml_webgpu_mul_mat_vec_pvr_fusion glu;
        glu.glu       = true;
        glu.stage_out = true;
        for (ggml_type type : { GGML_TYPE_Q4_K, GGML_TYPE_Q6_K }) {
            for (uint32_t ncol = 1; ncol <= WEBGPU_MUL_MAT_VEC_PVR_MAX_COLS; ncol++) {
                // the shapes for small and large matrices
                for (uint32_t m : { 1024u, 65536u }) {
                    auto shape = [&](const ggml_webgpu_mul_mat_vec_pvr_fusion & f) {
                        return ggml_webgpu_shader_lib::ggml_webgpu_mul_mat_vec_pvr_choose_shape(type, true, ncol, m, sg, f);
                    };
                    webgpu_ctx->shader_lib->get_mul_mat_vec_pvr_pipeline(type, true, ncol, sg, {}, shape({}));
                    webgpu_ctx->shader_lib->get_mul_mat_vec_pvr_pipeline(type, true, ncol, sg, add, shape(add));
                    webgpu_ctx->shader_lib->get_mul_mat_vec_pvr_pipeline(type, true, ncol, sg, add_in_dst,
                                                                         shape(add_in_dst));
                    if (type == GGML_TYPE_Q4_K) {
                        webgpu_ctx->shader_lib->get_mul_mat_vec_pvr_pipeline(type, true, ncol, sg, glu, shape(glu));
                    }
                }
            }
        }
        if (ggml_webgpu_pvr_subgroups_tile(webgpu_ctx->global_ctx, WEBGPU_FLASH_ATTN_PVR_MQ_ROWS * 2, 2)) {
            webgpu_ctx->shader_lib->get_flash_attn_pvr_mq_pipeline(64, 2, true, true);
        }
        if (ggml_webgpu_pvr_subgroups_tile(webgpu_ctx->global_ctx, WEBGPU_FLASH_ATTN_PVR_MQ_ROWS * 4, 4)) {
            webgpu_ctx->shader_lib->get_flash_attn_pvr_mq_pipeline(128, 4, true, true);
        }
        // LFM's fused short convolution for each batch size (the weights in their own buffer), of 3 taps
        // (LFM2) and 7 (LFM3)
        for (uint32_t width : { 3u, 7u }) {
            for (uint32_t n = 1; n <= 8; n++) {
                webgpu_ctx->shader_lib->get_shortconv_pvr_pipeline(n, width, true);
            }
        }
    }

#ifdef GGML_WEBGPU_GPU_PROFILE
    webgpu_ctx->batch_compute_passes = false;
    ggml_webgpu_create_buffer(
        webgpu_ctx->global_ctx->device, webgpu_ctx->profile_timestamp_dev_buf, WEBGPU_TIMESTAMP_QUERY_BUF_SIZE_BYTES,
        wgpu::BufferUsage::QueryResolve | wgpu::BufferUsage::CopySrc, "profile_timestamp_dev_buf");
    ggml_webgpu_create_buffer(webgpu_ctx->global_ctx->device, webgpu_ctx->profile_timestamp_host_buf,
                              WEBGPU_TIMESTAMP_QUERY_BUF_SIZE_BYTES,
                              wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead, "profile_timestamp_host_buf");
    wgpu::QuerySetDescriptor query_set_desc = {};
    query_set_desc.type                     = wgpu::QueryType::Timestamp;
    query_set_desc.count                    = WEBGPU_MAX_PROFILE_QUERY_COUNT;
    webgpu_ctx->profile_timestamp_query_set = webgpu_ctx->global_ctx->device.CreateQuerySet(&query_set_desc);
#endif

#ifdef GGML_WEBGPU_DEBUG
    // Initialize debug buffers
    ggml_webgpu_create_buffer(webgpu_ctx->global_ctx->device, webgpu_ctx->global_ctx->debug_host_buf,
                              WEBGPU_DEBUG_BUF_ELEMS * sizeof(uint32_t),
                              wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead, "debug_host_buf");
    ggml_webgpu_create_buffer(webgpu_ctx->global_ctx->device, webgpu_ctx->global_ctx->debug_dev_buf,
                              WEBGPU_DEBUG_BUF_ELEMS * sizeof(uint32_t),
                              wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc, "debug_dev_buf");
#endif
    return webgpu_ctx;
}

static ggml_backend_t ggml_backend_webgpu_backend_init(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_backend_init()");

    ggml_backend_webgpu_device_context * dev_ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);

    auto * backend_ctx      = new ggml_backend_webgpu_context();
    backend_ctx->name       = GGML_WEBGPU_NAME + std::string(": ") + dev_ctx->device_name;
    backend_ctx->webgpu_ctx = initialize_webgpu_context(dev);

    // See GGML Backend Interface section
    auto * backend = new ggml_backend();
    *backend       = {
        /* .guid      = */ ggml_backend_webgpu_guid(),
        /* .interface = */ ggml_backend_webgpu_i,
        /* .device    = */ dev,
        /* .context   = */ backend_ctx,
    };
    return backend;
}

static ggml_backend_buffer_type_t ggml_backend_webgpu_device_get_buffer_type(ggml_backend_dev_t dev) {
    // See GGML Backend Buffer Type Interface section

    static struct ggml_backend_buffer_type ggml_backend_webgpu_buffer_type = {
        /* .iface = */ {
                        /* .get_name            = */ ggml_backend_webgpu_buffer_type_get_name,
                        /* .alloc_buffer        = */ ggml_backend_webgpu_buffer_type_alloc_buffer,
                        /* .alloc_buffer_n      = */ NULL,
                        /* .get_alignment       = */ ggml_backend_webgpu_buffer_type_get_alignment,
                        /* .get_max_size        = */ ggml_backend_webgpu_buffer_type_get_max_size,
                        /* .get_alloc_size      = */ ggml_backend_webgpu_buffer_type_get_alloc_size,
                        /* .get_alloc_size_n    = */ NULL,
                        /* .is_host             = */ NULL,  // defaults to false
        },
        /* .device  = */
        dev,
        /* .context = */ NULL
    };

    return &ggml_backend_webgpu_buffer_type;
}

static bool ggml_backend_webgpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return buft->iface.get_name == ggml_backend_webgpu_buffer_type_get_name;
}

static bool ggml_webgpu_supported_qtype(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
            return true;
        default:
            return false;
    }
}

static bool ggml_backend_webgpu_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);

    ggml_tensor * src0 = op->src[0];
    ggml_tensor * src1 = op->src[1];
    ggml_tensor * src2 = op->src[2];

    // on smaller devices (or CI), tensors may be larger than the max storage buffer size
    if (ggml_nbytes(op) > ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize ||
        (src0 != nullptr &&
         ggml_nbytes(src0) > ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize) ||
        (src1 != nullptr &&
         ggml_nbytes(src1) > ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize)) {
        return false;
    }

    // PowerVR: tensors that may be repacked (see ggml_webgpu_pvr_repacked_marker)
    // are only read by matrix products through the PowerVR shaders, as their
    // weights; other ops on them fall back to the CPU, which reads them back
    // in their original layout.
    if (ggml_webgpu_is_pvr(ctx->webgpu_global_ctx) && !ggml_op_is_empty(op->op)) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            const ggml_tensor * src = op->src[i];
            if (!ggml_webgpu_pvr_reads_repackable(ctx->webgpu_global_ctx, src)) {
                continue;
            }
            const bool as_weights = op->op == GGML_OP_MUL_MAT && i == 0 && src->view_src == nullptr;
            if (!as_weights || src1 == nullptr || src1->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 ||
                src1->ne[2] != 1 || src1->ne[3] != 1 || src1->nb[0] != sizeof(float) ||
                op->nb[1] != op->ne[0] * sizeof(float) ||
                src1->ne[1] > ctx->webgpu_global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension ||
                (size_t) src1->ne[0] * src1->ne[1] * sizeof(float) >
                    ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize) {
                return false;
            }
        }
        if (ggml_webgpu_pvr_reads_repackable(ctx->webgpu_global_ctx, op)) {
            // an op writing such a tensor (the layout of what it writes is the original one)
            return false;
        }
    }

    bool supports_op = false;
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_RESHAPE:
            supports_op = true;
            break;
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type) &&
                          (src1->type == op->type);
            break;
        case GGML_OP_ADD_ID:
            supports_op = src0->type == GGML_TYPE_F32;
            break;
        case GGML_OP_CONCAT:
            supports_op = (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32);
            break;
        case GGML_OP_REPEAT:
            supports_op = (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_I32 ||
                           src0->type == GGML_TYPE_I16);
            break;
        case GGML_OP_DUP:
        case GGML_OP_CPY:
        case GGML_OP_CONT:
            supports_op = (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_I32) &&
                          (src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32);
            break;
        case GGML_OP_SET:
            supports_op = src0->type == src1->type && src0->type == op->type &&
                          (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_I32);
            break;
        case GGML_OP_SET_ROWS:
            supports_op = ((op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_Q8_0 ||
                            op->type == GGML_TYPE_Q4_0) &&
                           (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
                           (src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32));
            break;
        case GGML_OP_GET_ROWS:
            if (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 ||
                ggml_webgpu_supported_qtype(src0->type)) {
                supports_op = (op->type == GGML_TYPE_F32);
            } else if (src0->type == GGML_TYPE_I32) {
                supports_op = op->type == GGML_TYPE_I32;
            }
            break;
        case GGML_OP_MUL_MAT:
            {
                switch (src1->type) {
                    case GGML_TYPE_F16:
                        supports_op |= (src0->type == GGML_TYPE_F16);
                        break;
                    case GGML_TYPE_F32:
                        switch (src0->type) {
                            case GGML_TYPE_F32:
                            case GGML_TYPE_F16:
                            case GGML_TYPE_BF16:
                            case GGML_TYPE_Q1_0:
                            case GGML_TYPE_Q2_0:
                            case GGML_TYPE_Q4_0:
                            case GGML_TYPE_Q4_1:
                            case GGML_TYPE_Q5_0:
                            case GGML_TYPE_Q5_1:
                            case GGML_TYPE_Q8_0:
                            case GGML_TYPE_Q2_K:
                            case GGML_TYPE_Q3_K:
                            case GGML_TYPE_Q4_K:
                            case GGML_TYPE_Q5_K:
                            case GGML_TYPE_Q6_K:
                            case GGML_TYPE_IQ2_XXS:
                            case GGML_TYPE_IQ2_XS:
                            case GGML_TYPE_IQ2_S:
                            case GGML_TYPE_IQ3_XXS:
                            case GGML_TYPE_IQ3_S:
                            case GGML_TYPE_IQ1_S:
                            case GGML_TYPE_IQ1_M:
                            case GGML_TYPE_IQ4_NL:
                            case GGML_TYPE_IQ4_XS:
                            case GGML_TYPE_MXFP4:
                            case GGML_TYPE_NVFP4:
                                supports_op = true;
                                break;
                            default:
                                break;
                        }
                    default:
                        break;
                }
                break;
            }
        case GGML_OP_MUL_MAT_ID:
            switch (src1->type) {
                case GGML_TYPE_F16:
                    supports_op |= (src0->type == GGML_TYPE_F16);
                    break;
                case GGML_TYPE_F32:
                    switch (src0->type) {
                        case GGML_TYPE_F32:
                        case GGML_TYPE_F16:
                        case GGML_TYPE_BF16:
                        case GGML_TYPE_Q1_0:
                        case GGML_TYPE_Q4_0:
                        case GGML_TYPE_Q4_1:
                        case GGML_TYPE_Q5_0:
                        case GGML_TYPE_Q5_1:
                        case GGML_TYPE_Q8_0:
                        case GGML_TYPE_Q2_K:
                        case GGML_TYPE_Q3_K:
                        case GGML_TYPE_Q4_K:
                        case GGML_TYPE_Q5_K:
                        case GGML_TYPE_Q6_K:
                        case GGML_TYPE_IQ1_S:
                        case GGML_TYPE_IQ1_M:
                        case GGML_TYPE_IQ2_XXS:
                        case GGML_TYPE_IQ2_XS:
                        case GGML_TYPE_IQ2_S:
                        case GGML_TYPE_IQ3_XXS:
                        case GGML_TYPE_IQ3_S:
                        case GGML_TYPE_IQ4_NL:
                        case GGML_TYPE_IQ4_XS:
                        case GGML_TYPE_MXFP4:
                        case GGML_TYPE_NVFP4:
                            supports_op = true;
                            break;
                        default:
                            break;
                    }
                    break;
                default:
                    break;
            }
            if (ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
                supports_op = false;
            }
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            {
                // conservative support checks for whether the more resource-intensive shader paths
                // can be used, to avoid cases where flash_attn is assigned to the CPU later on
                // K/V row lists (ggml_flash_attn_ext_rows) are not supported
                supports_op = op->src[5] == nullptr && src0->type == GGML_TYPE_F32 &&
                              (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16 ||
                               src1->type == GGML_TYPE_Q4_0 || src1->type == GGML_TYPE_Q8_0) &&
                              (src2->type == GGML_TYPE_F32 || src2->type == GGML_TYPE_F16 ||
                               src2->type == GGML_TYPE_Q4_0 || src2->type == GGML_TYPE_Q8_0) &&
                              op->type == GGML_TYPE_F32;
                if (!supports_op) {
                    break;
                }
                if (ggml_webgpu_tensor_binding_overlap(ctx->webgpu_global_ctx, src1, src2) &&
                    src1->type != src2->type && !(ggml_is_quantized(src1->type) && ggml_is_quantized(src2->type))) {
                    supports_op = false;
                    break;
                }
                const auto & capabilities             = ctx->webgpu_global_ctx->capabilities;
                const size_t storage_offset_alignment = capabilities.limits.minStorageBufferOffsetAlignment;

                // subgroup matrix path requirements
                const bool use_subgroup_matrix = ggml_webgpu_flash_attn_can_use_subgroup_matrix_path(
                    capabilities.supports_subgroup_matrix, capabilities.sg_mat_k, capabilities.sg_mat_n, src0, src2);

                // tile path requirements
                const bool float_vec4_aligned =
                    ((src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_F32) ||
                     ggml_webgpu_flash_attn_float_vec4_aligned(src1, storage_offset_alignment)) &&
                    ((src2->type != GGML_TYPE_F16 && src2->type != GGML_TYPE_F32) ||
                     ggml_webgpu_flash_attn_float_vec4_aligned(src2, storage_offset_alignment));
                const uint32_t k_tile_head_align = (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) ?
                                                       GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH :
                                                       (uint32_t) ggml_blck_size(src1->type);
                const uint32_t v_tile_head_align = (src2->type == GGML_TYPE_F32 || src2->type == GGML_TYPE_F16) ?
                                                       GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH :
                                                       (uint32_t) ggml_blck_size(src2->type);
                const bool     tile_kv_head_dims_aligned =
                    src0->ne[0] % k_tile_head_align == 0 && src2->ne[0] % v_tile_head_align == 0;
                const bool tile_can_dispatch_all_q_rows =
                    capabilities.limits.maxComputeInvocationsPerWorkgroup >=
                    GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE * capabilities.max_subgroup_size;
                const bool use_tile = !use_subgroup_matrix && capabilities.supports_subgroups && float_vec4_aligned &&
                                      tile_kv_head_dims_aligned && tile_can_dispatch_all_q_rows;

                if (!use_subgroup_matrix && !use_tile) {
                    supports_op = false;
                    break;
                }
                const uint32_t q_tile =
                    use_subgroup_matrix ? capabilities.sg_mat_m : GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;
                const uint32_t kv_granularity = use_subgroup_matrix ? capabilities.sg_mat_n : 1u;
                const bool     kv_direct = use_subgroup_matrix ?
                                               ggml_webgpu_flash_attn_k_direct(src0, src1, capabilities.sg_mat_k) ||
                                                   ggml_webgpu_flash_attn_v_direct(src0, src2, capabilities.sg_mat_k) :
                                               false;
                const uint32_t max_kv_tile = ggml_webgpu_flash_attn_max_kv_tile(
                    capabilities.limits.maxComputeWorkgroupStorageSize, q_tile, kv_granularity, (uint32_t) src0->ne[0],
                    (uint32_t) src2->ne[0], op->src[3] != nullptr, kv_direct);
                supports_op = max_kv_tile > 0;
                break;
            }
        case GGML_OP_GROUP_NORM:
            {
                bool           valid_type     = (src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) ||
                                                (src0->type == GGML_TYPE_F16 && op->type == GGML_TYPE_F16);
                // One workgroup per normalization group within each batch item.
                const uint64_t num_workgroups = uint64_t(ggml_get_op_params_i32(op, 0)) * uint64_t(src0->ne[3]);
                const uint64_t max_per_dim =
                    ctx->webgpu_global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;
                const bool dispatch_fits = num_workgroups <= max_per_dim * max_per_dim;
                supports_op              = valid_type && dispatch_fits;
                break;
            }
        case GGML_OP_RMS_NORM:
        case GGML_OP_NORM:
        case GGML_OP_L2_NORM:
            supports_op = (op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32) && ggml_is_contiguous_rows(src0);
            break;
        case GGML_OP_ROPE:
            supports_op = ggml_rope_get_truncate(op) && (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    supports_op = op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16;
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    supports_op = op->type == GGML_TYPE_F32;
                    break;
                default:
                    break;
            }
            break;
        case GGML_OP_SCALE:
            supports_op = op->type == GGML_TYPE_F32;
            break;
        case GGML_OP_SOFT_MAX:
            supports_op = op->type == GGML_TYPE_F32;
            break;
        case GGML_OP_UNARY:
            {
                const ggml_unary_op UNARY_OP = ggml_get_unary_op(op);

                switch (UNARY_OP) {
                    case GGML_UNARY_OP_ABS:
                    case GGML_UNARY_OP_SGN:
                    case GGML_UNARY_OP_NEG:
                    case GGML_UNARY_OP_STEP:
                    case GGML_UNARY_OP_TANH:
                    case GGML_UNARY_OP_ELU:
                    case GGML_UNARY_OP_RELU:
                    case GGML_UNARY_OP_SIGMOID:
                    case GGML_UNARY_OP_GELU:
                    case GGML_UNARY_OP_GELU_QUICK:
                    case GGML_UNARY_OP_SILU:
                    case GGML_UNARY_OP_HARDSWISH:
                    case GGML_UNARY_OP_HARDSIGMOID:
                    case GGML_UNARY_OP_EXP:
                    case GGML_UNARY_OP_GELU_ERF:
                    case GGML_UNARY_OP_SOFTPLUS:
                    case GGML_UNARY_OP_EXPM1:
                    case GGML_UNARY_OP_FLOOR:
                    case GGML_UNARY_OP_CEIL:
                    case GGML_UNARY_OP_ROUND:
                    case GGML_UNARY_OP_TRUNC:
                    case GGML_UNARY_OP_XIELU:
                        supports_op =
                            (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
                        break;
                    default:
                        break;
                }
            }
            break;
        case GGML_OP_TRI:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
            break;
        case GGML_OP_DIAG:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
            break;
        case GGML_OP_SOLVE_TRI:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32;
            break;
        case GGML_OP_CONV_2D:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
                          (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
            break;
        case GGML_OP_CONV_2D_DW:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
                          (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) &&
                          (ggml_is_contiguous(src1) || ggml_is_contiguous_channels(src1));
            break;
        case GGML_OP_IM2COL:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
            break;
        case GGML_OP_SSM_CONV:
            supports_op = op->type == GGML_TYPE_F32;
            break;
        case GGML_OP_SSM_SCAN:
            supports_op = op->type == GGML_TYPE_F32 &&
                          src0->ne[0] <= ctx->webgpu_global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
            break;
        case GGML_OP_GATED_DELTA_NET:
            {
                const uint32_t s_v = (uint32_t) src2->ne[0];
                supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                              src2->type == GGML_TYPE_F32 && op->src[3]->type == GGML_TYPE_F32 &&
                              op->src[4]->type == GGML_TYPE_F32 && op->src[5]->type == GGML_TYPE_F32 &&
                              s_v <= ctx->webgpu_global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
            }
            break;
        case GGML_OP_CLAMP:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            break;
        case GGML_OP_FILL:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            break;
        case GGML_OP_LOG:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            break;
        case GGML_OP_SQR:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            break;
        case GGML_OP_SQRT:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            break;
        case GGML_OP_SIN:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            break;
        case GGML_OP_COS:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            break;
        case GGML_OP_ARANGE:
            supports_op = op->type == GGML_TYPE_F32 && ggml_is_contiguous(op);
            break;
        case GGML_OP_PAD:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
            break;
        case GGML_OP_ARGMAX:
            supports_op = op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32;
            break;
        case GGML_OP_ARGSORT:
            supports_op = op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(src0);
            break;
        case GGML_OP_TOP_K:
            supports_op = op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(src0);
            break;
        case GGML_OP_CUMSUM:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type;
            break;
        case GGML_OP_SUM:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type && ggml_is_contiguously_allocated(src0);
            break;
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type && ggml_is_contiguous_rows(src0);
            break;
        case GGML_OP_UPSCALE:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
            break;
        default:
            break;
    }
    if (ggml_nbytes(op) > ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize ||
        (src0 != nullptr &&
         ggml_nbytes(src0) > ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize) ||
        (src1 != nullptr &&
         ggml_nbytes(src1) > ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize) ||
        (src2 != nullptr &&
         ggml_nbytes(src2) > ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize)) {
        supports_op = false;
        WEBGPU_LOG_DEBUG("ggml_webgpu op not supported due to size: ");
    }

    if (!supports_op) {
        WEBGPU_LOG_DEBUG("ggml_webgpu op not supported: "
                         << ggml_op_name(op->op) << " with types dst: " << ggml_type_name(op->type)
                         << ", src0: " << (op->src[0] ? ggml_type_name(op->src[0]->type) : "null")
                         << ", src1: " << (op->src[1] ? ggml_type_name(op->src[1]->type) : "null"));
    } else {
        WEBGPU_LOG_DEBUG("ggml_webgpu op supported: "
                         << ggml_op_name(op->op) << " with types dst: " << ggml_type_name(op->type)
                         << ", src0: " << (op->src[0] ? ggml_type_name(op->src[0]->type) : "null")
                         << ", src1: " << (op->src[1] ? ggml_type_name(op->src[1]->type) : "null"));
    }
    return supports_op;
}

static struct ggml_backend_device_i ggml_backend_webgpu_device_i = {
    /* .get_name             = */ ggml_backend_webgpu_device_get_name,
    /* .get_description      = */ ggml_backend_webgpu_device_get_description,
    /* .get_memory           = */ ggml_backend_webgpu_device_get_memory,
    /* .get_type             = */ ggml_backend_webgpu_device_get_type,
    /* .get_props            = */ ggml_backend_webgpu_device_get_props,
    /* .init_backend         = */ ggml_backend_webgpu_backend_init,
    /* .get_buffer_type      = */ ggml_backend_webgpu_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_webgpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_webgpu_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ ggml_backend_webgpu_device_event_new,
    /* .event_free           = */ ggml_backend_webgpu_device_event_free,
    /* .event_synchronize    = */ ggml_backend_webgpu_device_event_synchronize,
};

/* End GGML Backend Device Interface */

/* GGML Backend Registration Interface */

static const char * ggml_backend_webgpu_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_webgpu_reg_context * ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);
    return ctx->name;
}

static size_t ggml_backend_webgpu_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_webgpu_reg_context * ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);
    return ctx->device_count;
}

// Only one device is supported for now
static ggml_backend_dev_t ggml_backend_webgpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    WEBGPU_LOG_DEBUG("ggml_backend_reg_get_device()");

    WEBGPU_CPU_PROFILE_TOTAL_START(reg_get_device);

    ggml_backend_webgpu_reg_context * reg_ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);

    create_webgpu_device(reg_ctx);

    static ggml_backend_webgpu_device_context device_ctx;
    device_ctx.device_name            = GGML_WEBGPU_NAME;
    device_ctx.device_desc            = GGML_WEBGPU_NAME;
    device_ctx.webgpu_global_ctx      = reg_ctx->webgpu_global_ctx;
    // See GGML Backend Device Interface section
    static ggml_backend_device device = {
        /* .iface   = */ ggml_backend_webgpu_device_i,
        /* .reg     = */ reg,
        /* .context = */ &device_ctx,
    };

    WEBGPU_CPU_PROFILE_TOTAL_END(reg_get_device, reg_ctx->webgpu_global_ctx);
    return &device;
}

static const struct ggml_backend_reg_i ggml_backend_webgpu_reg_i = {
    /* .get_name         = */ ggml_backend_webgpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_webgpu_reg_get_device_count,
    /* .get_device       = */ ggml_backend_webgpu_reg_get_device,
    /* .get_proc_address = */ NULL,
};

/* End GGML Backend Registration Interface */

ggml_backend_reg_t ggml_backend_webgpu_reg() {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_reg()");

    // Intentionally leak the global registry context to avoid crashing inside
    // Dawn/Vulkan static teardown during process exit.
    static ggml_backend_webgpu_reg_context * ctx = new ggml_backend_webgpu_reg_context();

    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_webgpu_reg_i,
        /* .context     = */ ctx,
    };

    ctx->name         = GGML_WEBGPU_NAME;
    ctx->device_count = 0;

    // Keep one Dawn/WebGPU instance alive for the lifetime of the static backend
    // registry. Recreating it on repeated registry lookups can invalidate
    // adapter/device references that are still held by the backend/device layer.
    if (ctx->webgpu_global_ctx != nullptr && ctx->webgpu_global_ctx->instance != nullptr) {
        return &reg;
    }

    wgpu::InstanceDescriptor               instance_descriptor{};
    std::vector<wgpu::InstanceFeatureName> instance_features = { wgpu::InstanceFeatureName::TimedWaitAny };
    instance_descriptor.requiredFeatures                     = instance_features.data();
    instance_descriptor.requiredFeatureCount                 = instance_features.size();

#if !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    const char * const          instanceEnabledToggles[] = { "allow_unsafe_apis" };
    wgpu::DawnTogglesDescriptor instanceTogglesDesc;
    instanceTogglesDesc.enabledToggles     = instanceEnabledToggles;
    instanceTogglesDesc.enabledToggleCount = 1;
    instance_descriptor.nextInChain        = &instanceTogglesDesc;
#endif

    wgpu::Instance inst              = wgpu::CreateInstance(&instance_descriptor);
    ctx->webgpu_global_ctx           = webgpu_global_context(new webgpu_global_context_struct());
    ctx->webgpu_global_ctx->instance = std::move(inst);

    // Probe for adapter support
    wgpu::Adapter adapter;
    if (ctx->webgpu_global_ctx->instance != nullptr) {
        ggml_backend_webgpu_request_adapter(ctx->webgpu_global_ctx->instance, adapter);
    }

    // WebGPU backend requires f16 support and, on native, implicit device synchronization.
    if (adapter != nullptr && adapter.HasFeature(wgpu::FeatureName::ShaderF16)
#if !defined(__EMSCRIPTEN__) && !defined(__wasi__)
        && adapter.HasFeature(wgpu::FeatureName::ImplicitDeviceSynchronization)
#endif
    ) {
        ctx->device_count = 1;
    }

    return &reg;
}

ggml_backend_t ggml_backend_webgpu_init(void) {
    ggml_backend_reg_t reg = ggml_backend_webgpu_reg();
    if (ggml_backend_reg_dev_count(reg) == 0) {
        return nullptr;
    }
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    return ggml_backend_webgpu_backend_init(dev, nullptr);
}

GGML_BACKEND_DL_IMPL(ggml_backend_webgpu_reg)
