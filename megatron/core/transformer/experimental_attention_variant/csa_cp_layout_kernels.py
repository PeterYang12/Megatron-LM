# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CuTeDSL kernels for DSv4 THD context parallel layout work.

This module contains the CuTeDSL compaction and final-index kernels retained by
the DSv4 CP path. ``csa_cp_utils.py`` owns row mapping and compressor
input layout; ``csa.py`` calls final-index lowering directly. Local THD RoPE
reuses MCore's fused MLA implementation.
"""

import math
from typing import Optional, Tuple

import torch

try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass import utils
    from cutlass.cute.runtime import from_dlpack, make_fake_stream

    _CUTE_AVAILABLE = True
except ImportError:
    cuda = None
    cutlass = None
    cute = None
    from_dlpack = None
    make_fake_stream = None
    _CUTE_AVAILABLE = False

# =============================================================================
# CuTeDSL Kernel Definitions
# =============================================================================
# Device kernels and CuTe launch functions. These blocks describe the GPU work;
# tensor allocation, validation, and autograd integration stay in the wrappers.
# =============================================================================

if _CUTE_AVAILABLE:

    def _launch_named(
        kernel, name: str, args: Tuple, grid: Tuple, block: Tuple, stream: cuda.CUstream
    ):
        """Launch one CSA CP CuTe kernel and keep its timeline prefix local."""
        kernel.set_name_prefix(name)
        launcher = kernel(*args)
        try:
            launcher.launch(grid=grid, block=block, stream=stream)
        finally:
            if hasattr(kernel, "_name_prefix"):
                kernel._name_prefix = None
            dsl = getattr(launcher, "dsl", None)
            if dsl is not None and hasattr(dsl, "_name_prefix"):
                dsl._name_prefix = None

    # Kernel contract:
    #   hidden_local: local hidden rows, shape (l_local, row_width), bf16/fp16/fp32.
    #   boundary_hidden: left boundary rows, shape (d_window, row_width).
    #   hidden_compact: output compact rows, shape (compact_len, row_width).
    #   comp_ids: original per-sequence compressed group ids, shape (c_cap,).
    #   Enumerates visible full compression groups in [global_start-d_comp,
    #   global_start+l_local), copies their ratio tokens from boundary/local
    #   hidden into hidden_compact, and emits compressed group ids.
    @cute.kernel
    def _compressor_input_compact_fwd_kernel(
        hidden_local: cute.Tensor,
        boundary_hidden: cute.Tensor,
        hidden_compact: cute.Tensor,
        cu_seqlens: cute.Tensor,
        comp_ids: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        ratio: cutlass.Constexpr,
        d_comp: cutlass.Constexpr,
        d_window: cutlass.Constexpr,
        compact_len: cutlass.Int32,
        row_width: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        row = bidx

        if cutlass.const_expr(hidden_compact.element_type.width == 16 and row_width % 4 == 0):
            vec_elems = cutlass.Int32(4)
            vec_cols = row_width // 4
        else:
            vec_elems = cutlass.Int32(1)
            vec_cols = row_width

        range_start = global_start
        range_end = global_start + l_local
        first_range_group_start = range_start - d_comp

        src_global = cutlass.Int32(-1)
        visible_comp_id = cutlass.Int32(-1)
        if row < compact_len:
            if tidx == 0 or tidx == 32:
                running_tokens = 0
                for seq in range(n_seq):
                    seq_start = cu_seqlens[seq]
                    seq_end = cu_seqlens[seq + 1]
                    local_seq_end = seq_end
                    if local_seq_end > range_end:
                        local_seq_end = range_end

                    if seq_start < local_seq_end and range_start < local_seq_end:
                        first_visible_numer = first_range_group_start - seq_start
                        if first_visible_numer < 0:
                            first_visible_numer = 0
                        first_visible_group = (first_visible_numer + ratio - 1) // ratio
                        stop_visible_group = (local_seq_end - seq_start) // ratio
                        visible_group_count = stop_visible_group - first_visible_group
                        if visible_group_count < 0:
                            visible_group_count = 0
                        visible_token_count = visible_group_count * ratio

                        if row >= running_tokens and row < running_tokens + visible_token_count:
                            local_visible_token = row - running_tokens
                            comp_id = first_visible_group + local_visible_token // ratio
                            token_in_group = (
                                local_visible_token - (local_visible_token // ratio) * ratio
                            )
                            src_global = seq_start + comp_id * ratio + token_in_group
                            visible_comp_id = comp_id
                        running_tokens = running_tokens + visible_token_count

            src_global = cute.arch.shuffle_sync(src_global, 0, mask=-1, mask_and_clamp=31)
            if row % ratio == 0 and tidx == 0:
                comp_ids[row // ratio] = visible_comp_id

            vec_col = tidx
            while vec_col < vec_cols:
                dst_offset = row * row_width + vec_col * vec_elems
                if cutlass.const_expr(
                    hidden_compact.element_type.width == 16 and row_width % 4 == 0
                ):
                    dst_ptr = cute.recast_ptr(
                        hidden_compact.iterator + dst_offset, dtype=cutlass.Int64
                    )
                    value = cutlass.Int64(0)
                    if src_global >= 0:
                        if src_global < range_start:
                            src_row = src_global - (range_start - d_window)
                            src_offset = src_row * row_width + vec_col * vec_elems
                            src_ptr = cute.recast_ptr(
                                boundary_hidden.iterator + src_offset, dtype=cutlass.Int64
                            )
                            value = cute.arch.load(
                                src_ptr.llvm_ptr,
                                cutlass.Int64,
                                level1_eviction_priority="evict_no_allocate",
                            )
                        else:
                            src_row = src_global - range_start
                            src_offset = src_row * row_width + vec_col * vec_elems
                            src_ptr = cute.recast_ptr(
                                hidden_local.iterator + src_offset, dtype=cutlass.Int64
                            )
                            value = cute.arch.load(
                                src_ptr.llvm_ptr,
                                cutlass.Int64,
                                level1_eviction_priority="evict_no_allocate",
                            )
                    cute.arch.store(dst_ptr.llvm_ptr, value, level1_eviction_priority="evict_first")
                else:
                    value = cutlass.Float32(0.0).to(hidden_compact.element_type)
                    if src_global >= 0:
                        if src_global < range_start:
                            src_row = src_global - (range_start - d_window)
                            value = boundary_hidden[src_row, vec_col]
                        else:
                            src_row = src_global - range_start
                            value = hidden_local[src_row, vec_col]
                    hidden_compact[row, vec_col] = value
                vec_col = vec_col + 64

    # Launch contract for _compressor_input_compact_fwd_kernel.
    #   One block copies each compact row; group-leading blocks also emit comp_ids.
    @cute.jit
    def _compressor_input_compact_fwd_launch(
        hidden_local: cute.Tensor,
        boundary_hidden: cute.Tensor,
        hidden_compact: cute.Tensor,
        cu_seqlens: cute.Tensor,
        comp_ids: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        ratio: cutlass.Constexpr,
        d_comp: cutlass.Constexpr,
        d_window: cutlass.Constexpr,
        compact_len: cutlass.Int32,
        row_width: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        _launch_named(
            _compressor_input_compact_fwd_kernel,
            "dsv4_cp_compressor_input_compact_fwd",
            (
                hidden_local,
                boundary_hidden,
                hidden_compact,
                cu_seqlens,
                comp_ids,
                n_seq,
                global_start,
                l_local,
                ratio,
                d_comp,
                d_window,
                compact_len,
                row_width,
            ),
            grid=(compact_len, 1, 1),
            block=(64, 1, 1),
            stream=stream,
        )

    # Kernel contract:
    #   grad_hidden_compact: gradient of compact rows, shape (compact_len, row_width).
    #   grad_hidden_local: output gradient, shape (l_local, row_width).
    #   grad_boundary_hidden: output gradient, shape (d_window, row_width).
    #   Reconstructs the same compact source mapping as forward and scatters each
    #   compact-row gradient back to either boundary or local hidden.
    @cute.kernel
    def _compressor_input_compact_bwd_kernel(
        grad_hidden_compact: cute.Tensor,
        grad_hidden_local: cute.Tensor,
        grad_boundary_hidden: cute.Tensor,
        cu_seqlens: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        ratio: cutlass.Constexpr,
        d_comp: cutlass.Constexpr,
        d_window: cutlass.Constexpr,
        compact_len: cutlass.Int32,
        row_width: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        out_row = bidx

        if cutlass.const_expr(grad_hidden_compact.element_type.width == 16 and row_width % 4 == 0):
            vec_elems = cutlass.Int32(4)
            vec_cols = row_width // 4
        else:
            vec_elems = cutlass.Int32(1)
            vec_cols = row_width

        compact_row = cutlass.Int32(-1)
        range_start = global_start
        range_end = global_start + l_local
        first_range_group_start = range_start - d_comp
        src_global = range_start + (out_row - d_window)
        dst_row = out_row - d_window
        if out_row < d_window:
            dst_row = out_row

        if tidx == 0 or tidx == 32:
            running_tokens = 0
            for seq in range(n_seq):
                seq_start = cu_seqlens[seq]
                seq_end = cu_seqlens[seq + 1]
                local_seq_end = seq_end
                if local_seq_end > range_end:
                    local_seq_end = range_end

                if seq_start < local_seq_end and range_start < local_seq_end:
                    first_visible_numer = first_range_group_start - seq_start
                    if first_visible_numer < 0:
                        first_visible_numer = 0
                    first_visible_group = (first_visible_numer + ratio - 1) // ratio
                    stop_visible_group = (local_seq_end - seq_start) // ratio
                    visible_group_count = stop_visible_group - first_visible_group
                    if visible_group_count < 0:
                        visible_group_count = 0
                    visible_token_count = visible_group_count * ratio

                    if src_global >= seq_start and src_global < seq_end:
                        seq_offset = src_global - seq_start
                        comp_id = seq_offset // ratio
                        token_in_group = seq_offset - comp_id * ratio
                        if comp_id >= first_visible_group and comp_id < stop_visible_group:
                            compact_row = (
                                running_tokens
                                + (comp_id - first_visible_group) * ratio
                                + token_in_group
                            )
                    running_tokens = running_tokens + visible_token_count

        compact_row = cute.arch.shuffle_sync(compact_row, 0, mask=-1, mask_and_clamp=31)

        vec_col = tidx
        while vec_col < vec_cols:
            dst_offset = dst_row * row_width + vec_col * vec_elems
            if cutlass.const_expr(
                grad_hidden_compact.element_type.width == 16 and row_width % 4 == 0
            ):
                value = cutlass.Int64(0)
                if compact_row >= 0 and compact_row < compact_len:
                    src_offset = compact_row * row_width + vec_col * vec_elems
                    src_ptr = cute.recast_ptr(
                        grad_hidden_compact.iterator + src_offset, dtype=cutlass.Int64
                    )
                    value = cute.arch.load(
                        src_ptr.llvm_ptr,
                        cutlass.Int64,
                        level1_eviction_priority="evict_no_allocate",
                    )
                if out_row < d_window:
                    dst_ptr = cute.recast_ptr(
                        grad_boundary_hidden.iterator + dst_offset, dtype=cutlass.Int64
                    )
                    cute.arch.store(dst_ptr.llvm_ptr, value, level1_eviction_priority="evict_first")
                else:
                    dst_ptr = cute.recast_ptr(
                        grad_hidden_local.iterator + dst_offset, dtype=cutlass.Int64
                    )
                    cute.arch.store(dst_ptr.llvm_ptr, value, level1_eviction_priority="evict_first")
            else:
                value = cutlass.Float32(0.0).to(grad_hidden_compact.element_type)
                if compact_row >= 0 and compact_row < compact_len:
                    value = grad_hidden_compact[compact_row, vec_col]
                if out_row < d_window:
                    grad_boundary_hidden[dst_row, vec_col] = value
                else:
                    grad_hidden_local[dst_row, vec_col] = value
            vec_col = vec_col + 64

    # Launch contract for _compressor_input_compact_bwd_kernel.
    #   total_work is the number of boundary + local gradient rows.
    @cute.jit
    def _compressor_input_compact_bwd_launch(
        grad_hidden_compact: cute.Tensor,
        grad_hidden_local: cute.Tensor,
        grad_boundary_hidden: cute.Tensor,
        cu_seqlens: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        ratio: cutlass.Constexpr,
        d_comp: cutlass.Constexpr,
        d_window: cutlass.Constexpr,
        compact_len: cutlass.Int32,
        row_width: cutlass.Constexpr,
        total_work: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _launch_named(
            _compressor_input_compact_bwd_kernel,
            "dsv4_cp_compressor_input_compact_bwd",
            (
                grad_hidden_compact,
                grad_hidden_local,
                grad_boundary_hidden,
                cu_seqlens,
                n_seq,
                global_start,
                l_local,
                ratio,
                d_comp,
                d_window,
                compact_len,
                row_width,
            ),
            grid=(total_work, 1, 1),
            block=(64, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def _build_attention_indices_kernel(
        cu_seqlens: cute.Tensor,
        cu_seqlens_compressed: cute.Tensor,
        indexer_topk: cute.Tensor,
        seq_to_rank_row: cute.Tensor,
        topk_idxs: cute.Tensor,
        topk_length: cute.Tensor,
        indexer_rank_major: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        d_window: cutlass.Int32,
        window_size: cutlass.Int32,
        ratio: cutlass.Int32,
        compressed_width: cutlass.Int32,
        seq_major_rows: cutlass.Int32,
        compressed_base: cutlass.Int32,
        total_width: cutlass.Int32,
        index_mode: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        linear = bidx * 128 + tidx

        # 0: selected top-k, 1: all visible compressed rows, 2: indexer loss.
        if cutlass.const_expr(index_mode == 0):
            row = linear
        else:
            row = bidx

        if row < l_local:
            global_q = global_start + row
            seq_start_found = cutlass.Int32(-1)
            seq_comp_start = cutlass.Int32(0)
            seq_comp_len = cutlass.Int32(0)
            if cutlass.const_expr(index_mode == 0):
                for seq in range(n_seq):
                    seq_start = cu_seqlens[seq]
                    seq_end = cu_seqlens[seq + 1]
                    if global_q >= seq_start and global_q < seq_end:
                        seq_start_found = seq_start
                        if ratio > 1 and compressed_width > 0:
                            seq_comp_start = cu_seqlens_compressed[seq]
                            seq_comp_len = cu_seqlens_compressed[seq + 1] - seq_comp_start
            else:
                shared = utils.SmemAllocator().allocate_tensor(cutlass.Int32, cute.make_layout(3))
                if tidx == 0:
                    for seq in range(n_seq):
                        seq_start = cu_seqlens[seq]
                        seq_end = cu_seqlens[seq + 1]
                        if global_q >= seq_start and global_q < seq_end:
                            seq_start_found = seq_start
                            if ratio > 1 and compressed_width > 0:
                                seq_comp_start = cu_seqlens_compressed[seq]
                                seq_comp_len = cu_seqlens_compressed[seq + 1] - seq_comp_start
                    shared[0] = seq_start_found
                    shared[1] = seq_comp_start
                    shared[2] = seq_comp_len
                cute.arch.sync_threads()
                seq_start_found = shared[0]
                seq_comp_start = shared[1]
                seq_comp_len = shared[2]

            if cutlass.const_expr(index_mode != 0):
                col = tidx
                while col < total_width:
                    topk_value = cutlass.Int32(-1)
                    rank_major_value = cutlass.Int32(-1)
                    length = cutlass.Int32(0)
                    if seq_start_found >= 0:
                        window_start = global_q - window_size + 1
                        if window_start < seq_start_found:
                            window_start = seq_start_found
                        window_count = global_q - window_start + 1
                        if cutlass.const_expr(index_mode == 2):
                            if col < compressed_width:
                                comp_id = indexer_topk[row, col]
                                if comp_id >= 0 and comp_id < seq_comp_len:
                                    seq_major_id = seq_comp_start + comp_id
                                    if seq_major_id < seq_major_rows:
                                        rank_major_value = seq_to_rank_row[seq_major_id]
                                        if rank_major_value >= 0:
                                            topk_value = compressed_base + rank_major_value
                            else:
                                window_col = col - compressed_width
                                if window_col < window_count:
                                    pos = window_start + window_col
                                    if pos < global_start:
                                        topk_value = pos - (global_start - d_window)
                                    else:
                                        topk_value = d_window + pos - global_start
                        else:
                            comp_count = cutlass.Int32(0)
                            if ratio > 1 and compressed_width > 0:
                                comp_count = (global_q - seq_start_found + 1) // ratio
                                if comp_count > compressed_width:
                                    comp_count = compressed_width
                                if comp_count > seq_comp_len:
                                    comp_count = seq_comp_len
                            length = window_count + comp_count
                            if col < window_count:
                                pos = window_start + col
                                if pos < global_start:
                                    topk_value = pos - (global_start - d_window)
                                else:
                                    topk_value = d_window + pos - global_start
                            elif col < length:
                                seq_major_id = seq_comp_start + col - window_count
                                if seq_major_id < seq_major_rows:
                                    rank_major_id = seq_to_rank_row[seq_major_id]
                                    if rank_major_id >= 0:
                                        topk_value = compressed_base + rank_major_id
                    elif total_width > 0 and cutlass.const_expr(index_mode != 2):
                        length = 1
                        if col == 0:
                            topk_value = 0
                    topk_idxs[row, col] = topk_value
                    if cutlass.const_expr(index_mode == 2):
                        if col < compressed_width:
                            indexer_rank_major[row, col] = rank_major_value
                    elif col == 0:
                        topk_length[row] = length
                    col = col + 128
            else:
                for out_col in range(total_width):
                    topk_idxs[row, out_col] = -1
                topk_length[row] = 0

                if seq_start_found >= 0:
                    write_col = cutlass.Int32(0)
                    window_start = global_q - window_size + 1
                    if window_start < seq_start_found:
                        window_start = seq_start_found
                    window_count = global_q - window_start + 1
                    for window_col in range(window_size):
                        if window_col < window_count:
                            pos = window_start + window_col
                            if pos < global_start:
                                topk_idxs[row, write_col] = pos - (global_start - d_window)
                            else:
                                topk_idxs[row, write_col] = d_window + pos - global_start
                            write_col = write_col + 1

                    if ratio > 1 and compressed_width > 0:
                        for compressed_col in range(compressed_width):
                            comp_id = indexer_topk[row, compressed_col]
                            if comp_id >= 0 and comp_id < seq_comp_len:
                                seq_major_id = seq_comp_start + comp_id
                                if seq_major_id < seq_major_rows:
                                    rank_major_id = seq_to_rank_row[seq_major_id]
                                    if rank_major_id >= 0:
                                        topk_idxs[row, write_col] = compressed_base + rank_major_id
                                        write_col = write_col + 1
                    topk_length[row] = write_col
                elif total_width > 0:
                    topk_idxs[row, 0] = 0
                    topk_length[row] = 1

    @cute.jit
    def _build_attention_indices_launch(
        cu_seqlens: cute.Tensor,
        cu_seqlens_compressed: cute.Tensor,
        indexer_topk: cute.Tensor,
        seq_to_rank_row: cute.Tensor,
        topk_idxs: cute.Tensor,
        topk_length: cute.Tensor,
        indexer_rank_major: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        d_window: cutlass.Int32,
        window_size: cutlass.Int32,
        ratio: cutlass.Int32,
        compressed_width: cutlass.Int32,
        seq_major_rows: cutlass.Int32,
        compressed_base: cutlass.Int32,
        total_width: cutlass.Int32,
        index_mode: cutlass.Constexpr,
        launch_work: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _launch_named(
            _build_attention_indices_kernel,
            "dsv4_cp_build_attention_indices",
            (
                cu_seqlens,
                cu_seqlens_compressed,
                indexer_topk,
                seq_to_rank_row,
                topk_idxs,
                topk_length,
                indexer_rank_major,
                n_seq,
                global_start,
                l_local,
                d_window,
                window_size,
                ratio,
                compressed_width,
                seq_major_rows,
                compressed_base,
                total_width,
                index_mode,
            ),
            grid=(cute.ceil_div(launch_work, 128), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )


# =============================================================================
# Torch Wrapper Functions
# =============================================================================
# Torch-facing entry points called by csa.py and csa_cp_utils.py. They validate
# inputs, allocate outputs, and dispatch the CuTeDSL kernels defined above.
# =============================================================================


def _require_cute(message: str, *tensors: Optional[torch.Tensor]) -> None:
    """Raise ``RuntimeError`` when a wrapper cannot use CuTeDSL kernels."""
    if not _cute_usable(*tensors):
        raise RuntimeError(message)


def _cute_usable(*tensors: Optional[torch.Tensor]) -> bool:
    """Whether the CuTeDSL kernels can run on these tensors (else the PyTorch fallback runs)."""
    return (
        _CUTE_AVAILABLE
        and torch.version.hip is None
        and all(tensor is None or tensor.is_cuda for tensor in tensors)
    )


_COMPILED_LAUNCH_CACHE = {}


# =============================================================================
# PyTorch fallback
# =============================================================================
# Same contracts as the CuTeDSL kernels above, written with gather/scatter and
# searchsorted so they run wherever PyTorch runs (ROCm, CPU, CUDA without
# CuTeDSL). Used automatically when the kernels are unavailable.
# =============================================================================


def _visible_group_spans(
    cu_seqlens: torch.Tensor, global_start: int, l_local: int, ratio: int, d_comp: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per sequence: start row, first visible compressed group, visible token count, token prefix sum.

    A compressed group is visible to this rank when its tokens lie in
    ``[global_start - d_comp, global_start + l_local)`` and inside the sequence.
    """
    cu = cu_seqlens.to(torch.int64)
    seq_start, seq_end = cu[:-1], cu[1:]
    range_start = int(global_start)
    range_end = range_start + int(l_local)
    local_seq_end = seq_end.clamp_max(range_end)
    active = (seq_start < local_seq_end) & (local_seq_end > range_start)
    first_group = ((range_start - int(d_comp) - seq_start).clamp_min(0) + int(ratio) - 1) // int(
        ratio
    )
    stop_group = (local_seq_end - seq_start) // int(ratio)
    count = torch.where(active, (stop_group - first_group).clamp_min(0), torch.zeros_like(stop_group))
    tokens = count * int(ratio)
    return seq_start, first_group, tokens, torch.cumsum(tokens, dim=0)


def _compact_source_rows(
    cu_seqlens: torch.Tensor,
    global_start: int,
    l_local: int,
    ratio: int,
    d_comp: int,
    compact_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Global source row (-1 when empty) and compressed group id (-1) of every compact row."""
    seq_start, first_group, tokens, prefix = _visible_group_spans(
        cu_seqlens, global_start, l_local, ratio, d_comp
    )
    n_seq = seq_start.shape[0]
    rows = torch.arange(int(compact_len), dtype=torch.int64, device=cu_seqlens.device)
    seq_idx = torch.searchsorted(prefix, rows, right=True)
    valid = seq_idx < n_seq
    seq_idx = seq_idx.clamp_max(max(n_seq - 1, 0))
    local_token = rows - (prefix[seq_idx] - tokens[seq_idx])
    comp_id = first_group[seq_idx] + local_token // int(ratio)
    src_global = seq_start[seq_idx] + comp_id * int(ratio) + local_token % int(ratio)
    src_global = torch.where(valid, src_global, torch.full_like(src_global, -1))
    comp_id = torch.where(valid, comp_id, torch.full_like(comp_id, -1))
    return src_global, comp_id


def _compressor_input_compact_torch_forward(
    ctx,
    hidden_local: torch.Tensor,
    boundary_hidden: torch.Tensor,
    cu_seqlens: torch.Tensor,
    global_start: int,
    ratio: int,
    d_comp: int,
    c_cap: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch forward of ``CompressorInputCompact``: one gather from (boundary | local | zero row)."""
    l_local = hidden_local.shape[0]
    d_window = boundary_hidden.shape[0]
    range_start = int(global_start)
    compact_len = int(c_cap) * int(ratio)
    src_global, comp_id = _compact_source_rows(
        cu_seqlens, global_start, l_local, ratio, d_comp, compact_len
    )
    # Row index into cat(boundary, local, zero): boundary rows cover
    # [range_start - d_window, range_start), local rows [range_start, range_start + l_local).
    gather = torch.where(
        src_global < 0,
        torch.full_like(src_global, d_window + l_local),
        torch.where(
            src_global < range_start,
            src_global - (range_start - d_window),
            src_global - range_start + d_window,
        ),
    )
    row_width = hidden_local[0].numel() if l_local else boundary_hidden[0].numel()
    source = torch.cat(
        (
            boundary_hidden.reshape(d_window, row_width),
            hidden_local.reshape(l_local, row_width),
            hidden_local.new_zeros((1, row_width)),
        ),
        dim=0,
    )
    hidden_compact = source.index_select(0, gather).reshape(
        (compact_len,) + tuple(hidden_local.shape[1:])
    )
    ctx.save_for_backward(gather)
    ctx.torch_shapes = (tuple(hidden_local.shape), tuple(boundary_hidden.shape))
    # One group id per compact group (the kernel writes it from each group's first row).
    return hidden_compact, comp_id[:: int(ratio)].to(torch.int32)


def _compressor_input_compact_torch_backward(ctx, grad_hidden_compact: torch.Tensor):
    """PyTorch backward of ``CompressorInputCompact``: scatter-add back through the same gather."""
    (gather,) = ctx.saved_tensors
    hidden_shape, boundary_shape = ctx.torch_shapes
    l_local, d_window = hidden_shape[0], boundary_shape[0]
    grad = grad_hidden_compact.reshape(grad_hidden_compact.shape[0], -1)
    grad_source = grad.new_zeros((d_window + l_local + 1, grad.shape[1]))
    grad_source.index_add_(0, gather, grad)
    grad_hidden = grad_source[d_window : d_window + l_local].reshape(hidden_shape)
    grad_boundary = grad_source[:d_window].reshape(boundary_shape)
    return (grad_hidden, grad_boundary, *([None] * 5))


def _build_attention_indices_torch(
    cu_seqlens: torch.Tensor,
    global_start: int,
    l_local: int,
    d_window: int,
    window_size: int,
    ratio: int,
    compressed_width: int,
    compressed_topk: Optional[torch.Tensor],
    cu_seqlens_compressed: torch.Tensor,
    seq_to_rank_row: torch.Tensor,
    for_indexer_loss: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """PyTorch version of ``build_attention_indices`` (same three modes as the kernel)."""
    device = cu_seqlens.device
    global_start, l_local = int(global_start), int(l_local)
    d_window, window_size = int(d_window), int(window_size)
    ratio, compressed_width = int(ratio), int(compressed_width)
    total_width = window_size + compressed_width
    compressed_base = d_window + l_local

    cu = cu_seqlens.to(torch.int64)
    n_seq = cu.shape[0] - 1
    rows = torch.arange(l_local, dtype=torch.int64, device=device)
    global_q = global_start + rows
    seq_idx = torch.searchsorted(cu[1:], global_q, right=True)
    found = seq_idx < n_seq
    seq_idx = seq_idx.clamp_max(max(n_seq - 1, 0))
    seq_start = cu[seq_idx]
    found = found & (global_q >= seq_start)

    use_comp = ratio > 1 and compressed_width > 0
    if use_comp:
        cu_comp = cu_seqlens_compressed.to(torch.int64)
        seq_comp_start = cu_comp[seq_idx]
        seq_comp_len = cu_comp[seq_idx + 1] - seq_comp_start
    else:
        seq_comp_start = torch.zeros_like(seq_start)
        seq_comp_len = torch.zeros_like(seq_start)
    seq_major_rows = seq_to_rank_row.shape[0]
    rank_rows = seq_to_rank_row.to(torch.int64)

    def rank_major_of(seq_major_id: torch.Tensor, ok: torch.Tensor):
        ok = ok & (seq_major_id >= 0) & (seq_major_id < seq_major_rows)
        if seq_major_rows == 0:
            return torch.full_like(seq_major_id, -1), ok
        rank_major = rank_rows[seq_major_id.clamp(0, seq_major_rows - 1)]
        return rank_major, ok & (rank_major >= 0)

    window_start = torch.maximum(global_q - window_size + 1, seq_start)
    window_count = global_q - window_start + 1
    window_cols = torch.arange(window_size, dtype=torch.int64, device=device)
    window_valid = (window_cols[None, :] < window_count[:, None]) & found[:, None]
    window_vals = window_start[:, None] + window_cols[None, :] - global_start + d_window
    topk_idxs = torch.full((l_local, total_width), -1, dtype=torch.int64, device=device)

    if for_indexer_loss:
        comp = compressed_topk.to(torch.int64)
        ok = (comp >= 0) & (comp < seq_comp_len[:, None]) & found[:, None]
        rank_major, ok = rank_major_of(seq_comp_start[:, None] + comp, ok)
        topk_idxs[:, :compressed_width] = torch.where(ok, compressed_base + rank_major, -1)
        topk_idxs[:, compressed_width:] = torch.where(window_valid, window_vals, -1)
        indexer_rank_major = torch.where(ok, rank_major, -1).to(torch.int32)
        return topk_idxs.to(torch.int32), None, indexer_rank_major

    topk_idxs[:, :window_size] = torch.where(window_valid, window_vals, -1)
    length = window_count.clone()
    if use_comp:
        row_idx = rows[:, None].expand(l_local, compressed_width)
        if compressed_topk is not None:
            # Selected top-k: valid entries are packed right after the window entries.
            comp = compressed_topk.to(torch.int64)
            ok = (comp >= 0) & (comp < seq_comp_len[:, None]) & found[:, None]
            rank_major, ok = rank_major_of(seq_comp_start[:, None] + comp, ok)
            position = window_count[:, None] + torch.cumsum(ok.to(torch.int64), dim=1) - 1
            topk_idxs[row_idx[ok], position[ok]] = compressed_base + rank_major[ok]
            length = length + ok.sum(dim=1)
        else:
            # All visible compressed rows, in sequence order.
            comp_cols = torch.arange(compressed_width, dtype=torch.int64, device=device)
            comp_count = torch.minimum(
                torch.minimum((global_q - seq_start + 1) // ratio, seq_comp_len),
                torch.full_like(seq_comp_len, compressed_width),
            )
            visible = (comp_cols[None, :] < comp_count[:, None]) & found[:, None]
            rank_major, ok = rank_major_of(seq_comp_start[:, None] + comp_cols[None, :], visible)
            vals = torch.where(ok, compressed_base + rank_major, -1)
            position = window_count[:, None] + comp_cols[None, :]
            topk_idxs[row_idx[visible], position[visible]] = vals[visible]
            length = length + torch.where(found, comp_count, torch.zeros_like(comp_count))
    if total_width > 0:
        topk_idxs[~found] = -1
        topk_idxs[~found, 0] = 0
        length = torch.where(found, length, torch.ones_like(length))
    else:
        length = torch.where(found, length, torch.zeros_like(length))
    return topk_idxs.to(torch.int32), length.to(torch.int32), None


def _run_compiled_launch(
    launch_fn,
    tensor_args: Tuple[torch.Tensor, ...],
    scalar_args: Tuple,
    static_arg_indices: Tuple[int, ...] = (),
) -> None:
    """Compile/cache a CuTe launch function and invoke it on the current stream."""
    static_arg_set = set(static_arg_indices)
    key = (
        launch_fn.__name__,
        tuple(
            (tensor.dtype, tuple(tensor.shape), tuple(tensor.stride())) for tensor in tensor_args
        ),
        tuple((i, scalar_args[i]) for i in static_arg_indices),
    )
    compiled = _COMPILED_LAUNCH_CACHE.get(key)
    if compiled is None:
        cap = torch.cuda.get_device_capability()
        arch = {(9, 0): "sm_90a", (10, 0): "sm_100a", (10, 3): "sm_103a"}.get(cap)
        if arch is None:
            raise RuntimeError(
                f"Unsupported GPU compute capability {cap} for CSA CP CuTe kernels; "
                "supported architectures: sm_90a, sm_100a, sm_103a."
            )
        cute_tensor_args = []
        for tensor in tensor_args:
            cute_tensor = from_dlpack(tensor.detach(), assumed_align=16, enable_tvm_ffi=True)
            if tensor.ndim != 0:
                cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=tensor.ndim - 1)
            cute_tensor_args.append(cute_tensor)
        compiled = cute.compile(
            launch_fn,
            *cute_tensor_args,
            *(
                arg if i in static_arg_set else cutlass.Int32(arg)
                for i, arg in enumerate(scalar_args)
            ),
            make_fake_stream(use_tvm_ffi_env_stream=False),
            options=f"--enable-tvm-ffi --gpu-arch {arch}",
        )
        _COMPILED_LAUNCH_CACHE[key] = compiled
    compiled(
        *tensor_args,
        *(cutlass.Int32(arg) for i, arg in enumerate(scalar_args) if i not in static_arg_set),
        cuda.CUstream(torch.cuda.current_stream(tensor_args[0].device).cuda_stream),
    )


class CompressorInputCompact(torch.autograd.Function):
    """Compact local and boundary hidden rows for compressor input.

    Inputs:
        hidden_local: CUDA tensor, shape ``(l_local, ...)``.
        boundary_hidden: CUDA tensor, shape ``(d_window, ...)``.
        cu_seqlens: int32 CUDA tensor, shape ``(n_seq + 1,)``.
        global_start: first global row in ``hidden_local``.
        ratio/d_comp/c_cap: compressor window and fixed group capacity.

    Outputs:
        ``hidden_compact`` shape ``(c_cap * ratio, ...)``, same dtype as hidden,
        and ``comp_ids`` int32 shape ``(c_cap,)``.
        The kernel copies the ratio source tokens for each visible compressed
        group and emits each row's compressed group id.
    """

    @staticmethod
    def forward(
        ctx,
        hidden_local: torch.Tensor,
        boundary_hidden: torch.Tensor,
        cu_seqlens: torch.Tensor,
        global_start: int,
        ratio: int,
        d_comp: int,
        c_cap: int,
    ):
        """Compact local and boundary hidden rows into compressor input rows."""
        ctx.use_torch = not _cute_usable(hidden_local, boundary_hidden, cu_seqlens)
        if ctx.use_torch:
            return _compressor_input_compact_torch_forward(
                ctx, hidden_local, boundary_hidden, cu_seqlens, global_start, ratio, d_comp, c_cap
            )
        l_local = hidden_local.shape[0]
        d_window = boundary_hidden.shape[0]
        ctx.hidden_shape = tuple(hidden_local.shape)
        ctx.boundary_shape = tuple(boundary_hidden.shape)
        ctx.compact_args = (int(global_start), int(l_local), int(ratio), int(d_comp), int(d_window))
        ctx.save_for_backward(cu_seqlens)

        compact_len = int(c_cap) * int(ratio)
        hidden_compact = hidden_local.new_empty((compact_len,) + tuple(hidden_local.shape[1:]))
        comp_ids = torch.empty((int(c_cap),), dtype=torch.int32, device=hidden_local.device)
        row_width = math.prod(hidden_local.shape[1:])
        _run_compiled_launch(
            _compressor_input_compact_fwd_launch,
            (
                hidden_local.reshape(int(l_local), row_width),
                boundary_hidden.reshape(int(d_window), row_width),
                hidden_compact.reshape(compact_len, row_width),
                cu_seqlens,
                comp_ids,
            ),
            (
                cu_seqlens.shape[0] - 1,
                int(global_start),
                int(l_local),
                int(ratio),
                int(d_comp),
                int(d_window),
                compact_len,
                row_width,
            ),
            static_arg_indices=(3, 4, 5, 7),
        )
        return hidden_compact, comp_ids

    @staticmethod
    def backward(ctx, grad_hidden_compact: torch.Tensor, _grad_comp_ids: torch.Tensor):
        """Scatter compacted compressor gradients to local and boundary rows."""
        if ctx.use_torch:
            return _compressor_input_compact_torch_backward(ctx, grad_hidden_compact)
        (cu_seqlens,) = ctx.saved_tensors
        global_start, l_local, ratio, d_comp, d_window = ctx.compact_args
        grad_hidden_compact = grad_hidden_compact.contiguous()
        grad_hidden = grad_hidden_compact.new_empty(ctx.hidden_shape)
        grad_boundary = grad_hidden_compact.new_empty(ctx.boundary_shape)
        compact_len = grad_hidden_compact.shape[0]
        row_width = math.prod(ctx.hidden_shape[1:])
        _run_compiled_launch(
            _compressor_input_compact_bwd_launch,
            (
                grad_hidden_compact.reshape(compact_len, row_width),
                grad_hidden.reshape(l_local, row_width),
                grad_boundary.reshape(d_window, row_width),
                cu_seqlens,
            ),
            (
                cu_seqlens.shape[0] - 1,
                global_start,
                l_local,
                ratio,
                d_comp,
                d_window,
                compact_len,
                row_width,
                l_local + d_window,
            ),
            static_arg_indices=(3, 4, 5, 7),
        )
        return (grad_hidden, grad_boundary, *([None] * 5))


def build_attention_indices(
    cu_seqlens: torch.Tensor,
    global_start: int,
    l_local: int,
    d_window: int,
    window_size: int,
    ratio: int,
    compressed_width: int,
    compressed_topk: Optional[torch.Tensor] = None,
    cu_seqlens_compressed: Optional[torch.Tensor] = None,
    seq_to_rank_row: Optional[torch.Tensor] = None,
    for_indexer_loss: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Build final sparse-attention and optional indexer-loss indices.

    Inputs:
        cu_seqlens: int32 CUDA tensor, shape ``(n_seq + 1,)``.
        global_start/l_local: this rank's global query start and row count.
        d_window/window_size: physical left-window capacity and per-query window width.
        ratio/compressed_width: compression mode and number of compressed columns.
        compressed_topk: optional per-sequence compressed ids, shape
            ``(l_local, compressed_width)``.
        cu_seqlens_compressed/seq_to_rank_row: compressed row mapping.
        for_indexer_loss: put compressed ids first and return their rank-major rows.

    Outputs:
        ``topk_idxs`` int32, shape ``(l_local, window_size + compressed_width)``;
        optional ``topk_length`` int32, shape ``(l_local,)``; and optional
        ``indexer_rank_major`` int32, shape ``(l_local, compressed_width)``.
        Normal attention puts window ids first. Indexer-loss mode puts selected
        compressed ids first and returns their rank-major rows.
    """
    if for_indexer_loss and compressed_topk is None:
        raise RuntimeError("DSv4 CP indexer-loss indices require compressed_topk.")
    global_start, l_local = int(global_start), int(l_local)
    if cu_seqlens_compressed is None:
        cu_seqlens_compressed = cu_seqlens
    if seq_to_rank_row is None:
        seq_to_rank_row = torch.empty((1,), dtype=torch.int32, device=cu_seqlens.device)
    if not _cute_usable(cu_seqlens, compressed_topk, cu_seqlens_compressed, seq_to_rank_row):
        return _build_attention_indices_torch(
            cu_seqlens,
            global_start,
            l_local,
            d_window,
            window_size,
            ratio,
            compressed_width,
            compressed_topk,
            cu_seqlens_compressed,
            seq_to_rank_row,
            for_indexer_loss,
        )

    total_width = window_size + compressed_width
    topk_idxs = torch.empty((l_local, total_width), dtype=torch.int32, device=cu_seqlens.device)
    index_mode = 2 if for_indexer_loss else int(compressed_topk is None)
    if compressed_topk is None:
        compressed_topk = torch.empty((1, 1), dtype=torch.int32, device=cu_seqlens.device)
    if for_indexer_loss:
        topk_length_kernel = torch.empty((1,), dtype=torch.int32, device=cu_seqlens.device)
        indexer_rank_major = torch.empty_like(compressed_topk)
    else:
        topk_length_kernel = torch.empty((l_local,), dtype=torch.int32, device=cu_seqlens.device)
        indexer_rank_major = torch.empty((1, 1), dtype=torch.int32, device=cu_seqlens.device)
    launch_work = l_local * 128 if index_mode else l_local
    compressed_base = int(d_window) + l_local
    _run_compiled_launch(
        _build_attention_indices_launch,
        (
            cu_seqlens,
            cu_seqlens_compressed,
            compressed_topk,
            seq_to_rank_row,
            topk_idxs,
            topk_length_kernel,
            indexer_rank_major,
        ),
        (
            cu_seqlens.shape[0] - 1,
            global_start,
            l_local,
            d_window,
            window_size,
            ratio,
            compressed_width,
            seq_to_rank_row.shape[0],
            compressed_base,
            total_width,
            index_mode,
            launch_work,
        ),
        static_arg_indices=(10,),
    )
    if for_indexer_loss:
        return topk_idxs, None, indexer_rank_major
    return topk_idxs, topk_length_kernel, None
