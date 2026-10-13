"""Collect chunk-aligned causal self-attention maps during video inference."""

from contextlib import contextmanager
import math

import torch


class AttentionMapCollector:
    """Average post-softmax attention over disjoint token groups.

    Each saved map has 26 query rows for the current chunk and 26 key columns
    per visible chunk. Every output cell averages a Cartesian product of
    non-overlapping query and key token groups. Softmax still sees all real KV
    cache keys before any pooling.
    """

    def __init__(self, query_size=26, per_head=False):
        if query_size <= 0:
            raise ValueError("query_size must be positive")
        self.query_size = query_size
        self.per_head = per_head
        self.num_heads = None
        self.maps = []
        self.query_tokens = []
        self.key_tokens = []
        self.frame_tokens = []
        self.sink_key_tokens = []
        self.key_chunk_indices = []
        self.key_chunk_token_counts = []
        self.timesteps = []
        self.chunk_frames = []
        self.map_chunk_indices = []
        self._active = False

    @contextmanager
    def attach(self, model):
        layers = [block.self_attn for block in model.blocks]
        previous = [getattr(layer, "att_map_collector", None) for layer in layers]
        self._num_layers = len(layers)
        try:
            for layer in layers:
                layer.att_map_collector = self
            yield self
        finally:
            self._active = False
            for layer, old in zip(layers, previous):
                layer.att_map_collector = old

    def add_context_chunk(self, num_frames):
        """Register an initial-latent chunk whose keys may appear in later maps."""
        if self._active:
            raise RuntimeError("Cannot add a context chunk during an attention step")
        self.chunk_frames.append(num_frames)

    def start_chunk(self, num_frames):
        if self._active:
            raise RuntimeError("Cannot start a chunk during an attention step")
        self.map_chunk_indices.append(len(self.chunk_frames))
        self.chunk_frames.append(num_frames)
        self.maps.append([])
        self.query_tokens.append([])
        self.key_tokens.append([])
        self.frame_tokens.append([])
        self.sink_key_tokens.append([])
        self.key_chunk_indices.append([])
        self.key_chunk_token_counts.append([])
        self.timesteps.append([])

    def start_step(self, timestep):
        if self._active or not self.maps:
            raise RuntimeError("Start a chunk before each attention step")
        self._active = True
        self._sum = None
        self._count = 0
        self._shape = None
        self._timestep = float(timestep.item()) if isinstance(timestep, torch.Tensor) else float(timestep)

    def record(self, query, key, frame_tokens, sink_key_tokens, key_start, current_start):
        if not self._active:
            return
        key_chunks = self._key_chunks(
            key.shape[1], query.shape[1], frame_tokens, sink_key_tokens,
            key_start, current_start + query.shape[1],
        )
        layout = tuple((index, tuple(spans)) for index, spans in key_chunks)
        if self.num_heads is None:
            self.num_heads = query.shape[2]
        elif query.shape[2] != self.num_heads:
            raise ValueError("Attention head count changed during collection")
        shape = (query.shape[1], key.shape[1], frame_tokens, sink_key_tokens, layout)
        if self._shape is None:
            self._shape = shape
        elif shape != self._shape:
            raise ValueError("Attention dimensions or key chunks differ across transformer layers")
        current = self._pool_attention(query, key, [spans for _, spans in key_chunks])
        self._sum = current if self._sum is None else self._sum + current
        self._count += 1

    def finish_step(self):
        if not self._active or self._count != self._num_layers:
            raise RuntimeError("Expected one attention map from every transformer layer")
        query_length, key_length, frame_tokens, sink_tokens, layout = self._shape
        self.maps[-1].append(self._sum / self._count)
        self.query_tokens[-1].append(query_length)
        self.key_tokens[-1].append(key_length)
        self.frame_tokens[-1].append(frame_tokens)
        self.sink_key_tokens[-1].append(sink_tokens)
        self.key_chunk_indices[-1].append([index for index, _ in layout])
        self.key_chunk_token_counts[-1].append([
            sum(end - start for start, end in spans) for _, spans in layout
        ])
        self.timesteps[-1].append(self._timestep)
        self._sum = None
        self._active = False

    def _key_chunks(self, key_length, query_length, frame_tokens, sink_tokens, key_start, current_end):
        """Map visible KV-cache positions back to generation chunk boundaries."""
        total_tokens = sum(self.chunk_frames) * frame_tokens
        if self.chunk_frames[-1] * frame_tokens != query_length:
            raise ValueError("Query length does not match the current chunk")
        origin = current_end - total_tokens
        # The cache has an optional fixed sink prefix followed by a contiguous
        # rolling suffix. Both ranges below use absolute token coordinates.
        sink_global = (key_start, key_start + sink_tokens)
        rolling_length = key_length - sink_tokens
        rolling_global = (current_end - rolling_length, current_end)
        chunks = []
        cursor = origin
        for index, frames in enumerate(self.chunk_frames):
            chunk_end = cursor + frames * frame_tokens
            spans = []
            for global_range, local_offset in (
                (sink_global, 0),
                (rolling_global, sink_tokens),
            ):
                left = max(cursor, global_range[0])
                right = min(chunk_end, global_range[1])
                if left < right:
                    spans.append((local_offset + left - global_range[0],
                                  local_offset + right - global_range[0]))
            if spans:
                chunks.append((index, spans))
            cursor = chunk_end
        if sum(end - start for _, spans in chunks for start, end in spans) != key_length:
            raise ValueError("Could not assign every visible key to a chunk")
        return chunks

    def _pool_attention(self, query, key, key_chunks):
        batch, query_length, heads, head_dim = query.shape
        key_length = key.shape[1]
        size = self.query_size
        if query_length < size:
            raise ValueError(f"A query chunk needs at least {size} tokens for disjoint pooling")

        # Integer edges partition Q exactly; no query token belongs to two rows.
        query_edges = torch.arange(size + 1, device=query.device) * query_length // size
        query_bins = torch.searchsorted(
            query_edges[1:], torch.arange(query_length, device=query.device), right=True
        )
        query_counts = (query_edges[1:] - query_edges[:-1]).float()
        if self.per_head:
            sums = torch.zeros((batch, heads, size, key_length), dtype=torch.float32, device=query.device)
        else:
            sums = torch.zeros((batch, size, key_length), dtype=torch.float32, device=query.device)

        # Bound peak memory by computing all-key softmax for small query/head
        # batches. Sum probabilities into query rows after softmax, not logits.
        for head_start in range(0, heads, 2):
            head_end = min(head_start + 2, heads)
            keys = key[:, :, head_start:head_end].permute(0, 2, 3, 1).float()
            for row_start in range(0, query_length, 64):
                row_end = min(row_start + 64, query_length)
                queries = query[:, row_start:row_end, head_start:head_end].permute(0, 2, 1, 3).float()
                logits = queries @ keys * (1.0 / math.sqrt(head_dim))
                probabilities = torch.softmax(logits, dim=-1)
                if self.per_head:
                    sums[:, head_start:head_end].index_add_(
                        2, query_bins[row_start:row_end], probabilities
                    )
                else:
                    sums.index_add_(
                        1, query_bins[row_start:row_end], probabilities.sum(dim=1) / heads
                    )
        row_means = (sums / query_counts[None, None, :, None] if self.per_head
                     else sums / query_counts[None, :, None])

        blocks = []
        for spans in key_chunks:
            values = torch.cat([row_means[..., start:end] for start, end in spans], dim=-1)
            token_count = values.shape[-1]
            if token_count < size:
                raise ValueError(f"A visible key chunk needs at least {size} tokens for disjoint pooling")
            # Partition this chunk independently into 26 disjoint key groups.
            # The next chunk starts a new 26-column block, even when the cache
            # has evicted part of this chunk or preserved a separate sink prefix.
            edges = torch.arange(size + 1, device=query.device) * token_count // size
            bins = torch.searchsorted(
                edges[1:], torch.arange(token_count, device=query.device), right=True
            )
            block = torch.zeros((*values.shape[:-1], size), dtype=torch.float32, device=query.device)
            bin_shape = (1,) * (values.ndim - 1) + (token_count,)
            block.scatter_add_(-1, bins.reshape(bin_shape).expand_as(values), values)
            count_shape = (1,) * (values.ndim - 1) + (size,)
            block /= (edges[1:] - edges[:-1]).float().reshape(count_shape)
            blocks.append(block)
        return torch.cat(blocks, dim=-1).cpu()

    def sample(self, index):
        """Return the original maps averaged over heads and layers."""
        return self._sample_data(
            [[step[index].mean(dim=0) if self.per_head else step[index]
              for step in chunk] for chunk in self.maps],
            "conditional causal self-attention; mean over heads and layers",
        )

    def sample_head(self, index, head_index):
        """Return one head's maps, averaged over transformer layers."""
        if not self.per_head:
            raise ValueError("Per-head maps were not collected")
        if not 0 <= head_index < self.num_heads:
            raise IndexError(f"Attention head {head_index} is out of range")
        data = self._sample_data(
            [[step[index, head_index] for step in chunk] for chunk in self.maps],
            f"conditional causal self-attention; head {head_index}; mean over layers",
        )
        data["head_index"] = head_index
        return data

    def _sample_data(self, maps, source):
        # maps[generated_chunk][step] is [26, 26 * visible_key_chunks].
        # Consecutive 26-column blocks follow key_chunk_indices[chunk][step].
        # Those indices refer to chunk_frames (which includes initial context
        # chunks); map_chunk_indices maps each saved row to its query chunk.
        # Within a block, columns average disjoint, chronological raw K token
        # ranges from only that chunk. Sink keys may be a separate cache prefix
        # but are grouped with their original chunk. key_chunk_token_counts
        # records how many retained keys contributed to each block.
        # frame_tokens spatial patches per frame are in row-major order.
        return {
            "maps": maps,
            "query_tokens": self.query_tokens,
            "key_tokens": self.key_tokens,
            "frame_tokens": self.frame_tokens,
            "sink_key_tokens": self.sink_key_tokens,
            "key_chunk_indices": self.key_chunk_indices,
            "key_chunk_token_counts": self.key_chunk_token_counts,
            "chunk_frames": self.chunk_frames,
            "map_chunk_indices": self.map_chunk_indices,
            "timesteps": self.timesteps,
            "query_size": self.query_size,
            "num_heads": self.num_heads,
            "format_version": 2,
            "pooling": "disjoint mean of post-softmax probabilities within each chunk",
            "source": source,
        }
