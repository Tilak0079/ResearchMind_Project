"""
Embedding generation using BAAI/bge-m3 via FlagEmbedding (the model's
official library). Produces both dense and sparse vectors from ONE model
call — this is what lets us drop a separate keyword/BM25 index (Section 2.3).

Note: we originally tried FastEmbed for this, but FastEmbed does not
support bge-m3 in any version checked (confirmed by inspecting its
supported-model list directly). FlagEmbedding is BGE-m3's own official
library and gives full native dense+sparse support with no workarounds.
"""


import logging
import os

# Force offline mode: we already have BGE-m3 cached locally from the first
# download. Without this, FlagEmbedding does an unnecessary online metadata
# check every time it loads, which fails if there's no internet connection.
os.environ["HF_HUB_OFFLINE"] = "1"

from FlagEmbedding import BGEM3FlagModel
from FlagEmbedding import BGEM3FlagModel

logger = logging.getLogger(__name__)

# Loaded once at module level — loading the model per-call would be very slow.
# use_fp16=True speeds up inference with negligible accuracy loss (safe on most machines).
_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)


def embed_text(text: str) -> dict:
    """
    Returns both dense and sparse embeddings for a piece of text in one call.

    Returns:
        {
            "dense": list[float]          # 1024-dim vector
            "sparse": {"indices": [...], "values": [...]}   # Qdrant sparse format
        }
    """
    output = _model.encode(
        [text],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,  # we don't need this third vector type
    )

    dense_vector = output["dense_vecs"][0].tolist()

    # FlagEmbedding returns sparse weights as {token_id: weight} dict —
    # convert to Qdrant's {indices, values} list format.
    sparse_weights: dict = output["lexical_weights"][0]
    indices = [int(k) for k in sparse_weights.keys()]
    values = [float(v) for v in sparse_weights.values()]

    return {
        "dense": dense_vector,
        "sparse": {"indices": indices, "values": values},
    }