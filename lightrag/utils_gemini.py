import os
from typing import Any, List
from lightrag.utils import Tokenizer, logger
from lightrag.exceptions import ChunkTokenLimitExceededError
from lightrag.llm.gemini import _get_gemini_client

class GeminiTokenizer(Tokenizer):
  def __init__(self, model_name: str = "gemini-2.5-flash"):
    """
    Initialize the GeminiTokenizer with the specified model and Vertex AI client.
    """
    self.model_name = model_name
    self.token_map = {}
    try:
      self.client = _get_gemini_client(api_key=None, base_url=None)
    except Exception as e:
      # Fallback or error handling
      print(f"Error initializing Gemini tokenizer: {e}")
      raise

  def encode(self, content: str) -> List[int]:
    """
    Encode the given string into a list of Gemini token IDs and cache the token bytes.
    """
    # returns the list of token IDs
    response = self.client.models.compute_tokens(
      model=self.model_name,
      contents=content,
    )
    # Store token_ids -> tokens(bytes) mapping for decoding
    if response.tokens_info:
      info = response.tokens_info[0]
      self.token_map.update(zip(info.token_ids, info.tokens))
      return info.token_ids
    return []

  def decode(self, tokens: List[int]) -> str:
    """
    Decode the given list of Gemini token IDs back into a string using the cached tokens.
    """
    # returns the original string using stored token map
    decoded_bytes = b"".join(self.token_map.get(t_id, b"") for t_id in tokens)
    return decoded_bytes.decode("utf-8", errors="replace")

def _get_token_data(tokenizer: Tokenizer, content: str):
  """
  Helper to get both token IDs and token bytes from Gemini API.
  Assumes tokenizer has .client and .model_name attributes (e.g. GeminiTokenizer).
  """
  if not hasattr(tokenizer, "client") or not hasattr(tokenizer, "model_name"):
    # Fallback to standard encode if not GeminiTokenizer
    # But this function is specifically for Gemini usage where decode is needed
    raise ValueError("Tokenizer must be a GeminiTokenizer with .client and .model_name")

  response = tokenizer.client.models.compute_tokens(
    model=tokenizer.model_name,
    contents=content,
  )
  # response.tokens_info is a list, usually with 1 element for simple content
  if not response.tokens_info:
    return [], []

  info = response.tokens_info[0]
  # token_ids is list[int], tokens is list[bytes]
  return info.token_ids, info.tokens

def chunking_by_gemini_token_size(
  tokenizer: Tokenizer,
  content: str,
  split_by_character: str | None = None,
  split_by_character_only: bool = False,
  chunk_overlap_token_size: int = 100,
  chunk_token_size: int = 1200,
) -> List[dict[str, Any]]:
  """
  Custom chunking function for Gemini models using google-genai SDK.
  Replicates chunking_by_token_size logic but handles decoding via token bytes.
  """

  # helper for decoding bytes list to string
  def decode_tokens(token_bytes: List[bytes]) -> str:
    return b"".join(token_bytes).decode("utf-8", errors="replace")

  results: List[dict[str, Any]] = []

  # Initial tokenization
  # We need both IDs (for length check) and Bytes (for reconstruction)
  token_ids, token_bytes = _get_token_data(tokenizer, content)

  if split_by_character:
    raw_chunks = content.split(split_by_character)
    new_chunks = []

    if split_by_character_only:
      for chunk in raw_chunks:
        # Get token count for sub-chunk
        # Note: tokenizing parts separately might differ slightly from whole, 
        # but split_by_character strategies accept this trade-off.
        sub_ids, sub_bytes = _get_token_data(tokenizer, chunk)

        if len(sub_ids) > chunk_token_size:
          logger.warning(
            "Chunk split_by_character exceeds token limit: len=%d limit=%d",
            len(sub_ids),
            chunk_token_size,
          )
          raise ChunkTokenLimitExceededError(
            chunk_tokens=len(sub_ids),
            chunk_token_limit=chunk_token_size,
            chunk_preview=chunk[:120],
          )
        new_chunks.append((len(sub_ids), chunk))
    else:
      for chunk in raw_chunks:
        sub_ids, sub_bytes = _get_token_data(tokenizer, chunk)

        if len(sub_ids) > chunk_token_size:
          # Need to split further using sliding window on TOKENS
          for start in range(
            0, len(sub_ids), chunk_token_size - chunk_overlap_token_size
          ):
            end = start + chunk_token_size
            chunk_bytes_slice = sub_bytes[start:end]
            chunk_content = decode_tokens(chunk_bytes_slice)

            tokens_len = min(chunk_token_size, len(sub_ids) - start)
            new_chunks.append((tokens_len, chunk_content))
        else:
          new_chunks.append((len(sub_ids), chunk))

    for index, (_len, chunk) in enumerate(new_chunks):
      results.append(
        {
          "tokens": _len,
          "content": chunk.strip(),
          "chunk_order_index": index,
        }
      )
  else:
    # No split_by_character, just sliding window on the whole content tokens
    for index, start in enumerate(
      range(0, len(token_ids), chunk_token_size - chunk_overlap_token_size)
    ):
      end = start + chunk_token_size
      chunk_bytes_slice = token_bytes[start:end]
      chunk_content = decode_tokens(chunk_bytes_slice)

      tokens_len = min(chunk_token_size, len(token_ids) - start)
      results.append(
        {
          "tokens": tokens_len,
          "content": chunk_content.strip(),
          "chunk_order_index": index,
        }
      )

  return results
