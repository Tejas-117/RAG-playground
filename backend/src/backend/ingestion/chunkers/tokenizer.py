"""Offline adapter for the pinned multilingual chunking tokenizer."""

import hashlib
from functools import lru_cache
from pathlib import Path

from tokenizers import Tokenizer

from backend.ingestion.chunkers.models import TokenizedText

TOKENIZER_IDENTIFIER = "bert-base-multilingual-cased"
TOKENIZER_REVISION = "0fcb34d393e71211e8d72b52c31a46e7b7597068"
TOKENIZER_SHA256 = "f4a4d5bf7301717e261fafbe26e1eb967f6ba4cb3ae0ab7a29f4642ec229f386"
SPECIAL_TOKENS_POLICY = "add_special_tokens=false;truncation=false;padding=false"
TOKENIZER_PATH = (
    Path(__file__).with_name("assets") / TOKENIZER_IDENTIFIER / "tokenizer.json"
)


class TokenizerAssetError(RuntimeError):
    """Report a missing or modified pinned tokenizer asset."""


class MultilingualBertTokenizer:
    """Expose deterministic token offsets from a repository-local JSON asset."""

    identifier = TOKENIZER_IDENTIFIER
    revision = TOKENIZER_REVISION
    special_tokens_policy = SPECIAL_TOKENS_POLICY

    def __init__(
        self,
        asset_path: Path = TOKENIZER_PATH,
        expected_sha256: str = TOKENIZER_SHA256,
    ) -> None:
        """Validate and load one local tokenizer asset.

        Args:
            asset_path: Path to a Hugging Face tokenizer JSON file.
            expected_sha256: Required hexadecimal SHA-256 asset digest.

        Returns:
            None. The initialized adapter stores the loaded tokenizer.

        Raises:
            TokenizerAssetError: If the asset is absent or has the wrong digest.
        """
        self.asset_sha256 = _calculate_asset_digest(asset_path)

        # Reject unreviewed tokenizer changes before they alter chunk boundaries.
        if self.asset_sha256 != expected_sha256:
            raise TokenizerAssetError(
                "The chunking tokenizer asset does not match its pinned SHA-256."
            )

        # Load from the verified local file without contacting a model registry.
        self._tokenizer = Tokenizer.from_file(str(asset_path))
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()

    def encode(self, text: str) -> TokenizedText:
        """Tokenize text without model-only special tokens.

        Args:
            text: Unicode source text whose boundaries should be measured.

        Returns:
            Ordered zero-based, end-exclusive Unicode character offsets.
        """
        encoding = self._tokenizer.encode(text, add_special_tokens=False)

        token_starts: list[int] = []
        token_ends: list[int] = []

        # Copy valid boundaries into compact parallel arrays. This avoids allocating a
        # separate Python TokenOffset dataclass for every token in a large document.
        for start, end in encoding.offsets:
            # Ignore zero-width entries defensively if a future tokenizer emits them.
            if end > start:
                token_starts.append(start)
                token_ends.append(end)

        return TokenizedText(
            token_starts=tuple(token_starts),
            token_ends=tuple(token_ends),
        )


def _calculate_asset_digest(asset_path: Path) -> str:
    """Calculate the SHA-256 identity of a tokenizer asset.

    Args:
        asset_path: Local file whose immutable identity should be checked.

    Returns:
        Lowercase hexadecimal SHA-256 digest.

    Raises:
        TokenizerAssetError: If the configured asset does not exist.
    """
    # Convert filesystem errors into a stable chunking-domain failure.
    if not asset_path.is_file():
        raise TokenizerAssetError("The chunking tokenizer asset is missing.")

    digest = hashlib.sha256()

    # Stream the asset so validation does not create another full in-memory copy.
    with asset_path.open("rb") as asset_file:
        while block := asset_file.read(64 * 1024):
            digest.update(block)

    return digest.hexdigest()


@lru_cache(maxsize=1)
def get_chunking_tokenizer() -> MultilingualBertTokenizer:
    """Load and cache the process-wide immutable chunking tokenizer.

    Args:
        None.

    Returns:
        The verified offline multilingual BERT tokenizer adapter.
    """
    # Tokenizer instances are immutable during encoding and expensive to reload.
    return MultilingualBertTokenizer()
