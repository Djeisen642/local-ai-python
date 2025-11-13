"""Model selection and optimization logic for Whisper models."""


def apply_model_optimizations(
    model_size: str,
    use_english_only: bool = True,
    use_distilled: bool = True,
) -> str:
    """
    Apply model optimizations (distilled and/or English-only variants).

    Distilled models offer better speed/accuracy tradeoff and allow upgrading:
    - distil-medium.en: ~2x faster than medium.en, better than small.en
    - distil-large-v3: ~6x faster than large-v3, competitive with medium

    When distilled is enabled, we upgrade model sizes for optimal performance:
    - tiny/small/medium → distil-medium.en (faster than small.en, more accurate)
    - large → distil-large-v3 (much faster than large-v3)

    English-only models (.en suffix) are faster and more accurate for English speech.

    Args:
        model_size: Base model size (e.g., "tiny", "small", "medium", "large")
        use_english_only: Whether to use English-only variant (.en suffix)
        use_distilled: Whether to use distilled variant (distil- prefix)

    Returns:
        Optimized model name
    """
    # Distilled models available: medium, large-v3
    # English-only available for: tiny, base, small, medium (not large)
    # Strategy: Upgrade to distilled models when enabled for better performance

    if use_distilled:
        # Upgrade to distilled models when available for better performance
        if model_size in ["tiny", "small", "medium"]:
            # distil-medium.en is faster than small.en with better accuracy
            # Upgrade tiny/small to distil-medium for optimal speed/accuracy
            return "distil-medium.en" if use_english_only else "distil-medium"
        if model_size == "large":
            # distil-large-v3 is much faster than large-v3
            return "distil-large-v3"

    # Fall back to standard models with English-only suffix
    if use_english_only and model_size in ["tiny", "base", "small", "medium"]:
        return f"{model_size}.en"

    return model_size
