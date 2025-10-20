"""Tests for audio filtering data models."""

from typing import List

import pytest

from local_ai.speech_to_text.audio_filtering.models import (
    AudioProfile,
    FilterStats,
    NoiseType,
)
from local_ai.speech_to_text.models import AudioChunk


@pytest.mark.unit
class TestEnhancedAudioChunk:
    """Test cases for enhanced AudioChunk with filtering metadata."""

    def test_enhanced_audio_chunk_creation_with_original_fields(self) -> None:
        """Test enhanced AudioChunk can be created with original fields."""
        chunk = AudioChunk(
            data=b"test_audio_data",
            timestamp=1234567890.0,
            sample_rate=16000,
            duration=1.0,
        )

        assert chunk.data == b"test_audio_data"
        assert chunk.timestamp == 1234567890.0
        assert chunk.sample_rate == 16000
        assert chunk.duration == 1.0
        # Check default values for new fields
        assert chunk.noise_level == 0.0
        assert chunk.signal_level == 0.0
        assert chunk.snr_db == 0.0
        assert chunk.is_filtered is False

    def test_enhanced_audio_chunk_creation_with_filtering_metadata(self) -> None:
        """Test enhanced AudioChunk can be created with filtering metadata."""
        chunk = AudioChunk(
            data=b"filtered_audio_data",
            timestamp=1234567890.0,
            sample_rate=16000,
            duration=1.0,
            noise_level=0.3,
            signal_level=0.8,
            snr_db=15.5,
            is_filtered=True,
        )

        assert chunk.data == b"filtered_audio_data"
        assert chunk.timestamp == 1234567890.0
        assert chunk.sample_rate == 16000
        assert chunk.duration == 1.0
        assert chunk.noise_level == 0.3
        assert chunk.signal_level == 0.8
        assert chunk.snr_db == 15.5
        assert chunk.is_filtered is True

    def test_enhanced_audio_chunk_negative_snr(self) -> None:
        """Test enhanced AudioChunk handles negative SNR values."""
        chunk = AudioChunk(
            data=b"noisy_audio_data",
            timestamp=1234567890.0,
            sample_rate=16000,
            duration=1.0,
            noise_level=0.9,
            signal_level=0.2,
            snr_db=-10.5,
            is_filtered=False,
        )

        assert chunk.snr_db == -10.5
        assert chunk.noise_level == 0.9
        assert chunk.signal_level == 0.2
        assert chunk.is_filtered is False

    def test_enhanced_audio_chunk_zero_duration_edge_case(self) -> None:
        """Test enhanced AudioChunk handles zero duration edge case."""
        chunk = AudioChunk(
            data=b"",
            timestamp=1234567890.0,
            sample_rate=16000,
            duration=0.0,
            noise_level=0.0,
            signal_level=0.0,
            snr_db=0.0,
            is_filtered=False,
        )

        assert chunk.duration == 0.0
        assert chunk.data == b""
        assert chunk.noise_level == 0.0


@pytest.mark.unit
class TestFilterStats:
    """Test cases for FilterStats data model."""

    def test_filter_stats_creation(self) -> None:
        """Test FilterStats can be created with all required fields."""
        stats = FilterStats(
            noise_reduction_db=12.5,
            signal_enhancement_db=3.2,
            processing_latency_ms=25.0,
            filters_applied=["noise_reduction", "normalization"],
            audio_quality_score=0.85,
        )

        assert stats.noise_reduction_db == 12.5
        assert stats.signal_enhancement_db == 3.2
        assert stats.processing_latency_ms == 25.0
        assert stats.filters_applied == ["noise_reduction", "normalization"]
        assert stats.audio_quality_score == 0.85

    def test_filter_stats_empty_filters_list(self) -> None:
        """Test FilterStats with empty filters list."""
        stats = FilterStats(
            noise_reduction_db=0.0,
            signal_enhancement_db=0.0,
            processing_latency_ms=5.0,
            filters_applied=[],
            audio_quality_score=1.0,
        )

        assert stats.filters_applied == []
        assert stats.noise_reduction_db == 0.0
        assert stats.audio_quality_score == 1.0

    def test_filter_stats_multiple_filters(self) -> None:
        """Test FilterStats with multiple filters applied."""
        filters = [
            "noise_reduction",
            "spectral_enhancement",
            "normalization",
            "echo_cancellation",
        ]
        stats = FilterStats(
            noise_reduction_db=15.8,
            signal_enhancement_db=5.1,
            processing_latency_ms=45.0,
            filters_applied=filters,
            audio_quality_score=0.92,
        )

        assert len(stats.filters_applied) == 4
        assert "noise_reduction" in stats.filters_applied
        assert "echo_cancellation" in stats.filters_applied
        assert stats.processing_latency_ms == 45.0

    def test_filter_stats_negative_values_edge_case(self) -> None:
        """Test FilterStats handles negative enhancement values."""
        stats = FilterStats(
            noise_reduction_db=-2.0,  # Could happen if filter makes things worse
            signal_enhancement_db=-1.5,
            processing_latency_ms=10.0,
            filters_applied=["failed_filter"],
            audio_quality_score=0.3,
        )

        assert stats.noise_reduction_db == -2.0
        assert stats.signal_enhancement_db == -1.5
        assert stats.audio_quality_score == 0.3


@pytest.mark.unit
class TestNoiseType:
    """Test cases for NoiseType enum."""

    def test_noise_type_values(self) -> None:
        """Test NoiseType enum has correct values."""
        assert NoiseType.STATIONARY.value == "stationary"
        assert NoiseType.TRANSIENT.value == "transient"
        assert NoiseType.MECHANICAL.value == "mechanical"
        assert NoiseType.SPEECH.value == "speech"
        assert NoiseType.MIXED.value == "mixed"

    def test_noise_type_enum_members(self) -> None:
        """Test NoiseType enum has all expected members."""
        expected_members = {"STATIONARY", "TRANSIENT", "MECHANICAL", "SPEECH", "MIXED"}
        actual_members = {member.name for member in NoiseType}
        assert actual_members == expected_members

    def test_noise_type_from_string(self) -> None:
        """Test NoiseType can be created from string values."""
        assert NoiseType("stationary") == NoiseType.STATIONARY
        assert NoiseType("transient") == NoiseType.TRANSIENT
        assert NoiseType("mechanical") == NoiseType.MECHANICAL
        assert NoiseType("speech") == NoiseType.SPEECH
        assert NoiseType("mixed") == NoiseType.MIXED

    def test_noise_type_invalid_value(self) -> None:
        """Test NoiseType raises ValueError for invalid values."""
        with pytest.raises(ValueError):
            NoiseType("invalid_noise_type")


@pytest.mark.unit
class TestAudioProfile:
    """Test cases for AudioProfile data model."""

    def test_audio_profile_creation(self) -> None:
        """Test AudioProfile can be created with all required fields."""
        profile = AudioProfile(
            snr_db=20.5,
            dominant_frequencies=[440.0, 880.0, 1320.0],
            noise_type=NoiseType.STATIONARY,
            speech_presence=0.75,
            recommended_filters=["noise_reduction", "spectral_enhancement"],
        )

        assert profile.snr_db == 20.5
        assert profile.dominant_frequencies == [440.0, 880.0, 1320.0]
        assert profile.noise_type == NoiseType.STATIONARY
        assert profile.speech_presence == 0.75
        assert profile.recommended_filters == ["noise_reduction", "spectral_enhancement"]

    def test_audio_profile_low_snr(self) -> None:
        """Test AudioProfile with low SNR and mixed noise."""
        profile = AudioProfile(
            snr_db=-5.2,
            dominant_frequencies=[60.0, 120.0],  # Low frequency noise
            noise_type=NoiseType.MIXED,
            speech_presence=0.1,
            recommended_filters=["aggressive_noise_reduction", "high_pass_filter"],
        )

        assert profile.snr_db == -5.2
        assert profile.noise_type == NoiseType.MIXED
        assert profile.speech_presence == 0.1
        assert "aggressive_noise_reduction" in profile.recommended_filters

    def test_audio_profile_empty_frequencies(self) -> None:
        """Test AudioProfile with empty dominant frequencies list."""
        profile = AudioProfile(
            snr_db=10.0,
            dominant_frequencies=[],
            noise_type=NoiseType.TRANSIENT,
            speech_presence=0.9,
            recommended_filters=[],
        )

        assert profile.dominant_frequencies == []
        assert profile.recommended_filters == []
        assert profile.noise_type == NoiseType.TRANSIENT

    def test_audio_profile_high_speech_presence(self) -> None:
        """Test AudioProfile with high speech presence."""
        profile = AudioProfile(
            snr_db=25.0,
            dominant_frequencies=[300.0, 1000.0, 3000.0],  # Speech frequencies
            noise_type=NoiseType.SPEECH,
            speech_presence=0.95,
            recommended_filters=["speech_enhancement"],
        )

        assert profile.speech_presence == 0.95
        assert profile.noise_type == NoiseType.SPEECH
        assert 300.0 in profile.dominant_frequencies
        assert 3000.0 in profile.dominant_frequencies

    def test_audio_profile_mechanical_noise(self) -> None:
        """Test AudioProfile for mechanical noise scenario."""
        profile = AudioProfile(
            snr_db=8.5,
            dominant_frequencies=[50.0, 100.0, 200.0],  # Mechanical frequencies
            noise_type=NoiseType.MECHANICAL,
            speech_presence=0.4,
            recommended_filters=["notch_filter", "spectral_subtraction"],
        )

        assert profile.noise_type == NoiseType.MECHANICAL
        assert profile.speech_presence == 0.4
        assert "notch_filter" in profile.recommended_filters
        assert "spectral_subtraction" in profile.recommended_filters

    def test_audio_profile_data_validation_edge_cases(self) -> None:
        """Test AudioProfile with edge case values."""
        # Test with extreme values
        profile = AudioProfile(
            snr_db=0.0,  # Exactly equal signal and noise
            dominant_frequencies=[20000.0],  # High frequency
            noise_type=NoiseType.TRANSIENT,
            speech_presence=1.0,  # Maximum speech presence
            recommended_filters=["single_filter"],
        )

        assert profile.snr_db == 0.0
        assert profile.speech_presence == 1.0
        assert profile.dominant_frequencies == [20000.0]
