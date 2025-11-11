"""Comparison test for audio filtering effectiveness on scripted audio."""

# ruff: noqa: T201  # Print statements are intentional for test output

import time
from pathlib import Path

import pytest
from src.local_ai.speech_to_text.audio_filtering.audio_filter_pipeline import (
    AudioFilterPipeline,
)
from src.local_ai.speech_to_text.models import AudioChunk
from src.local_ai.speech_to_text.transcriber import WhisperTranscriber

# WER thresholds for interpretation
WER_SIGNIFICANT_IMPROVEMENT = 0.05  # 5% improvement
WER_MODEST_IMPROVEMENT = 0.01  # 1% improvement
WER_MINIMAL_CHANGE = -0.01  # Less than 1% degradation


@pytest.mark.integration
class TestScriptedAudioComparison:
    """Compare transcription quality with and without audio filtering."""

    @pytest.fixture
    def test_audio_dir(self) -> Path:
        """Get the test audio directory."""
        return Path(__file__).parent.parent / "test_data" / "audio" / "scripted"

    @pytest.fixture
    def transcriber(self) -> WhisperTranscriber:
        """Create a WhisperTranscriber instance."""
        return WhisperTranscriber(model_size="small", device="cpu", compute_type="int8")

    @pytest.fixture
    def audio_filter(self) -> AudioFilterPipeline:
        """Create an AudioFilterPipeline instance."""
        return AudioFilterPipeline(sample_rate=16000)

    def load_audio_file(self, file_path: Path) -> bytes:
        """Load audio file as bytes."""
        with open(file_path, "rb") as f:
            return f.read()

    def load_expected_text(self, file_path: Path) -> str:
        """Load expected transcription text."""
        with open(file_path, encoding="utf-8") as f:
            return f.read().strip()

    def calculate_word_error_rate(self, reference: str, hypothesis: str) -> float:
        """
        Calculate Word Error Rate (WER) between reference and hypothesis.

        WER = (S + D + I) / N
        where S = substitutions, D = deletions, I = insertions, N = words in reference
        """
        # Simple tokenization (split on whitespace and punctuation)
        import re

        def tokenize(text: str) -> list[str]:
            # Remove markdown formatting
            text = re.sub(r"\*\*", "", text)
            # Convert to lowercase and split on whitespace/punctuation
            return re.findall(r"\b\w+\b", text.lower())

        ref_words = tokenize(reference)
        hyp_words = tokenize(hypothesis)

        # Simple Levenshtein distance for word sequences
        n = len(ref_words)
        m = len(hyp_words)

        if n == 0:
            return 0.0 if m == 0 else 1.0

        # Dynamic programming matrix
        dp = [[0] * (m + 1) for _ in range(n + 1)]

        # Initialize
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j

        # Fill matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(
                        dp[i - 1][j] + 1,  # deletion
                        dp[i][j - 1] + 1,  # insertion
                        dp[i - 1][j - 1] + 1,  # substitution
                    )

        return dp[n][m] / n if n > 0 else 0.0

    async def transcribe_with_filtering(
        self,
        transcriber: WhisperTranscriber,
        audio_filter: AudioFilterPipeline,
        audio_data: bytes,
    ) -> tuple[str, float]:
        """Transcribe audio with filtering applied."""
        start_time = time.time()

        # Parse WAV file to get duration
        import io
        import wave

        duration = 0.0
        try:
            with wave.open(io.BytesIO(audio_data), "rb") as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
                duration = frames / float(rate)
        except Exception:
            # Fallback duration if parsing fails
            duration = 1.0

        # Create AudioChunk
        chunk = AudioChunk(
            data=audio_data,
            timestamp=time.time(),
            sample_rate=16000,
            duration=duration,
            is_filtered=False,
        )

        # Apply filtering
        filtered_chunk = await audio_filter.process_audio_chunk(chunk)

        # Transcribe
        result = await transcriber.transcribe_audio_with_result(filtered_chunk.data)

        processing_time = time.time() - start_time
        return result.text, processing_time

    async def transcribe_without_filtering(
        self,
        transcriber: WhisperTranscriber,
        audio_data: bytes,
    ) -> tuple[str, float]:
        """Transcribe audio without filtering."""
        start_time = time.time()

        # Transcribe directly
        result = await transcriber.transcribe_audio_with_result(audio_data)

        processing_time = time.time() - start_time
        return result.text, processing_time

    @pytest.mark.asyncio
    async def test_script1_comparison(
        self,
        test_audio_dir: Path,
        transcriber: WhisperTranscriber,
        audio_filter: AudioFilterPipeline,
    ) -> None:
        """Compare transcription quality for script1 (dates, numbers, times)."""
        audio_file = test_audio_dir / "script1.wav"
        text_file = test_audio_dir / "script1.txt"

        if not audio_file.exists():
            pytest.skip(f"Audio file not found: {audio_file}")

        audio_data = self.load_audio_file(audio_file)
        expected_text = self.load_expected_text(text_file)

        # Transcribe without filtering
        text_no_filter, time_no_filter = await self.transcribe_without_filtering(
            transcriber, audio_data
        )

        # Transcribe with filtering
        text_with_filter, time_with_filter = await self.transcribe_with_filtering(
            transcriber, audio_filter, audio_data
        )

        # Calculate WER for both
        wer_no_filter = self.calculate_word_error_rate(expected_text, text_no_filter)
        wer_with_filter = self.calculate_word_error_rate(expected_text, text_with_filter)

        # Print results
        print("\n" + "=" * 80)
        print("SCRIPT 1 COMPARISON (Dates, Numbers, Times)")
        print("=" * 80)
        print(f"\nExpected text:\n{expected_text}\n")
        print(f"Without filtering:\n{text_no_filter}")
        print(f"WER: {wer_no_filter:.2%} | Time: {time_no_filter:.2f}s\n")
        print(f"With filtering:\n{text_with_filter}")
        print(f"WER: {wer_with_filter:.2%} | Time: {time_with_filter:.2f}s\n")

        improvement = wer_no_filter - wer_with_filter
        print(f"WER Improvement: {improvement:+.2%}")
        print("=" * 80)

    @pytest.mark.asyncio
    async def test_script2_comparison(
        self,
        test_audio_dir: Path,
        transcriber: WhisperTranscriber,
        audio_filter: AudioFilterPipeline,
    ) -> None:
        """Compare transcription quality for script2 (punctuation, capitalization)."""
        audio_file = test_audio_dir / "script2.wav"
        text_file = test_audio_dir / "script2.txt"

        if not audio_file.exists():
            pytest.skip(f"Audio file not found: {audio_file}")

        audio_data = self.load_audio_file(audio_file)
        expected_text = self.load_expected_text(text_file)

        # Transcribe without filtering
        text_no_filter, time_no_filter = await self.transcribe_without_filtering(
            transcriber, audio_data
        )

        # Transcribe with filtering
        text_with_filter, time_with_filter = await self.transcribe_with_filtering(
            transcriber, audio_filter, audio_data
        )

        # Calculate WER for both
        wer_no_filter = self.calculate_word_error_rate(expected_text, text_no_filter)
        wer_with_filter = self.calculate_word_error_rate(expected_text, text_with_filter)

        # Print results
        print("\n" + "=" * 80)
        print("SCRIPT 2 COMPARISON (Punctuation, Capitalization)")
        print("=" * 80)
        print(f"\nExpected text:\n{expected_text}\n")
        print(f"Without filtering:\n{text_no_filter}")
        print(f"WER: {wer_no_filter:.2%} | Time: {time_no_filter:.2f}s\n")
        print(f"With filtering:\n{text_with_filter}")
        print(f"WER: {wer_with_filter:.2%} | Time: {time_with_filter:.2f}s\n")

        improvement = wer_no_filter - wer_with_filter
        print(f"WER Improvement: {improvement:+.2%}")
        print("=" * 80)

    @pytest.mark.asyncio
    async def test_script3_comparison(
        self,
        test_audio_dir: Path,
        transcriber: WhisperTranscriber,
        audio_filter: AudioFilterPipeline,
    ) -> None:
        """Compare transcription quality for script3 (homophones, technical vocab)."""
        audio_file = test_audio_dir / "script3.wav"
        text_file = test_audio_dir / "script3.txt"

        if not audio_file.exists():
            pytest.skip(f"Audio file not found: {audio_file}")

        audio_data = self.load_audio_file(audio_file)
        expected_text = self.load_expected_text(text_file)

        # Transcribe without filtering
        text_no_filter, time_no_filter = await self.transcribe_without_filtering(
            transcriber, audio_data
        )

        # Transcribe with filtering
        text_with_filter, time_with_filter = await self.transcribe_with_filtering(
            transcriber, audio_filter, audio_data
        )

        # Calculate WER for both
        wer_no_filter = self.calculate_word_error_rate(expected_text, text_no_filter)
        wer_with_filter = self.calculate_word_error_rate(expected_text, text_with_filter)

        # Print results
        print("\n" + "=" * 80)
        print("SCRIPT 3 COMPARISON (Homophones, Technical Vocabulary)")
        print("=" * 80)
        print(f"\nExpected text:\n{expected_text}\n")
        print(f"Without filtering:\n{text_no_filter}")
        print(f"WER: {wer_no_filter:.2%} | Time: {time_no_filter:.2f}s\n")
        print(f"With filtering:\n{text_with_filter}")
        print(f"WER: {wer_with_filter:.2%} | Time: {time_with_filter:.2f}s\n")

        improvement = wer_no_filter - wer_with_filter
        print(f"WER Improvement: {improvement:+.2%}")
        print("=" * 80)

    @pytest.mark.asyncio
    async def test_all_scripts_summary(
        self,
        test_audio_dir: Path,
        transcriber: WhisperTranscriber,
        audio_filter: AudioFilterPipeline,
    ) -> None:
        """Run all scripts and provide a summary comparison."""
        results = []

        for script_num in [1, 2, 3]:
            audio_file = test_audio_dir / f"script{script_num}.wav"
            text_file = test_audio_dir / f"script{script_num}.txt"

            if not audio_file.exists():
                continue

            audio_data = self.load_audio_file(audio_file)
            expected_text = self.load_expected_text(text_file)

            # Transcribe both ways
            text_no_filter, time_no_filter = await self.transcribe_without_filtering(
                transcriber, audio_data
            )
            text_with_filter, time_with_filter = await self.transcribe_with_filtering(
                transcriber, audio_filter, audio_data
            )

            # Calculate WER
            wer_no_filter = self.calculate_word_error_rate(expected_text, text_no_filter)
            wer_with_filter = self.calculate_word_error_rate(
                expected_text, text_with_filter
            )

            results.append(
                {
                    "script": f"script{script_num}",
                    "wer_no_filter": wer_no_filter,
                    "wer_with_filter": wer_with_filter,
                    "time_no_filter": time_no_filter,
                    "time_with_filter": time_with_filter,
                    "improvement": wer_no_filter - wer_with_filter,
                }
            )

        # Print summary
        print("\n" + "=" * 80)
        print("OVERALL SUMMARY - AUDIO FILTERING EFFECTIVENESS")
        print("=" * 80)
        print(
            f"\n{'Script':<12} {'WER (No Filter)':<18} "
            f"{'WER (Filtered)':<18} {'Improvement':<15}"
        )
        print("-" * 80)

        total_improvement = 0.0
        for result in results:
            print(
                f"{result['script']:<12} "
                f"{result['wer_no_filter']:>15.2%}  "
                f"{result['wer_with_filter']:>15.2%}  "
                f"{result['improvement']:>+13.2%}"
            )
            total_improvement += result["improvement"]

        avg_improvement = total_improvement / len(results) if results else 0.0

        print("-" * 80)
        print(f"{'Average':<12} {'':<18} {'':<18} {avg_improvement:>+13.2%}")
        print("=" * 80)

        # Interpretation
        print("\nINTERPRETATION:")
        if avg_improvement > WER_SIGNIFICANT_IMPROVEMENT:
            print("✅ Audio filtering shows SIGNIFICANT improvement (>5% WER reduction)")
        elif avg_improvement > WER_MODEST_IMPROVEMENT:
            print("⚠️  Audio filtering shows MODEST improvement (1-5% WER reduction)")
        elif avg_improvement > WER_MINIMAL_CHANGE:
            print("⚠️  Audio filtering shows MINIMAL impact (<1% change)")
        else:
            print("❌ Audio filtering shows DEGRADATION (WER increased)")
            print("   Consider disabling or adjusting filter parameters")

        print("\nNOTE: WER (Word Error Rate) measures transcription accuracy.")
        print("      Lower WER = better accuracy. Negative improvement = worse accuracy.")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    # Run the summary test
    pytest.main([__file__, "-v", "-s", "-k", "test_all_scripts_summary"])
