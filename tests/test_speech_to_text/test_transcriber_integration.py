"""Integration tests for WhisperTranscriber using real audio files."""

import pytest
from pathlib import Path
from local_ai.speech_to_text.transcriber import WhisperTranscriber

# Test data directory
TEST_AUDIO_DIR = Path(__file__).parent.parent / "test_data" / "audio"


class TestWhisperTranscriberIntegration:
    """Integration tests using real audio files."""

    @pytest.fixture
    def transcriber(self) -> WhisperTranscriber:
        """Create a WhisperTranscriber instance for testing."""
        return WhisperTranscriber()

    async def _transcribe_file(self, transcriber: WhisperTranscriber, file_path: Path) -> str:
        """Helper method to transcribe an audio file."""
        if not file_path.exists():
            pytest.skip(f"Audio file not found: {file_path}")
        
        with open(file_path, "rb") as f:
            audio_data = f.read()
        
        return await transcriber.transcribe_audio(audio_data)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_transcribe_hello_world(self, transcriber: WhisperTranscriber) -> None:
        """Test transcription of 'hello world' audio."""
        file_path = TEST_AUDIO_DIR / "hello_world.wav"
        result = await self._transcribe_file(transcriber, file_path)
        
        # Check that the transcription contains expected words
        result_lower = result.lower()
        assert "hello" in result_lower, f"Expected 'hello' in transcription: '{result}'"
        assert "world" in result_lower, f"Expected 'world' in transcription: '{result}'"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_transcribe_numbers(self, transcriber: WhisperTranscriber) -> None:
        """Test transcription of numbers audio."""
        file_path = TEST_AUDIO_DIR / "numbers.wav"
        result = await self._transcribe_file(transcriber, file_path)
        
        # Check for number words (Whisper might output digits or words)
        result_lower = result.lower()
        
        # Accept either digit form or word form
        expected_numbers = [
            ("1", "one"),
            ("2", "two"), 
            ("3", "three"),
            ("4", "four"),
            ("5", "five")
        ]
        
        for digit, word in expected_numbers:
            assert digit in result or word in result_lower, \
                f"Expected '{digit}' or '{word}' in transcription: '{result}'"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_transcribe_alphabet(self, transcriber: WhisperTranscriber) -> None:
        """Test transcription of alphabet audio."""
        file_path = TEST_AUDIO_DIR / "alphabet.wav"
        result = await self._transcribe_file(transcriber, file_path)
        
        # Check for some letters (Whisper might not get all perfectly)
        result_upper = result.upper()
        expected_letters = ["A", "B", "C", "D", "E"]
        
        found_letters = sum(1 for letter in expected_letters if letter in result_upper)
        assert found_letters >= 3, \
            f"Expected at least 3 letters from {expected_letters} in transcription: '{result}'"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_transcribe_short_sentence(self, transcriber: WhisperTranscriber) -> None:
        """Test transcription of a short sentence."""
        file_path = TEST_AUDIO_DIR / "short_sentence.wav"
        result = await self._transcribe_file(transcriber, file_path)
        
        # Should contain some key words from "This is a test sentence"
        result_lower = result.lower()
        key_words = ["this", "test", "sentence"]
        
        found_words = sum(1 for word in key_words if word in result_lower)
        assert found_words >= 2, \
            f"Expected at least 2 words from {key_words} in transcription: '{result}'"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_transcribe_very_short_audio(self, transcriber: WhisperTranscriber) -> None:
        """Test transcription of very short audio."""
        file_path = TEST_AUDIO_DIR / "edge_cases" / "very_short.wav"
        result = await self._transcribe_file(transcriber, file_path)
        
        # Very short audio should still return a string (might be empty or contain the word)
        assert isinstance(result, str), f"Expected string result, got: {type(result)}"
        # Don't assert specific content as very short audio can be tricky

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_transcribe_silence(self, transcriber: WhisperTranscriber) -> None:
        """Test transcription of silence."""
        file_path = TEST_AUDIO_DIR / "edge_cases" / "silence.wav"
        result = await self._transcribe_file(transcriber, file_path)
        
        # Silence should return empty string or minimal content
        assert isinstance(result, str), f"Expected string result, got: {type(result)}"
        assert len(result.strip()) <= 10, \
            f"Expected minimal content for silence, got: '{result}'"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_transcribe_high_quality_audio(self, transcriber: WhisperTranscriber) -> None:
        """Test transcription of high quality audio."""
        file_path = TEST_AUDIO_DIR / "quality" / "high_quality_16khz.wav"
        result = await self._transcribe_file(transcriber, file_path)
        
        # High quality audio should produce good results
        assert isinstance(result, str), f"Expected string result, got: {type(result)}"
        assert len(result.strip()) > 0, "Expected non-empty transcription for high quality audio"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_transcribe_low_quality_audio(self, transcriber: WhisperTranscriber) -> None:
        """Test transcription of low quality audio."""
        file_path = TEST_AUDIO_DIR / "quality" / "low_quality_8khz.wav"
        result = await self._transcribe_file(transcriber, file_path)
        
        # Low quality audio should still work, but might be less accurate
        assert isinstance(result, str), f"Expected string result, got: {type(result)}"
        # Don't assert accuracy for low quality

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_transcribe_mp3_file(self, transcriber: WhisperTranscriber) -> None:
        """Test transcription of MP3 file."""
        file_path = TEST_AUDIO_DIR / "quality" / "compressed_mp3.mp3"
        result = await self._transcribe_file(transcriber, file_path)
        
        # MP3 should work (Whisper handles various formats)
        assert isinstance(result, str), f"Expected string result, got: {type(result)}"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_transcribe_command_audio(self, transcriber: WhisperTranscriber) -> None:
        """Test transcription of command audio."""
        file_path = TEST_AUDIO_DIR / "scenarios" / "command.wav"
        result = await self._transcribe_file(transcriber, file_path)
        
        # Should contain command-related words
        result_lower = result.lower()
        command_words = ["start", "stop", "record", "now"]
        
        found_words = sum(1 for word in command_words if word in result_lower)
        assert found_words >= 1, \
            f"Expected at least 1 command word from {command_words} in transcription: '{result}'"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_transcribe_question_audio(self, transcriber: WhisperTranscriber) -> None:
        """Test transcription of question audio."""
        file_path = TEST_AUDIO_DIR / "scenarios" / "question.wav"
        result = await self._transcribe_file(transcriber, file_path)
        
        # Should contain question-related words
        result_lower = result.lower()
        question_words = ["what", "time", "when", "how", "?"]
        
        found_words = sum(1 for word in question_words if word in result_lower)
        assert found_words >= 1, \
            f"Expected at least 1 question word from {question_words} in transcription: '{result}'"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_transcribe_dictation_audio(self, transcriber: WhisperTranscriber) -> None:
        """Test transcription of dictation audio."""
        file_path = TEST_AUDIO_DIR / "scenarios" / "dictation.wav"
        result = await self._transcribe_file(transcriber, file_path)
        
        # Should contain dictation-related words
        result_lower = result.lower()
        dictation_words = ["please", "transcribe", "message", "dictation"]
        
        found_words = sum(1 for word in dictation_words if word in result_lower)
        assert found_words >= 1, \
            f"Expected at least 1 dictation word from {dictation_words} in transcription: '{result}'"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_model_availability_with_real_model(self, transcriber: WhisperTranscriber) -> None:
        """Test that model is actually available for real transcription."""
        # This tests the real model loading, not mocked
        available = transcriber.is_model_available()
        
        if not available:
            pytest.skip("Whisper model not available - this is expected in CI environments")
        
        assert available is True

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_model_info_with_real_model(self, transcriber: WhisperTranscriber) -> None:
        """Test getting model info from real model."""
        if not transcriber.is_model_available():
            pytest.skip("Whisper model not available")
        
        info = transcriber.get_model_info()
        
        # Should have basic model information
        assert isinstance(info, dict)
        assert "model_size" in info
        assert info["model_size"] == "small"  # Default model size

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_transcription_performance(self, transcriber: WhisperTranscriber) -> None:
        """Test that transcription completes in reasonable time."""
        import time
        
        file_path = TEST_AUDIO_DIR / "hello_world.wav"
        if not file_path.exists():
            pytest.skip(f"Audio file not found: {file_path}")
        
        start_time = time.time()
        result = await self._transcribe_file(transcriber, file_path)
        end_time = time.time()
        
        # Should complete within reasonable time (adjust as needed)
        duration = end_time - start_time
        assert duration < 30.0, f"Transcription took too long: {duration:.2f} seconds"
        assert isinstance(result, str)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_multiple_transcriptions(self, transcriber: WhisperTranscriber) -> None:
        """Test multiple transcriptions with same transcriber instance."""
        files = [
            TEST_AUDIO_DIR / "hello_world.wav",
            TEST_AUDIO_DIR / "numbers.wav"
        ]
        
        results = []
        for file_path in files:
            if file_path.exists():
                result = await self._transcribe_file(transcriber, file_path)
                results.append(result)
        
        # Should handle multiple transcriptions
        assert len(results) > 0, "No audio files were found for testing"
        for result in results:
            assert isinstance(result, str)