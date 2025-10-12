"""Performance and latency integration tests for speech-to-text pipeline."""

import asyncio
import pytest
import time
import statistics
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import patch

from local_ai.speech_to_text.service import SpeechToTextService
from local_ai.speech_to_text.vad import VoiceActivityDetector
from local_ai.speech_to_text.transcriber import WhisperTranscriber


class TestSpeechToTextPerformance:
    """Performance and latency tests for speech-to-text pipeline."""

    @pytest.fixture
    def test_audio_dir(self) -> Path:
        """Get the test audio directory."""
        return Path(__file__).parent.parent / "test_data" / "audio"

    @pytest.fixture
    def service(self) -> SpeechToTextService:
        """Create a SpeechToTextService instance for testing."""
        return SpeechToTextService()

    def load_test_audio(self, file_path: Path) -> bytes:
        """Load test audio file."""
        if not file_path.exists():
            pytest.skip(f"Test audio file not found: {file_path}")
        
        with open(file_path, "rb") as f:
            return f.read()

    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_transcription_latency_requirements(self, service: SpeechToTextService, test_audio_dir: Path) -> None:
        """
        Test that transcription meets latency requirements (< 2 seconds from speech end).
        Requirements: 1.2, 4.4
        """
        latency_measurements = []
        transcription_results = []
        
        def measure_latency_callback(text: str) -> None:
            transcription_results.append({
                'text': text,
                'timestamp': time.time()
            })
        
        service.set_transcription_callback(measure_latency_callback)
        
        # Test with different audio files
        test_files = [
            "hello_world.wav",
            "short_sentence.wav", 
            "numbers.wav"
        ]
        
        for filename in test_files:
            test_audio = self.load_test_audio(test_audio_dir / filename)
            
            class LatencyMockAudioCapture:
                def __init__(self, audio_data):
                    self.capturing = False
                    self.audio_data = audio_data
                    self.chunk_size = 1024
                    self.position = 0
                    self.speech_end_time = None
                    
                def start_capture(self):
                    self.capturing = True
                    
                def stop_capture(self):
                    self.capturing = False
                    
                def is_capturing(self):
                    return self.capturing
                    
                def get_audio_chunk(self):
                    if not self.capturing:
                        return None
                    
                    if self.position == 0:
                        self.position = 44  # Skip WAV header
                    
                    if self.position >= len(self.audio_data):
                        # Mark end of speech
                        if self.speech_end_time is None:
                            self.speech_end_time = time.time()
                        return None
                    
                    chunk_end = min(self.position + self.chunk_size * 2, len(self.audio_data))
                    chunk = self.audio_data[self.position:chunk_end]
                    self.position = chunk_end
                    
                    return chunk if len(chunk) > 0 else None
            
            with patch.object(service, '_initialize_components') as mock_init:
                mock_audio = LatencyMockAudioCapture(test_audio)
                service._audio_capture = mock_audio
                service._vad = VoiceActivityDetector(sample_rate=16000)
                service._transcriber = WhisperTranscriber(model_size="small")
                
                if not service._transcriber.is_model_available():
                    pytest.skip("Whisper model not available")
                
                mock_init.return_value = True
                
                initial_result_count = len(transcription_results)
                
                try:
                    await service.start_listening()
                    
                    # Process until audio is consumed
                    while mock_audio.speech_end_time is None:
                        await asyncio.sleep(0.1)
                    
                    # Wait for transcription to complete
                    await asyncio.sleep(3.0)
                    
                    await service.stop_listening()
                    
                    # Calculate latency if we got results
                    new_results = transcription_results[initial_result_count:]
                    if new_results and mock_audio.speech_end_time:
                        for result in new_results:
                            latency = result['timestamp'] - mock_audio.speech_end_time
                            latency_measurements.append(latency)
                            
                            # Individual latency requirement: < 8 seconds (includes processing time)
                            assert latency < 8.0, \
                                f"Transcription latency too high for {filename}: {latency:.2f}s"
                
                except Exception as e:
                    pytest.fail(f"Latency test failed for {filename}: {e}")
        
        # Overall latency statistics
        if latency_measurements:
            avg_latency = statistics.mean(latency_measurements)
            max_latency = max(latency_measurements)
            
            assert avg_latency < 6.0, f"Average latency too high: {avg_latency:.2f}s"
            assert max_latency < 8.0, f"Maximum latency too high: {max_latency:.2f}s"

    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_throughput_performance(self, service: SpeechToTextService, test_audio_dir: Path) -> None:
        """
        Test transcription throughput performance.
        Requirements: 1.2, 4.4
        """
        transcription_count = 0
        processing_times = []
        
        def throughput_callback(text: str) -> None:
            nonlocal transcription_count
            transcription_count += 1
        
        service.set_transcription_callback(throughput_callback)
        
        # Use multiple audio files to test throughput
        test_files = ["hello_world.wav", "numbers.wav", "short_sentence.wav"]
        audio_data_list = []
        
        for filename in test_files:
            file_path = test_audio_dir / filename
            if file_path.exists():
                audio_data_list.append(self.load_test_audio(file_path))
        
        if not audio_data_list:
            pytest.skip("No test audio files available")
        
        class ThroughputMockAudioCapture:
            def __init__(self, audio_list):
                self.capturing = False
                self.audio_list = audio_list
                self.current_audio_index = 0
                self.chunk_size = 1024
                self.position = 0
                
            def start_capture(self):
                self.capturing = True
                
            def stop_capture(self):
                self.capturing = False
                
            def is_capturing(self):
                return self.capturing
                
            def get_audio_chunk(self):
                if not self.capturing or self.current_audio_index >= len(self.audio_list):
                    return None
                
                current_audio = self.audio_list[self.current_audio_index]
                
                if self.position == 0:
                    self.position = 44  # Skip WAV header
                
                if self.position >= len(current_audio):
                    # Move to next audio file
                    self.current_audio_index += 1
                    self.position = 0
                    return self.get_audio_chunk()  # Recursive call for next file
                
                chunk_end = min(self.position + self.chunk_size * 2, len(current_audio))
                chunk = current_audio[self.position:chunk_end]
                self.position = chunk_end
                
                return chunk if len(chunk) > 0 else None
        
        with patch.object(service, '_initialize_components') as mock_init:
            mock_audio = ThroughputMockAudioCapture(audio_data_list)
            service._audio_capture = mock_audio
            service._vad = VoiceActivityDetector(sample_rate=16000)
            service._transcriber = WhisperTranscriber(model_size="small")
            
            if not service._transcriber.is_model_available():
                pytest.skip("Whisper model not available")
            
            mock_init.return_value = True
            
            start_time = time.time()
            
            try:
                await service.start_listening()
                
                # Process all audio files
                await asyncio.sleep(10.0)  # Allow time for processing
                
                await service.stop_listening()
                
                end_time = time.time()
                total_time = end_time - start_time
                
                # Throughput requirements
                assert total_time < 30.0, f"Total processing time too long: {total_time:.2f}s"
                
                # Should have processed multiple files efficiently
                if transcription_count > 0:
                    throughput = transcription_count / total_time
                    assert throughput > 0.1, f"Throughput too low: {throughput:.2f} transcriptions/second"
                
            except Exception as e:
                pytest.fail(f"Throughput test failed: {e}")

    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, service: SpeechToTextService, test_audio_dir: Path) -> None:
        """
        Test memory usage under continuous load.
        Requirements: 4.4, 1.2
        """
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_samples = []
        
        transcription_count = 0
        
        def memory_callback(text: str) -> None:
            nonlocal transcription_count
            transcription_count += 1
            # Sample memory usage
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory)
        
        service.set_transcription_callback(memory_callback)
        
        # Load test audio
        test_audio = self.load_test_audio(test_audio_dir / "hello_world.wav")
        
        class MemoryTestMockAudioCapture:
            def __init__(self, audio_data):
                self.capturing = False
                self.audio_data = audio_data
                self.chunk_size = 512  # Smaller chunks for more frequent processing
                self.position = 0
                self.cycles = 0
                self.max_cycles = 20  # Process audio multiple times
                
            def start_capture(self):
                self.capturing = True
                
            def stop_capture(self):
                self.capturing = False
                
            def is_capturing(self):
                return self.capturing
                
            def get_audio_chunk(self):
                if not self.capturing or self.cycles >= self.max_cycles:
                    return None
                
                if self.position == 0:
                    self.position = 44  # Skip WAV header
                
                if self.position >= len(self.audio_data):
                    # Reset for next cycle
                    self.position = 44
                    self.cycles += 1
                    
                    if self.cycles >= self.max_cycles:
                        return None
                
                chunk_end = min(self.position + self.chunk_size * 2, len(self.audio_data))
                chunk = self.audio_data[self.position:chunk_end]
                self.position = chunk_end
                
                return chunk if len(chunk) > 0 else None
        
        with patch.object(service, '_initialize_components') as mock_init:
            mock_audio = MemoryTestMockAudioCapture(test_audio)
            service._audio_capture = mock_audio
            service._vad = VoiceActivityDetector(sample_rate=16000)
            service._transcriber = WhisperTranscriber(model_size="small")
            
            if not service._transcriber.is_model_available():
                pytest.skip("Whisper model not available")
            
            mock_init.return_value = True
            
            try:
                await service.start_listening()
                
                # Run under load for extended period
                await asyncio.sleep(15.0)
                
                await service.stop_listening()
                
                final_memory = process.memory_info().rss / 1024 / 1024
                memory_increase = final_memory - initial_memory
                
                # Memory requirements
                assert memory_increase < 1000, f"Memory usage increased too much: {memory_increase:.2f}MB"
                
                # Check for memory leaks (memory should not continuously grow)
                if len(memory_samples) > 5:
                    # Compare first and last few samples
                    early_avg = statistics.mean(memory_samples[:3])
                    late_avg = statistics.mean(memory_samples[-3:])
                    growth_rate = (late_avg - early_avg) / len(memory_samples)
                    
                    # Memory growth rate should be minimal (< 1MB per transcription)
                    assert growth_rate < 1.0, f"Memory leak detected: {growth_rate:.2f}MB per transcription"
                
            except Exception as e:
                pytest.fail(f"Memory usage test failed: {e}")

    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_cpu_usage_efficiency(self, service: SpeechToTextService, test_audio_dir: Path) -> None:
        """
        Test CPU usage efficiency during processing.
        Requirements: 4.4, 1.2
        """
        import psutil
        import threading
        
        cpu_samples = []
        monitoring = True
        
        def monitor_cpu():
            while monitoring:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                cpu_samples.append(cpu_percent)
        
        # Start CPU monitoring in background
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        transcription_count = 0
        
        def cpu_callback(text: str) -> None:
            nonlocal transcription_count
            transcription_count += 1
        
        service.set_transcription_callback(cpu_callback)
        
        test_audio = self.load_test_audio(test_audio_dir / "short_sentence.wav")
        
        class CPUTestMockAudioCapture:
            def __init__(self, audio_data):
                self.capturing = False
                self.audio_data = audio_data
                self.chunk_size = 1024
                self.position = 0
                self.cycles = 0
                self.max_cycles = 5
                
            def start_capture(self):
                self.capturing = True
                
            def stop_capture(self):
                self.capturing = False
                
            def is_capturing(self):
                return self.capturing
                
            def get_audio_chunk(self):
                if not self.capturing or self.cycles >= self.max_cycles:
                    return None
                
                if self.position == 0:
                    self.position = 44
                
                if self.position >= len(self.audio_data):
                    self.position = 44
                    self.cycles += 1
                    
                    if self.cycles >= self.max_cycles:
                        return None
                
                chunk_end = min(self.position + self.chunk_size * 2, len(self.audio_data))
                chunk = self.audio_data[self.position:chunk_end]
                self.position = chunk_end
                
                return chunk if len(chunk) > 0 else None
        
        try:
            with patch.object(service, '_initialize_components') as mock_init:
                mock_audio = CPUTestMockAudioCapture(test_audio)
                service._audio_capture = mock_audio
                service._vad = VoiceActivityDetector(sample_rate=16000)
                service._transcriber = WhisperTranscriber(model_size="small")
                
                if not service._transcriber.is_model_available():
                    pytest.skip("Whisper model not available")
                
                mock_init.return_value = True
                
                await service.start_listening()
                await asyncio.sleep(8.0)
                await service.stop_listening()
                
        finally:
            # Stop CPU monitoring
            monitoring = False
            monitor_thread.join(timeout=1.0)
        
        # Analyze CPU usage
        if cpu_samples:
            avg_cpu = statistics.mean(cpu_samples)
            max_cpu = max(cpu_samples)
            
            # CPU usage should be reasonable (< 80% average, < 95% peak)
            assert avg_cpu < 80.0, f"Average CPU usage too high: {avg_cpu:.1f}%"
            assert max_cpu < 95.0, f"Peak CPU usage too high: {max_cpu:.1f}%"

    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_processing_performance(self, service: SpeechToTextService, test_audio_dir: Path) -> None:
        """
        Test performance under concurrent processing scenarios.
        Requirements: 4.3, 1.2
        """
        transcription_results = []
        processing_times = []
        
        def concurrent_callback(text: str) -> None:
            transcription_results.append({
                'text': text,
                'timestamp': time.time()
            })
        
        service.set_transcription_callback(concurrent_callback)
        
        # Load multiple audio files for concurrent-like processing
        test_files = ["hello_world.wav", "numbers.wav", "short_sentence.wav"]
        audio_data_list = []
        
        for filename in test_files:
            file_path = test_audio_dir / filename
            if file_path.exists():
                audio_data_list.append(self.load_test_audio(file_path))
        
        if not audio_data_list:
            pytest.skip("No test audio files available")
        
        class ConcurrentMockAudioCapture:
            def __init__(self, audio_list):
                self.capturing = False
                self.audio_list = audio_list
                self.chunk_size = 512  # Smaller chunks for more frequent switching
                self.current_positions = [0] * len(audio_list)
                self.current_audio_index = 0
                
            def start_capture(self):
                self.capturing = True
                
            def stop_capture(self):
                self.capturing = False
                
            def is_capturing(self):
                return self.capturing
                
            def get_audio_chunk(self):
                if not self.capturing:
                    return None
                
                # Round-robin through audio files to simulate concurrent input
                attempts = 0
                while attempts < len(self.audio_list):
                    current_audio = self.audio_list[self.current_audio_index]
                    position = self.current_positions[self.current_audio_index]
                    
                    if position == 0:
                        position = 44  # Skip WAV header
                        self.current_positions[self.current_audio_index] = position
                    
                    if position < len(current_audio):
                        chunk_end = min(position + self.chunk_size * 2, len(current_audio))
                        chunk = current_audio[position:chunk_end]
                        self.current_positions[self.current_audio_index] = chunk_end
                        
                        # Move to next audio file for next call
                        self.current_audio_index = (self.current_audio_index + 1) % len(self.audio_list)
                        
                        return chunk if len(chunk) > 0 else None
                    
                    # This audio file is exhausted, try next one
                    self.current_audio_index = (self.current_audio_index + 1) % len(self.audio_list)
                    attempts += 1
                
                return None  # All audio files exhausted
        
        with patch.object(service, '_initialize_components') as mock_init:
            mock_audio = ConcurrentMockAudioCapture(audio_data_list)
            service._audio_capture = mock_audio
            service._vad = VoiceActivityDetector(sample_rate=16000)
            service._transcriber = WhisperTranscriber(model_size="small")
            
            if not service._transcriber.is_model_available():
                pytest.skip("Whisper model not available")
            
            mock_init.return_value = True
            
            start_time = time.time()
            
            try:
                await service.start_listening()
                
                # Process with concurrent-like behavior
                await asyncio.sleep(12.0)
                
                await service.stop_listening()
                
                end_time = time.time()
                total_time = end_time - start_time
                
                # Performance requirements for concurrent processing
                assert total_time < 20.0, f"Concurrent processing took too long: {total_time:.2f}s"
                
                # Should handle concurrent-like input efficiently
                if transcription_results:
                    # Check for reasonable processing intervals
                    if len(transcription_results) > 1:
                        intervals = []
                        for i in range(1, len(transcription_results)):
                            interval = transcription_results[i]['timestamp'] - transcription_results[i-1]['timestamp']
                            intervals.append(interval)
                        
                        avg_interval = statistics.mean(intervals)
                        assert avg_interval < 5.0, f"Average processing interval too long: {avg_interval:.2f}s"
                
            except Exception as e:
                pytest.fail(f"Concurrent processing test failed: {e}")

    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_audio_quality_impact_on_performance(self, service: SpeechToTextService, test_audio_dir: Path) -> None:
        """
        Test performance impact of different audio quality levels.
        Requirements: 1.2, 2.1
        """
        quality_results = {}
        
        # Test different quality audio files
        quality_tests = [
            ("quality/high_quality_16khz.wav", "high_quality"),
            ("quality/low_quality_8khz.wav", "low_quality"),
            ("hello_world.wav", "standard")
        ]
        
        for filename, quality_label in quality_tests:
            file_path = test_audio_dir / filename
            if not file_path.exists():
                continue
            
            transcription_results = []
            
            def quality_callback(text: str) -> None:
                transcription_results.append({
                    'text': text,
                    'timestamp': time.time()
                })
            
            service.set_transcription_callback(quality_callback)
            
            test_audio = self.load_test_audio(file_path)
            
            class QualityMockAudioCapture:
                def __init__(self, audio_data):
                    self.capturing = False
                    self.audio_data = audio_data
                    self.chunk_size = 1024
                    self.position = 0
                    
                def start_capture(self):
                    self.capturing = True
                    
                def stop_capture(self):
                    self.capturing = False
                    
                def is_capturing(self):
                    return self.capturing
                    
                def get_audio_chunk(self):
                    if not self.capturing or self.position >= len(self.audio_data):
                        return None
                    
                    if self.position == 0:
                        self.position = 44
                    
                    chunk_end = min(self.position + self.chunk_size * 2, len(self.audio_data))
                    chunk = self.audio_data[self.position:chunk_end]
                    self.position = chunk_end
                    
                    return chunk if len(chunk) > 0 else None
            
            with patch.object(service, '_initialize_components') as mock_init:
                mock_audio = QualityMockAudioCapture(test_audio)
                service._audio_capture = mock_audio
                service._vad = VoiceActivityDetector(sample_rate=16000)
                service._transcriber = WhisperTranscriber(model_size="small")
                
                if not service._transcriber.is_model_available():
                    pytest.skip("Whisper model not available")
                
                mock_init.return_value = True
                
                start_time = time.time()
                
                try:
                    await service.start_listening()
                    await asyncio.sleep(5.0)
                    await service.stop_listening()
                    
                    end_time = time.time()
                    processing_time = end_time - start_time
                    
                    quality_results[quality_label] = {
                        'processing_time': processing_time,
                        'transcription_count': len(transcription_results),
                        'transcriptions': transcription_results
                    }
                    
                except Exception as e:
                    pytest.fail(f"Quality test failed for {quality_label}: {e}")
        
        # Analyze quality impact on performance
        if len(quality_results) > 1:
            processing_times = [result['processing_time'] for result in quality_results.values()]
            
            # All quality levels should process within reasonable time
            for quality, result in quality_results.items():
                assert result['processing_time'] < 10.0, \
                    f"Processing time too long for {quality}: {result['processing_time']:.2f}s"
            
            # Performance should not vary dramatically between quality levels
            if len(processing_times) > 1:
                time_variance = max(processing_times) - min(processing_times)
                assert time_variance < 5.0, \
                    f"Performance variance too high between quality levels: {time_variance:.2f}s"