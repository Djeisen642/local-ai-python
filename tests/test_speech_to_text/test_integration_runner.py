"""Integration test runner with comprehensive reporting."""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


class IntegrationTestRunner:
    """Runs integration tests and generates comprehensive reports."""

    def __init__(self, test_dir: Path | None = None):
        """Initialize the test runner."""
        self.test_dir = test_dir or Path(__file__).parent
        self.results = {}
        self.start_time = None
        self.end_time = None

    def run_integration_tests(self, verbose: bool = True) -> dict[str, Any]:
        """
        Run all integration tests and collect results.

        Args:
            verbose: Whether to print verbose output

        Returns:
            Dictionary containing test results and performance metrics
        """
        self.start_time = time.time()

        if verbose:
            pass

        # Define test categories and their files
        test_categories = {
            "end_to_end": "test_end_to_end_integration.py",
            "performance": "test_performance_integration.py",
            "transcriber_integration": "test_transcriber_integration.py",
            "vad_integration": "test_vad_integration.py",
        }

        results = {}

        for category, test_file in test_categories.items():
            if verbose:
                pass

            test_path = self.test_dir / test_file
            if not test_path.exists():
                if verbose:
                    pass
                continue

            category_results = self._run_test_file(test_path, verbose)
            results[category] = category_results

        self.end_time = time.time()
        self.results = results

        if verbose:
            self._print_summary()

        return self._generate_report()

    def _run_test_file(self, test_path: Path, verbose: bool) -> dict[str, Any]:
        """Run tests from a specific file and collect results."""
        start_time = time.time()

        # Run pytest with integration markers
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(test_path),
            "-v",
            "-m",
            "integration",
            "--tb=short",
            "--json-report",
            "--json-report-file=/tmp/pytest_report.json",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout per test file
            )

            end_time = time.time()

            # Parse pytest JSON report if available
            report_data = self._parse_pytest_json_report()

            return {
                "duration": end_time - start_time,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0,
                "test_details": report_data,
            }

        except subprocess.TimeoutExpired:
            return {
                "duration": 300,
                "return_code": -1,
                "stdout": "",
                "stderr": "Test execution timed out",
                "success": False,
                "test_details": None,
            }
        except Exception as e:
            return {
                "duration": 0,
                "return_code": -1,
                "stdout": "",
                "stderr": str(e),
                "success": False,
                "test_details": None,
            }

    def _parse_pytest_json_report(self) -> dict[str, Any] | None:
        """Parse pytest JSON report if available."""
        try:
            report_path = Path("/tmp/pytest_report.json")
            if report_path.exists():
                with open(report_path) as f:
                    return json.load(f)
        except Exception:
            pass
        return None

    def _print_summary(self) -> None:
        """Print test execution summary."""

        (self.end_time - self.start_time if self.start_time and self.end_time else 0)

        for result in self.results.values():
            "PASS" if result["success"] else "FAIL"
            result["duration"]

            if not result["success"] and result["stderr"]:
                pass

        # Overall status
        all(result["success"] for result in self.results.values())

    def _generate_report(self) -> dict[str, Any]:
        """Generate comprehensive test report."""
        total_duration = (
            self.end_time - self.start_time if self.start_time and self.end_time else 0
        )

        # Count test results
        passed_categories = sum(
            1 for result in self.results.values() if result["success"]
        )
        total_categories = len(self.results)

        # Extract performance metrics
        performance_metrics = self._extract_performance_metrics()

        return {
            "summary": {
                "total_duration": total_duration,
                "categories_passed": passed_categories,
                "categories_total": total_categories,
                "success_rate": passed_categories / total_categories
                if total_categories > 0
                else 0,
                "overall_success": passed_categories == total_categories,
            },
            "categories": self.results,
            "performance_metrics": performance_metrics,
            "timestamp": time.time(),
            "requirements_coverage": self._analyze_requirements_coverage(),
        }

    def _extract_performance_metrics(self) -> dict[str, Any]:
        """Extract performance metrics from test results."""
        metrics = {"latency": {}, "throughput": {}, "memory_usage": {}, "cpu_usage": {}}

        # Parse performance data from test outputs
        performance_result = self.results.get("performance", {})
        if performance_result.get("success") and performance_result.get("stdout"):
            stdout = performance_result["stdout"]

            # Extract latency information
            if "latency" in stdout.lower():
                metrics["latency"]["tests_run"] = True
                metrics["latency"]["status"] = "measured"

            # Extract throughput information
            if "throughput" in stdout.lower():
                metrics["throughput"]["tests_run"] = True
                metrics["throughput"]["status"] = "measured"

            # Extract memory usage information
            if "memory" in stdout.lower():
                metrics["memory_usage"]["tests_run"] = True
                metrics["memory_usage"]["status"] = "measured"

        return metrics

    def _analyze_requirements_coverage(self) -> dict[str, Any]:
        """Analyze which requirements are covered by the tests."""
        # Requirements from the task specification
        target_requirements = [
            "4.3",  # Modular and testable code
            "1.2",  # Real-time processing with minimal delay
            "2.1",  # Local AI model usage
            "1.4",  # Error handling and graceful fallback
            "2.3",  # Clear error messages and instructions
            "4.4",  # Logging for debugging
        ]

        covered_requirements = set()

        # Analyze test files for requirement coverage
        for category, result in self.results.items():
            if result.get("success"):
                # Map test categories to requirements
                if category == "end_to_end":
                    covered_requirements.update(["4.3", "1.2", "2.1", "1.4"])
                elif category == "performance":
                    covered_requirements.update(["1.2", "4.4"])
                elif category == "transcriber_integration":
                    covered_requirements.update(["2.1", "2.3"])
                elif category == "vad_integration":
                    covered_requirements.update(["1.2", "4.4"])

        return {
            "target_requirements": target_requirements,
            "covered_requirements": list(covered_requirements),
            "coverage_percentage": len(covered_requirements)
            / len(target_requirements)
            * 100,
            "missing_requirements": [
                req for req in target_requirements if req not in covered_requirements
            ],
        }

    def save_report(self, output_path: Path) -> None:
        """Save test report to file."""
        report = self._generate_report()

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

    def run_specific_test_category(
        self, category: str, verbose: bool = True
    ) -> dict[str, Any]:
        """Run tests for a specific category only."""
        test_files = {
            "end_to_end": "test_end_to_end_integration.py",
            "performance": "test_performance_integration.py",
            "transcriber": "test_transcriber_integration.py",
            "vad": "test_vad_integration.py",
        }

        if category not in test_files:
            raise ValueError(f"Unknown test category: {category}")

        test_file = test_files[category]
        test_path = self.test_dir / test_file

        if not test_path.exists():
            raise FileNotFoundError(f"Test file not found: {test_path}")

        if verbose:
            pass

        self.start_time = time.time()
        result = self._run_test_file(test_path, verbose)
        self.end_time = time.time()

        self.results = {category: result}

        if verbose:
            self._print_summary()

        return self._generate_report()


def main():
    """Main entry point for running integration tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Run speech-to-text integration tests")
    parser.add_argument(
        "--category",
        choices=["end_to_end", "performance", "transcriber", "vad"],
        help="Run specific test category only",
    )
    parser.add_argument("--output", type=Path, help="Save report to file")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    args = parser.parse_args()

    runner = IntegrationTestRunner()

    try:
        if args.category:
            report = runner.run_specific_test_category(
                args.category, verbose=not args.quiet
            )
        else:
            report = runner.run_integration_tests(verbose=not args.quiet)

        if args.output:
            runner.save_report(args.output)

        # Exit with error code if tests failed
        if not report["summary"]["overall_success"]:
            sys.exit(1)

    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
