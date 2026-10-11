import sys
import unittest
from unittest.mock import MagicMock, patch
from core.generator import generate_answer


class TestGenerator(unittest.TestCase):
    def test_empty_context_returns_default_message(self):
        result = generate_answer("What is X?", "")
        self.assertIn("No relevant context found", result)

        result_whitespace = generate_answer("What is X?", "   \n\t  ")
        self.assertIn("No relevant context found", result_whitespace)

    def test_missing_ollama_raises_import_error(self):
        with patch.dict(sys.modules, {"ollama": None}):
            with self.assertRaises(ImportError):
                generate_answer("What is X?", "Some valid context")

    def test_successful_answer_generation(self):
        mock_ollama = MagicMock()
        mock_ollama.chat.return_value = {
            "message": {"content": "This is the generated answer."}
        }

        with patch.dict(sys.modules, {"ollama": mock_ollama}):
            answer = generate_answer("What is load balancing?", "Load balancing distributes traffic.")
            self.assertEqual(answer, "This is the generated answer.")
            mock_ollama.chat.assert_called_once()
            call_kwargs = mock_ollama.chat.call_args[1]
            self.assertEqual(call_kwargs["model"], "phi3")

    def test_ollama_runtime_failure_raises_runtime_error(self):
        mock_ollama = MagicMock()
        mock_ollama.chat.side_effect = Exception("Connection refused to Ollama server")

        with patch.dict(sys.modules, {"ollama": mock_ollama}):
            with self.assertRaises(RuntimeError):
                generate_answer("Question", "Context")


if __name__ == "__main__":
    unittest.main()
