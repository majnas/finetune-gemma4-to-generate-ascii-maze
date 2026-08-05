import tempfile
import unittest
from pathlib import Path

from .build_dataset import split_counts, write_jsonl
from .dataset import build_sample, format_path
from .evaluate import parse_prediction, score_record


class Fixed4x4SolverTests(unittest.TestCase):
    def test_sample_is_deterministic_and_has_expected_shape(self):
        sample = build_sample(1934)
        self.assertEqual(sample, build_sample(1934))
        self.assertEqual(sample["meta"]["start"], [0, 0])
        self.assertEqual(sample["meta"]["end"], [3, 3])
        answer = sample["conversations"][1]["content"]
        self.assertEqual(answer, format_path([tuple(c) for c in sample["meta"]["path_cells"]]))
        self.assertTrue(answer.startswith("S,"))
        self.assertTrue(answer.endswith(",E"))

    def test_parser(self):
        self.assertEqual(parse_prediction("S,A2,B2,E"), ["S", "A2", "B2", "E"])
        with self.assertRaises(ValueError):
            parse_prediction("A1,A2,E")
        with self.assertRaises(ValueError):
            parse_prediction("S,Z9,E")

    def test_scoring(self):
        sample = build_sample(7)
        expected = sample["conversations"][1]["content"]
        self.assertEqual(score_record({**sample, "prediction": expected}), (True, None))
        self.assertEqual(score_record({**sample, "prediction": "S,E"}), (False, "wrong path"))

    def test_split_and_jsonl(self):
        self.assertEqual(split_counts(7000), {"train": 6300, "val": 350, "test": 350})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.jsonl"
            write_jsonl(path, [build_sample(1)])
            self.assertEqual(len(path.read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
