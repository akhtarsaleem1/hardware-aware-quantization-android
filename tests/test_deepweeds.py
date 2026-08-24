import csv
import unittest

from src.data.deepweeds import load_official_split, validate_disjoint_splits


def write_labels(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Filename", "Label", "Species"])
        writer.writeheader()
        writer.writerows(rows)


class DeepWeedsTests(unittest.TestCase):
    def test_load_official_split_and_disjoint_validation(self):
        with self.subTest("valid official schema"):
            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                train_path = root / "train.csv"
                test_path = root / "test.csv"
                write_labels(train_path, [{"Filename": "20170101-120000-0.jpg", "Label": 1, "Species": "Lantana"}])
                write_labels(test_path, [{"Filename": "20170101-120001-0.jpg", "Label": 8, "Species": "Negative"}])
                train = load_official_split(train_path)
                test = load_official_split(test_path)
                validate_disjoint_splits({"train": train, "test": test})
                self.assertEqual(train[0].label, 1)

    def test_rejects_filename_cross_split_overlap(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            write_labels(path, [{"Filename": "20170101-120000-0.jpg", "Label": 1, "Species": "Lantana"}])
            rows = load_official_split(path)
            with self.assertRaisesRegex(ValueError, "both train and test"):
                validate_disjoint_splits({"train": rows, "test": rows})

    def test_rejects_unsafe_filename(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            write_labels(path, [{"Filename": "../escape.jpg", "Label": 1, "Species": "Lantana"}])
            with self.assertRaisesRegex(ValueError, "unsafe Filename"):
                load_official_split(path)


if __name__ == "__main__":
    unittest.main()
