"""Pure validation-selection regression tests; no model downloads."""
import unittest

from absa_enhanced.selection import checkpoint_directory, selection_key


def metrics(accuracy, macro_f1, end_to_end=0.0):
    return {"gold_aspect_sentiment": {"accuracy": accuracy, "macro_f1": macro_f1},
            "end_to_end": {"f1": end_to_end}}


class SelectionTests(unittest.TestCase):
    def test_accuracy_and_macro_f1_choose_different_epochs(self):
        epochs = [metrics(.8253676471, .6745713325), metrics(.8327205882, .7338638914),
                  metrics(.8363970588, .7301894061), metrics(.8327205882, .7328173912)]
        best_accuracy = max(range(4), key=lambda i: selection_key(epochs[i], "gold_accuracy"))
        best_macro = max(range(4), key=lambda i: selection_key(epochs[i], "gold_macro_f1"))
        self.assertEqual(best_accuracy + 1, 3)
        self.assertEqual(best_macro + 1, 2)

    def test_accuracy_tie_uses_macro_f1(self):
        self.assertGreater(selection_key(metrics(.85, .78), "gold_accuracy"),
                           selection_key(metrics(.85, .76), "gold_accuracy"))

    def test_macro_f1_tie_uses_accuracy(self):
        self.assertGreater(selection_key(metrics(.86, .78), "gold_macro_f1"),
                           selection_key(metrics(.85, .78), "gold_macro_f1"))

    def test_primary_and_secondary_directory_mapping(self):
        self.assertEqual(checkpoint_directory("gold_accuracy", "gold_accuracy"), "best")
        self.assertEqual(checkpoint_directory("gold_macro_f1", "gold_accuracy"), "best_macro_f1")
        self.assertEqual(checkpoint_directory("gold_accuracy", "gold_macro_f1"), "best_accuracy")


if __name__ == "__main__":
    unittest.main()
