import unittest
import numpy as np
from graphgen.data import data

class TestClean(unittest.TestCase):

    def test_length_threshold(self):
        traces = np.array([np.linspace(start=1, stop=10, num=25)])
        cleaned_traces = data.clean(traces, 50, 0)
        self.assertEqual(len(cleaned_traces), 0)

if __name__ == "__main__":
    unittest.main()
