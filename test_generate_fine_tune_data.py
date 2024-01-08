import unittest
from e_generate_fine_tune_data import generate_fine_tune_data_point, replace_bug_text

class TestGenerateFineTuneData(unittest.TestCase):
    def test_replace_bug_text(self):
        original_text = 'for (int item = 0; item < itemCount; item++) { <BUG>  </BUG> lvalue = intervalXYData.getStartXValue(series, item);'
        replacement_text = 'double value = intervalXYData.getXValue(series, item);'
        result = replace_bug_text(original_text, replacement_text)
        actual_result = 'for (int item = 0; item < itemCount; item++) { double value = intervalXYData.getXValue(series, item); lvalue = intervalXYData.getStartXValue(series, item);'
        self.assertEqual(result, actual_result)
    
    
