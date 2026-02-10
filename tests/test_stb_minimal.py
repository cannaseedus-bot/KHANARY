import unittest

from tools import stb


class TestStbHelpers(unittest.TestCase):
    def test_decode_load_bin_tensor_payload(self):
        self.assertEqual(stb.decode_load_bin_tensor_payload(0xAB), (0xA, 0xB))

    def test_decode_payload_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            stb.decode_load_bin_tensor_payload(256)

    def test_read_write_raise_clear_error_without_numpy(self):
        if stb.np is not None:
            self.skipTest("NumPy present in environment")
        with self.assertRaises(stb.StbDependencyError):
            stb.read_stb("any.stb")
        with self.assertRaises(stb.StbDependencyError):
            stb.write_stb("any.stb", [])


if __name__ == "__main__":
    unittest.main()
