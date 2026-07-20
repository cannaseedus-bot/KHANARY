import unittest

from tools import stb


class TestStbHelpers(unittest.TestCase):
    def test_decode_load_bin_tensor_payload(self):
        self.assertEqual(stb.decode_load_bin_tensor_payload(0xAB), (0xA, 0xB))

    def test_decode_payload_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            stb.decode_load_bin_tensor_payload(256)

    def test_round_trip_with_unaligned_descriptor_table(self):
        # Regression: when 32 + N*32 is not 64-aligned, the data region is padded; the reader
        # must seek to data_offset or every tensor shifts. N=2 -> 96 -> pad 32.
        if stb.np is None:
            self.skipTest("NumPy required")
        import tempfile, os
        np = stb.np
        a = np.arange(10, dtype=np.float32) * 1.5
        b = (np.arange(6, dtype=np.float32) + 100).reshape(2, 3)
        path = os.path.join(tempfile.gettempdir(), "stb_unaligned.stb")
        stb.write_stb(path, [{"tensor_id": 0, "array": a}, {"tensor_id": 1, "array": b}])
        back = stb.read_stb(path)
        assert np.array_equal(np.asarray(back[0]["array"]), a)
        assert np.array_equal(np.asarray(back[1]["array"]), b)

    def test_read_write_raise_clear_error_without_numpy(self):
        if stb.np is not None:
            self.skipTest("NumPy present in environment")
        with self.assertRaises(stb.StbDependencyError):
            stb.read_stb("any.stb")
        with self.assertRaises(stb.StbDependencyError):
            stb.write_stb("any.stb", [])


if __name__ == "__main__":
    unittest.main()
