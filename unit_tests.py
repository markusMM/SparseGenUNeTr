import unittest

if __name__ == "__main__":
    loader = unittest.TestLoader()
    tests = loader.discover('src_test/', pattern='*_test.py')
    testRunner = unittest.TextTestRunner()
    testRunner.run(tests)
