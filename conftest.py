# conftest.py
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """ This hook is called after all tests are executed. """
    passed = terminalreporter.stats.get('passed', [])
    failed = terminalreporter.stats.get('failed', [])
    for test in passed:
        print(f"{test.nodeid} - PASSED")
    for test in failed:
        print(f"{test.nodeid} - FAILED")

