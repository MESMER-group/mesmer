from setuptools import setup
from setuptools.command.test import test as TestCommand

import versioneer


class Mesmer(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest

        pytest.main(self.test_args)


cmdclass = versioneer.get_cmdclass()
cmdclass.update({"test": Mesmer})

setup(version=versioneer.get_version(), cmdclass=cmdclass)
