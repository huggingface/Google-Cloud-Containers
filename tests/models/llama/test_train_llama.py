import unittest

from ..test_model import ModelTrainingTestMixin, TestCasePlus


class LLamaModelTrainingTest(ModelTrainingTestMixin, TestCasePlus):

    # WIP
    def get_training_script(self):
        raise NotImplementedError

    @unittest.skip("WIP")
    def prepare_training_command(self, **kwargs):
        output_dir = kwargs.get("output_dir", "my_dir")

        testargs = f""""""

        return testargs
