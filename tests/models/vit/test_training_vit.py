import os
import sys

from ..test_model import ModelTrainingTestMixin, TestCasePlus


# `TRANSFORMERS_DIR` is an environment variable pointing to a `transformers` source directory (containing `examples`)
SRC_DIRS = [os.path.join(os.getenv("TRANSFORMERS_DIR"), "examples/pytorch", dirname) for dirname in ["image-classification"]]
sys.path.extend(SRC_DIRS)


if SRC_DIRS is not None:
    import run_image_classification


class ViTTrainingTest(ModelTrainingTestMixin, TestCasePlus):

    def get_training_script(self):
        return run_image_classification

    def prepare_training_command(self, **kwargs):
        output_dir = kwargs.get("output_dir", "my_dir")

        testargs = f"""
            run_image_classification.py
            --output_dir {output_dir}
            --model_name_or_path google/vit-base-patch16-224-in21k
            --dataset_name hf-internal-testing/cats_vs_dogs_sample
            --do_train
            --do_eval
            --learning_rate 1e-4
            --per_device_train_batch_size 2
            --per_device_eval_batch_size 1
            --remove_unused_columns False
            --overwrite_output_dir True
            --dataloader_num_workers 16
            --metric_for_best_model accuracy
            --max_steps 10
            --train_val_split 0.1
            --seed 42
            --label_column_name labels
        """

        return testargs
