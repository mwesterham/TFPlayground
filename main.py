from image_classifier_experiment.classifier_experiment import run_classifier_experiment
from text_binary_classifier_experiment.binary_classifier_experiment import run_text_classifier_experiment
from image_generator_experiment.generator_experiment import run_dcgan_experiment

if __name__ == '__main__':
    # Image classifier experiment
    # run_classifier_experiment()

    # Text classifier experiment
    # run_text_classifier_experiment()

    # DCGAN experiment
    run_dcgan_experiment()