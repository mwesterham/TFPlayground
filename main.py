from classifier_experiment.classifier_experiment import ClassifierExperiment


if __name__ == '__main__':
    # Based off of https://www.tensorflow.org/tutorials/keras/classification
    experiment1 = ClassifierExperiment(config={
        'EPOCHS': 10
    })
    experiment1.execute()