# Tensorflow Playground

This repository serves as demonstration of tensorflow model implementations and ML concepts in the form of different
experiments. A series of base classes are additionally provided in order to facilitate each experiment and generify the
training process. An explanation of each experiment is provided below.

### Classifier Experiment

The classifier experiment is based off of Tensorflow's documentation
found [here](https://www.tensorflow.org/tutorials/keras/classification). In this experiment we implement a model that
classifies images into 1 of 10 different classes of clothing. The framework I provide gives the user access to a number
of configurability options that allow us to easily modify the training process. Additionally, a custom directory with
images taken from the internet is also used in the evaluation of the model. The results are provided below for where the
original image is show and the probabilities of each of the categories that image falls into
based off the trained model.

 - **Results**

| Train Dataset | Test Dataset  | EPOCHS | Accuracy           | Loss               |
|---------------|:--------------|:-------|:-------------------|--------------------|
| fashion mnist | fashion mnist | 1      | 0.836899995803833  | 65.11700439453125  |
| fashion mnist | fashion mnist | 10     | 0.8583999872207642 | 59.760101318359375 |
| fashion mnist | fashion mnist | 100    | 0.8655999898910522 | 219.7647247314453  |
| fashion mnist | custom        | 1      | 0.6000000238418579 | 160.0536651611328  |
| fashion mnist | custom        | 10     | 0.4000000059604645 | 162.7378692626953  |
| fashion mnist | custom        | 100    | 0.4000000059604645 | 765.4063110351562  |

Observation: Accuracy decreases as the number of EPOCHS increase and loss increases as EPOCHS increase.

Interpretation: The model performs worse the longer it is trained. This may suggest over fitting. Loss can also be seen
to be much higher on the custom dataset suggesting that when testing images from the internet (that do not strictly
adhere to the formatting of the fashion mnist dataset) the model performs worse.

 - **Visualization for 100 EPOCH Fashion MNIST test dataset**

![](generated_plots/100EPOCHS%20First%2015%20Image%20Results%20test_loss%20219_7647247314453%20test_acc%200_8655999898910522-1659984932.png)

 - **Visualization for 10 EPOCH Custom test dataset**

![](generated_plots/10EPOCHS%20Custom%20Image%20Results%20test_loss%20162_7378692626953%20test_acc%200_4000000059604645-1659984698.png)
