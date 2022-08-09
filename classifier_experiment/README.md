# Classifier Experiment on Fashion MNIST

The classifier experiment is based off of Tensorflow's documentation
found [here](https://www.tensorflow.org/tutorials/keras/classification). In this experiment we implement a model that
classifies images into 1 of 10 different classes of clothing. The framework I provide gives the user access to a number
of configurability options that allow us to easily modify the training process. Additionally, I compiled a few custom
images myself to be used in the evaluation process.

- **Results**

The overall results are given below. The model was trained with a variable number of epochs and the accuracy and loss
were recorded.

| Train Dataset | Test Dataset  | EPOCHS | Accuracy           | Loss               |
|---------------|:--------------|:-------|:-------------------|--------------------|
| fashion mnist | fashion mnist | 1      | 0.836899995803833  | 65.11700439453125  |
| fashion mnist | fashion mnist | 10     | 0.8583999872207642 | 59.760101318359375 |
| fashion mnist | fashion mnist | 100    | 0.8655999898910522 | 219.7647247314453  |
| fashion mnist | custom        | 1      | 0.6000000238418579 | 160.0536651611328  |
| fashion mnist | custom        | 10     | 0.4000000059604645 | 162.7378692626953  |
| fashion mnist | custom        | 100    | 0.4000000059604645 | 765.4063110351562  |

Observations: Accuracy decreases as the number of EPOCHS increase and loss increases as EPOCHS increase.

Interpretation: The model performs worse the longer it is trained. This may suggest over fitting. Loss can also be seen
to be much higher on the custom dataset suggesting that when testing images from the internet (that do not strictly
adhere to the formatting of the fashion mnist dataset) the model performs worse.

- **Visualization**

Each image is accompanied by a bar graph that shows what probabilities each class was assigned by the trained model.
Additionally, each image has a description with the predicted label (left hand side) and the real label (right hand side
in parentheses). The label and bar graph are colored blue if the predicted label matches the real label; they are 
colored red if the predicted and real labels do not match.

[100 epochs using the Fashion MNIST as test data]

![](generated_plots/fashion_mnist%20100EPOCHS%20First%2015%20Image%20Results%20test_loss%20219_7647247314453%20test_acc%200_8655999898910522-1659984932.png)

[10 epochs using custom images as test data]

![](generated_plots/fashion_mnist%2010EPOCHS%20Custom%20Image%20Results%20test_loss%20162_7378692626953%20test_acc%200_4000000059604645-1659984698.png)
