# Classifier Experiment

The classifier experiments are based off of Tensorflow's documentation
found [here](https://www.tensorflow.org/tutorials/keras/classification). In this experiment we implement a model that
classifies images into different classes for two datasets: Fashion MNIST and MNIST. Once these models were trained, they
were evaluated with the test dataset provided by Tensorflow and a small custom dataset I compiled myself for each. The
results from these two datasets are compared and analyzed.

### Experiment: Classifier on Fashion MNIST dataset

- **Results**

The overall results are given below. The model was trained with a variable number of epochs and the accuracy and loss
were recorded.

| Train Dataset | Test Dataset  | EPOCHS | Loss               | Accuracy            |
|---------------|:--------------|:-------|--------------------|:--------------------|
| fashion mnist | fashion mnist | 1      | 61.53921127319336  | 0.8395000100135803  |
| fashion mnist | fashion mnist | 10     | 60.90907669067383  | 0.8600000143051147  |
| fashion mnist | fashion mnist | 100    | 212.41879272460938 | 0.857699990272522   |
| fashion mnist | custom        | 1      | 165.23138427734375 | 0.20000000298023224 |
| fashion mnist | custom        | 10     | 170.42318725585938 | 0.20000000298023224 |
| fashion mnist | custom        | 100    | 721.8863525390625  | 0.4000000059604645  |

Observations: Loss increases as EPOCHS increase. Loss is significantly higher when testing on a dataset I compiled
myself.

Interpretation: The model performs worse the longer it is trained. This may suggest over fitting. Since loss can also be
seen to be much higher on the custom dataset, we can infer when testing images from the internet (that do not strictly
adhere to the formatting of the fashion mnist dataset) the model performs worse.

- **Visualization**

Each image is accompanied by a bar graph that shows what probabilities each class was assigned by the trained model.
Additionally, each image has a description with the predicted label (left hand side) and the real label (right hand side
in parentheses). The label and bar graph are colored blue if the predicted label matches the real label; they are
colored red if the predicted and real labels do not match.

[100 epochs using the Fashion MNIST as test data]

![](./generated_assets/fashion_mnist/100EPOCHS%20First%2015%20Image%20Results-1660071658.png)

[100 epochs using custom images as test data]

![](./generated_assets/fashion_mnist/100EPOCHS%20Custom%20Image%20Results-1660071659.png)

### Experiment: Classifier on MNIST dataset

- **Results**

The overall results are given below. The model was trained with a variable number of epochs and the accuracy and loss
were recorded.

| Train Dataset | Test Dataset | EPOCHS | Loss               | Accuracy           |
|---------------|:-------------|:-------|--------------------|:-------------------|
| mnist         | mnist        | 1      | 19.780439376831055 | 0.95660001039505   |
| mnist         | mnist        | 10     | 15.685311317443848 | 0.9776999950408936 |
| mnist         | mnist        | 100    | 59.741580963134766 | 0.9789000153541565 |
| mnist         | custom       | 1      | 647.4493408203125  | 0.6000000238418579 |
| mnist         | custom       | 10     | 1428.693603515625  | 0.6000000238418579 |
| mnist         | custom       | 100    | 7887.95947265625   | 0.4000000059604645 |

Observations: For the mnist test dataset, loss decreased from 1 to 10 EPOCHS indicating performance improved but at
100 EPOCHS the loss jumped to 59.74 suggesting overfitting. For the custom test dataset, loss continuously increased
to significantly higher levels than the loss of the test dataset. This is likely due to it not following the
same patterns as the downloaded datset (I drew these images myself).

Interpretation: Identical to the Fashion MNIST interpretation. The model performs worse the longer it is trained. This
may suggest over fitting. Since loss can also be seen to be much higher on the custom dataset, we can infer when testing
images from the internet (that do not strictly adhere to the formatting of the fashion mnist dataset) the model performs
worse.

- **Visualization**

[100 epochs using the MNIST as test data]

![](./generated_assets/mnist/100EPOCHS%20First%2015%20Image%20Results-1660071849.png)

[100 epochs using custom images as test data]

![](./generated_assets/mnist/100EPOCHS%20Custom%20Image%20Results-1660071851.png)

### Analysis and Conclusion

The model performed better when training with simpler data. This can be seen in the results below where the model
trained on the Fashion MNIST dataset had less accuracy and more loss than the model trained with MNIST.

However, the model trained on MNIST performed significantly worse on the custom datasets provided. This may indicate
that although the model learned patterns of the simpler data much better, it will perform significantly worse on images
that don't follow the learned patterns. This could be due to human error in creation of the datasets; it is also
possible
that the model trained on MNIST is worse at predicting data that it was not trained for, where the Fashion MNIST trained
model did better since the patterns were more complex and likely more generalized.
