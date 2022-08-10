# Classifier Experiment

The classifier experiments are based off of Tensorflow's documentation
found [here](https://www.tensorflow.org/tutorials/keras/classification). In this experiment we implement a model that
classifies images into different classes for two datasets: Fashion MNIST and MNIST. Once these models were trained, they
were evaluated with the test dataset provided by Tensorflow and a small custom dataset I compiled myself for each. The
results from these two datasets are compared and analyzed.

### Experiment: Classifier on Fashion MNIST dataset

The model was trained with a variable number of epochs and the accuracy and loss
were recorded for both default and custom datasets. The overall results are given below. 

<table>
    <tr>
      <th rowspan="2">EPOCHS</th>
      <th colspan="2">Default Test Data</th>
      <th colspan="2">Custom Test Data</th>
    </tr>
    <tr>
        <th>Loss</th>
        <th>Accuracy</th>
        <th>Loss</th>
        <th>Accuracy</th>
    </tr>
    <tr>
        <td>0</td>
        <td>2.4360</td>
        <td>0.1856</td>
        <td>2.2657</td>
        <td>0.0909</td>
    </tr>
    <tr>
        <td>5</td>
        <td>0.3452</td>
        <td>0.8754</td>
        <td>2.8864</td>
        <td>0.4545</td>
    </tr>
    <tr>
        <td>10</td>
        <td>0.3250</td>
        <td>0.8870</td>
        <td>4.0661</td>
        <td>0.4545</td>
    </tr>
    <tr>
        <td>15</td>
        <td>0.3508</td>
        <td>0.8841</td>
        <td>5.0543</td>
        <td>0.2727</td>
    </tr>
    <tr>
        <td>20</td>
        <td>0.3540</td>
        <td>0.8859</td>
        <td>4.5577</td>
        <td>0.4545</td>
    </tr>
    <tr>
        <td>25</td>
        <td>0.3852</td>
        <td>0.8889</td>
        <td>6.8401</td>
        <td>0.4545</td>
    </tr>
    <tr>
        <td>30</td>
        <td>0.3983</td>
        <td>0.8913</td>
        <td>9.4489</td>
        <td>0.4545</td>
    </tr>
    <tr>
        <td>40</td>
        <td>0.4361</td>
        <td>0.8924</td>
        <td>10.1575</td>
        <td>0.4545</td>
    </tr>
    <tr>
        <td>50</td>
        <td>0.5193</td>
        <td>0.8866</td>
        <td>11.4156</td>
        <td>0.4545</td>
    </tr>
    <tr>
        <td>60</td>
        <td>0.5951</td>
        <td>0.8817</td>
        <td>15.1648</td>
        <td>0.0909</td>
    </tr>
    <tr>
        <td>70</td>
        <td>0.6228</td>
        <td>0.8838</td>
        <td>13.7696</td>
        <td>0.2727</td>
    </tr>
    <tr>
        <td>80</td>
        <td>0.6612</td>
        <td>0.8891</td>
        <td>17.1835</td>
        <td>0.3636</td>
    </tr>
    <tr>
        <td>100</td>
        <td>0.8147</td>
        <td>0.8869</td>
        <td>17.8112</td>
        <td>0.5454</td>
    </tr>
</table>

Observations: Loss increases as EPOCHS increase. Loss is significantly higher when testing on a dataset I compiled
myself.

Interpretation: The model's performance improves until some threshold (~10 EPOCHS) after which the longer it is trained, 
the worse it will become. This may suggest over fitting. Since loss can also be
seen to be much higher on the custom dataset, we can infer when testing images from the internet (that do not strictly
adhere to the formatting of the fashion mnist dataset) the model performs worse.

### Experiment: Classifier on MNIST dataset

The overall results are given below. The model was trained with a variable number of epochs and the accuracy and loss
were recorded.

<table>
    <tr>
      <th rowspan="2">EPOCHS</th>
      <th colspan="2">Default Test Data</th>
      <th colspan="2">Custom Test Data</th>
    </tr>
    <tr>
        <th>Loss</th>
        <th>Accuracy</th>
        <th>Loss</th>
        <th>Accuracy</th>
    </tr>
    <tr>
        <td>0</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>5</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>10</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>15</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>20</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>25</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>30</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>40</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>50</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>60</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>70</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>80</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>100</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
</table>

Observations: For the mnist test dataset, loss decreased from 1 to 10 EPOCHS indicating performance improved but at
100 EPOCHS the loss jumped to 59.74 suggesting overfitting. For the custom test dataset, loss continuously increased
to significantly higher levels than the loss of the test dataset. This is likely due to it not following the
same patterns as the downloaded datset (I drew these images myself).

Interpretation: Identical to the Fashion MNIST interpretation. The model performs worse the longer it is trained. This
may suggest over fitting. Since loss can also be seen to be much higher on the custom dataset, we can infer when testing
images from the internet (that do not strictly adhere to the formatting of the mnist dataset) the model performs
worse.

### Comparison and Analysis

![](./readme%20images/Test%20Data%20Fashion%20MNIST%20vs%20MNIST.png)
![](./readme%20images/Custom%20Data%20Fashion%20MNIST%20vs%20MNIST.png)

The model performed better when training with simpler data. This can be seen in the results below where the model
trained on the Fashion MNIST dataset had less accuracy and more loss than the model trained with MNIST.

However, the model trained on MNIST performed significantly worse on the custom datasets provided. This may indicate
that although the model learned patterns of the simpler data much better, it will perform significantly worse on images
that don't follow the learned patterns. This could be due to human error in creation of the datasets; it is also
possible
that the model trained on MNIST is worse at predicting data that it was not trained for, where the Fashion MNIST trained
model did better since the patterns were more complex and likely more generalized.
