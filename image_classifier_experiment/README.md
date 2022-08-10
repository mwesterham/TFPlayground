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
      <th colspan="5">Fashion MNIST Dataset</th>
    </tr>
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
the worse it will become. This may suggest over-fitting. Loss can also be seen to be increasing with the custom dataset, 
we can observe when testing
images from the internet (that do not strictly adhere to the formatting of the fashion mnist dataset) the model performs
worse and worse as it is trained more. This makes sense since these custom images have variations that the model
has never seen and was never trained for.

### Experiment: Classifier on MNIST dataset

The model was trained with a variable number of epochs and the accuracy and loss
were recorded for both default and custom datasets. The overall results are given below. 

<table>
    <tr>
      <th colspan="5">MNIST Dataset</th>
    </tr>
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
        <td>2.3050</td>
        <td>0.1516</td>
        <td>2.2848</td>
        <td>0.2727</td>
    </tr>
    <tr>
        <td>5</td>
        <td>0.0821</td>
        <td>0.9753</td>
        <td>3.1627</td>
        <td>0.6363</td>
    </tr>
    <tr>
        <td>10</td>
        <td>0.0804</td>
        <td>0.9771</td>
        <td>3.5345</td>
        <td>0.7272</td>
    </tr>
    <tr>
        <td>15</td>
        <td>0.0947</td>
        <td>0.9772</td>
        <td>5.7048</td>
        <td>0.7272</td>
    </tr>
    <tr>
        <td>20</td>
        <td>0.1042</td>
        <td>0.9797</td>
        <td>6.5124</td>
        <td>0.7272</td>
    </tr>
    <tr>
        <td>25</td>
        <td>0.1160</td>
        <td>0.9797</td>
        <td>8.5554</td>
        <td>0.7272</td>
    </tr>
    <tr>
        <td>30</td>
        <td>0.1439</td>
        <td>0.9764</td>
        <td>7.5284</td>
        <td>0.8181</td>
    </tr>
    <tr>
        <td>40</td>
        <td>0.1577</td>
        <td>0.9779</td>
        <td>8.2524</td>
        <td>0.7272</td>
    </tr>
    <tr>
        <td>50</td>
        <td>0.1977</td>
        <td>0.9763</td>
        <td>13.0131</td>
        <td>0.7272</td>
    </tr>
    <tr>
        <td>60</td>
        <td>0.1721</td>
        <td>0.9768</td>
        <td>9.8029</td>
        <td>0.8181</td>
    </tr>
    <tr>
        <td>70</td>
        <td>0.2294</td>
        <td>0.9753</td>
        <td>12.9371</td>
        <td>0.7272</td>
    </tr>
    <tr>
        <td>80</td>
        <td>0.1895</td>
        <td>0.9787</td>
        <td>9.3681</td>
        <td>0.7272</td>
    </tr>
    <tr>
        <td>100</td>
        <td>0.2563</td>
        <td>0.9775</td>
        <td>13.8714</td>
        <td>0.6363</td>
    </tr>
</table>

Observations: For the mnist test dataset, loss decreased from 0 to 10 EPOCHS indicating performance improved but after
this range the loss only rose, which suggests over-fitting. For the custom test dataset, loss continuously increased
to significantly higher levels than the loss of the test dataset. This is likely due to it not following the
same patterns as the downloaded dataset (I drew these images myself).

Interpretation: Identical to the Fashion MNIST interpretation. The model performs worse the longer it is trained. 
Loss can also be seen to be increasing with the custom dataset, we can observe when testing
images from the internet (that do not strictly adhere to the formatting of the mnist dataset) the model performs
worse and worse as it is trained more.

### Comparison and Analysis

![](./readme%20images/Test%20Data%20Fashion%20MNIST%20vs%20MNIST.png)
![](./readme%20images/Custom%20Data%20Fashion%20MNIST%20vs%20MNIST.png)

The model performed better when training with the MNIST dataset. This can be seen in the results where the model
trained on the Fashion MNIST dataset had higher loss than the model trained on the MNIST dataset. This can be observed 
in both the default and custom test datasets.

Both models trained on their datasets performed poorly on the custom data I provided. This is likely due to the fact 
that these custom images did not closely adhere to the patterns of images in the training dataset. Additionally, 
as more EPOCHs passed, the models did worse and worse on the custom dataset. This makes sense since as over-fitting 
became more prevalent, the custom images' differences with the training set would also become more obvious.
