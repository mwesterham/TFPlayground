# Text Classifier Experiment

The text classifier experiment is based off of Tensorflow's documentation
found [here](https://www.tensorflow.org/tutorials/keras/text_classification). The aim of this experiment is
to train a binary classifier that performs sentiment analysis on the Large Movie Review Dataset (aclImdb).
Additionally, I provide model checkpointing and workflow abstraction to demonstrate these common ML processes.

### Experiment

The model was trained for 25 EPOCHs and the accuracy and loss can be seen below:

![](./readme%20images/Training%20and%20validation%20loss-25EPOCHS-1662146481.png)
![](./readme%20images/Training%20and%20validation%20accuracy-25EPOCHS-1662146482.png)

### Observation and Analysis

Loss and accuracy improve up to a certain point (approximately 10 EPOCHs) after which the accuracy
plateaus and the validation loss begins to worsen. This may suggest over-fitting. Training a model for a long time
generally leads to over-fitting because the model starts learning trends that do not generalize the data. This can
be amended a couple of ways one of which is to find better data that represents the general trends in the dataset.
An alternative is to make the model smaller (or limit how much information it can memorize).