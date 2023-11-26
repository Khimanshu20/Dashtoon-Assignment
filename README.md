# Dashtoon-Assignment

A loss function (nn.CrossEntropyLoss()) and an optimizer (optim.SGD) are defined.
The optimizer is set up to update the parameters of a neural network model (vgg19) using stochastic gradient descent (SGD) with momentum.
The training loop runs for a specified number of epochs (num_epochs), which defines how many times the entire dataset is passed through the model for training.
Within each epoch, the model is set to training mode (vgg19.train()).
The training dataset is iterated through using train_loader, which loads batches of input data (inputs) and corresponding labels (labels).
For each batch of data:
The model's gradients are zeroed (optimizer.zero_grad()).
Forward pass: The input data is passed through the model (vgg19(inputs)) to obtain predictions.
Loss calculation: The loss function compares the model predictions to the true labels to compute the loss.
Backward pass: Gradients of the loss with respect to the model parameters are calculated (loss.backward()).
Optimizer step: The optimizer updates the model's parameters based on the computed gradients (optimizer.step()).
Running loss: The loss for each batch is accumulated to track the total loss for the epoch.
Epoch Summary:
At the end of each epoch, the average training loss across all batches is computed (running_loss/len(train_loader)) and printed as an indicator of the model's performance on the training data for that epoch.
