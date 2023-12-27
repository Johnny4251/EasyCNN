from EasyModel import EasyCNN
from EasyModel import ExampleModelV2 as EM

# Color image(128x128) w/ class count of 3
model = EM(3, 3, 128)

# Using the default train_transform in EasyCNN
easycnn = EasyCNN(model, train_transform=EasyCNN.example_train_transform)
easycnn.train_model(1, 32, save_model=True, model_name="ExampleModel")


