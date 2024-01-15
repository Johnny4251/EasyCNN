from EasyModel import EasyCNN
from EasyModel import ExampleModelV2 as EM
from EasyModel import VGG16

# Color image(128x128) w/ class count of 3
model = EM(3, 3, 128)
vgg_model = VGG16(5)

# Using the default train_transform in EasyCNN
#easycnn = EasyCNN(model, train_transform=EasyCNN.example_train_transform)
#easycnn.train_model(1, 32, save_model=True, model_name="ExampleModel")

# Training a model using built-in vgg16 model
vgg_trainer = EasyCNN(vgg_model, EasyCNN.vgg_train_transform, grayscale=False)
vgg_trainer.train_model(total_epochs=1, batch_size=64, save_model=True, model_name="CustomVGGModel")