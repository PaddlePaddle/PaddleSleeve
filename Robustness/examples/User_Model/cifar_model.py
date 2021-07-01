""" Example of using paddle to train a classification model on the cifar10 dataset """

import paddle
from paddle.metric import Accuracy
from paddle.vision import transforms as T


def show_image():
    import matplotlib.pyplot as plt
    plt.axis('off')
    test = paddle.vision.datasets.Cifar10(mode='test')
    train_data0, train_label_0 = test[2][0], test[2][1]
    plt.imshow(train_data0)
    plt.show()
    print('train_data0 label is: ' + str(train_label_0))
    exit(-1)


# show_image()


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
transform_train = T.Compose([T.Resize((32, 32)),
                             T.Transpose(),
                             T.Normalize(
                                 mean=[0, 0, 0],
                                 std=[255, 255, 255]),
                             T.Normalize(mean=MEAN,
                                         std=STD)
                             ])
transform_eval = T.Compose([T.Resize((32, 32)),
                            T.Transpose(),
                            T.Normalize(
                                mean=[0, 0, 0],
                                std=[255, 255, 255]),
                            T.Normalize(mean=MEAN,
                                        std=STD)
                            ])

print('download training data and load training data')
train_dataset = paddle.vision.datasets.Cifar10(mode='train', transform=transform_train)
test_dataset = paddle.vision.datasets.Cifar10(mode='test', transform=transform_eval)
print('load finished')

network = paddle.vision.models.resnet50(pretrained=True, num_classes=10)
model = paddle.Model(network)
# model.summary((-1, 3, 32, 32))

optim = paddle.optimizer.Adam(learning_rate=0.0005, parameters=model.parameters())

model.prepare(
    optim,
    paddle.nn.CrossEntropyLoss(),
    Accuracy()
)

# model.load('checkpoint/test')

model.fit(train_dataset,
          epochs=30,
          batch_size=64,
          verbose=1,
          save_dir='./cifar',
          save_freq=2
          )

model.save('checkpoint/test')  # save for training

model.evaluate(test_dataset, batch_size=64, verbose=1)
