import pandas as pd
import string

import torch
from pytoune.framework import Model
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from dataset import EmojifyDataset
from constants import LABELS, PAD
from glove import Glove
from module import Emojify
import matplotlib.pyplot as plt

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    glove = Glove()
    glove.summary()
    train_dataset = EmojifyDataset('data/train_emoji.csv', glove)
    train = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=train_dataset.collate_fn)
    test_dataset = EmojifyDataset('data/test_emoji.csv', glove)
    test = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=test_dataset.collate_fn)
    module = Emojify(glove)
    module.summary()
    optimizer = Adam(filter(lambda p: p.requires_grad, module.parameters()))
    criterion = CrossEntropyLoss(ignore_index=PAD)
    model = Model(module, optimizer, criterion, metrics=['accuracy']).to(device)
    history = model.fit_generator(train, test, epochs=50)

    module.eval()  # setting model to eval mode disables drop out
    wrong_x = []
    wrong_y = []
    real_y = []
    for x, y in test:
        x, y = x.to(device), y.to(device)
        y_hat = module(x)
        y_hat_labels = y_hat.argmax(dim=-1)
        wrong_x.append(x[y_hat_labels != y])
        wrong_y.append(y_hat_labels[y_hat_labels != y])
        real_y.append(y[y_hat_labels != y])
    wrong_x = torch.cat(wrong_x, dim=0)
    wrong_y = torch.cat(wrong_y, dim=0)
    real_y = torch.cat(real_y, dim=0)
    print(f'Mislabeled({len(wrong_x)}):')
    for sentence, predicted_label, actual_label in zip(glove.to_sentences(wrong_x[:, :-1]), wrong_y, real_y):
        print(f'{sentence} -> predicted: {LABELS[predicted_label.item()]} actual: {LABELS[actual_label.item()]}')

    while True:
        sentence = input('Your own sentence:').translate(str.maketrans('', '', string.punctuation))
        if sentence == '':
            break
        x = test_dataset.sentence_to_tensor(sentence, device).unsqueeze(0)
        y_hat = module(x)
        print(glove.to_sentence(x[0][:-1]) + ' ' + LABELS[y_hat[0].argmax(dim=-1).item()])

    pd.DataFrame(history).set_index('epoch').plot(subplots=True)
    plt.show()
