import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tqdm import tqdm


def train(space, model, train_loader, device):
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=space['lr'])

    if space['loss_type'] == 'CL':
        criterion = nn.BCEWithLogitsLoss(reduction='sum')
    elif space['loss_type'] == 'SL':
        criterion = nn.MSELoss(reduction='sum')
    else:
        raise ValueError(f"Invalid loss type: {space['loss_type']}")

    last_loss = 0.
    early_stopping_counter = 0
    stop = False

    model.train()
    for epoch in range(1, space['epochs'] + 1):
        if stop:
            break
        current_loss = 0.
        # set process bar display
        pbar = tqdm(train_loader)
        pbar.set_description(f'[Epoch {epoch:03d}]')
        for user, item, label in pbar:
            user = user.to(device)
            item = item.to(device)
            label = label.to(device)

            model.zero_grad()
            prediction = model(user, item)
            loss = criterion(prediction, label)
            # TODO: IMPLEMEMNT REGULARIZATIONS

            if torch.isnan(loss):
                raise ValueError(f'Loss=Nan or Infinity: current settings does not fit the recommender')

            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())
            current_loss += loss.item()

        if (last_loss < current_loss):
            early_stopping_counter += 1
            if early_stopping_counter == 10:
                print('Satisfy early stop mechanism')
                stop = True
        else:
            early_stopping_counter = 0
        last_loss = current_loss

    # TODO: EVALUATION OF VALIDATION AND RETURN METRICS
    # TODO: TENSORBOARD HR and NDCG
    # TODO: TENSORBOARD LOSS last_loss
