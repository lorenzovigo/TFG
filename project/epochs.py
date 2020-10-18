import math
from statistics import mean

import torch

from utils import getHitRatio, getNDCG, getRMSE


def train(model, optimizer, criterion, data_loader, device):
    """
    One epoch training for our given model.
    :param model: Model to be trained.
    :param optimizer: Optimizer to be used in the training.
    :param data_loader: Data loader to be used in the training.
    :param criterion: Criterion to be used in the training, our loss function.
    :param device: Device to be used.
    :return: Average loss computed among all interactions.
    """
    # We tell our model we are training it.
    model.train()
    data_loader.dataset.negative_sampling()
    total_loss = []

    for i, (interactions) in enumerate(data_loader):
        model.zero_grad()

        # We calculate our predictions and the loss value between them and targets
        interactions = interactions.to(device)
        targets = interactions[:, 2]
        predictions = model(interactions[:, :2])

        # We seek to optimize the loss function and do so with the given optimizer
        loss = criterion(predictions, targets.float())
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

    return mean(total_loss)


def test(model, test_set, device, topk=10):
    """
    Carries out the testing of a model.
    :param model: Model to be tested.
    :param test_set: Set with which we will test our model.
    :param device: Device to be used.
    :param topk: Number of recommendations to return. (Top k scores)
    :return: Two metric values (Hit Ratio and NDGC) of the model.
    """
    # Test the HR and NDCG for the model @topK
    # We tell our model we are testing it.
    model.eval()

    HR, NDCG, RMSE = [], [], []

    for user_test in test_set:
        # For each user in the test set we get the ground truth item
        gt_item = user_test[0][1]

        # We compute the predictions using our model and retrieve our recommendations (those with best score)
        predictions = model.predict(user_test, device)
        _, indices = torch.topk(predictions, topk)
        recommend_list = user_test[indices.cpu().detach().numpy()][:, 1]

        # We compute the hit ratio and NDCG
        HR.append(getHitRatio(recommend_list, gt_item))
        NDCG.append(getNDCG(recommend_list, gt_item))
        RMSE.append(getRMSE(predictions))
    return mean(HR), mean(NDCG), math.sqrt(mean(RMSE))


def run(model, optimizer, criterion, data_loader, full_dataset, writer, device, tb=True, epochs=100, top_k=10):
    for epoch_i in range(epochs):
        # We train our model in every epoch and compute our metrics afterwards.
        # TODO data_loader.dataset.negative_sampling() dentro de train o aquí, donde es correcto?
        train_loss = train(model, optimizer, criterion, data_loader, device)
        hr, ndcg, rmse = test(model, full_dataset.test_set, device, topk=top_k)

        print(f'epoch {epoch_i}:')
        print(
            f'training loss = {train_loss:.4f} | Eval: HR@{top_k} = {hr:.4f}, NDCG@{top_k} = {ndcg:.4f}, RMSE@{top_k} = {rmse:.4f} ')
        print('\n')
        if tb:
            writer.add_scalar('train/loss', train_loss, epoch_i)
            writer.add_scalar('eval/HR@{top_k}', hr, epoch_i)
            writer.add_scalar('eval/NDCG@{top_k}', ndcg, epoch_i)
            writer.add_scalar('eval/RMSE@{top_k}', rmse, epoch_i)
