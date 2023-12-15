import torch
import numpy as np


def max_sharpe(y_return, weights):
    print('\n-------------------\n')
    print('y_return.shape: ', y_return.shape)
    print('weights.shape: ', weights.shape)
    weights = torch.unsqueeze(weights, 1)
    meanReturn = torch.unsqueeze(torch.mean(y_return, axis=1), 2)
    print('meanReturn.shape: ', meanReturn.shape)
    covmat = torch.Tensor([np.cov(batch.cpu().T, ddof=0) for batch in y_return]).to("cuda")
    print('covmat.shape: ', covmat.shape)
    portReturn = torch.matmul(weights, meanReturn)
    print('portReturn.shape: ', portReturn.shape)
    portVol = torch.matmul(
        weights, torch.matmul(covmat, torch.transpose(weights, 2, 1))
    )
    print('portVol.shape: ', portVol.shape)
    objective = (portReturn * 12 - 0.02) / (torch.sqrt(portVol * 12))
    print('objective.shape: ', objective.shape)
    return -objective.mean()


def equal_risk_parity(y_return, weights):
    B = y_return.shape[0]
    F = y_return.shape[2]
    weights = torch.unsqueeze(weights, 1).to("cuda")
    covmat = torch.Tensor(
        [np.cov(batch.cpu().T, ddof=0) for batch in y_return]
    )  # (batch, 50, 50)
    covmat = covmat.to("cuda")
    sigma = torch.sqrt(
        torch.matmul(weights, torch.matmul(covmat, torch.transpose(weights, 2, 1)))
    )
    mrc = (1 / sigma) * (covmat @ torch.transpose(weights, 2, 1))
    rc = weights.view(B, F) * mrc.view(B, F)
    target = (torch.ones((B, F)) * (1 / F)).to("cuda")
    risk_diffs = rc - target
    sum_risk_diffs_squared = torch.mean(torch.square(risk_diffs))
    return sum_risk_diffs_squared


if __name__ == "__main__":
    pass
