import torch

def compute_hr_ndcg(model, test_loader, device, k=10):
    model.eval()
    model.to(device)
    hr_total, ndcg_total = 0, 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            scores = model(inputs)
            _, indices = scores.topk(k, dim=1)

            for i in range(len(targets)):
                target = targets[i].item()
                top_k = indices[i].tolist()
                if target in top_k:
                    hr_total += 1
                    rank = top_k.index(target)
                    ndcg_total += 1 / torch.log2(torch.tensor(rank + 2.0))

    hr = hr_total / len(test_loader.dataset)
    ndcg = ndcg_total / len(test_loader.dataset)
    return hr.item(), ndcg.item()