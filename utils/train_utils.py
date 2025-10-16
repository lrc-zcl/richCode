def calculate_detailed_accuracy(predictions, targets):
    """
    计算详细的准确率信息

    Returns:
        batch_avg_acc: batch平均准确率
        position_acc: 每个位置的准确率 [7]
    """
    pred_classes = predictions.argmax(dim=-1)  # [batch_size, 7]
    correct = (pred_classes == targets).float()  # [batch_size, 7]
    sample_accuracies = correct.sum(dim=1) / 7.0  # [batch_size]
    batch_avg_acc = sample_accuracies.mean().item()
    position_acc = correct.mean(dim=0)
    return batch_avg_acc, position_acc.cpu().numpy()  # 每个位置上的准确率
