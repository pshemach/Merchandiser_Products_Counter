from torch import nn

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss = (label * euclidean_distance.pow(2) +
                (1 - label) * nn.functional.relu(self.margin - euclidean_distance).pow(2))
        return loss.mean()

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        positive_distance = nn.functional.pairwise_distance(anchor, positive)
        negative_distance = nn.functional.pairwise_distance(anchor, negative)
        loss = nn.functional.relu(positive_distance - negative_distance + self.margin)
        return loss.mean()