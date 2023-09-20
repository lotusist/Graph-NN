import time
import math
import torch
import torch.nn.functional as F
from torch import optim
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.loader import NeighborLoader, HGTLoader
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

torch.manual_seed(0)

###### Adjust the inputs ######

PATH_load = './data_lotus_20230828_USA.pt'
PATH_save = './'

MODEL_VERSION_NAME = 'baseline_HGNN'

device = 'cuda:0'

BatchSize = 128
TRAIN_EPOCHS = 5

###### End of adjustable inputs ######


def display_data_types(data):
    # Display data types for node features
    print("Node Features (x):")
    for node_type, x in data.x_dict.items():
        print(f"  {node_type}: {x.dtype}")
    
    # Display data types for edge indices
    print("\nEdge Indices:")
    for edge_type, edge_index in data.edge_index_dict.items():
        print(f"  {edge_type}: {edge_index.dtype}")
    
    # If available, display data types for node labels
    if hasattr(data, 'y_dict'):
        print("\nNode Labels (y):")
        for node_type, y in data.y_dict.items():
            print(f"  {node_type}: {y.dtype}")


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


def train(model, n_iters, print_every=1000, plot_every=1):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  
    plot_loss_total = 0
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    pltcount = 0
    prtcount = 0
    cp = 0

    for iter in range(1, n_iters+1): 
        print(f'EPOCH {iter} ...')
        for batch in loader:
            pltcount += 1
            prtcount += 1
            model.zero_grad()
            batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict)
            loss = F.cross_entropy(out['org'][:BatchSize], batch['org'].y[:BatchSize])#.to(device)
            loss.backward()
            optimizer.step()

            ls = loss.detach().item()
            print_loss_total +=ls
            plot_loss_total += ls
        
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / prtcount
            print_loss_total = 0 # resume 
            prtcount = 0 # resume
            print('%s (%d %d%%) %f' % (timeSince(start, iter / n_iters),
                                            iter, 
                                            iter / n_iters * 100, 
                                            print_loss_avg))
        
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / pltcount
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0 # resume
            pltcount = 0 # resume

    return plot_losses


def showPlot(points, _save_path):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=100)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig(_save_path + 'showPlot.png')
    # plt.show()


##### main #####

data = torch.load(PATH_load)
data = T.ToUndirected()(data)
data = T.RandomNodeSplit(num_val = 0.1, num_test = 0.1)(data)

print(data)
display_data_types(data)

model = GNN(hidden_channels=64, out_channels=2)
model = to_hetero(model, data.metadata(), aggr='sum')

print(model)

loader = NeighborLoader(
    data,
    num_neighbors=[5],
    batch_size=128,
    input_nodes=('org', data['org'].train_mask))

next(iter(loader))

model.to(device)
loss = train(model, n_iters = TRAIN_EPOCHS)
showPlot(loss, PATH_save)

torch.save(model.state_dict(), PATH_save + MODEL_VERSION_NAME + '.pth')
print(f'MODEL VERSION NAME : {MODEL_VERSION_NAME}\n')
print(f'{MODEL_VERSION_NAME}.pt has been saved!')

# model evaluation

model = model.to(device)
data = data.to(device)

model.eval()
pred_logits = model(data.x_dict, data.edge_index_dict)
pred_all = pred_logits['org'].argmax(dim=1)

pred = pred_all[data['org'].test_mask]
true_labels = data['org'].y[data['org'].test_mask]

# Compute accuracy
correct = (pred == true_labels).sum()
acc = int(correct) / len(true_labels)
print(f'Accuracy: {acc:.4f}')

# Compute F1 score
f1 = f1_score(true_labels.cpu().numpy(), pred.cpu().numpy(), average='macro')
print(f'F1 Score (Macro): {f1:.4f}')

# Compute confusion matrix
conf_matrix = confusion_matrix(true_labels.cpu().numpy(), pred.cpu().numpy())
print('Confusion Matrix:')
print(conf_matrix)
