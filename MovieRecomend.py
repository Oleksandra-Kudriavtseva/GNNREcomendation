import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from sklearn.preprocessing import LabelEncoder

#дані (тільки айді користувачів та фільмів і оцінки)
ratings = pd.read_csv("data/ratings.csv")
ratings = ratings.head(20000)


user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()

ratings["user"] = user_encoder.fit_transform(ratings["userId"])
ratings["movie"] = movie_encoder.fit_transform(ratings["movieId"])

num_users = ratings["user"].nunique()
num_movies = ratings["movie"].nunique()

ratings["movie"] = ratings["movie"] + num_users

src = torch.tensor(ratings["user"].values)
dst = torch.tensor(ratings["movie"].values)

graph = dgl.graph((src, dst))
graph = dgl.add_self_loop(graph)

num_nodes = num_users + num_movies


features = torch.randn(num_nodes, 16)

labels = torch.tensor(ratings["rating"].values, dtype=torch.float32)




#тут для GCN, бо він простіший, але коли буде більше типів вузлів буду використовувати R GCN
class GCN(nn.Module):
    def __init__(self, in_feats, hidden, out_feats):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_feats, hidden)
        self.conv2 = dglnn.GraphConv(hidden, out_feats)

    def forward(self, g, x):
        x = self.conv1(g, x)
        x = F.relu(x)
        x = self.conv2(g, x)
        return x

model = GCN(16, 32, 16)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()


for epoch in range(30):
    model.train()
    node_embeddings = model(graph, features)


    user_emb = node_embeddings[src]
    movie_emb = node_embeddings[dst]


    preds = (user_emb * movie_emb).sum(dim=1)
    loss = loss_fn(preds, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"ітерація {epoch+1}, похибка: {loss.item():.4f}")
