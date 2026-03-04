import pandas as pd
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn

moviesGenre = pd.read_csv("movies.csv")
movieRatings = pd.read_csv("small_movie_user_rating.csv", nrows=500)

booksGenre = []
with open("goodreads_book_genres_initial.json", "r", encoding="utf-8") as f:
    for line in f:
        booksGenre.append(json.loads(line))

bookRatings = pd.read_csv("small_book_user_rating.csv", nrows=500)
games = pd.read_csv("vgsales.csv")
games = games.reset_index()
games.rename(columns={"index": "gameId"}, inplace=True)

numFakeUsers = 50
fakeGameRatings = []
gameIdsList = games["gameId"].values
for u in range(numFakeUsers):
    for _ in range(random.randint(5, 15)):
        game = random.choice(gameIdsList)
        rating = random.uniform(0.0, 5.0)
        fakeGameRatings.append((u, game, rating))

gameRatings = pd.DataFrame(fakeGameRatings, columns=["userId", "gameId", "rating"])

userIds = (
    set(movieRatings["userId"].unique())
    | set(bookRatings["user_id"].unique())
    | set(gameRatings["userId"].unique())
)

movieIds = set(moviesGenre["movieId"].unique())
bookIds = set(bookRatings["book_id"].unique())
gameIds = set(games["gameId"].unique())

userMap = {id_: i for i, id_ in enumerate(sorted(userIds))}
movieMap = {id_: i for i, id_ in enumerate(sorted(movieIds))}
bookMap = {id_: i for i, id_ in enumerate(sorted(bookIds))}
gameMap = {id_: i for i, id_ in enumerate(sorted(gameIds))}

allGenres = set()

movieGenres = {}
for _, row in moviesGenre.iterrows():
    genres = row["genres"].split("|")
    movieGenres[row["movieId"]] = genres
    allGenres.update(genres)

bookGenres = {}
for b in booksGenre:
    bookId = int(b["book_id"])
    if bookId in bookIds:
        genres = list(b["genres"].keys())
        bookGenres[bookId] = genres
        allGenres.update(genres)

gameGenres = {}
for _, row in games.iterrows():
    if pd.notna(row["Genre"]):
        genre = row["Genre"].strip()
        gameGenres[row["gameId"]] = [genre]
        allGenres.add(genre)

genreMap = {g: i for i, g in enumerate(allGenres)}

edges = {}

userMovieSrc, userMovieDst = [], []
for _, row in movieRatings.iterrows():
    if row["movieId"] in movieMap:
        userMovieSrc.append(userMap[row["userId"]])
        userMovieDst.append(movieMap[row["movieId"]])
edges[('user', 'ratesMovie', 'movie')] = (torch.tensor(userMovieSrc), torch.tensor(userMovieDst))

userBookSrc, userBookDst = [], []
for _, row in bookRatings.iterrows():
    if row["book_id"] in bookMap:
        userBookSrc.append(userMap[row["user_id"]])
        userBookDst.append(bookMap[row["book_id"]])
edges[('user', 'ratesBook', 'book')] = (torch.tensor(userBookSrc), torch.tensor(userBookDst))

userGameSrc, userGameDst = [], []
for _, row in gameRatings.iterrows():
    if row["gameId"] in gameMap:
        userGameSrc.append(userMap[row["userId"]])
        userGameDst.append(gameMap[row["gameId"]])
edges[('user', 'ratesGame', 'game')] = (torch.tensor(userGameSrc), torch.tensor(userGameDst))

def addGenreEdges(itemGenres, itemMap, edgeName, nodeType):
    src, dst = [], []
    for itemId, genres in itemGenres.items():
        if itemId in itemMap:
            for g in genres:
                src.append(itemMap[itemId])
                dst.append(genreMap[g])
    edges[(nodeType, edgeName, 'genre')] = (torch.tensor(src), torch.tensor(dst))

addGenreEdges(movieGenres, movieMap, 'hasGenre', 'movie')
addGenreEdges(bookGenres, bookMap, 'hasGenre', 'book')
addGenreEdges(gameGenres, gameMap, 'hasGenre', 'game')

g = dgl.heterograph(edges)
print(g)


trainUsers, trainItems, trainTypes, trainRatings = [], [], [], []

def addTrainingSamples(ratingsDf, userCol, itemCol, itemMap, itemType):
    for _, row in ratingsDf.iterrows():
        if row[itemCol] in itemMap:
            trainUsers.append(userMap[row[userCol]])
            trainItems.append(itemMap[row[itemCol]])
            trainTypes.append(itemType)
            trainRatings.append(float(row["rating"]))

addTrainingSamples(movieRatings, 'userId', 'movieId', movieMap, 'movie')
addTrainingSamples(bookRatings, 'user_id', 'book_id', bookMap, 'book')
addTrainingSamples(gameRatings, 'userId', 'gameId', gameMap, 'game')

trainUsers = torch.tensor(trainUsers)
trainItems = torch.tensor(trainItems)
trainRatings = torch.tensor(trainRatings, dtype=torch.float32)

hiddenDim = 32
for nodeType in g.ntypes:
    g.nodes[nodeType].data['feat'] = torch.randn(g.num_nodes(nodeType), hiddenDim)

class HeteroGNN(nn.Module):
    def __init__(self, hiddenDim, relationNames, ntypes):
        super().__init__()
        self.ntypes = ntypes
        self.conv1 = dglnn.HeteroGraphConv(
            {rel: dglnn.SAGEConv(hiddenDim, hiddenDim, 'mean') for rel in relationNames},
            aggregate='sum'
        )
        self.conv2 = dglnn.HeteroGraphConv(
            {rel: dglnn.SAGEConv(hiddenDim, hiddenDim, 'mean') for rel in relationNames},
            aggregate='sum'
        )

    def forward(self, graph, features):
        h = self.conv1(graph, features)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        for nt in self.ntypes:
            if nt not in h:
                h[nt] = features[nt]
        return h

class RatingPredictor(nn.Module):
    def __init__(self, hiddenDim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hiddenDim * 2, hiddenDim),
            nn.ReLU(),
            nn.Linear(hiddenDim, 1)
        )

    def forward(self, userEmb, itemEmb):
        x = torch.cat([userEmb, itemEmb], dim=1)
        return self.mlp(x).squeeze()

gnn = HeteroGNN(hiddenDim, g.etypes, g.ntypes)
predictor = RatingPredictor(hiddenDim)

optimizer = torch.optim.Adam(list(gnn.parameters()) + list(predictor.parameters()), lr=0.01)

for epoch in range(50):
    gnn.train()
    predictor.train()

    embeddings = gnn(g, {nt: g.nodes[nt].data['feat'] for nt in g.ntypes})

    userEmbList, itemEmbList = [], []
    for i in range(len(trainUsers)):
        userEmb = embeddings['user'][trainUsers[i]]
        itemEmb = embeddings[trainTypes[i]][trainItems[i]]
        userEmbList.append(userEmb)
        itemEmbList.append(itemEmb)

    userEmbTensor = torch.stack(userEmbList)
    itemEmbTensor = torch.stack(itemEmbList)

    predictions = predictor(userEmbTensor, itemEmbTensor)
    loss = F.mse_loss(predictions, trainRatings)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

#збереження
torch.save(gnn.state_dict(), "gnn_model.pth")
torch.save(predictor.state_dict(), "predictor_model.pth")



gnn.load_state_dict(torch.load("gnn_model.pth", map_location=torch.device('cpu')))
predictor.load_state_dict(torch.load("predictor_model.pth", map_location=torch.device('cpu')))
gnn.eval()
predictor.eval()









#рекомендація
def recommend_for_user(user_id, top_k=3):
    if user_id not in userMap:
        return {"error": "User not found"}

    with torch.no_grad():
        embeddings = gnn(g, {nt: g.nodes[nt].data['feat'] for nt in g.ntypes})

        user_idx = userMap[user_id]
        user_emb = embeddings['user'][user_idx]

        recommendations = {}

        for item_type, item_map in [('movie', movieMap),
                                    ('book', bookMap),
                                    ('game', gameMap)]:

            scores = []
            for original_id, mapped_id in item_map.items():
                item_emb = embeddings[item_type][mapped_id]
                score = predictor(
                    user_emb.unsqueeze(0),
                    item_emb.unsqueeze(0)
                ).item()

                scores.append((int(original_id), float(score)))

            scores.sort(key=lambda x: x[1], reverse=True)
            recommendations[item_type] = scores[:top_k]

        return recommendations








print(recommend_for_user(0))