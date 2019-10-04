import torch
import torch.nn as nn
import torch.nn.functional as F

class MovePredict(nn.Module):
    """
    Move予測するネットワーク
    """
    def __init__(self, num_layers=1, hidden_size=32, input_shape=29):
        super(MovePredict, self).__init__()
        import pickle
        with open('/mnt/aoni02/katayama/nwjc_sudachi_full_abc_w2v/embedding_metrix.pkl', "rb") as f:
            embedding_metrix = pickle.load(f)
        self.emb = nn.Embedding(num_embeddings=embedding_metrix.shape[0]+1,
                                embedding_dim=embedding_metrix.shape[1])

        self.emb.from_pretrained(torch.Tensor(embedding_metrix))

        self.lstm_word = torch.nn.LSTM(
            input_size=324,#88,  # 入力size
            hidden_size=hidden_size,  # 出力size
            num_layers=num_layers,  # stackする数
            dropout=0.5,
            batch_first=True,  # given_data.shape = (batch , frames , input_size)
            bidirectional=False
        )

        self.lstm_sent = torch.nn.LSTM(
            input_size= hidden_size+1,  # 入力size
            hidden_size= hidden_size,  # 出力size
            num_layers=num_layers,  # stackする数
            dropout=0.5,
            batch_first=True,  # given_data.shape = (batch , frames , input_size)
            bidirectional=False
        )
        self.fc1 = nn.Linear(12, 12)
        self.fc2 = nn.Linear(input_shape, 12)
        self.fc3 = nn.Linear(300,64)
        self.fc4 = nn.Linear(hidden_size,hidden_size)
        self.fc5 = nn.Linear(hidden_size,5)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, x, xp, xp2, xi, h0 = None, c0 = None):
        x = self.emb(x)
        #x = x.unsqueeze(0)
        #print(x.size())
        assert len(x.shape) == 3, print('data shape is incorrect.')
        batch_size, frames = x.shape[:2]

        _,_=self.reset_state(batch_size)

        xp = xp.view(len(xp)*frames, -1)
        xp2 = xp2.view(len(xp2)*frames, -1)
        #x = F.dropout(F.relu(self.fc3(x.view(len(x)*frames,-1))),p=0.5)
        xp = F.dropout(F.relu(self.fc1(xp)),p=0.5)
        xp2 = F.dropout(F.relu(self.fc2(xp2)),p=0.5)
        #x = x.view(-1,frames,64)
        xp = xp.view(-1,frames,12)
        xp2 = xp2.view(-1, frames, 12)
        x = torch.cat((x,xp,xp2),dim=2)
        h, _ = self.lstm_word(x, (self.h0, self.c0))
        if h0 is None:
            h0, c0 = self.reset_state(batch_size)
            print('reset state!!')
        xi = xi.view(1,1,1)
        
        #h = torch.cat((h[:,-1:,:],xi),dim=2) #言語は最終層の出力を抽出
        #h, (h0, c0) = self.lstm_sent(h, (h0, c0))
        #h = F.dropout(F.relu(self.fc4(h[:, -1, :])), p=0.5)
        y = self.fc5(h[:,-1,:])

        return y, h0, c0

    def reset_state(self, batch_size):
        self.h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        self.c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        return self.h0, self.c0
