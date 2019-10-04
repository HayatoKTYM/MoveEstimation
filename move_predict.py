from nlp_utils import *
from cnn_model import MovePredict
import numpy as np
import glob
import torch
import torch.optim as optim
import torch.nn as nn
from collections import Counter

def train(net, dataloaders_dict, criterion, optimizer, num_epochs):
    # 学習ループ
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    net.to(device)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-------------')
        # epochごとの訓練と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # モデルを訓練モードに
            else:
                net.eval()  # モデルを検証モードに

            epoch_loss = 0.0  # epochの損失和
            epoch_corrects = 0  # epochの正解数

            if (epoch == 0) and (phase == 'train'):  # 未学習時の検証性能を確かめるため、epoch=0の訓練は省略
                continue
            
            cnt = 0
            for inputs,inputs2,inputs3,inputs4, labels in dataloaders_dict[phase]:
                inputs = inputs.to(device)
                inputs2 = inputs2.to(device)
                inputs3 = inputs3.to(device)
                inputs4 = inputs4.to(device)
                if labels.data.numpy()[0] == -1:
                    h, c = net.reset_state(1)
                    print('reset state! conversation changed!')
                    continue
                labels = labels.to(device)
                if cnt == 0 :
                    out, h, c = net(inputs,inputs2,inputs3,inputs4, None, None)  # 順伝播
                    cnt += 1
                else:
                    #out, h, c = net(inputs,inputs2,inputs3,inputs4, h.detach(), c.detach())  # 順伝播
                    out, h, c = net(inputs,inputs2,inputs3,inputs4, h, c)  # 順伝播
                    cnt += 1
                #weights = torch.tensor([2.0,3.01,15.0,9.5,9.5]).to(device)
                loss = criterion(out, labels) if cnt % 64 == 1 else loss + criterion(out, labels) # ロスの計算
                _, preds = torch.max(out, 1)  # ラベルを予測
                #print(loss.data,end=',')
                if cnt % 64 == 0:
                    # 50会入力したら纏めて誤差逆伝播
                    if phase == 'train':  # 訓練時はバックプロパゲーション
                        #loss = loss / 64
                        optimizer.zero_grad()  # 勾配の初期化
                        loss.backward()  # retain_graph=True) # 勾配の計算
                        optimizer.step()  # パラメータの更新
                        #cnt = 0
                        h = h.detach()
                        c = c.detach()

                    epoch_loss += loss.item() * inputs.size(0)  # lossの合計を更新
                epoch_corrects += torch.sum(preds == labels.data)  # 正解数の合計を更新

            # epochごとのlossと正解率を表示
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print('')

    y_true, y_pred = np.array([]), np.array([])
    cnt = 0
    for inputs,inputs2,inputs3,inputs4, labels in dataloaders_dict['test']:
        inputs = inputs.to(device)
        inputs2 = inputs2.to(device)
        inputs3 = inputs3.to(device)
        inputs4 = inputs4.to(device)
        if labels.data.numpy()[0] == -1:
            h, c = net.reset_state(1)
            print('reset state! conversation changed!')
            continue
        if cnt == 0:
            out, h, c = net(inputs,inputs2,inputs3,inputs4, None, None)
            cnt += 1
        else:
            out, h, c = net(inputs,inputs2,inputs3,inputs4, h.detach(), c.detach())  # 順伝播
        # loss = criterion(out, labels) # ロスの計算
        _, preds = torch.max(out, 1)  # ラベルを予測

        y_true = np.append(y_true, labels.data.numpy())
        y_pred = np.append(y_pred, preds.cpu().data.numpy())

    from sklearn.metrics import accuracy_score, confusion_matrix
    print(accuracy_score(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
    #torch.save(net.state_dict(), './move_model.pth')


def main():
    text_files = glob.glob('/mnt/aoni02/katayama/dataset/Trans_withLabel/*txt')
    
    feature, X_pos, X_pos2, X_id, label = encode_sentence(text_files,start=0,end=12,MAX_SIZE=20)
    print(np.shape(feature),np.shape(X_pos),np.shape(X_pos2),np.shape(X_id),np.shape(label))
    
    feature = torch.tensor(feature,dtype=torch.long)
    batch_size = 1
    train_dataloader = torch.utils.data.DataLoader(
        list(zip(feature[:5000],X_pos,X_pos2,X_id,label)), batch_size=batch_size, shuffle=False)

    val_dataloader = torch.utils.data.DataLoader(
        list(zip(feature[5000:],X_pos[5000:],X_pos2[5000:],X_id[5000:],label[5000:])), batch_size=batch_size, shuffle=False)

    test_dataloader = torch.utils.data.DataLoader(
        list(zip(feature[5000:],X_pos[5000:],X_pos2[5000:],X_id[5000:],label[5000:])), batch_size=batch_size, shuffle=False)

    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader, "test": test_dataloader}

    net = MovePredict(hidden_size=128)
    print(net)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = torch.tensor([2.1,3.5,15.0,9.8,9.5]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)#,reduce=False)
    for name, param in net.named_parameters():
        if 'fc' in name or 'lstm' in name:
            param.requires_grad = True
            print("勾配計算あり。学習する：", name)
        else:
            param.requires_grad = False
            print("勾配計算なし。学習しない：", name)
    optimizer = optim.Adam(net.parameters(), lr=1e-03)
    train(net=net, dataloaders_dict=dataloaders_dict, criterion=criterion, optimizer=optimizer, num_epochs=30)

if __name__ == '__main__':
    main()
