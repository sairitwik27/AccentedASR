import torch
from torch import nn

class AccentClassifier(torch.nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        self.classifier_linear1 = nn.Linear(512, 256)
        self.classifier_linear2 = nn.Linear(256, 128)
        self.act = nn.ReLU()
        self.classifier_output = nn.Linear(128, num_classes)
        self.classifier_loss_fun = nn.CrossEntropyLoss()
        
    def forward(self,feat,labels):
        mean = torch.mean(feat,1)
        std = torch.std(feat,1)
        stat_pooling = torch.cat((mean,std),1)
        # stat_pooling = mean
        linear1_out = self.act(self.classifier_linear1(stat_pooling))
        linear2_out = self.act(self.classifier_linear2(linear1_out))
        self.predictions = self.classifier_output(linear2_out)
        classifier_loss = self.classifier_loss_fun(self.predictions,labels)
        self.loss = classifier_loss
        
        return self.predictions,self.loss
    
    def encode(self,feat):
        mean = torch.mean(feat,1)
        std = torch.std(feat,1)
        stat_pooling = torch.cat((mean,std),1)
        # stat_pooling = mean
        linear1_out = self.act(self.classifier_linear1(stat_pooling))
        linear2_out = self.act(self.classifier_linear2(linear1_out))

        return linear2_out.squeeze(0)

# class LSTMAccentClassifier(torch.nn.Module):
#     def __init__(self,batch_size,embed_dim,hidden_size,num_layers=4,num_classes=10):
#         super().__init__()
#         self.lstm_module = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim,num_layers=num_layers,batch_first=True)
#         self.classifier_linear1 = nn.Linear(hidden_size, hidden_size//2)
#         self.act1 = nn.ReLU()
#         self.classifier_output = nn.Linear(hidden_size//2, num_classes)
#         self.classifier_loss_fun = nn.CrossEntropyLoss()

#     def forward(self,feat,labels):
#         lstm_embed = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim,num_layers=num_layers,batch_first=True)
#         mean = torch.mean(lstm_embed,1)
#         stat_pooling = mean
#         linear1_out = self.act1(self.classifier_linear1(stat_pooling))
#         self.predictions = self.classifier_output(linear1_out)
#         classifier_loss = self.classifier_loss_fun(self.predictions,labels)
#         self.loss = classifier_loss
#         return self.predictions,self.loss

# class ReDATGMM:
#     def __init__(self,embeddings,embedding_dim,n_clusters):
        
#         self.n_clusters = n_clusters
#         self.n_features = embedding_dim
    
#     def fitGaussian(self):
#         model = GaussianMixture(n_components, x.size(1), covariance_type="diag").cuda()
#         model.fit(x)

#         # check that y_p has dimensions (n, k)
#         y_p = model.predict(x, probs=True)

#         return y_p

# class ReDATKMeans:
#     def __init__(self,embedding_dim=128,n_clusters=20):
#         self.n_clusters = n_clusters
#         self.n_features = embedding_dim
    
#     def cluster(self,feat_matrix):
#         pass

