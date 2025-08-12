import torch
import torch.nn as nn
import torch.nn.init as init
import math
from torchvision import models

class LipNet(torch.nn.Module):
    def __init__(self, dropout_p=0.5):
        super(LipNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        
        self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), (1, 1, 1), (1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        
        self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1))     
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        
        self.gru1  = nn.GRU(96*4*8, 256, 1, bidirectional=True)
        self.gru2  = nn.GRU(512, 256, 1, bidirectional=True)
        
        self.FC    = nn.Linear(512, 27+1)
        self.dropout_p  = dropout_p

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_p)        
        self.dropout3d = nn.Dropout3d(self.dropout_p)  
        self._init()
    
    def _init(self):
        
        init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        init.constant_(self.conv1.bias, 0)
        
        init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        init.constant_(self.conv2.bias, 0)
        
        init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
        init.constant_(self.conv3.bias, 0)        
        
        init.kaiming_normal_(self.FC.weight, nonlinearity='sigmoid')
        init.constant_(self.FC.bias, 0)
        
        for m in (self.gru1, self.gru2):
            stdv = math.sqrt(2 / (96 * 3 * 6 + 256))
            for i in range(0, 256 * 3, 256):
                init.uniform_(m.weight_ih_l0[i: i + 256],
                            -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(m.weight_hh_l0[i: i + 256])
                init.constant_(m.bias_ih_l0[i: i + 256], 0)
                init.uniform_(m.weight_ih_l0_reverse[i: i + 256],
                            -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(m.weight_hh_l0_reverse[i: i + 256])
                init.constant_(m.bias_ih_l0_reverse[i: i + 256], 0)
        
        
    def forward(self, x, coords=None):
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout3d(x)        
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout3d(x)        
        x = self.pool3(x)
        
        # (B, C, T, H, W)->(T, B, C, H, W)
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        # (B, C, T, H, W)->(T, B, C*H*W)
        x = x.view(x.size(0), x.size(1), -1)
        
        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()
        
        x, h = self.gru1(x)        
        x = self.dropout(x)
        x, h = self.gru2(x)   
        x = self.dropout(x)
                
        x = self.FC(x)
        x = x.permute(1, 0, 2).contiguous()
        return x


#LipCoordFormer - model with LipNet + Coords - not used in runs
class LipCoordFormer(torch.nn.Module):
    def __init__(self, dropout_p=0.5, coord_input_dim=40, coord_hidden_dim=128):
        super(LipCoordFormer, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), (1, 1, 1), (1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.gru1 = nn.GRU(96 * 4 * 8, 256, 1, bidirectional=True)
        self.gru2 = nn.GRU(512, 256, 1, bidirectional=True)

        self.FC = nn.Linear(512 + 2 * coord_hidden_dim, 27 + 1)
        self.dropout_p = dropout_p

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_p)
        self.dropout3d = nn.Dropout3d(self.dropout_p)

        # New GRU layer for lip coordinates
        self.coord_gru = nn.GRU(
            coord_input_dim, coord_hidden_dim, 1, bidirectional=True
        )

        self._init()

    def _init(self):
        init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        init.constant_(self.conv1.bias, 0)

        init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        init.constant_(self.conv2.bias, 0)

        init.kaiming_normal_(self.conv3.weight, nonlinearity="relu")
        init.constant_(self.conv3.bias, 0)

        init.kaiming_normal_(self.FC.weight, nonlinearity="sigmoid")
        init.constant_(self.FC.bias, 0)

        for m in (self.gru1, self.gru2):
            stdv = math.sqrt(2 / (96 * 3 * 6 + 256))
            for i in range(0, 256 * 3, 256):
                init.uniform_(
                    m.weight_ih_l0[i : i + 256],
                    -math.sqrt(3) * stdv,
                    math.sqrt(3) * stdv,
                )
                init.orthogonal_(m.weight_hh_l0[i : i + 256])
                init.constant_(m.bias_ih_l0[i : i + 256], 0)
                init.uniform_(
                    m.weight_ih_l0_reverse[i : i + 256],
                    -math.sqrt(3) * stdv,
                    math.sqrt(3) * stdv,
                )
                init.orthogonal_(m.weight_hh_l0_reverse[i : i + 256])
                init.constant_(m.bias_ih_l0_reverse[i : i + 256], 0)

    def forward(self, x, coords):
        # branch 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool3(x)

        # (B, C, T, H, W)->(T, B, C, H, W)
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        # (B, C, T, H, W)->(T, B, C*H*W)
        x = x.view(x.size(0), x.size(1), -1)

        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()

        x, h = self.gru1(x)
        x = self.dropout(x)
        x, h = self.gru2(x)
        x = self.dropout(x)

        # branch 2
        # Process lip coordinates through GRU
        self.coord_gru.flatten_parameters()

        # (B, T, N, C)->(T, B, C, N, C)
        coords = coords.permute(1, 0, 2, 3).contiguous()
        # (T, B, C, N, C)->(T, B, C, N*C)
        coords = coords.view(coords.size(0), coords.size(1), -1)
        coords, _ = self.coord_gru(coords)
        coords = self.dropout(coords)

        # combine the two branches
        combined = torch.cat((x, coords), dim=2)

        x = self.FC(combined)
        x = x.permute(1, 0, 2).contiguous()
        return x

#LipCoordFormer2 is our first model with coord and GRU - run 3 on slides
class LipCoordFormer2(torch.nn.Module):
    def __init__(self, original_lipnet, dropout_p=0.5, coord_input_dim=40, coord_hidden_dim=128):
        super(LipCoordFormer2, self).__init__()
        # self.conv1 = nn.Conv3d(3, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        # self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        # self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), (1, 1, 1), (1, 2, 2))
        # self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        # self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        # self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.feature_extractor = nn.Sequential(
            original_lipnet.conv1,
            original_lipnet.relu,
            original_lipnet.dropout3d,
            original_lipnet.pool1,
            
            original_lipnet.conv2,
            original_lipnet.relu,
            original_lipnet.dropout3d,
            original_lipnet.pool2,
            
            original_lipnet.conv3,
            original_lipnet.relu,
            original_lipnet.dropout3d,
            original_lipnet.pool3
        )

        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.gru1 = nn.GRU(96 * 4 * 8, 256, 1, bidirectional=True)
        self.gru2 = nn.GRU(512, 256, 1, bidirectional=True)

        self.FC = nn.Linear(512 + 2 * coord_hidden_dim, 27 + 1)
        self.dropout_p = dropout_p

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_p)
        self.dropout3d = nn.Dropout3d(self.dropout_p)

        # New GRU layer for lip coordinates
        self.coord_gru = nn.GRU(
            coord_input_dim, coord_hidden_dim, 1, bidirectional=True
        )

        self._init()

    def _init(self):
        for m in (self.gru1, self.gru2):
            stdv = math.sqrt(2 / (96 * 3 * 6 + 256))
            for i in range(0, 256 * 3, 256):
                init.uniform_(
                    m.weight_ih_l0[i : i + 256],
                    -math.sqrt(3) * stdv,
                    math.sqrt(3) * stdv,
                )
                init.orthogonal_(m.weight_hh_l0[i : i + 256])
                init.constant_(m.bias_ih_l0[i : i + 256], 0)
                init.uniform_(
                    m.weight_ih_l0_reverse[i : i + 256],
                    -math.sqrt(3) * stdv,
                    math.sqrt(3) * stdv,
                )
                init.orthogonal_(m.weight_hh_l0_reverse[i : i + 256])
                init.constant_(m.bias_ih_l0_reverse[i : i + 256], 0)

    def forward(self, x, coords):
        x = self.feature_extractor(x)

        # (B, C, T, H, W)->(T, B, C, H, W)
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        # (B, C, T, H, W)->(T, B, C*H*W)
        x = x.view(x.size(0), x.size(1), -1)

        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()

        x, h = self.gru1(x)
        x = self.dropout(x)
        x, h = self.gru2(x)
        x = self.dropout(x)

        # branch 2
        # Process lip coordinates through GRU
        self.coord_gru.flatten_parameters()

        # (B, T, N, C)->(T, B, C, N, C)
        coords = coords.permute(1, 0, 2, 3).contiguous()
        # (T, B, C, N, C)->(T, B, C, N*C)
        coords = coords.view(coords.size(0), coords.size(1), -1)
        coords, _ = self.coord_gru(coords)
        coords = self.dropout(coords)

        # combine the two branches
        combined = torch.cat((x, coords), dim=2)

        x = self.FC(combined)
        x = x.permute(1, 0, 2).contiguous()
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

# LipCoordFormer3 - Transformer Decoder - Attention Mechanism - Run 5 on slides
class LipCoordFormer3(torch.nn.Module):
    def __init__(self, original_lipnet, dropout_p=0.5, coord_input_dim=40, coord_hidden_dim=128, d_model=512, nhead=8, num_layers=3):
        super(LipCoordFormer3, self).__init__()
        self.feature_extractor = nn.Sequential(
            original_lipnet.conv1,
            original_lipnet.relu,
            original_lipnet.dropout3d,
            original_lipnet.pool1,
            
            original_lipnet.conv2,
            original_lipnet.relu,
            original_lipnet.dropout3d,
            original_lipnet.pool2,
            
            original_lipnet.conv3,
            original_lipnet.relu,
            original_lipnet.dropout3d,
            original_lipnet.pool3
        )
        
        self.gru1 = nn.GRU(96 * 4 * 8, 256, 1, bidirectional=True)
        self.gru2 = nn.GRU(512, 256, 1, bidirectional=True)

        self.FC = nn.Linear(512 + 2 * coord_hidden_dim, 27 + 1)
        self.FC2 = nn.Linear(d_model, 27 + 1)
        
        self.dropout_p = dropout_p

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_p)
        self.dropout3d = nn.Dropout3d(self.dropout_p)

        # New GRU layer for lip coordinates
        self.coord_gru = nn.GRU(
            coord_input_dim, coord_hidden_dim, 1, bidirectional=True
        )
        
        # Transformer components
        self.d_model = d_model
        gru_hidden_size = original_lipnet.gru2.hidden_size
        
        # Projection layers
        self.branch1_proj = nn.Linear(512, d_model)  # CNN+GRU features
        self.branch2_proj = nn.Linear(256, d_model)  # Coordinates features
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=False
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        

        self._init()
        

    def _init(self):
        for m in (self.gru1, self.gru2):
            stdv = math.sqrt(2 / (96 * 3 * 6 + 256))
            for i in range(0, 256 * 3, 256):
                init.uniform_(
                    m.weight_ih_l0[i : i + 256],
                    -math.sqrt(3) * stdv,
                    math.sqrt(3) * stdv,
                )
                init.orthogonal_(m.weight_hh_l0[i : i + 256])
                init.constant_(m.bias_ih_l0[i : i + 256], 0)
                init.uniform_(
                    m.weight_ih_l0_reverse[i : i + 256],
                    -math.sqrt(3) * stdv,
                    math.sqrt(3) * stdv,
                )
                init.orthogonal_(m.weight_hh_l0_reverse[i : i + 256])
                init.constant_(m.bias_ih_l0_reverse[i : i + 256], 0)

    def forward(self, x, coords):
        # branch 1
        x = self.feature_extractor(x) 
        # (B, C, T, H, W)->(T, B, C, H, W)
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        # (B, C, T, H, W)->(T, B, C*H*W)
        x = x.view(x.size(0), x.size(1), -1)
        
        # Process through GRUs
        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters() 
        x, h = self.gru1(x)
        x = self.dropout(x)
        x, h = self.gru2(x)
        x = self.dropout(x) # (seq_len, batch, 2*gru_hidden_size)

        # branch 2
        # Process lip coordinates through GRU
        self.coord_gru.flatten_parameters() 
        # (B, T, N, C)->(T, B, C, N, C)
        coords = coords.permute(1, 0, 2, 3).contiguous()
        # (T, B, C, N, C)->(T, B, C, N*C)
        coords = coords.view(coords.size(0), coords.size(1), -1)
        coords, _ = self.coord_gru(coords)
        coords = self.dropout(coords)

        # Debugging: Check shapes
        #")
        #print(f"coords shape: {coords.shape}")


        # combine the two branches
        #combined = torch.cat((x, coords), dim=2)
        
        # Project to common dimension space
        branch1 = self.branch1_proj(x)  # (seq_len, batch, d_model)
        #print(f"branch1 shape: {branch1.shape}")
        branch1 = self.pos_encoder(branch1)
        branch2 = self.branch2_proj(coords)  # (seq_len, batch, d_model)
        
        # Debugging: Check shapes after projection
        #print(f"branch1 shape: {branch1.shape}")
        #print(f"branch2 shape: {branch2.shape}")

        # Transformer processing
        decoder_output = self.transformer_decoder(
            tgt=branch1,  # Self-attention on lip features
            memory=branch2  # Cross-attention with coordinates
        ) 
        

        #x = self.FC(combined)
        x = self.FC2(decoder_output)
        x = x.permute(1, 0, 2).contiguous()
        return x

#LipCoordFormer4 is our model with coords + Bi-LSTM - Run 4 on slides
class LipCoordFormer4(torch.nn.Module):
    def __init__(self, original_lipnet, dropout_p=0.5, coord_input_dim=40, coord_hidden_dim=128):
        super(LipCoordFormer4, self).__init__()
        self.feature_extractor = nn.Sequential(
            original_lipnet.conv1,
            original_lipnet.relu,
            original_lipnet.dropout3d,
            original_lipnet.pool1,
            
            original_lipnet.conv2,
            original_lipnet.relu,
            original_lipnet.dropout3d,
            original_lipnet.pool2,
            
            original_lipnet.conv3,
            original_lipnet.relu,
            original_lipnet.dropout3d,
            original_lipnet.pool3
        )
        
        self.bilstm1 = nn.LSTM(96 * 4 * 8, 256, 1, bidirectional=True, batch_first = True)
        self.bilstm2 = nn.LSTM(512, 256, 1, bidirectional=True, batch_first = True) #batch_first set to True to ensure (B,T,C) is output - common ML practice

        self.FC = nn.Linear(512 + 2 * coord_hidden_dim, 27 + 1)
        self.dropout_p = dropout_p

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_p)
        self.dropout3d = nn.Dropout3d(self.dropout_p)

        # New GRU layer for lip coordinates
        self.coord_gru = nn.GRU(
            coord_input_dim, coord_hidden_dim, 1, bidirectional=True
        )

        self._init()

    def _init(self):
        
        for m in (self.bilstm1, self.bilstm2):
            stdv = math.sqrt(2 / (96 * 3 * 6 + 256))
            for i in range(0, 256 * 3, 256):
                init.uniform_(
                    m.weight_ih_l0[i : i + 256],
                    -math.sqrt(3) * stdv,
                    math.sqrt(3) * stdv,
                )
                init.orthogonal_(m.weight_hh_l0[i : i + 256])
                init.constant_(m.bias_ih_l0[i : i + 256], 0)
                init.uniform_(
                    m.weight_ih_l0_reverse[i : i + 256],
                    -math.sqrt(3) * stdv,
                    math.sqrt(3) * stdv,
                )
                init.orthogonal_(m.weight_hh_l0_reverse[i : i + 256])
                init.constant_(m.bias_ih_l0_reverse[i : i + 256], 0)

    def forward(self, x, coords):
        x = self.feature_extractor(x)

        # (B, C, T, H, W)->(T, B, C, H, W)
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        # (B, C, T, H, W)->(T, B, C*H*W)
        x = x.view(x.size(0), x.size(1), -1)

        self.bilstm1.flatten_parameters()
        self.bilstm2.flatten_parameters()

        x, h = self.bilstm1(x)
        x = self.dropout(x)
        x, h = self.bilstm2(x)
        x = self.dropout(x)

        # branch 2
        # Process lip coordinates through GRU
        self.coord_gru.flatten_parameters()

        # (B, T, N, C)->(T, B, C, N, C)
        coords = coords.permute(1, 0, 2, 3).contiguous()
        # (T, B, C, N, C)->(T, B, C, N*C)
        coords = coords.view(coords.size(0), coords.size(1), -1)
        coords, _ = self.coord_gru(coords)
        coords = self.dropout(coords)

        # combine the two branches
        combined = torch.cat((x, coords), dim=2)

        x = self.FC(combined)
        x = x.permute(1, 0, 2).contiguous()
        return x


# LipCoordFormer5 - Transformer Decoder - Attention Mechanism with Bi-LSTM - Run 6 on slides
class LipCoordFormer5(torch.nn.Module):
    def __init__(self, original_lipnet, dropout_p=0.5, coord_input_dim=40, coord_hidden_dim=128, d_model=512, nhead=8, num_layers=3):
        super(LipCoordFormer5, self).__init__()
        self.feature_extractor = nn.Sequential(
            original_lipnet.conv1,
            original_lipnet.relu,
            original_lipnet.dropout3d,
            original_lipnet.pool1,
            
            original_lipnet.conv2,
            original_lipnet.relu,
            original_lipnet.dropout3d,
            original_lipnet.pool2,
            
            original_lipnet.conv3,
            original_lipnet.relu,
            original_lipnet.dropout3d,
            original_lipnet.pool3
        )

        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.bilstm1 = nn.LSTM(96 * 4 * 8, 256, 1, bidirectional=True, batch_first = True)
        self.bilstm2 = nn.LSTM(512, 256, 1, bidirectional=True, batch_first = True) #might need to set batch_first = false just in case - but it should be fine

        self.FC = nn.Linear(512 + 2 * coord_hidden_dim, 27 + 1)
        self.FC2 = nn.Linear(d_model, 27 + 1)
        
        self.dropout_p = dropout_p

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_p)
        self.dropout3d = nn.Dropout3d(self.dropout_p)

        # New GRU layer for lip coordinates
        self.coord_gru = nn.GRU(
            coord_input_dim, coord_hidden_dim, 1, bidirectional=True
        )
        
        # Transformer components
        self.d_model = d_model
        gru_hidden_size = original_lipnet.gru2.hidden_size
        
        # Projection layers
        self.branch1_proj = nn.Linear(512, d_model)  # CNN+GRU features
        self.branch2_proj = nn.Linear(256, d_model)  # Coordinates features
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=False
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        

        self._init()
        

    def _init(self):
        for m in (self.bilstm1, self.bilstm2):
            stdv = math.sqrt(2 / (96 * 3 * 6 + 256))
            for i in range(0, 256 * 3, 256):
                init.uniform_(
                    m.weight_ih_l0[i : i + 256],
                    -math.sqrt(3) * stdv,
                    math.sqrt(3) * stdv,
                )
                init.orthogonal_(m.weight_hh_l0[i : i + 256])
                init.constant_(m.bias_ih_l0[i : i + 256], 0)
                init.uniform_(
                    m.weight_ih_l0_reverse[i : i + 256],
                    -math.sqrt(3) * stdv,
                    math.sqrt(3) * stdv,
                )
                init.orthogonal_(m.weight_hh_l0_reverse[i : i + 256])
                init.constant_(m.bias_ih_l0_reverse[i : i + 256], 0)

    def forward(self, x, coords):
        # branch 1
        x = self.feature_extractor(x) 
        # (B, C, T, H, W)->(T, B, C, H, W)
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        # (B, C, T, H, W)->(T, B, C*H*W)
        x = x.view(x.size(0), x.size(1), -1)
        
        # Process through GRUs
        self.bilstm1.flatten_parameters()
        self.bilstm2.flatten_parameters() 
        x, h = self.bilstm1(x)
        x = self.dropout(x)
        x, h = self.bilstm2(x)
        x = self.dropout(x) # (seq_len, batch, 2*gru_hidden_size)
 

        # branch 2
        # Process lip coordinates through GRU
        self.coord_gru.flatten_parameters() 
        # (B, T, N, C)->(T, B, C, N, C)
        coords = coords.permute(1, 0, 2, 3).contiguous()
        # (T, B, C, N, C)->(T, B, C, N*C)
        coords = coords.view(coords.size(0), coords.size(1), -1)
        coords, _ = self.coord_gru(coords)
        coords = self.dropout(coords)

        # Debugging: Check shapes
        #") 


        # combine the two branches
        #combined = torch.cat((x, coords), dim=2)
        
        # Project to common dimension space
        branch1 = self.branch1_proj(x)  # (seq_len, batch, d_model)
        #print(f"branch1 shape: {branch1.shape}")
        branch1 = self.pos_encoder(branch1)
        branch2 = self.branch2_proj(coords)  # (seq_len, batch, d_model)
        
        # Debugging: Check shapes after projection
        #print(f"branch1 shape: {branch1.shape}")
        #print(f"branch2 shape: {branch2.shape}")

        # Transformer processing
        decoder_output = self.transformer_decoder(
            tgt=branch1,  # Self-attention on lip features
            memory=branch2  # Cross-attention with coordinates
        ) 
        

        #x = self.FC(combined)
        x = self.FC2(decoder_output)
        x = x.permute(1, 0, 2).contiguous()
        return x


# LipNet 2
class LipNet2(torch.nn.Module):
    def __init__(self, dropout_p=0.5):
        super(LipNet2, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        
        self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), (1, 1, 1), (1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        
        self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1))     
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        
        self.gru1  = nn.GRU(96*4*8, 256, 1, bidirectional=True)
        self.gru2  = nn.GRU(512, 256, 1, bidirectional=True)
        
        self.FC    = nn.Linear(512, 27+1)
        self.dropout_p  = dropout_p

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_p)        
        self.dropout3d = nn.Dropout3d(self.dropout_p)  

        self.bilstm1 = nn.LSTM(96 * 4 * 8, 256, 1, bidirectional=True, batch_first = True)
        self.bilstm2 = nn.LSTM(512, 256, 1, bidirectional=True, batch_first = True) #batch_first set to True to ensure (B,T,C) is output - common ML practice

        
        self._init()
    
    def _init(self):
        
         for m in (self.bilstm1, self.bilstm2):
            stdv = math.sqrt(2 / (96 * 3 * 6 + 256))
            for i in range(0, 256 * 3, 256):
                init.uniform_(
                    m.weight_ih_l0[i : i + 256],
                    -math.sqrt(3) * stdv,
                    math.sqrt(3) * stdv,
                )
                init.orthogonal_(m.weight_hh_l0[i : i + 256])
                init.constant_(m.bias_ih_l0[i : i + 256], 0)
                init.uniform_(
                    m.weight_ih_l0_reverse[i : i + 256],
                    -math.sqrt(3) * stdv,
                    math.sqrt(3) * stdv,
                )
                init.orthogonal_(m.weight_hh_l0_reverse[i : i + 256])
                init.constant_(m.bias_ih_l0_reverse[i : i + 256], 0)
        
        
    def forward(self, x, coords=None):
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout3d(x)        
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout3d(x)        
        x = self.pool3(x)
        
        # (B, C, T, H, W)->(T, B, C, H, W)
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        # (B, C, T, H, W)->(T, B, C*H*W)
        x = x.view(x.size(0), x.size(1), -1)
        
        # self.gru1.flatten_parameters()
        # self.gru2.flatten_parameters()
        
        # x, h = self.gru1(x)        
        # x = self.dropout(x)
        # x, h = self.gru2(x)   
        # x = self.dropout(x)

        self.bilstm1.flatten_parameters()
        self.bilstm2.flatten_parameters()

        x, h = self.bilstm1(x)
        x = self.dropout(x)
        x, h = self.bilstm2(x)
        x = self.dropout(x)
                
        x = self.FC(x)
        x = x.permute(1, 0, 2).contiguous()
        return x


class ResNetBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels * self.expansion),
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out
    
def make_layer(block, in_channels, out_channels, blocks, stride=1):
    layers = []
    layers.append(block(in_channels, out_channels, stride))
    for _ in range(1, blocks):
        layers.append(block(out_channels, out_channels))
    return nn.Sequential(*layers)

# LipCoordFormer6 - ResNet34 + GRU/LSTM + Transformer
class LipCoordFormer6(torch.nn.Module):
    def __init__(self, landmark=True, LSTM=False, transformer=False, dropout_p=0.5, coord_input_dim=40, coord_hidden_dim=128, d_model=512, nhead=8, num_layers=3):
        super(LipCoordFormer6, self).__init__()
        self.transformer = transformer
        self.landmark = landmark
        self.LSTM = LSTM
        
        # --- Branch 1: Frozen ResNet34 (feature extractor) ---
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        
        # Keep layers up to GAP (avgpool), exclude fc
        self.feature_extractor = nn.Sequential(
            *list(resnet.children())[:-1],  # Includes conv layers + avgpool
            nn.Flatten()  # Flatten output of GAP to [B*T, 512]
        )
        
        # Freeze all weights (no gradients)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        if self.LSTM:
            print("Using LSTM")
            self.bilstm1 = nn.LSTM(512, 256, 1, bidirectional=True, batch_first = True)
            self.bilstm2 = nn.LSTM(512, 256, 1, bidirectional=True, batch_first = True) #might need to set batch_first = false just in case - but it should be fine
            # GRU layer for lip coordinates
            self.coord_gru = nn.LSTM(
                coord_input_dim, coord_hidden_dim, 1, bidirectional=True, batch_first = True
            )
        else:
            print("Using GRU")
            self.gru1 = nn.GRU(96 * 4 * 8, 256, 1, bidirectional=True)
            self.gru2 = nn.GRU(512, 256, 1, bidirectional=True)
            # GRU layer for lip coordinates
            self.coord_gru = nn.GRU(
                coord_input_dim, coord_hidden_dim, 1, bidirectional=True
            )

        
        self.dropout_p = dropout_p

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_p)
        self.dropout3d = nn.Dropout3d(self.dropout_p)


        
        if transformer:
            print(f"Using Transformer with d_model={d_model}, nhead={nhead}, num_layers={num_layers}") 
            # Transformer components
            self.d_model = d_model
            
            # Projection layers
            self.branch1_proj = nn.Linear(512, d_model)  # CNN+GRU features
            self.branch2_proj = nn.Linear(256, d_model)  # Coordinates features
            self.pos_encoder = PositionalEncoding(d_model)
            
            # Transformer decoder
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                batch_first=False
            )
            self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
            self.FC2 = nn.Linear(d_model, 27 + 1)
        else:
            print("Using FC layer")

        if landmark:
            print("Using lip landmark")
            self.FC = nn.Linear(512 + 2 * coord_hidden_dim, 27 + 1)
        else:
            print("NO lip landmark")
            self.FC = nn.Linear(512, 27+1)
        
        self._init()

    def _init(self):
        if self.LSTM:
            for m in (self.bilstm1, self.bilstm2):
                stdv = math.sqrt(2 / (96 * 3 * 6 + 256))
                for i in range(0, 256 * 3, 256):
                    init.uniform_(
                        m.weight_ih_l0[i : i + 256],
                        -math.sqrt(3) * stdv,
                        math.sqrt(3) * stdv,
                    )
                    init.orthogonal_(m.weight_hh_l0[i : i + 256])
                    init.constant_(m.bias_ih_l0[i : i + 256], 0)
                    init.uniform_(
                        m.weight_ih_l0_reverse[i : i + 256],
                        -math.sqrt(3) * stdv,
                        math.sqrt(3) * stdv,
                    )
                    init.orthogonal_(m.weight_hh_l0_reverse[i : i + 256])
                    init.constant_(m.bias_ih_l0_reverse[i : i + 256], 0)
        else:
            for m in (self.gru1, self.gru2):
                stdv = math.sqrt(2 / (96 * 3 * 6 + 256))
                for i in range(0, 256 * 3, 256):
                    init.uniform_(
                        m.weight_ih_l0[i : i + 256],
                        -math.sqrt(3) * stdv,
                        math.sqrt(3) * stdv,
                    )
                    init.orthogonal_(m.weight_hh_l0[i : i + 256])
                    init.constant_(m.bias_ih_l0[i : i + 256], 0)
                    init.uniform_(
                        m.weight_ih_l0_reverse[i : i + 256],
                        -math.sqrt(3) * stdv,
                        math.sqrt(3) * stdv,
                    )
                    init.orthogonal_(m.weight_hh_l0_reverse[i : i + 256])
                    init.constant_(m.bias_ih_l0_reverse[i : i + 256], 0)

    def forward(self, x, coords): 
        # branch 1
        B, C, T, H, W = x.shape  # x: [40, 3, 75, 64, 128]
        
        # --- Branch 1: Process video frames ---
        # Reshape to [B*T, C, H, W] for frame-wise processing
        x = x.permute(0, 2, 1, 3, 4)  # [40, 75, 3, 64, 128]
        x = x.reshape(B * T, C, H, W)  # [3000, 3, 64, 128]
        
        # Extract features (frozen ResNet)
        with torch.no_grad():
            x = self.feature_extractor(x)  # [3000, 512]
        
        # Reshape to [T, B, 512]
        x = x.view(B, T, -1).permute(1, 0, 2)  # [75, 40, 512]

        if self.LSTM:
            self.bilstm1.flatten_parameters()
            self.bilstm2.flatten_parameters()
    
            x, h = self.bilstm1(x)
            x = self.dropout(x)
            x, h = self.bilstm2(x)
            x = self.dropout(x)

        if self.landmark:
            # branch 2
            # Process lip coordinates through GRU
            self.coord_gru.flatten_parameters()
    
            # (B, T, N, C)->(T, B, C, N, C)
            coords = coords.permute(1, 0, 2, 3).contiguous()
            # (T, B, C, N, C)->(T, B, C, N*C)
            coords = coords.view(coords.size(0), coords.size(1), -1)
            coords, _ = self.coord_gru(coords)
            coords = self.dropout(coords)
    
            # Output shape should be [T, B, 256]
            #print(f"Shape of coords: {coords.shape}")

        
        if self.transformer:
            
            # Project to common dimension space
            branch1 = self.branch1_proj(x)  # (seq_len, batch, d_model)
            #print(f"branch1 shape: {branch1.shape}")
            branch1 = self.pos_encoder(branch1)
            branch2 = self.branch2_proj(coords)  # (seq_len, batch, d_model)
            
            # Debugging: Check shapes after projection
            #print(f"branch1 shape: {branch1.shape}")
            #print(f"branch2 shape: {branch2.shape}")

            # Transformer processing
            decoder_output = self.transformer_decoder(
                tgt=branch1,  # Self-attention on lip features
                memory=branch2  # Cross-attention with coordinates
            ) 
            

            #x = self.FC(combined)
            x = self.FC2(decoder_output)
        else:
            if self.landmark:
                # Combine the two branches
                # Output shape should be [T, B, 768]
                combined = torch.cat((x, coords), dim=2) 
                x = self.FC(combined)
            else:
                x = self.FC(x)
            
        x = x.permute(1, 0, 2).contiguous()
        return x
