import torch
import torch.nn as nn
import torch.nn.functional as F

from RVEnet.data_loader import TaskTypes

class SelfAttention(nn.Module):
    "Self attention layer for `n_channels`."
    def __init__(self, n_channels):
        super(SelfAttention,self).__init__()
        self.query,self.key,self.value = [self._conv(n_channels, c) for c in (n_channels//8,n_channels//8,n_channels)]

    def _conv(self, n_in: int, n_out: int) -> nn.Module:
        return nn.Conv1d(n_in, n_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.size()
        x = x.view(*size[:2],-1)
        f,g,h = self.query(x),self.key(x),self.value(x)
        beta = F.softmax(torch.bmm(f.transpose(1,2), g), dim=1)
        #o = self.gamma * torch.bmm(h, beta) + x
        o = torch.bmm(h, beta)
        return o.view(*size).contiguous()

class EFNet(nn.Module):

    def __init__(self, features: nn.Module, batch_size: int, DICOM_frame_nbr: int, input_image_size: int, 
                 task: TaskTypes, activation_function: str, self_attention: bool=False, 
                 self_attention_feature_reduction: int=256, dropout_prob: float=0.1):

        super(EFNet, self).__init__()

        self.task = task
        self.frame_nbr = DICOM_frame_nbr
        self.batch_size = batch_size
        self.dropout_prob = dropout_prob
        self.features = features
        self.activation_function = activation_function
        self.self_attention = self_attention

        if self.activation_function == 'ReLU':
            self.activation = nn.ReLU()
        elif self.activation_function == 'SELU':
            self.activation = nn.SELU()

        sample_tensor = torch.randn(batch_size, 3, input_image_size, input_image_size)
        output = features(sample_tensor)
        feature_out_channel = output.size()[1]
        feature_out_H = output.size()[2]
        feature_out_W = output.size()[3]

        if self.self_attention:
            self.self_attention_feature_reduction = 256
            self.frame_feature_reduction = nn.Conv2d(feature_out_channel, self.self_attention_feature_reduction, (1, 1))

            self.selfAttention = SelfAttention(self.self_attention_feature_reduction * DICOM_frame_nbr)
            self.dimension_reduction_conv = nn.Conv2d(self.self_attention_feature_reduction * DICOM_frame_nbr, 256, (1, 1))
            self.FC1 = nn.Linear(self_attention_feature_reduction * feature_out_H * feature_out_W, 1024)

        else:
            self.temporal_conv_layer = nn.Conv2d(feature_out_channel * DICOM_frame_nbr, 256, (1, 1))
            self.batch_norm_conv_layer = nn.BatchNorm2d(256)
            self.FC1 = nn.Linear(256 * feature_out_H * feature_out_W, 1024)

        self.dropout = nn.Dropout(self.dropout_prob)
        
        self.batch_norm_FC1 = nn.BatchNorm1d(1024)

        if self.task.task_type == TaskTypes.CLASSIFICATION or self.task.task_type == TaskTypes.BINARY_CLASSIFICATION:
            self.FC2 = nn.Linear(1024, task.output_nbr)
        elif self.task.task_type == TaskTypes.REGRESSION:
            self.FC2 = nn.Linear(1024, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)

        if self.self_attention:
            x = self.frame_feature_reduction(x)

        x = self.split_dicom_frames(x)

        if self.self_attention:
            x = self.selfAttention(x)
            x = self.activation(self.dimension_reduction_conv(x))
        else:
            x = self.activation(self.batch_norm_conv_layer(self.temporal_conv_layer(x)))

        x = x.view(-1, self.num_flat_features(x))

        x = self.activation(self.batch_norm_FC1(self.FC1(x)))

        x = self.dropout(x)

        x = self.FC2(x)

        x = self.dropout(x)

        return x

    def split_dicom_frames(self, input_tensor: torch.Tensor) -> torch.Tensor:

        s = list(input_tensor.shape)
        s[0] = int(s[0] / self.frame_nbr)
        s[1] = int(s[1] * self.frame_nbr)
        reshaped_input_tensor = input_tensor.reshape(s)

        return reshaped_input_tensor

    def num_flat_features(self, x: torch.Tensor) -> int:
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
