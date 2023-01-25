import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(in_dim, in_dim//8)
        self.key = nn.Linear(in_dim, in_dim//8)
        self.value = nn.Linear(in_dim, in_dim)
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        B, C, T , _= x.size()
        query = self.query(x).view(B, -1, T).permute(0, 2, 1)
        key = self.key(x).view(B, -1, T)
        energy = torch.bmm(query, key)
        attention = torch.softmax(energy, dim=-1)
        value = self.value(x).view(B, -1, T)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, T)
        out = self.gamma * out + x
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, in_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = in_dim
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        B, C, T, _ = x.size()
        query = self.query(x).view(B, -1, T, self.num_heads)
        key = self.key(x).view(B, -1, T, self.num_heads)
        value = self.value(x).view(B, -1, T, self.num_heads)
        query = query.permute(0, 3, 2, 1)
        key = key.permute(0, 3, 1, 2)
        energy = torch.matmul(query, key)
        attention = torch.softmax(energy, dim=-1)
        value = value.permute(0, 3, 2, 1)
        out = torch.matmul(attention, value)
        out = out.permute(0, 2, 3, 1).contiguous()
        out = out.view(B, T, -1)
        out = self.gamma * out + x
        return out
    
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1).squeeze(-1))
        out = avg_out + max_out
        return torch.sigmoid(out).unsqueeze(-1).unsqueeze(-1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd!"
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channel, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class SELayer(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SENet(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SENet, self).__init__()
        self.se = SELayer(in_channels, reduction_ratio)

    def forward(self, x):
        return self.se(x)

class ECA(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.k_size = k_size
        self.conv = nn.Conv1d(channel, channel, kernel_size=k_size, bias=False, groups=channel)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = nn.functional.unfold(y.transpose(-1, -3), kernel_size=(1, self.k_size), padding=(0, (self.k_size - 1) // 2))
        y = self.conv(y.transpose(-1, -2)).unsqueeze(-1)
        y = self.sigmoid(y)
        x = x * y.expand_as(x)
        return x


class SE_CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SE_CBAM, self).__init__()
        self.cbam = CBAM(in_channels)
        self.se = SENet(in_channels, reduction_ratio)

    def forward(self, x):
        y = self.cbam(x)
        z = self.se(x)
        return y*z
