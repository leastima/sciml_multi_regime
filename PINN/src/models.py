import torch
import torch.nn as nn

from src.curves import *

'''
Implementation of PINNs. 

Source: https://github.com/AdityaLab/pinnsformer/blob/main/model/pinn.py
'''
class PINN(nn.Module):
  def __init__(self, in_dim, hidden_dim, out_dim, num_layer):
    super(PINN, self).__init__()

    layers = []
    for i in range(num_layer-1):
      if i == 0:
        layers.append(nn.Linear(in_features=in_dim, out_features=hidden_dim))
        layers.append(nn.Tanh())
      else:
        layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
        layers.append(nn.Tanh())

    layers.append(nn.Linear(in_features=hidden_dim, out_features=out_dim))

    self.linear = nn.Sequential(*layers)

  def forward(self, x, t):
    src = torch.cat((x,t), dim=-1)
    return self.linear(src)


# Unified version for both opt_for_pinns and PINNacle
class UnifiedPINN(nn.Module):
    """
    Unified PINN model compatible with both opt_for_pinns and PINNacle frameworks.
    Extends the basic PINN with more flexibility for different activation functions
    and initialization strategies.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layer, activation='tanh', init_method='xavier_normal'):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layer = num_layer
        self.activation = activation
        self.init_method = init_method
        
        layers = []
        for i in range(num_layer):
            if i == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
            elif i == num_layer - 1:
                layers.append(nn.Linear(hidden_dim, out_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            
            # Add activation (except for output layer)
            if i < num_layer - 1:
                if activation == 'tanh':
                    layers.append(nn.Tanh())
                elif activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'sigmoid':
                    layers.append(nn.Sigmoid())
                else:
                    layers.append(nn.Tanh())  # Default fallback
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights using specified method"""
        if isinstance(m, nn.Linear):
            if self.init_method == 'xavier_normal':
                nn.init.xavier_normal_(m.weight)
            elif self.init_method == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight)
            elif self.init_method == 'kaiming_normal':
                nn.init.kaiming_normal_(m.weight)
            else:
                nn.init.xavier_normal_(m.weight)  # Default
            nn.init.zeros_(m.bias)
    
    def forward(self, x, t=None):
        """
        Forward pass - compatible with both single input and (x,t) format
        """
        if t is not None:
            inputs = torch.cat([x, t], dim=-1)
        else:
            inputs = x
        return self.network(inputs)

# Factory function for model creation (PINNacle-style convenience)
def create_pinn_model(model_type='unified', **kwargs):
    """
    Factory function to create PINN models
    
    Args:
        model_type: 'basic' for original PINN, 'unified' for UnifiedPINN
        **kwargs: model parameters
    """
    if model_type == 'basic':
        return PINN(**kwargs)
    elif model_type == 'unified':
        return UnifiedPINN(**kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
'''
原ResnetCurve
因MLP仅有Linear层带有需要训练的参数，这里仅对linear层进行替换，激活函数层不做处理
'''
class PINNCurve(nn.Module):
  def __init__(self, in_dim, hidden_dim, out_dim, num_layer, fix_points):
    super(PINNCurve, self).__init__()

    layers = []
    for i in range(num_layer - 1):
      if i == 0:
        layers.append(Linear(in_dim, hidden_dim, fix_points=fix_points))
        layers.append(nn.Tanh())
      else:
        layers.append(Linear(hidden_dim, hidden_dim, fix_points=fix_points))
        layers.append(nn.Tanh())

    layers.append(Linear(hidden_dim, out_dim, fix_points=fix_points))
    self.linear = nn.Sequential(*layers)

  def forward(self, x, t):
    src = torch.cat((x, t), dim=-1)
    return self.linear(src)
