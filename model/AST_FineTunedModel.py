from transformers import ASTForAudioClassification
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Implementation taken from tutorial 9
class LowRankLayer(nn.Module):
    def __init__(self, linear, rank, alpha, use_dora=True):
        super().__init__()
        # rank: controls the inner dimension of the matrices A and B; controls the number of additional param
        # a key factor in determining the balance between model adaptability and parameter efficiency.
        # alpha: a scaling hyper-parameter applied to the output of the low-rank adaptation, 
        # controls the extent to which the adapted layer's output is allowed to influence the original output

        self.use_dora = use_dora
        self.rank = rank # low-rank
        self.alpha = alpha # scaling hyper-parameter
        self.linear = linear
        self.in_dim = linear.in_features
        self.out_dim = linear.out_features

        # weights
        std_dev = 1 / torch.sqrt(torch.tensor(self.rank).float())
        self.A = nn.Parameter(torch.randn(self.in_dim, self.rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(self.rank, self.out_dim))

        if self.use_dora:
            self.m = nn.Parameter(
            self.linear.weight.norm(p=2, dim=0, keepdim=True))
        else:
            self.m = None

    def forward(self, x):
        lora = self.A @ self.B # combine LoRA matrices
        if self.use_dora:
            numerator = self.linear.weight + self.alpha * lora.T
            denominator = numerator.norm(p=2, dim=0, keepdim=True)
            directional_component = numerator / denominator
            new_weight = self.m * directional_component
            return F.linear(x, new_weight, self.linear.bias)
        else:
            # combine LoRA with orig. weights
            combined_weight = self.linear.weight + self.alpha * lora.T
            return F.linear(x, combined_weight, self.linear.bias)

class AST(ASTForAudioClassification):
    def __init__(self,config):
        super().__init__(config)
        #Uploading the pretrained AST model
        model = ASTForAudioClassification.from_pretrained(config._name_or_path)
        self.audio_spectrogram_transformer=model.audio_spectrogram_transformer
        self.classifier=model.classifier
        self.requires_grad_(False)
        #Add Dora wrapper to linear layers within AST backbone model
        for i in np.arange(len(config.to_low_rank_layer_idx)):
            layer_idx=config.to_low_rank_layer_idx[i]
            rank=config.low_rank_layer_rank[i]
            alpha=config.low_rank_layer_alpha[i]
            self.audio_spectrogram_transformer.encoder.layer[layer_idx].output.dense=LowRankLayer(self.audio_spectrogram_transformer.encoder.layer[layer_idx].output.dense,rank,alpha)
        #Update the classifier architecture
        self.classifier.dense= nn.Sequential(nn.Dropout(p=config.dropout_rat[0]),
                                             nn.Linear(self.classifier.dense.in_features,config.classifier_hidden_layer_size),
                                             nn.GELU(),
                                             nn.Dropout(p=config.dropout_rat[1]),
                                             nn.Linear(config.classifier_hidden_layer_size,config.num_labels)
                                            )
