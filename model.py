import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor
#from timm.models.layers import DropPath
from timm.layers import DropPath
from torchvision.models import resnet50, ResNet50_Weights

class GRTSNet(nn.Module):
    def __init__(self, num_classes, feature_dim=768, dropout=0.4, drop_path=0.2):
        super(GRTSNet, self).__init__()
        # Model A: Frozen DINOv2 (teacher)
        self.model_a = AutoModel.from_pretrained('facebook/dinov2-base')
        self.processor_a = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        for param in self.model_a.parameters():
            param.requires_grad = False

        # Model B: Trainable ResNet-50 (student) with drop path
        #self.model_b = resnet50(pretrained=True)
        self.model_b = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model_b.fc = nn.Identity()  # Remove final FC
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Projection for B to match dim
        self.proj_b = nn.Linear(2048, feature_dim)

        # Gated Fusion
        self.mlp_gate = nn.Sequential(
            nn.Linear(2 * feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1)
        )
        self.ln_a = nn.LayerNorm(feature_dim)
        self.ln_b = nn.LayerNorm(feature_dim)

        # Classifier Head
        self.head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_classes)
        )

        # Auxiliary Head for B
        self.aux_head = nn.Linear(2048, num_classes)

    def forward(self, x):
        # ---- DINOv2 teacher ----
        with torch.no_grad():
            inputs_a = self.processor_a(x, return_tensors="pt", do_rescale=False)
            inputs_a = {k: v.to(x.device, non_blocking=True) for k, v in inputs_a.items()}
            x_a = self.model_a(**inputs_a).last_hidden_state[:, 0]  # CLS

        # ---- ResNet student ----
        x_b = self.model_b(x)
        x_b = self.drop_path(x_b)
        x_b_proj = self.proj_b(x_b)

        # ---- Gated Fusion ----
        ln_a = self.ln_a(x_a)
        ln_b = self.ln_b(x_b_proj)

        gate_input = torch.cat([ln_a, ln_b], dim=1)
        g = torch.sigmoid(self.mlp_gate(gate_input))
        fused = x_a + g * (x_b_proj - x_a)

        # ---- Heads ----
        out = self.head(fused)
        aux_out = self.aux_head(x_b)

        return out, aux_out, x_b_proj, x_a


def grts_loss(final_out, aux_out, student_feats, teacher_feats, labels, args, lambda1=0.3, lambda2=0.7, T=3.0):
    ce_final = F.cross_entropy(final_out, labels, label_smoothing=args.smoothing)
    ce_aux = F.cross_entropy(aux_out, labels, label_smoothing=args.smoothing)
    kl = F.kl_div(F.log_softmax(student_feats / T, dim=1), F.softmax(teacher_feats / T, dim=1), reduction='batchmean') * (T ** 2)
    return ce_final + lambda1 * ce_aux + lambda2 * kl