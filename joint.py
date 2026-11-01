import torch
import torch.nn as nn
import torch.nn.functional as F

# بردن به فضای مشترک
# L2 Normalization تا ضرب داخلی دو بردار دقیقاً برابر با Cosine Similarity
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, projection_dim=128):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, projection_dim)
        )

    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, p=2, dim=-1)  # L2 normalization


def contrastive_loss(proj_gcn, proj_trans, temperature=0.07):
    """
    Symmetric Contrastive Loss (Graph-to-Seq and Seq-to-Graph)
    proj_gcn: (batch_size, projection_dim)
    proj_trans: (batch_size, projection_dim)
    """
    batch_size = proj_gcn.size(0)
    device = proj_gcn.device

    # محاسبه ماتریس شباهت کسینوسی (چون قبلاً نرمالیزه شده‌اند، فقط ضرب ماتریسی کافیست)
    # This temperature is for making differences bigger
    similarity_matrix = torch.matmul(proj_gcn, proj_trans.T) / temperature  # (B, B)

    labels = torch.arange(batch_size, device=device)

    # Loss از سمت گراف به توالی
    loss_g2s = F.cross_entropy(similarity_matrix, labels)

    # Loss از سمت توالی به گراف
    loss_s2g = F.cross_entropy(similarity_matrix.T, labels)

    # Symmetric Contrastive Loss
    loss_contrastive = (loss_g2s + loss_s2g) / 2

    # Positive-pair cosine loss (جهت ترغیب نزدیک‌تر شدن جفت‌های مثبت)
    # Cosine similarity برای جفت‌های مثبت به طور ایده‌آل باید ۱ باشد
    pos_sim = torch.diag(similarity_matrix) * temperature  # خارج کردن تقسیم بر دما تا به شباهت کسینوسی بین 1 تا -1 برگرده
    loss_pair = torch.mean(1.0 - pos_sim)

    return loss_contrastive, loss_pair



