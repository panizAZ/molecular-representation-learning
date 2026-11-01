import numpy as np
import torch
import matplotlib.pyplot as plt

def token_accuracy(logits, targets, pad_idx):
    preds = logits.argmax(dim=-1)
    mask = targets != pad_idx
    correct = ((preds == targets) & mask).sum().item()
    total = mask.sum().item()
    return correct / max(total, 1)


def perplexity_from_loss(loss_value):
    # error handling: if loss becomes too large it will crash
    return float(np.exp(min(loss_value, 20.0)))


@torch.no_grad()
def collect_test_embeddings(gcn, transformer, gcn_proj, trans_proj, dataloader, device):
    gcn.eval()
    transformer.eval()
    gcn_proj.eval()
    trans_proj.eval()

    all_g = []
    all_t = []

    for batch in dataloader:
        #Graph and inputs contain the same molecule in the same position
        graph_batch = batch["graph"].to(device)
        input_ids = batch["input_ids"].to(device)
        eos_positions = batch["eos_positions"].to(device)

        x_in = input_ids[:, :-1]
        shifted_eos_positions = (eos_positions - 1).clamp(min=0)

        g_rep = gcn(graph_batch)
        t_rep = transformer.get_representation(x_in, eos_positions=shifted_eos_positions)

        g_proj = gcn_proj(g_rep)
        t_proj = trans_proj(t_rep)

        all_g.append(g_proj)
        all_t.append(t_proj)

    return torch.cat(all_g, dim=0), torch.cat(all_t, dim=0)


#می‌سنجیم که آیا مدل می‌فهمد گراف مولکول A به رشته‌ی SELFIES همان مولکول A ربط دارد یا نه
@torch.no_grad()
def retrieval_metrics(graph_emb, seq_emb):
    sim = graph_emb @ seq_emb.T
    n = sim.size(0)

    g2s_rank = sim.argsort(dim=1, descending=True)
    s2g_rank = sim.T.argsort(dim=1, descending=True)

    targets = torch.arange(n, device=sim.device)

    # is the first recommended sequence for the graph we gave it?
    g2s_top1 = (g2s_rank[:, 0] == targets).float().mean().item()
    g2s_top5 = (g2s_rank[:, :5] == targets.unsqueeze(1)).any(dim=1).float().mean().item()

    s2g_top1 = (s2g_rank[:, 0] == targets).float().mean().item()
    s2g_top5 = (s2g_rank[:, :5] == targets.unsqueeze(1)).any(dim=1).float().mean().item()

    # میانگین شباهت جفت‌های درست (قطر اصلی ماتریس). هر چه به ۱ نزدیک‌تر، بهتر
    pos_cos = sim.diag().mean().item()

    # میانگین شباهت جفت‌های غلط (سایر خانه‌ها). هر چه به ۰ نزدیک‌تر، بهتر
    neg_mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
    neg_cos = sim[neg_mask].mean().item()

    # If model worked randomly
    random_top1 = 1.0 / n
    random_top5 = min(5, n) / n

    return {
        "g2s_top1": g2s_top1,
        "g2s_top5": g2s_top5,
        "s2g_top1": s2g_top1,
        "s2g_top5": s2g_top5,
        "avg_matching_cosine": pos_cos,
        "avg_nonmatching_cosine": neg_cos,
        "random_top1": random_top1,
        "random_top5": random_top5,
    }

def plot_training_curves(history):
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.plot(history["warmup_train_loss"], label="warmup_train_loss")
    plt.plot(history["warmup_val_loss"], label="warmup_val_loss")
    plt.plot(history["joint_train_total"], label="joint_train_total")
    plt.plot(history["joint_val_total"], label="joint_val_total")
    plt.legend()
    plt.title("Warm-up / Joint Total Loss")

    plt.subplot(2, 2, 2)
    plt.plot(history["joint_train_token"], label="train_token_loss")
    plt.plot(history["joint_val_token"], label="val_token_loss")
    plt.plot(history["joint_train_contrastive"], label="train_contrastive_loss")
    plt.plot(history["joint_val_contrastive"], label="val_contrastive_loss")
    plt.plot(history["joint_train_pair"], label="train_pair_loss")
    plt.plot(history["joint_val_pair"], label="val_pair_loss")
    plt.legend()
    plt.title("Joint Loss Components")

    plt.subplot(2, 2, 3)
    plt.plot(history["joint_train_acc"], label="train_token_acc")
    plt.plot(history["joint_val_acc"], label="val_token_acc")
    plt.legend()
    plt.title("Token Accuracy")

    plt.subplot(2, 2, 4)
    plt.plot(history["joint_train_ppl"], label="train_perplexity")
    plt.plot(history["joint_val_ppl"], label="val_perplexity")
    plt.legend()
    plt.title("Perplexity")

    plt.tight_layout()
    plt.show()