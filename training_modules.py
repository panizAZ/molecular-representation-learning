import torch
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from eval_plot import perplexity_from_loss, token_accuracy
from joint import contrastive_loss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# -------------
# Warm-Up epoch
#--------------
def train_transformer_warmup_epoch(
    transformer,
    dataloader,
    optimizer,
    scheduler,
    scaler,
    device,
    pad_idx,
    max_grad_norm=1.0,
):
    transformer.train()

    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)

        x_in = input_ids[:, :-1]
        y_out = input_ids[:, 1:]

        # for starting a new batch
        optimizer.zero_grad(set_to_none=True)

        # یعنی بعضی محاسبات با precision کمتر انجام می‌شوند تا:
        # آموزش سریع‌تر شود
        # حافظه کمتر مصرف شود
        # for mixed precision
        with autocast():
            logits, _ = transformer(x_in)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y_out.reshape(-1),
                ignore_index=pad_idx,
            )

        # چون mixed precision استفاده شده، GradScaler کمک می‌کند underflow رخ ندهد
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        acc = token_accuracy(logits, y_out, pad_idx)

        total_loss += loss.item()
        total_acc += acc
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_acc = total_acc / max(num_batches, 1)

    return {
        "loss": avg_loss,
        "token_acc": avg_acc,
        "perplexity": perplexity_from_loss(avg_loss),
    }

# -------------
# Validation Epoch
#--------------
# این تابع تقریباً همان warm-up است، ولی:
# بدون backprop
# بدون optimizer step
# بدون scheduler step
@torch.no_grad()
def validate_transformer_epoch(transformer, dataloader, device, pad_idx):
    transformer.eval()

    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)

        x_in = input_ids[:, :-1]
        y_out = input_ids[:, 1:]

        logits, _ = transformer(x_in)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y_out.reshape(-1),
            ignore_index=pad_idx,
        )

        acc = token_accuracy(logits, y_out, pad_idx)

        total_loss += loss.item()
        total_acc += acc
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_acc = total_acc / max(num_batches, 1)

    return {
        "loss": avg_loss,
        "token_acc": avg_acc,
        "perplexity": perplexity_from_loss(avg_loss),
    }

# -------------
# Joint Training Epoch
#--------------
# Trains GCN + Transformer + Projection Heads together
def train_joint_epoch(
    gcn,
    transformer,
    gcn_proj,
    trans_proj,
    dataloader,
    optimizer,
    scheduler,
    scaler,
    device,
    pad_idx,
    lambda_contrastive=1.0,
    gamma_pair=0.25,
    max_grad_norm=1.0,
):
    gcn.train()
    transformer.train()
    gcn_proj.train()
    trans_proj.train()

    meters = {
        "total_loss": 0.0,
        "token_loss": 0.0,
        "contrastive_loss": 0.0,
        "pair_loss": 0.0,
        "token_acc": 0.0,
    }
    num_batches = 0

    for batch in dataloader:
        graph_batch = batch["graph"].to(device)
        input_ids = batch["input_ids"].to(device)
        eos_positions = batch["eos_positions"].to(device)

        x_in = input_ids[:, :-1]
        y_out = input_ids[:, 1:]
        # for x_in that was shifted
        shifted_eos_positions = eos_positions - 1
        # for eos position to not be negative
        shifted_eos_positions = shifted_eos_positions.clamp(min=0)

        optimizer.zero_grad(set_to_none=True)

        # for mixed precision
        with autocast():
            gcn_rep = gcn(graph_batch)
            seq_logits, _ = transformer(x_in)
            # نمایش فشرده توالی SELFIES برای هر مولکول
            trans_rep = transformer.get_representation(x_in, eos_positions=shifted_eos_positions)

            # Going to same dimension
            proj_gcn = gcn_proj(gcn_rep)
            proj_trans = trans_proj(trans_rep)

            loss_token = F.cross_entropy(
                seq_logits.reshape(-1, seq_logits.size(-1)),
                y_out.reshape(-1),
                ignore_index=pad_idx,
            )
            loss_contrastive, loss_pair = contrastive_loss(proj_gcn, proj_trans)
            loss = loss_token + lambda_contrastive * loss_contrastive + gamma_pair * loss_pair

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(gcn.parameters()) +
            list(transformer.parameters()) +
            list(gcn_proj.parameters()) +
            list(trans_proj.parameters()),
            max_grad_norm,
        )
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        acc = token_accuracy(seq_logits, y_out, pad_idx)

        meters["total_loss"] += loss.item()
        meters["token_loss"] += loss_token.item()
        meters["contrastive_loss"] += loss_contrastive.item()
        meters["pair_loss"] += loss_pair.item()
        meters["token_acc"] += acc
        num_batches += 1

    for k in meters:
        meters[k] /= max(num_batches, 1)
    # perplexity is for token loss
    meters["perplexity"] = perplexity_from_loss(meters["token_loss"])

    return meters

# -------------
# Joint Validation Epoch
#--------------
@torch.no_grad()
def validate_joint_epoch(
    gcn,
    transformer,
    gcn_proj,
    trans_proj,
    dataloader,
    device,
    pad_idx,
    lambda_contrastive=1.0,
    gamma_pair=0.25,
):
    gcn.eval()
    transformer.eval()
    gcn_proj.eval()
    trans_proj.eval()

    meters = {
        "total_loss": 0.0,
        "token_loss": 0.0,
        "contrastive_loss": 0.0,
        "pair_loss": 0.0,
        "token_acc": 0.0,
    }
    num_batches = 0

    for batch in dataloader:
        graph_batch = batch["graph"].to(device)
        input_ids = batch["input_ids"].to(device)
        eos_positions = batch["eos_positions"].to(device)

        x_in = input_ids[:, :-1]
        y_out = input_ids[:, 1:]
        shifted_eos_positions = (eos_positions - 1).clamp(min=0)

        gcn_rep = gcn(graph_batch)
        seq_logits, _ = transformer(x_in)
        trans_rep = transformer.get_representation(x_in, eos_positions=shifted_eos_positions)

        proj_gcn = gcn_proj(gcn_rep)
        proj_trans = trans_proj(trans_rep)

        loss_token = F.cross_entropy(
            seq_logits.reshape(-1, seq_logits.size(-1)),
            y_out.reshape(-1),
            ignore_index=pad_idx,
        )
        loss_contrastive, loss_pair = contrastive_loss(proj_gcn, proj_trans)
        loss = loss_token + lambda_contrastive * loss_contrastive + gamma_pair * loss_pair

        acc = token_accuracy(seq_logits, y_out, pad_idx)

        meters["total_loss"] += loss.item()
        meters["token_loss"] += loss_token.item()
        meters["contrastive_loss"] += loss_contrastive.item()
        meters["pair_loss"] += loss_pair.item()
        meters["token_acc"] += acc
        num_batches += 1

    for k in meters:
        meters[k] /= max(num_batches, 1)
    meters["perplexity"] = perplexity_from_loss(meters["token_loss"])

    return meters

# -------------
# Validation based Early Stopping
#--------------
class EarlyStopping:
    def __init__(self, patience=8, mode="min"):
        # after 8 epochs default, if there's no improvement it early stops
        self.patience = patience
        self.mode = mode
        self.best = None
        self.bad_epochs = 0

    def step(self, value):
        improved = False

        if self.best is None:
            improved = True
        elif self.mode == "min" and value < self.best:
            improved = True
        elif self.mode == "max" and value > self.best:
            improved = True

        if improved:
            self.best = value
            self.bad_epochs = 0
            # return stop, improved
            return False, True

        self.bad_epochs += 1
        stop = self.bad_epochs >= self.patience
        # return stop, improved
        return stop, False

#---------------
# Main train driver
#---------------
def fit_all(
    gcn,
    transformer,
    gcn_proj,
    trans_proj,
    train_loader,
    val_loader,
    device,
    pad_idx,
    warmup_epochs=15,
    joint_epochs=40,
    lr=3e-4,
    weight_decay=1e-4,
    lambda_contrastive=1.0,
    gamma_pair=0.25,
):
    # For plotting and report
    history = {
        "warmup_train_loss": [],
        "warmup_val_loss": [],
        "joint_train_total": [],
        "joint_val_total": [],
        "joint_train_token": [],
        "joint_val_token": [],
        "joint_train_contrastive": [],
        "joint_val_contrastive": [],
        "joint_train_pair": [],
        "joint_val_pair": [],
        "joint_train_acc": [],
        "joint_val_acc": [],
        "joint_train_ppl": [],
        "joint_val_ppl": [],
    }

    # for mixed precision
    scaler = GradScaler()

    # AdamW is good for transformers
    warmup_optimizer = AdamW(transformer.parameters(), lr=lr, weight_decay=weight_decay)
    # Cosine learning rate decay
    warmup_scheduler = CosineAnnealingLR(
        warmup_optimizer,
        #  happens warmup_epochs×num_batches
        T_max=warmup_epochs * len(train_loader),
    )

    for epoch in range(warmup_epochs):
        train_metrics = train_transformer_warmup_epoch(
            transformer, train_loader, warmup_optimizer, warmup_scheduler, scaler, device, pad_idx
        )
        val_metrics = validate_transformer_epoch(transformer, val_loader, device, pad_idx)

        history["warmup_train_loss"].append(train_metrics["loss"])
        history["warmup_val_loss"].append(val_metrics["loss"])

        print(
            f"[Warmup {epoch+1:02d}/{warmup_epochs}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['token_acc']:.4f} "
            f"val_ppl={val_metrics['perplexity']:.4f}"
        )

    # joint training: add all parameters so all train with one optimizer
    joint_params = (
        list(gcn.parameters()) +
        list(transformer.parameters()) +
        list(gcn_proj.parameters()) +
        list(trans_proj.parameters())
    )
    joint_optimizer = AdamW(joint_params, lr=lr, weight_decay=weight_decay)
    joint_scheduler = CosineAnnealingLR(
        joint_optimizer,
        T_max=joint_epochs * len(train_loader),
    )

    stopper = EarlyStopping(patience=8, mode="min")
    best_state = None

    for epoch in range(joint_epochs):
        train_metrics = train_joint_epoch(
            gcn, transformer, gcn_proj, trans_proj, train_loader,
            joint_optimizer, joint_scheduler, scaler, device, pad_idx,
            lambda_contrastive=lambda_contrastive,
            gamma_pair=gamma_pair,
        )
        val_metrics = validate_joint_epoch(
            gcn, transformer, gcn_proj, trans_proj, val_loader, device, pad_idx,
            lambda_contrastive=lambda_contrastive,
            gamma_pair=gamma_pair,
        )

        history["joint_train_total"].append(train_metrics["total_loss"])
        history["joint_val_total"].append(val_metrics["total_loss"])
        history["joint_train_token"].append(train_metrics["token_loss"])
        history["joint_val_token"].append(val_metrics["token_loss"])
        history["joint_train_contrastive"].append(train_metrics["contrastive_loss"])
        history["joint_val_contrastive"].append(val_metrics["contrastive_loss"])
        history["joint_train_pair"].append(train_metrics["pair_loss"])
        history["joint_val_pair"].append(val_metrics["pair_loss"])
        history["joint_train_acc"].append(train_metrics["token_acc"])
        history["joint_val_acc"].append(val_metrics["token_acc"])
        history["joint_train_ppl"].append(train_metrics["perplexity"])
        history["joint_val_ppl"].append(val_metrics["perplexity"])

        print(
            f"[Joint {epoch+1:02d}/{joint_epochs}] "
            f"train_total={train_metrics['total_loss']:.4f} "
            f"val_total={val_metrics['total_loss']:.4f} "
            f"val_token={val_metrics['token_loss']:.4f} "
            f"val_ctr={val_metrics['contrastive_loss']:.4f} "
            f"val_pair={val_metrics['pair_loss']:.4f} "
            f"val_acc={val_metrics['token_acc']:.4f} "
            f"val_ppl={val_metrics['perplexity']:.4f}"
        )

        stop, improved = stopper.step(val_metrics["total_loss"])
        if improved:
            best_state = {
                "gcn": gcn.state_dict(),
                "transformer": transformer.state_dict(),
                "gcn_proj": gcn_proj.state_dict(),
                "trans_proj": trans_proj.state_dict(),
            }

        if stop:
            print("Early stopping triggered.")
            break

    return history, best_state