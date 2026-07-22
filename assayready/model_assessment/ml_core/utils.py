import math
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup_and_cooldown(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cooldown_steps: int = 0,
    min_lr_ratio: float = 0.1,
    num_cycles: float = 0.5
):
    """Create a schedule with linear warmup, cosine annealing, and optional cooldown."""
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        if num_cooldown_steps > 0 and current_step >= num_training_steps - num_cooldown_steps:
            cooldown_progress = (current_step - (num_training_steps - num_cooldown_steps)) / num_cooldown_steps
            return max(min_lr_ratio, 1.0 - (1.0 - min_lr_ratio) * cooldown_progress)
        
        progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps - num_cooldown_steps)
        progress = min(progress, 1.0)
        
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        return max(min_lr_ratio, cosine_decay)
    
    return LambdaLR(optimizer, lr_lambda)


def get_one_cycle_schedule(
    optimizer,
    max_lr: float,
    num_training_steps: int,
    pct_start: float = 0.3,
    div_factor: float = 25.0,
    final_div_factor: float = 1e4
):
    """Create a one-cycle schedule that ramps up and then down."""
    def lr_lambda(current_step: int):
        if current_step < pct_start * num_training_steps:
            progress = current_step / (pct_start * num_training_steps)
            return (1.0 - progress) / div_factor + progress
        
        else:
            progress = (current_step - pct_start * num_training_steps) / ((1.0 - pct_start) * num_training_steps)
            return (1.0 - progress) + progress / final_div_factor
    
    return LambdaLR(optimizer, lr_lambda)


class EMA:
    """Exponential moving average with linear decay warmup."""
    def __init__(self, model, decay=0.999, warmup_steps=2000):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.num_updates = 0
        self.shadow = {name: param.clone().detach().to(param.device)                      for name, param in model.named_parameters()                      if param.requires_grad}
        self.backup = {}

    def get_decay(self):
        """Compute decay with warmup."""
        if self.num_updates < self.warmup_steps:
            # Linear warmup
            return self.decay * (self.num_updates / self.warmup_steps)
        return self.decay

    def update(self, model):
        """Update shadow weights with current model weights."""
        self.num_updates += 1
        decay = self.get_decay()
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    if self.shadow[name].device != param.device:
                        self.shadow[name] = self.shadow[name].to(param.device)
                    self.shadow[name].mul_(decay).add_(param.data, alpha=1 - decay)

    def apply_shadow(self, model):
        """Apply shadow weights to model (for evaluation)."""
        self.backup = {name: param.clone().detach() 
                      for name, param in model.named_parameters() 
                      if param.requires_grad}
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore original weights from backup."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.backup:
                    param.data.copy_(self.backup[name])
        self.backup = {}


@torch.no_grad()
def evaluate(
    model,
    eval_loader,
    device,
    amp_enabled,
    epoch,
    tok,
    kmer_rc_tables=None,
    lambda_rc=0.0,
    max_batches: int | None = 500,
    strict_eval: bool = True,
):
    """Evaluate masked-token accuracy, calibration, and sequence baselines."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    strict_total_loss = 0.0
    strict_total_correct = 0
    strict_total_top5 = 0
    strict_total_count = 0
    unigram_counts = None
    log2 = math.log(2.0)
    
    def _single_base_token_id(tokenizer, base: str) -> int | None:
        backend = getattr(tokenizer, "tokenizer", None)
        if backend is not None and hasattr(backend, "token_to_id"):
            return backend.token_to_id(base)

        if getattr(tokenizer, "tokenizer_mode", None) == "base":
            try:
                encoded = tokenizer.encode(base, max_length=3, return_tensors=None, padding=False)
            except Exception:
                return None
            for token_id in encoded.get("input_ids", []):
                token_id = int(token_id)
                if token_id > int(getattr(tokenizer, "special_tokens_max_id", 4)):
                    return token_id
        return None

    # Single-base token metrics. For BPE, these only cover literal single-base
    # BPE labels; for base tokenization, they cover the canonical A/C/G/T labels.
    base_names = ["A", "C", "G", "T"]
    base_token_ids = [_single_base_token_id(tok, base) for base in base_names]
    per_class_correct = [0, 0, 0, 0]
    per_class_total = [0, 0, 0, 0]
    vocab_size = int(getattr(tok, "vocab_size", 4096))
    strict_base_tokens = getattr(tok, "tokenizer_mode", None) == "base" and vocab_size == 4
    markov_stats = {}
    if strict_base_tokens:
        for order in (1, 2, 3):
            contexts = vocab_size ** order
            markov_stats[order] = {
                "train": torch.zeros((contexts, vocab_size), dtype=torch.float64),
                "score": torch.zeros((contexts, vocab_size), dtype=torch.float64),
            }

    def _loss_to_bits(loss_value: float) -> float:
        return float(loss_value) / log2

    def _update_markov_stats(
        original_ids: torch.Tensor,
        attn_mask: torch.Tensor,
        score_mask: torch.Tensor,
    ) -> None:
        if not markov_stats:
            return
        seq_len = int(original_ids.size(1))
        attn_bool = attn_mask.bool()
        score_bool = score_mask.bool()
        for order, stats in markov_stats.items():
            if seq_len <= order:
                continue
            target_ids = original_ids[:, order:]
            valid = attn_bool[:, order:] & (target_ids >= 0) & (target_ids < vocab_size)
            context_idx = torch.zeros_like(target_ids, dtype=torch.long)
            width = seq_len - order
            for offset in range(order):
                prev = original_ids[:, offset : offset + width]
                valid = valid & attn_bool[:, offset : offset + width] & (prev >= 0) & (prev < vocab_size)
                context_idx = context_idx * vocab_size + prev.clamp(0, vocab_size - 1).long()

            if valid.any():
                flat = (context_idx[valid] * vocab_size + target_ids[valid].long()).reshape(-1)
                counts = torch.bincount(flat, minlength=stats["train"].numel()).to(
                    device="cpu",
                    dtype=torch.float64,
                )
                stats["train"] += counts.reshape_as(stats["train"])

            score_valid = valid & score_bool[:, order:]
            if score_valid.any():
                flat_score = (context_idx[score_valid] * vocab_size + target_ids[score_valid].long()).reshape(-1)
                score_counts = torch.bincount(flat_score, minlength=stats["score"].numel()).to(
                    device="cpu",
                    dtype=torch.float64,
                )
                stats["score"] += score_counts.reshape_as(stats["score"])
    
    confidence_buckets = [0.0] * 10  # 10 buckets for confidence
    accuracy_buckets = [0.0] * 10
    counts_buckets = [0] * 10

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

    eval_generator = torch.Generator(device=device).manual_seed(42 + epoch)
    
    max_steps = None if max_batches is None or max_batches <= 0 else int(max_batches)
    evaluated_batches = 0

    for batch_idx, batch in enumerate(eval_loader):
        if max_steps is not None and batch_idx >= max_steps:
            break
        evaluated_batches += 1
            
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        kmer_ids = batch.get("kmer_ids")
        if kmer_ids is not None:
            kmer_ids = kmer_ids.to(device, non_blocking=True)

        from .masking import gpu_span_mask
        from ..data_core.config import IGNORE_INDEX

        strict_generator = None
        if strict_eval:
            try:
                strict_state = eval_generator.get_state()
                strict_generator = torch.Generator(device=device)
                strict_generator.set_state(strict_state)
            except Exception:
                strict_generator = torch.Generator(device=device).manual_seed(420_000 + epoch * 100_000 + batch_idx)

        mask_token_id = None if strict_base_tokens else getattr(tok, "mask_token_id", 4)
        x_masked, labels, kmer_ids_masked, mask_bool = gpu_span_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
            kmer_ids=kmer_ids,
            kmer_k=getattr(tok, "k_mer_size", None),
            kmer_pad_id=getattr(tok, "kmer_pad_id", None),
            vocab_size=vocab_size,
            mask_fraction=0.20,
            mean_span_length=3.0,
            generator=eval_generator,
            min_frac=0.20,
            max_frac=0.25,
            mask_kmer_ids=True,
            special_tokens_max_id=getattr(tok, "special_tokens_max_id", 4),
            mask_token_id=mask_token_id,
            mask_replace_prob=0.0 if strict_base_tokens else 0.8,
            random_replace_prob=0.0 if strict_base_tokens else 0.1,
            leave_unmasked_prob=1.0 if strict_base_tokens else 0.1,
        )

        with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype, enabled=bool(amp_enabled)):
            outputs = model(
                x_masked, 
                attention_mask=attention_mask, 
                kmer_ids=kmer_ids_masked,
                input_mask_bool=mask_bool if strict_base_tokens else None,
            )
            logits = outputs.get("logits")

            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
            valid_tokens = (labels != IGNORE_INDEX).sum()
            if valid_tokens == 0:
                loss = torch.tensor(0.0, device=loss.device)

        total_loss += float(loss.item())

        pred = logits.argmax(dim=-1)
        masked = (labels != IGNORE_INDEX) & (attention_mask == 1)
        correct = (pred == labels) & masked

        total_correct += int(correct.sum().item())
        total_count += int(masked.sum().item())
        
        # ECE Tracking
        if masked.any():
            m_logits = logits[masked]
            m_labels = labels[masked]
            
            probs = torch.softmax(m_logits, dim=-1)
            conf, pred_m = probs.max(dim=-1)
            correct_m = (pred_m == m_labels)
            
            b_idx = (conf * 10).long().clamp(max=9)
            for b_id in range(10):
                in_bucket = (b_idx == b_id)
                if in_bucket.any():
                    confidence_buckets[b_id] += conf[in_bucket].sum().item()
                    accuracy_buckets[b_id] += correct_m[in_bucket].sum().item()
                    counts_buckets[b_id] += in_bucket.sum().item()
        
        # Per-class accuracy
        for i, token_id in enumerate(base_token_ids):
            if token_id is None:
                continue
            class_mask = (labels == int(token_id)) & masked
            per_class_correct[i] += int((pred == int(token_id))[class_mask].sum().item())
            per_class_total[i] += int(class_mask.sum().item())

        if strict_eval:
            x_strict, strict_labels, strict_kmer_ids, _ = gpu_span_mask(
                input_ids=input_ids,
                attention_mask=attention_mask,
                kmer_ids=kmer_ids,
                kmer_k=getattr(tok, "k_mer_size", None),
                kmer_pad_id=getattr(tok, "kmer_pad_id", None),
                vocab_size=vocab_size,
                mask_fraction=0.20,
                mean_span_length=3.0,
                generator=strict_generator,
                min_frac=0.20,
                max_frac=0.25,
                mask_kmer_ids=True,
                special_tokens_max_id=getattr(tok, "special_tokens_max_id", 4),
                mask_token_id=mask_token_id,
                mask_replace_prob=0.0 if strict_base_tokens else 1.0,
                random_replace_prob=0.0,
                leave_unmasked_prob=1.0 if strict_base_tokens else 0.0,
            )
            with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype, enabled=bool(amp_enabled)):
                strict_outputs = model(
                    x_strict,
                    attention_mask=attention_mask,
                    kmer_ids=strict_kmer_ids,
                    input_mask_bool=(strict_labels != IGNORE_INDEX) if strict_base_tokens else None,
                )
                strict_logits = strict_outputs.get("logits")
                strict_loss = torch.nn.functional.cross_entropy(
                    strict_logits.view(-1, strict_logits.size(-1)),
                    strict_labels.view(-1),
                    ignore_index=IGNORE_INDEX,
                    reduction="sum",
                )
                strict_valid_tokens = (strict_labels != IGNORE_INDEX).sum()
                if strict_valid_tokens == 0:
                    strict_loss = torch.tensor(0.0, device=strict_loss.device)

            strict_masked = (strict_labels != IGNORE_INDEX) & (attention_mask == 1)
            strict_total_loss += float(strict_loss.item())
            strict_count = int(strict_masked.sum().item())
            strict_total_count += strict_count
            _update_markov_stats(input_ids, attention_mask, strict_masked)
            if strict_count > 0:
                strict_pred = strict_logits.argmax(dim=-1)
                strict_total_correct += int(((strict_pred == strict_labels) & strict_masked).sum().item())

                masked_logits = strict_logits[strict_masked]
                masked_labels = strict_labels[strict_masked]
                top_k = min(5, int(masked_logits.size(-1)))
                topk = masked_logits.topk(top_k, dim=-1).indices
                strict_total_top5 += int((topk == masked_labels.unsqueeze(-1)).any(dim=-1).sum().item())

                batch_counts = torch.bincount(
                    masked_labels.detach().to("cpu"),
                    minlength=int(masked_logits.size(-1)),
                )
                if unigram_counts is None:
                    unigram_counts = torch.zeros_like(batch_counts)
                if unigram_counts.numel() < batch_counts.numel():
                    unigram_counts = torch.nn.functional.pad(
                        unigram_counts,
                        (0, batch_counts.numel() - unigram_counts.numel()),
                    )
                unigram_counts[: batch_counts.numel()] += batch_counts

    metrics = {
        "loss": total_loss / max(1, total_count),
        "accuracy": total_correct / max(1, total_count),
        "evaluated_batches": evaluated_batches,
        "max_batches": max_steps,
    }
    metrics["loss_bits_per_token"] = _loss_to_bits(metrics["loss"])
    if strict_base_tokens:
        metrics["loss_bits_per_base"] = metrics["loss_bits_per_token"]

    ece = 0.0
    if total_count > 0:
        for b_id in range(10):
            if counts_buckets[b_id] > 0:
                avg_conf = confidence_buckets[b_id] / counts_buckets[b_id]
                avg_acc = accuracy_buckets[b_id] / counts_buckets[b_id]
                ece += (counts_buckets[b_id] / total_count) * abs(avg_acc - avg_conf)
    metrics["ece"] = ece

    metrics["single_base_token_count"] = int(sum(per_class_total))
    for c, name in enumerate(base_names):
        total = int(per_class_total[c])
        if total > 0:
            metrics[f"accuracy_{name}"] = per_class_correct[c] / total
            metrics[f"accuracy_{name}_count"] = total

    metrics["acc"] = metrics["accuracy"]
    try:
        metrics["ppl"] = math.exp(metrics["loss"])
    except OverflowError:
        metrics["ppl"] = float("inf")

    if strict_eval:
        strict_loss = strict_total_loss / max(1, strict_total_count)
        metrics["strict_eval_loss"] = strict_loss
        metrics["strict_eval_bits_per_token"] = _loss_to_bits(strict_loss)
        if strict_base_tokens:
            metrics["strict_eval_bits_per_base"] = metrics["strict_eval_bits_per_token"]
        metrics["strict_eval_acc"] = strict_total_correct / max(1, strict_total_count)
        metrics["strict_eval_top5"] = strict_total_top5 / max(1, strict_total_count)
        metrics["strict_eval_count"] = strict_total_count
        try:
            metrics["strict_eval_ppl"] = math.exp(strict_loss)
        except OverflowError:
            metrics["strict_eval_ppl"] = float("inf")

        if unigram_counts is not None and int(unigram_counts.sum().item()) > 0:
            counts = unigram_counts.to(dtype=torch.float64)
            total = float(counts.sum().item())
            probs = counts / total
            nonzero = probs > 0
            metrics["unigram_loss"] = float((-(counts[nonzero] * probs[nonzero].log()).sum() / total).item())
            metrics["unigram_bits_per_token"] = _loss_to_bits(metrics["unigram_loss"])
            if strict_base_tokens:
                metrics["unigram_bits_per_base"] = metrics["unigram_bits_per_token"]
            metrics["unigram_top1_acc"] = float(counts.max().item() / total)
            top_k = min(5, int(counts.numel()))
            metrics["unigram_top5_acc"] = float(torch.topk(counts, top_k).values.sum().item() / total)
            try:
                metrics["unigram_ppl"] = math.exp(metrics["unigram_loss"])
            except OverflowError:
                metrics["unigram_ppl"] = float("inf")

        for order, stats in markov_stats.items():
            score_total = float(stats["score"].sum().item())
            if score_total <= 0:
                continue
            smoothed = stats["train"] + 1.0
            probs = smoothed / smoothed.sum(dim=-1, keepdim=True).clamp_min(1.0)
            loss = float((-(stats["score"] * probs.log()).sum() / score_total).item())
            context_ids = torch.arange(stats["train"].size(0))
            top1 = stats["train"].argmax(dim=-1)
            top1_correct = float(stats["score"][context_ids, top1].sum().item())
            prefix = f"markov_order{order}"
            metrics[f"{prefix}_loss"] = loss
            metrics[f"{prefix}_bits_per_token"] = _loss_to_bits(loss)
            metrics[f"{prefix}_top1_acc"] = top1_correct / score_total
            metrics[f"{prefix}_count"] = int(score_total)
            if strict_base_tokens:
                metrics[f"{prefix}_bits_per_base"] = metrics[f"{prefix}_bits_per_token"]

        if "strict_eval_bits_per_base" in metrics and "unigram_bits_per_base" in metrics:
            metrics["strict_eval_bits_per_base_gain_vs_unigram"] = (
                metrics["unigram_bits_per_base"] - metrics["strict_eval_bits_per_base"]
            )
        if "strict_eval_bits_per_base" in metrics:
            for order in (1, 2, 3):
                key = f"markov_order{order}_bits_per_base"
                if key in metrics:
                    metrics[f"strict_eval_bits_per_base_gain_vs_markov_order{order}"] = (
                        metrics[key] - metrics["strict_eval_bits_per_base"]
                    )

    return metrics

def get_scheduler(optimizer, num_warmup_steps, num_training_steps, kind: str = "linear"):
    """Backward compatibility wrapper."""
    if kind == "cosine":
        return get_cosine_schedule_with_warmup_and_cooldown(
            optimizer, num_warmup_steps, num_training_steps,
            num_cooldown_steps=int(num_training_steps * 0.1),  # 10% cooldown
            min_lr_ratio=0.1
        )
    elif kind == "onecycle":
        base_lr = optimizer.param_groups[0]['lr']
        return get_one_cycle_schedule(optimizer, base_lr, num_training_steps)
    else:
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(0.0, float(num_training_steps - current_step) / 
                      float(max(1, num_training_steps - num_warmup_steps)))
        return LambdaLR(optimizer, lr_lambda)


def parameter_groups(model: nn.Module, weight_decay: float):
    """Separate weight-decayed parameters from biases and normalization weights."""
    decay = []
    no_decay = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if "bias" in name or "norm" in name or "embedding" in name:
            no_decay.append(param)
        else:
            decay.append(param)
            
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0}
    ]
