"""Counterfactual-aware contrastive loss helpers for TRACER."""

import torch
import torch.nn.functional as F


def info_nce_lg_cf(z_local_w, z_global_w, z_cf_w, projection, tau: float, omega: float) -> torch.Tensor:
    """
    z_*_w: outputs of w_cl (shape [B, h_dim]), same as inputs to projection_model in get_loss_conv.
    projection: model.projection_model.
    """
    device = z_local_w.device
    dtype = z_local_w.dtype
    pl = F.normalize(projection(z_local_w), p=2, dim=1)
    pg = F.normalize(projection(z_global_w), p=2, dim=1)
    pcf = F.normalize(projection(z_cf_w), p=2, dim=1)

    b = pl.shape[0]
    if b == 0:
        return torch.zeros((), device=device, dtype=dtype)

    sim = torch.mm(pl, pg.t()) / tau
    sim_cf = (pl * pcf).sum(dim=1) / tau

    log_omega = torch.log(torch.tensor(max(omega, 1e-12), device=device, dtype=dtype))

    total = torch.zeros((), device=device, dtype=dtype)
    for i in range(b):
        pos = sim[i, i]
        neg_global = torch.cat([sim[i, :i], sim[i, i + 1 :]], dim=0)
        logits = torch.cat([pos.unsqueeze(0), (log_omega + sim_cf[i]).unsqueeze(0), neg_global], dim=0)
        total = total + -(pos - torch.logsumexp(logits, dim=0))

    return total / b


def cf_margin_rank_loss(scores_fact, scores_cf, targets, sample_mask, quality, margin, weight):
    """Intervention-aware margin ranking on true object logits."""
    if weight <= 0 or scores_fact is None or scores_cf is None:
        device = targets.device
        return torch.zeros((), device=device)
    idx = torch.arange(targets.shape[0], device=targets.device)
    fact_o = scores_fact[idx, targets]
    cf_o = scores_cf[idx, targets]
    diff = fact_o - cf_o
    per = F.relu(float(margin) - diff)
    w = sample_mask.float() * quality.float()
    denom = torch.clamp(w.sum(), min=1.0)
    return float(weight) * (w * per).sum() / denom


def cf_consistency_kl(scores_fact, scores_cf, sample_mask, quality, weight):
    """KL on unaffected samples (sample_mask == 0)."""
    if weight <= 0 or scores_fact is None or scores_cf is None:
        return torch.zeros((), device=sample_mask.device)
    p = F.softmax(scores_fact, dim=1)
    q = F.softmax(scores_cf, dim=1)
    kl = (p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))).sum(dim=1)
    w = (1.0 - sample_mask.float()) * quality.float()
    denom = torch.clamp(w.sum(), min=1.0)
    return float(weight) * (w * kl).sum() / denom


def cf_quality_from_delta(cf_delta, triples, tau, beta, no_gate=False):
    """Per-sample quality gate from entity-level CF delta magnitude."""
    b = triples.shape[0]
    device = cf_delta.device
    if no_gate or b == 0:
        return torch.ones(b, device=device)
    ent_div = torch.mean(torch.abs(cf_delta), dim=1)
    triple_div = 0.5 * (ent_div[triples[:, 0]] + ent_div[triples[:, 2]])
    return torch.sigmoid(float(beta) * (triple_div - float(tau)))
