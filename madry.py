"""Minimal reproduction of Madry et al., "Towards Deep Learning Models Resistant
to Adversarial Attacks" (ICLR 2018), on MNIST with an L-infinity threat model.

Everything the notebook needs lives in this one file:
  data      -> load_mnist, batches
  model     -> MadryCNN
  attacks   -> fgsm, pgd, pgd_loss_curves
  training  -> train            (natural / FGSM / PGD adversarial training)
  metrics   -> evaluate, transfer_accuracy
"""
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 0.3      # L-inf radius on [0, 1] pixels (paper, MNIST)
ALPHA = 0.01   # PGD step size (paper, MNIST)


# --------------------------------------------------------------------------
# Data: MNIST is tiny, so we keep the whole set on the GPU as two tensors.
# This is much faster than a DataLoader and easier to read.
# --------------------------------------------------------------------------
def load_mnist(root="data", device="cuda"):
    from torchvision.datasets import MNIST

    def to_tensors(ds):
        x = ds.data.float().div(255.0).unsqueeze(1)   # (N, 1, 28, 28), values in [0, 1]
        y = ds.targets.long()
        return x.to(device), y.to(device)

    train = to_tensors(MNIST(root, train=True, download=True))
    test = to_tensors(MNIST(root, train=False, download=True))
    return train, test


def batches(x, y, batch_size, generator=None):
    """Yield mini-batches. If a CPU generator is given, the order is shuffled."""
    n = len(x)
    if generator is not None:
        idx = torch.randperm(n, generator=generator).to(x.device)
    else:
        idx = torch.arange(n, device=x.device)
    for i in range(0, n, batch_size):
        j = idx[i:i + batch_size]
        yield x[j], y[j]


# --------------------------------------------------------------------------
# Model: the paper's MNIST network.
#   scale=16 -> conv(32) -> pool -> conv(64) -> pool -> fc(1024) -> fc(10)
#   (exactly the architecture of the paper / official MadryLab code).
#   scale=s  -> 2s, 4s filters and 64s hidden units: the capacity family of
#   Section 4 of the paper (s = 1, 2, 4, 8, 16).
# No BatchNorm / Dropout, so train() and eval() modes behave identically.
# --------------------------------------------------------------------------
class MadryCNN(nn.Module):
    def __init__(self, scale=16):
        super().__init__()
        c1, c2, hidden = 2 * scale, 4 * scale, 64 * scale
        self.conv1 = nn.Conv2d(1, c1, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(7 * 7 * c2, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)   # 28x28 -> 14x14
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)   # 14x14 -> 7x7
        x = F.relu(self.fc1(x.flatten(1)))
        return self.fc2(x)                           # logits


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------
# Attacks. Both maximise the cross-entropy loss inside the L-inf ball
#   S = { delta : ||delta||_inf <= eps }  and keep pixels in [0, 1].
# torch.autograd.grad only returns d loss / d input, so the model's
# parameter .grad fields are never touched by the attack.
# --------------------------------------------------------------------------
def project(x_adv, x, eps):
    """Projection onto the eps-ball around x, then onto the valid image box."""
    x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
    return x_adv.clamp(0.0, 1.0)


def fgsm(model, x, y, eps=EPS):
    """Fast Gradient Sign Method: one step of size eps (Goodfellow et al.)."""
    x_req = x.clone().requires_grad_(True)
    loss = F.cross_entropy(model(x_req), y)
    grad, = torch.autograd.grad(loss, x_req)
    return (x + eps * grad.sign()).clamp(0.0, 1.0).detach()


def pgd(model, x, y, eps=EPS, alpha=ALPHA, steps=40, random_start=True):
    """Projected Gradient Descent (on the negative loss), Madry et al. Eq. (3):
         x^{t+1} = Proj_{x+S}( x^t + alpha * sign(grad_x L(theta, x^t, y)) )
    starting from a uniformly random point in the eps-ball."""
    if random_start:
        x_adv = project(x + torch.empty_like(x).uniform_(-eps, eps), x, eps)
    else:
        x_adv = x.clone()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad, = torch.autograd.grad(loss, x_adv)
        x_adv = project(x_adv.detach() + alpha * grad.sign(), x, eps)
    return x_adv.detach()


def pgd_loss_curves(model, x1, y1, restarts=1000, steps=40, eps=EPS, alpha=ALPHA):
    """Run `restarts` independent PGD attacks on ONE image (batched) and return
    the per-example loss after every step: tensor of shape (restarts, steps+1).
    Used for the paper's Figure 1 (loss values concentrate across restarts)."""
    x = x1.unsqueeze(0).repeat(restarts, 1, 1, 1)
    y = y1.view(1).repeat(restarts)
    x_adv = project(x + torch.empty_like(x).uniform_(-eps, eps), x, eps)
    curves = []
    for t in range(steps + 1):
        x_adv.requires_grad_(True)
        losses = F.cross_entropy(model(x_adv), y, reduction="none")
        curves.append(losses.detach())
        if t == steps:
            break
        grad, = torch.autograd.grad(losses.sum(), x_adv)
        x_adv = project(x_adv.detach() + alpha * grad.sign(), x, eps)
    return torch.stack(curves, dim=1).cpu()


# --------------------------------------------------------------------------
# Training: min_theta E[ max_{delta in S} L(theta, x + delta, y) ]  (Eq. 2)
#   inner max  -> approximated by the attack (PGD-40 for the paper's method)
#   outer min  -> Adam step on the loss of the adversarial batch
# A checkpoint is written after every epoch; calling train() again with the
# same ckpt_path resumes from the last finished epoch.
# --------------------------------------------------------------------------
def make_training_batch(model, x, y, mode, steps, alpha, eps):
    if mode == "natural":
        return x
    if mode == "fgsm":
        return fgsm(model, x, y, eps)
    if mode == "pgd":
        return pgd(model, x, y, eps, alpha, steps, random_start=True)
    raise ValueError(f"unknown mode {mode}")


def train(mode, scale, seed, epochs, ckpt_path, train_data,
          batch_size=50, lr=1e-4, steps=40, alpha=ALPHA, eps=EPS,
          log_every=0, device="cuda"):
    x_train, y_train = train_data
    torch.manual_seed(seed)                       # init + attack random starts
    model = MadryCNN(scale).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    order = torch.Generator().manual_seed(seed)   # data order
    history, start = [], 0

    if os.path.exists(ckpt_path):
        # map to CPU: RNG states must stay CPU ByteTensors; load_state_dict
        # copies weights / Adam moments to the model's device by itself.
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        order.set_state(state["order_rng"])
        torch.set_rng_state(state["cpu_rng"])
        if torch.cuda.is_available() and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state(state["cuda_rng"])
        history, start = state["history"], state["epoch"]
        if start < epochs:
            print(f"  resuming {os.path.basename(ckpt_path)} from epoch {start}")

    for epoch in range(start, epochs):
        t0 = time.time()
        loss_sum, correct, seen = 0.0, 0, 0
        for it, (xb, yb) in enumerate(batches(x_train, y_train, batch_size, order)):
            xa = make_training_batch(model, xb, yb, mode, steps, alpha, eps)
            logits = model(xa)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            seen += len(yb)
            if log_every and (it + 1) % log_every == 0:
                print(f"    it {it + 1}: train loss {loss_sum / seen:.3f}, "
                      f"train acc (on {mode} inputs) {correct / seen:.3f}")
        history.append({"epoch": epoch + 1, "train_loss": loss_sum / seen,
                        "train_acc": correct / seen, "seconds": time.time() - t0})
        print(f"  [{mode} s={scale} seed={seed}] epoch {epoch + 1}/{epochs} "
              f"loss {history[-1]['train_loss']:.4f} acc {history[-1]['train_acc']:.4f} "
              f"({history[-1]['seconds']:.0f}s)")
        state = {"model": model.state_dict(), "opt": opt.state_dict(),
                 "epoch": epoch + 1, "history": history,
                 "order_rng": order.get_state(), "cpu_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                 "config": {"mode": mode, "scale": scale, "seed": seed, "lr": lr,
                            "batch_size": batch_size, "steps": steps, "alpha": alpha, "eps": eps}}
        tmp = ckpt_path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, ckpt_path)               # atomic: a crash never leaves a broken file
    return model, history


def load_model(ckpt_path, device="cuda"):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MadryCNN(state["config"]["scale"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


# --------------------------------------------------------------------------
# Evaluation.
# An example counts as robust only if it is classified correctly
#   (a) without attack and (b) after EVERY random restart.
# --------------------------------------------------------------------------
def evaluate(model, x, y, attack="none", eps=EPS, alpha=ALPHA, steps=100,
             restarts=1, batch_size=1000, seed=12345):
    torch.manual_seed(seed)          # same random starts for every model
    model.eval()
    correct = 0
    for xb, yb in batches(x, y, batch_size):
        with torch.no_grad():
            ok = model(xb).argmax(1) == yb
        if attack == "fgsm":
            xa = fgsm(model, xb, yb, eps)
            with torch.no_grad():
                ok &= model(xa).argmax(1) == yb
        elif attack == "pgd":
            for _ in range(restarts):
                xa = pgd(model, xb, yb, eps, alpha, steps, random_start=True)
                with torch.no_grad():
                    ok &= model(xa).argmax(1) == yb
        elif attack != "none":
            raise ValueError(attack)
        correct += ok.sum().item()
    return correct / len(x)


def transfer_accuracy(source, target, x, y, eps=EPS, alpha=ALPHA, steps=100,
                      batch_size=1000, seed=12345):
    """Black-box (transfer) attack: craft PGD examples on `source`,
    measure the accuracy of `target` on them."""
    torch.manual_seed(seed)
    correct = 0
    for xb, yb in batches(x, y, batch_size):
        xa = pgd(source, xb, yb, eps, alpha, steps, random_start=True)
        with torch.no_grad():
            correct += (target(xa).argmax(1) == yb).sum().item()
    return correct / len(x)


# --------------------------------------------------------------------------
# Small helpers for saving results
# --------------------------------------------------------------------------
def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def load_json(path):
    with open(path) as f:
        return json.load(f)
