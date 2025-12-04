import copy
import torch
from torch import nn
import torch.nn.functional as F
import itertools
import tqdm
from advertorch.attacks import L2PGDAttack 

from .utils import keys, eval_opt, plot_unlearn_remain_acc_figure, evaluate_model_on_all_loaders
from utils import *
from trainer import *
import log_utils

def compute_fisher_information(loader, model, device="cuda"):
    """
    Computes the Diagonal Fisher Information Matrix (FIM).
    This represents the 'Riemannian Metric' of the parameter space.
    
    F_ii = E [ (dLoss/dTheta_i)^2 ]
    
    High F_ii means moving this parameter increases the Geodesic Distance massively.
    Low F_ii means the parameter is 'flat' and we can move freely (short geodesic).
    """
    fisher = {n: torch.zeros_like(p, device=device) for n, p in model.named_parameters() if p.requires_grad}
    model.eval()
    
    # We only need a subset to approximate the curvature
    num_samples = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        
        # We assume the model predicts a distribution. 
        # For classification, we sample from the predicted distribution (or use labels as proxy).
        # Using true labels is a standard 'Empirical Fisher' approximation.
        loss = F.cross_entropy(outputs, targets)
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.grad is not None:
                    # Accumulate squared gradients
                    fisher[n] += p.grad.pow(2) * len(targets)
        
        num_samples += len(targets)
        if num_samples > 1000: # Limit samples for speed
            break
            
    # Normalize
    fisher = {n: (f / num_samples) for n, f in fisher.items()}
    
    # Normalize to [0, 1] range for stability in optimization
    # (Optional but recommended for hyperparameter tuning)
    all_vals = torch.cat([f.view(-1) for f in fisher.values()])
    f_max = all_vals.max()
    fisher = {n: f / (f_max + 1e-8) for n, f in fisher.items()}
    
    return fisher

def geodesic_dist_loss(model, origin_params, fisher_metric):
    """
    Calculates the Squared Geodesic Distance (Local Quadratic Approximation).
    
    Dist_G^2(theta, theta*) approx (theta - theta*)^T * Fisher * (theta - theta*)
    
    This penalizes movement strictly according to the manifold curvature.
    """
    loss = 0
    for n, p in model.named_parameters():
        if n in origin_params and n in fisher_metric:
            p_diff = p - origin_params[n]
            # Fisher-weighted L2 distance
            loss += torch.sum(fisher_metric[n] * p_diff.pow(2))
    return loss

def get_valid_attack_label(y, num_classes):
    shift = torch.randint(1, num_classes, size=y.shape, device=y.device)
    return (y + shift) % num_classes

def adv_attack(model, adversary, data, target, num_classes):
    model.eval()
    data, target = data.to("cuda"), target.to("cuda")
    attack_label = get_valid_attack_label(target, num_classes)
    adv_example = adversary.perturb(data, attack_label)
    return adv_example.detach(), attack_label

@timer
def geodesic_ul(ori_model, train_forget_loader, train_remain_loader, num_classes,
                unlearn_epoch, unlearn_rate,
                logger, console_handler,
                loader_dict, experiment_path,
                eval_opt=eval_opt,
                adv_eps=0.4, 
                adv_lambda=0.1,  # Weight for Adversarial Attack (Push OFF forget manifold)
                geo_lambda=10.0, # Weight for Geodesic Preservation (Stay ON remain manifold)
                disable_bn=False):
    
    logger.info(f"Starting Geodesic Unlearning (GeodesicUL)")
    logger.info(f"Metric: Fisher-Rao Geodesic Distance. Geo Lambda: {geo_lambda}")

    test_model = copy.deepcopy(ori_model).to("cuda")
    unlearn_model = copy.deepcopy(ori_model).to("cuda")
    
    # --- 1. Compute the Riemannian Metric (Fisher Information) on Remain Set ---
    # This defines the "Curve" of the manifold we must stay on.
    logger.info("Computing Fisher Information Matrix (Geodesic Metric)...")
    fisher_metric = compute_fisher_information(train_remain_loader, test_model)
    
    # Store original parameters (The anchor point on the manifold)
    origin_params = {n: p.clone().detach() for n, p in unlearn_model.named_parameters() if p.requires_grad}

    # --- 2. Setup Adversary (L2 PGD) ---
    # We use L2 attack to push 'off' the forget manifold.
    test_model.eval()
    adversary = L2PGDAttack(test_model, eps=adv_eps, eps_iter=adv_eps/4, nb_iter=10, rand_init=True, targeted=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=unlearn_rate, momentum=0.9)
    
    accs_dict = {k: [] for k in keys}
    log_utils.enable_console_logging(logger, console_handler, False)

    # --- 3. Training Loop ---
    for epoch in tqdm.trange(unlearn_epoch):
        for x, y in train_forget_loader:
            x, y = x.to("cuda"), y.to("cuda")
            
            # A. Adversarial Attack (Push away from Forget Class)
            # The attack vector helps us find the direction "off" the forget manifold
            test_model.eval()
            x_adv, y_adv = adv_attack(test_model, adversary, x, y, num_classes)

            unlearn_model.train()
            if disable_bn:
                for m in unlearn_model.modules():
                    if isinstance(m, nn.BatchNorm2d): m.eval()
            
            optimizer.zero_grad()
            logits = unlearn_model(x)
            
            # B. Forward on Adv
            unlearn_model.eval()
            logits_adv = unlearn_model(x_adv)
            unlearn_model.train()

            # --- LOSS FUNCTION ---
            
            # 1. Forgetting (KL to Uniform) - The "Push"
            # Move along the manifold away from the "Target Class" region
            uni = torch.ones_like(logits) / num_classes
            loss_forget = F.kl_div(F.log_softmax(logits, dim=1), uni, reduction='batchmean')
            
            # 2. Adversarial Regularization
            loss_adv = criterion(logits_adv, y_adv) * adv_lambda
            
            # 3. Geodesic Preservation - The "Pull"
            # Keeps us on the "Remain Data" manifold using the Fisher metric
            loss_geo = geodesic_dist_loss(unlearn_model, origin_params, fisher_metric) * geo_lambda
            
            total_loss = loss_forget + loss_adv + loss_geo
            
            total_loss.backward()
            optimizer.step()

        logger.info(f"Epoch {epoch+1}: Loss {total_loss.item():.4f}")
        
        cur = evaluate_model_on_all_loaders(unlearn_model, loader_dict, eval_opt, logger)
        for k in keys: accs_dict[k].append(cur[k])
        plot_unlearn_remain_acc_figure(epoch+1, accs_dict, experiment_path)

    log_utils.enable_console_logging(logger, console_handler, True)
    return unlearn_model
