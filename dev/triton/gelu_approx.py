import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import triton
import triton.language as tl

# import wandb
from torch.optim import Adam, SGD

twoseven = torch.tensor(27.0, requires_grad=True, dtype=torch.float64)
nine = torch.tensor(9.0, requires_grad=True, dtype=torch.float64)


def tanh_approx(x):
    appr = x * ((twoseven + (x * x)) / (twoseven + (nine * (x * x))))
    return torch.clamp(appr, -1, 1)


half = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
sqrt_hlf_pi = torch.tensor(0.7978845608, dtype=torch.float64, requires_grad=True)
magic = torch.tensor(0.044715, dtype=torch.float64, requires_grad=True)


def gelu_approx(x):
    return half * x * (1 + tanh_approx(sqrt_hlf_pi * (x + magic * x * x * x)))

if __name__ == '__main__':
    x = torch.linspace(-6, 6, 10000, dtype=torch.float16, device='cuda')

    # wandb.init(project="gelu_approx")

    params = [twoseven, nine, half, sqrt_hlf_pi, magic]

    # perform optimization
    optimizer = Adam(params, lr=1e-4)
    for step in range(0, 100_000, 1):
        err = torch.pow(gelu_approx(x) - F.gelu(x), 2).sum()
        if step % 1_000 == 0:
            # wandb.log({"error": err, "step": step})
            print("error: ", err.item())
        err.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(params)
    torch.save(params, "gelu_approx_params.pt")

    gelu_approx_values = gelu_approx(x)
    gelu_torch_values = F.gelu(x)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(x.cpu().detach().float().numpy(), gelu_approx_values.cpu().detach().float().numpy(), label='GELU (Approx)',
             linestyle='--')
    plt.plot(x.cpu().detach().float().numpy(), gelu_torch_values.cpu().detach().float().numpy(), label='GELU (Torch)',
             linestyle='-')
    plt.title('Comparison of GELU implementations in float16')
    plt.xlabel('Input values')
    plt.ylabel('GELU output')
    plt.legend()
    plt.grid(True)
    plt.savefig('gelu_comparison.png')
    plt.show()
