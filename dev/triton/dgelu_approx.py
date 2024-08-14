import math
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
# import wandb

half = torch.tensor(0.5, requires_grad=True, dtype=torch.float64)
inv_sqrt_two_pi = torch.tensor(1.0 / math.sqrt(2 * math.pi), requires_grad=True, dtype=torch.float64)
inv_sqrt_two = torch.tensor(1.0 / math.sqrt(2.0), requires_grad=True, dtype=torch.float64)
twoseven = torch.tensor(27.0, requires_grad=True, dtype=torch.float64)
nine = torch.tensor(9.0, requires_grad=True, dtype=torch.float64)
p1 = torch.tensor(1.0, requires_grad=True, dtype=torch.float64)
p2 = torch.tensor(0.0, requires_grad=True, dtype=torch.float64)
one = torch.tensor(1.0, requires_grad=True, dtype=torch.float64)

# Additional parameters for better approximation
a1 = torch.tensor(1.0, requires_grad=True, dtype=torch.float64)
a2 = torch.tensor(1.0, requires_grad=True, dtype=torch.float64)
b1 = torch.tensor(1.0, requires_grad=True, dtype=torch.float64)
b2 = torch.tensor(1.0, requires_grad=True, dtype=torch.float64)


def erf_approx(x):
    appr = x * ((twoseven + (x * x)) / (twoseven + (nine * (x * x))))
    return torch.clamp(appr, -1, 1)


def approx_gauss(x):
    num = a1 + a2 * x * x
    den = one + b1 * x * x + b2 * x * x * x * x
    return num / den


def dgelu_approx(x):
    return half * (1 + erf_approx(x * inv_sqrt_two)) + x * inv_sqrt_two_pi * torch.exp((-x * x) / 2)


def dgelu(x):
    upstream_grad = torch.ones_like(x)
    return torch.ops.aten.gelu_backward(upstream_grad, x)



if __name__ == '__main__':
    # wandb.init(project="dgelu_approx")

    x = torch.linspace(-6, 6, 1000, dtype=torch.float64)

    # Added new parameters to the list for optimization
    params = [half, inv_sqrt_two, inv_sqrt_two_pi, twoseven, nine, a1, a2, b1, b2]

    # perform optimization
    optimizer = Adam(params, lr=1e-4)
    for step in range(0, 100_000, 1):
        err = torch.pow(dgelu_approx(x) - dgelu(x), 2).sum()
        if step % 1_000 == 0:
            # wandb.log({"error": err, "step": step})
            print("error: ", err.item())
        err.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(params)
    torch.save(params, "dgelu_approx_params.pt")

    gelu_triton_values = dgelu_approx(x)
    gelu_torch_values = dgelu(x)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(x.detach().float().numpy(), gelu_triton_values.detach().float().numpy(), label='dGELU (Approx)',
             linestyle='--')
    plt.plot(x.detach().float().numpy(), gelu_torch_values.detach().float().numpy(), label='dGELU (Torch)',
             linestyle='-')
    plt.title('Comparison of dGELU implementations in float16')
    plt.xlabel('Input values')
    plt.ylabel('dGELU output')
    plt.legend()
    plt.grid(True)
    plt.savefig('dgelu_comparison.png')
    plt.show()
