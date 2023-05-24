import torch
from torch import Tensor
from torch.distributions import Normal
import matplotlib.pyplot as plt


dist_mean = Tensor([0.2, 1, 3, 8.1, 3])
dist_std = Tensor([0.2, 1, 3, 0.1, 5])

norm = Normal(dist_mean, dist_std)
norm_samples = Normal(0, 1).rsample(sample_shape=torch.Size([1280]))

dist_samples = norm_samples.unsqueeze(-1) * dist_std + dist_mean
dist_probs = torch.exp(Normal(dist_mean, dist_std).log_prob(dist_samples))
mixture_probs = torch.exp(Normal(dist_mean, dist_std).log_prob(dist_samples.unsqueeze(-1)))


# Under specifikt den distributionen som samplas från, vad är deras prob i mixturen?
mixture_probs = torch.mean(mixture_probs, dim=[2])

dist_per_dist = torch.sqrt((mixture_probs / dist_probs)).mean(0)

mean_dist = dist_per_dist.mean()


color = 'red', 'green', 'blue', 'darkgoldenrod', 'navy'
fig, ax = plt.subplots(3, figsize=(10, 10), sharex=True)
for i, c in enumerate(color):
    ax[0].hist(dist_samples[:, i].detach().numpy(), color=c, bins=51, alpha=0.5)

ax[1].hist(dist_samples.flatten().detach().numpy(), color='k', bins=51, alpha=0.5)
mm_dist = Normal(dist_samples.mean(), dist_samples.std())
X = torch.linspace(-5, 5, 501)
mm_pdf = torch.exp(mm_dist.log_prob(X))
ax[2].plot(X.detach().numpy(), mm_pdf.detach().numpy(), color='k')

print(torch.sqrt(1 - dist_per_dist))
print(mean_dist)
plt.show()
