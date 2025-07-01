import torch

class LinearNoiseScheduler:
    def __init__(self, beta_min: float = 0.0001, beta_max: float = 0.02, T: int = 1000):
        self.betas = torch.linspace(beta_min, beta_max, T)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)

    def add_noise(self, x0: torch.tensor, t: torch.tensor, noise_sample: torch.tensor = None): # type: ignore
        """
        Given a x0 and t return x_t
        Args:
            x0: the shape will be (B, C, H, W)
            noise_sample: the shape will be (B, C, H, W)
            t: it should be a list of indices (B, )
        Returns:
            x_t: the noisy version of image at t-th timestep
        """
        if noise_sample is None:
            noise_sample = torch.randn(x0.shape).to(x0.device)
        sqrt_minus_alpha_bars = self.sqrt_one_minus_alpha_bars[t.cpu()].to(x0.device)
        sqrt_alpha_bars = self.sqrt_alpha_bars[t.cpu()].to(x0.device)
        sqrt_minus_alpha_bars = sqrt_minus_alpha_bars.view(x0.shape[0], 1, 1, 1)
        sqrt_alpha_bars = sqrt_alpha_bars.view(x0.shape[0], 1, 1, 1)
        return sqrt_minus_alpha_bars*noise_sample + sqrt_alpha_bars*x0
    
    def sample(self, xt: torch.tensor, t: int, noise_prediction: torch.tensor): # type: ignore
        """
        Given a sample estimate and x_t return x_t-1
        Args:
            xt: dimension is (B, C, H, W)
            noise_estimate: dimension is (B, C, H, W)
            t: timestep
        Returns:
            xt_minus_1 and x0
        """
        x0 = (xt - self.sqrt_one_minus_alpha_bars[t]*noise_prediction)/self.sqrt_alpha_bars[t]
        mean = (xt - self.betas[t]*noise_prediction /self.sqrt_one_minus_alpha_bars[t])/torch.sqrt(self.alphas[t])
        if t==0:
            return mean, x0
        else:
            variance = (1 - self.alpha_bars[t - 1]) / (1.0 - self.alpha_bars[t])
            variance = variance * self.betas[t]
            sigma = variance**0.5
            xt_minus_1 = mean + sigma*torch.randn(xt.shape).to(xt.device)
            return xt_minus_1, x0