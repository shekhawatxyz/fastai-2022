"""E25: regularized actuator/output disturbance inference around a frozen CWM.
Only the correction changes; upstream CEM, encoder, predictor and readout stay fixed.
All output coordinates are standardized with the upstream evaluation scale.
"""
import torch

ARMS = ('trained_predictor', 'counterfactual', 'scalar', 'bias', 'actuator', 'joint')

class Correction:
    def __init__(self, model, sigma, limit, noise, alpha, arm):
        if arm not in ARMS:
            raise ValueError(arm)
        self.m, self.arm, self.limit = model, arm, limit
        self.sigma = torch.as_tensor(sigma, device=model.device, dtype=torch.float32)
        self.noise = torch.as_tensor(noise, device=model.device, dtype=torch.float32)
        if self.noise.shape != self.sigma.shape or not torch.all(self.noise > 0) or not torch.all(self.sigma > 0):
            raise ValueError('positive per-coordinate sigma and nominal residual standard deviation required')
        self.alpha = float(alpha)
        self.a, self.d = model.a, len(self.sigma)
        # Dimensionless gain and offset; output bias in standardized coordinates.
        prior = torch.cat([self.sigma.new_full((2*self.a,), .25), self.sigma.new_full((self.d,), .25)])
        mask = {'bias': [False]*(2*self.a)+[True]*self.d,
                'actuator': [True]*(2*self.a)+[False]*self.d}.get(arm, [True]*(2*self.a+self.d))
        self.mask = torch.tensor(mask, device=model.device)
        self.prec = torch.diag(prior[self.mask].reciprocal().square())
        self.rhs = torch.zeros_like(prior[self.mask])
        self.last = torch.zeros_like(self.sigma)
        self.raw_last = torch.zeros_like(self.sigma)

    def mean(self, z, actions):
        return self.m.decode(self.m.predict(z, actions))

    @torch.no_grad()
    def observe(self, z, action, decoded_outcome):
        action = action.reshape(-1)
        pred = self.mean(z, action[None])[0]
        self.raw_last = decoded_outcome - pred
        self.last = self.raw_last / self.sigma
        if self.arm not in ('bias', 'actuator', 'joint'):
            return
        h = .01 * self.limit
        eye = torch.eye(self.a, device=self.m.device, dtype=action.dtype) * h
        plus = (action[None]+eye).clamp(-self.limit, self.limit)
        minus = (action[None]-eye).clamp(-self.limit, self.limit)
        denom = (plus-minus).diagonal().clamp_min(1e-8)
        # Derivative of decoded frozen predictor, never a simulator derivative.
        j = ((self.mean(z, plus)-self.mean(z, minus))/denom[:,None]).T / self.sigma[:,None]
        design = torch.cat([j*action, j*self.limit, torch.eye(self.d, device=self.m.device)], dim=1)[:,self.mask]
        design = design / self.noise[:,None]
        target = self.last / self.noise
        self.prec += design.T @ design
        self.rhs += design.T @ target

    @torch.no_grad()
    def parameters(self):
        p = self.sigma.new_zeros(2*self.a+self.d)
        p[self.mask] = torch.linalg.solve(self.prec, self.rhs)
        return p

    @torch.no_grad()
    def predict(self, z, actions):
        if self.arm == 'trained_predictor':
            return self.mean(z, actions)
        if self.arm in ('counterfactual', 'scalar'):
            return self.mean(z, actions) + self.raw_last*(1 if self.arm=='counterfactual' else self.alpha)
        p = self.parameters()
        # Conservative extrapolation: no action inputs outside the trained interval.
        effective = (actions*(1+p[:self.a])+self.limit*p[self.a:2*self.a]).clamp(-self.limit,self.limit)
        return self.mean(z, effective) + p[2*self.a:]*self.sigma


def calibrate(records, sigma):
    """Nominal error scale and pooled previous→next residual coefficient.
    records: disjoint upstream replan logs with condition and observed_residuals.
    Labels used only here; never passed to Correction at test time.
    """
    sigma = torch.as_tensor(sigma, dtype=torch.float64)
    nominal, prev, nxt = [], [], []
    for r in records:
        x = torch.as_tensor(r['observed_residuals'], dtype=torch.float64)/sigma
        if r['condition']=='none': nominal.append(x)
        prev.append(x[:-1]); nxt.append(x[1:])
    if not nominal or not prev:
        raise ValueError('disjoint calibration logs required, including nominal episodes')
    x, y = torch.cat(prev), torch.cat(nxt)
    alpha = ((x*y).sum()/x.square().sum().clamp_min(1e-12)).clamp(0,1)
    noise = torch.cat(nominal).square().mean(0).sqrt().clamp_min(.02)
    return noise.float(), float(alpha)


class PlannerView:
    """Expose corrected physical predictions to upstream CEM without modifying its search."""
    def __init__(self, correction):
        self.c = correction
        self.a, self.device = correction.m.a, correction.m.device
    def predict(self, z, actions):
        return self.c.predict(z, actions)
    def decode(self, physical):
        return physical
