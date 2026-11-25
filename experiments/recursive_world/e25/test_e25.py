import unittest
import torch
from e25 import Correction, PlannerView, calibrate, ARMS

class Frozen:
    a=2; device='cpu'
    def predict(self,z,a):
        return a @ z.T
    def decode(self,x):
        return x

class Tests(unittest.TestCase):
    def setUp(self):
        self.z=torch.eye(2); self.a=torch.tensor([.3,-.2]); self.m=Frozen()
    def make(self, arm): return Correction(self.m,[1,1],1,[.03,.03],.4,arm)
    def test_zero_evidence(self):
        for arm in ARMS:
            torch.testing.assert_close(self.make(arm).predict(self.z,self.a[None]),self.a[None])
    def test_zero_residual(self):
        for arm in ARMS:
            c=self.make(arm);c.observe(self.z,self.a,self.a)
            torch.testing.assert_close(c.predict(self.z,self.a[None]),self.a[None])
    def test_last_and_scalar(self):
        for arm,scale in [('counterfactual',1),('scalar',.4)]:
            c=self.make(arm);c.observe(self.z,self.a,self.a+.1)
            torch.testing.assert_close(c.predict(self.z,self.a[None]),self.a[None]+scale*.1)
    def test_gain_reversal(self):
        c=self.make('actuator')
        for a in [self.a,-self.a,torch.tensor([.4,.5]),torch.tensor([-.5,-.4])]:
            c.observe(self.z,a,a*1.3)
        torch.testing.assert_close(c.predict(self.z,self.a[None]),1.3*self.a[None],atol=.007,rtol=0)
    def test_joint_identification(self):
        c=self.make('joint');b=torch.tensor([.08,-.06])
        # Vary J to disambiguate actuator offsets and output bias.
        for z in [self.z,self.z*2,self.z*.5]:
            for a in [self.a,-self.a,torch.tensor([.5,.4]),torch.tensor([-.4,-.5])]:
                c.observe(z,a,(a*1.2+torch.tensor([.04,.03]))@z.T+b)
        torch.testing.assert_close(c.predict(self.z,self.a[None]),1.2*self.a[None]+torch.tensor([.04,.03])+b,atol=.005,rtol=0)
    def test_boundary_finite(self):
        c=self.make('joint');a=torch.ones(2);c.observe(self.z,a,a+.1)
        self.assertTrue(torch.isfinite(c.predict(self.z,a[None])).all())
    def test_planner_interface(self):
        p=PlannerView(self.make('joint'));torch.testing.assert_close(p.decode(p.predict(self.z,self.a[None])),self.a[None])
    def test_calibration(self):
        noise,alpha=calibrate([dict(condition='none',observed_residuals=[[.1,.1],[.1,.1],[.1,.1]])],[1,1]);self.assertAlmostEqual(alpha,1);torch.testing.assert_close(noise,torch.tensor([.1,.1]))

if __name__=='__main__': unittest.main(verbosity=2)
