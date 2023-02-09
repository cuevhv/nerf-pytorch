import torch


class RefinePose(torch.nn.Module):
    def __init__(self):
        super(RefinePose, self).__init__()

    def forward(self, refine_pose, prior_pose):
        refine_pose = se3_to_SE3(refine_pose)
        return compose_pair(refine_pose, prior_pose)


def skew_symmetric(w):
    w0,w1,w2 = w.unbind(dim=-1)
    O = torch.zeros_like(w0)
    wx = torch.stack([torch.stack([O,-w2,w1],dim=-1),
                        torch.stack([w2,O,-w0],dim=-1),
                        torch.stack([-w1,w0,O],dim=-1)],dim=-2)
    return wx

def se3_to_SE3(wu): # [...,3]
    w,u = wu.split([3,3],dim=-1)
    wx = skew_symmetric(w)
    theta = w.norm(dim=-1)[...,None,None]
    I = torch.eye(3,device=w.device,dtype=torch.float32)
    A = taylor_A(theta)
    B = taylor_B(theta)
    C = taylor_C(theta)
    R = I+A*wx+B*wx@wx
    V = I+B*wx+C*wx@wx
    Rt = torch.cat([R,(V@u[...,None])],dim=-1)
    return Rt

def taylor_A(x,nth=10):
    # Taylor expansion of sin(x)/x
    ans = torch.zeros_like(x)
    denom = 1.
    for i in range(nth+1):
        if i>0: denom *= (2*i)*(2*i+1)
        ans = ans+(-1)**i*x**(2*i)/denom
    return ans

def taylor_B(x,nth=10):
    # Taylor expansion of (1-cos(x))/x**2
    ans = torch.zeros_like(x)
    denom = 1.
    for i in range(nth+1):
        denom *= (2*i+1)*(2*i+2)
        ans = ans+(-1)**i*x**(2*i)/denom
    return ans

def taylor_C(x,nth=10):
    # Taylor expansion of (x-sin(x))/x**3
    ans = torch.zeros_like(x)
    denom = 1.
    for i in range(nth+1):
        denom *= (2*i+2)*(2*i+3)
        ans = ans+(-1)**i*x**(2*i)/denom
    return ans

def compose_pair(pose_a,pose_b):
    # pose_new(x) = pose_b o pose_a(x)
    R_a,t_a = pose_a[...,:3],pose_a[...,3:]
    R_b,t_b = pose_b[...,:3],pose_b[...,3:]
    R_new = R_b@R_a
    t_new = (R_b@t_a+t_b)[...,0]
    pose = torch.cat([R_new,t_new[...,None]],dim=-1) # [...,3,4]
    assert(pose.shape[-2:]==(3,4))
    return pose

    # pose_new = self(R=R_new,t=t_new)
    # return pose_new
