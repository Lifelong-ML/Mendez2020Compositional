import torch
import torch.nn.functional as F

from torch.optim.optimizer import Optimizer

from collections import Counter
from functools import reduce

'''
Heavily based on https://github.com/Thrandis/EKFAC-pytorch
'''

def kronecker(A, B):
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))

class KFAC_EWC(Optimizer):

    def __init__(self, net, ewc_lambda=1e-3, eps=None, sua=False, pi=False, update_freq=1,
                 alpha=1.0, constraint_norm=False):
        """ K-FAC Preconditionner for Linear and Conv2d layers.
        Computes the K-FAC of the second moment of the gradients.
        It works for Linear and Conv2d layers and silently skip other layers.
        Args:
            net (torch.nn.Module): Network to precondition.
            eps (float): Tikhonov regularization parameter for the inverses.
            sua (bool): Applies SUA approximation.
            pi (bool): Computes pi correction for Tikhonov regularization.
            update_freq (int): Perform inverses every update_freq updates.
            alpha (float): Running average parameter (if == 1, no r. ave.).
            constraint_norm (bool): Scale the gradients by the squared
                fisher norm.
        """
        self.ewc_lambda = ewc_lambda
        self.eps = eps
        self.sua = sua
        self.pi = pi
        self.update_freq = update_freq
        self.alpha = alpha
        self.constraint_norm = constraint_norm
        self.params = []
        self._fwd_handles = []
        self._bwd_handles = []
        self._iteration_counter = Counter()
        for mod in net.modules():
            mod_class = mod.__class__.__name__
            if mod_class in ['Linear', 'Conv2d']:
                handle = mod.register_forward_pre_hook(self._save_input)
                self._fwd_handles.append(handle)
                handle = mod.register_backward_hook(self._save_grad_output)
                self._bwd_handles.append(handle)
                params = [mod.weight]
                if mod.bias is not None:
                    params.append(mod.bias)
                d = {'params': params, 'mod': mod, 'layer_type': mod_class}
                self.params.append(d)
        super().__init__(self.params, {})

    def add_param_group(self, mod):
        mod_class = mod.__class__.__name__
        if mod_class in ['Linear', 'Conv2d']:
            handle = mod.register_forward_pre_hook(self._save_input)
            self._fwd_handles.append(handle)
            handle = mod.register_backward_hook(self._save_grad_output)
            self._bwd_handles.append(handle)
            params = [mod.weight]
            if mod.bias is not None:
                params.append(mod.bias)
            d = {'params': params, 'mod': mod, 'layer_type': mod_class}
            self.params.append(d)
            super().add_param_group(d)


    def step(self, task_id, update_stats=True, update_params=True):
        """Performs one step of preconditioning."""
        fisher_norm = 0.
        for group in self.param_groups:
            # Getting parameters
            if group['params'][0].grad is None:
                continue
            if len(group['params']) == 2:
                weight, bias = group['params']
            else:
                weight = group['params'][0]
                bias = None
            state = self.state[weight]
            # Update convariances and inverses
            if update_stats:
                if self._iteration_counter[task_id] % self.update_freq == 0:
                    self._compute_covs(group, state, task_id)
                else:
                    if self.alpha != 1:
                        self._compute_covs(group, state, task_id)
            if update_params:
                self._precond(weight, bias, group, state)
                gw, gb = self._precond(weight, bias, group, state)
                # Updating gradients
                if self.constraint_norm:
                    fisher_norm += (weight.grad * gw).sum()
                weight.grad.data = gw
                if bias is not None:
                    if self.constraint_norm:
                        fisher_norm += (bias.grad * gb).sum()
                    bias.grad.data = gb
            # Cleaning
            if 'x' in self.state[group['mod']]:
                del self.state[group['mod']]['x']
            if 'gy' in self.state[group['mod']]:
                del self.state[group['mod']]['gy']
        if update_stats:
            self._iteration_counter[task_id] += 1

    def _save_input(self, mod, i):
        """Saves input of layer to compute covariance."""
        if mod.training:
            self.state[mod]['x'] = i[0]

    def _save_grad_output(self, mod, grad_input, grad_output):
        """Saves grad on output of layer to compute covariance."""
        if mod.training:
            self.state[mod]['gy'] = grad_output[0] * grad_output[0].size(0)

    def _precond(self, weight, bias, group, state):
        """Applies preconditioning."""
        if 'Htheta_sum' not in state:
            return weight.grad.data, bias.grad.data
        g = self._get_grad_vector(group, weight, bias)
        theta = self._get_param_vector(group, weight, bias)
        
        H_sum_theta = torch.zeros_like(state['Htheta_sum'])
        for xxt, ggt in zip(state['xxt'].values(), state['ggt'].values()):
            tmp = ggt.mm(theta).mm(xxt)
            H_sum_theta += tmp.transpose(1, 0).contiguous().view(-1, 1) # vec operator
        g = g.transpose(1, 0).contiguous().view(-1, 1) + self.ewc_lambda * (H_sum_theta - state['Htheta_sum'])
        
        if group['layer_type'] == 'Conv2d':
            g /= state['num_locations']

        sw = weight.shape
        sb = bias.shape
        s1 = 1 + reduce(lambda x, y: x * y, sw[1:], 1)  # product of sw1[1] ... sw[n] (+ 1 for bias)
        g = g.view(s1, sw[0]).transpose(1, 0)
        
        if group['layer_type'] == 'Conv2d':
            if self.sua:
                '''
                fw: 
                    sw1 sw0 sw2 sw3
                    if bias
                        bias: 1 sum(sb) 1 1
                        bias: 1 sum(sb) sw2 sw3
                        sw1+1 sw0 sw2 sw3
                    sw1+1 sw0 sw2 sw3
                    sw0 sw1+1 sw2 sw3
                    sw0 (sw+1+sw2+sw3)
                bw:
                    s0, s1+1, sw2, sw3
                    sw1+1 sw0 sw2 sw3
                    if bias
                        bias: 1, sum(sb) sw2 sw3
                        bias: sum(sb)
                    sw1 sw0 sw2 sw3
                    sw0 sw1 sw2 sw3
                '''
                g = g.view(sw[0], sw[1]+1, sw[2], sw[3])
                g = g.permute(1, 0, 2, 3)
                if bias is not None:
                    gb = g[-1]
                    gb = g[0, :, 0, 0]
                g = g[:-1]
                g = g.permute(1, 0, 2, 3)
            else:
                if bias is not None:
                    gb = g[:, -1]
                g = g[:, :-1]
                g = g.view(sw[0], sw[1], sw[2], sw[3])
        else:
            if bias is not None:
                gb = g[:, -1]
                g = g[:, :-1]
        
        assert gb.data.shape == bias.data.shape
        assert g.data.shape == weight.data.shape
        
        return g, gb

    def _precond_sua(self, weight, bias, group, state):
        """Preconditioning for KFAC SUA."""
        g = weight.grad.data
        s = g.shape
        mod = group['mod']
        g = g.permute(1, 0, 2, 3).contiguous()
        if bias is not None:
            gb = bias.grad.view(1, -1, 1, 1).expand(1, -1, s[2], s[3])
            g = torch.cat([g, gb], dim=0)        

        g = g + self.ewc_lambda * (state['H_sum'].mm(g) - state['Htheta_sum'])
        g = g.view(s[0], -1, s[2], s[3])
        g /= state['num_locations']
        if bias is not None:
            gb = g[:, -1, s[2]//2, s[3]//2]
            g = g[:, :-1]
        else:
            gb = None
        return g, gb

    def _get_grad_vector(self, group, weight, bias):
        g = weight.grad.data
        s = g.shape
        if group['layer_type'] == 'Conv2d' and self.sua:
            mod = group['mod']
            g = g.permute(1, 0, 2, 3).contiguous()
            if bias is not None:
                gb = bias.grad.view(1, -1, 1, 1).expand(1, -1, s[2], s[3])
                g = torch.cat([g, gb], dim=0)
        else:
            if group['layer_type'] == 'Conv2d':
                g = g.contiguous().view(s[0], s[1]*s[2]*s[3])
            if bias is not None:
                gb = bias.grad.data
                g = torch.cat([g, gb.view(gb.shape[0], 1)], dim=1)
        return g.clone()

    def _get_param_vector(self, group, weight, bias):
        params = weight.data
        s = params.shape
        if group['layer_type'] == 'Conv2d':
            if self.sua:
                params = params.permute(1, 0, 2, 3).contiguous()
                if bias is not None:
                    bias = bias.data.view(1, -1, 1, 1).expand(1, -1, s[2], s[3])
                    params = torch.cat([params, bias], dim=0)
                params = params.view(-1, s[0], s[2], s[3]).permute(1, 0, 2, 3).contiguous()
                params = params.view(s[0], -1)
                return params.clone()       # a little messy to return here, but trying to mimick original code somewhat
            params = params.contiguous().view(s[0], s[1]*s[2]*s[3])
        if bias is not None:
            params = torch.cat([params, bias.data.view(bias.shape[0], 1)], dim=1)
        return params.clone()

    def _compute_covs(self, group, state, task_id):
        """Computes the covariances."""
        mod = group['mod']
        x = self.state[mod]['x']
        gy = self.state[mod]['gy']
        # Computation of xxt
        if group['layer_type'] == 'Conv2d':
            if not self.sua:
                x = F.unfold(x, mod.kernel_size, padding=mod.padding,
                             stride=mod.stride)
            else:
                x = x.view(x.shape[0], x.shape[1], -1)
            x = x.data.permute(1, 0, 2).contiguous().view(x.shape[1], -1)
        else:
            x = x.data.t()
        if mod.bias is not None:
            ones = torch.ones_like(x[:1])
            x = torch.cat([x, ones], dim=0)
        if self._iteration_counter[task_id] == 0:
            if 'xxt' not in state:
                state['xxt'] = {task_id : torch.mm(x, x.t()) / float(x.shape[1])}
            else:
                state['xxt'][task_id] = torch.mm(x, x.t()) / float(x.shape[1])
        else:
            params = state['params'][task_id]
            xxt = state['xxt'][task_id]
            ggt = state['ggt'][task_id]
            tmp = ggt.mm(params).mm(xxt)
            state['Htheta_sum'] -= tmp.transpose(1, 0).contiguous().view(-1, 1) # vec operator
            state['xxt'][task_id].addmm_(mat1=x, mat2=x.t(),
                                beta=(1. - self.alpha),
                                alpha=self.alpha / float(x.shape[1]))
        # Computation of ggt
        if group['layer_type'] == 'Conv2d':
            gy = gy.data.permute(1, 0, 2, 3)
            state['num_locations'] = gy.shape[2] * gy.shape[3]
            gy = gy.contiguous().view(gy.shape[0], -1)
        else:
            gy = gy.data.t()
            state['num_locations'] = 1
        if self._iteration_counter[task_id] == 0:
            if 'ggt' not in state:
                state['ggt'] = {task_id : torch.mm(gy, gy.t()) / float(gy.shape[1])}
            else:
                state['ggt'][task_id] = torch.mm(gy, gy.t()) / float(gy.shape[1])
        else:
            state['ggt'][task_id].addmm_(mat1=gy, mat2=gy.t(),
                                beta=(1. - self.alpha),
                                alpha=self.alpha / float(gy.shape[1]))

        if 'params' not in state:
            state['params'] = {task_id : self._get_param_vector(group, mod.weight, mod.bias)}
        else:
            state['params'][task_id] = self._get_param_vector(group, mod.weight, mod.bias)

        params = state['params'][task_id]
        xxt = state['xxt'][task_id]
        ggt = state['ggt'][task_id]
        if 'Htheta_sum' not in state:
            tmp = ggt.mm(params).mm(xxt)
            state['Htheta_sum'] = tmp.transpose(1, 0).contiguous().view(-1, 1) # vec operator
        else:
            tmp = ggt.mm(params).mm(xxt)
            state['Htheta_sum'] += tmp.transpose(1, 0).contiguous().view(-1, 1) # vec operator
        

    def _inv_covs(self, xxt, ggt, num_locations):
        """Inverses the covariances."""
        # Computes pi
        pi = 1.0
        if self.pi:
            tx = torch.trace(xxt) * ggt.shape[0]
            tg = torch.trace(ggt) * xxt.shape[0]
            pi = (tx / tg)
        # Regularizes and inverse
        eps = self.eps / num_locations
        diag_xxt = xxt.new(xxt.shape[0]).fill_((eps * pi) ** 0.5)
        diag_ggt = ggt.new(ggt.shape[0]).fill_((eps / pi) ** 0.5)
        ixxt = (xxt + torch.diag(diag_xxt)).inverse()
        iggt = (ggt + torch.diag(diag_ggt)).inverse()
        return ixxt, iggt
    
    def __del__(self):
        for handle in self._fwd_handles + self._bwd_handles:
            handle.remove()