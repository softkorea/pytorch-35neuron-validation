"""35-neuron RecurrentMLP in PyTorch — cross-validation of NumPy implementation.

Architecture: Input(10) -> H1(10) -> H2(10) -> Output(5)
with output->H1 recurrent feedback via tanh(y/tau).
Total: 35 neurons, 325 parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurrentMLP(nn.Module):
    def __init__(self, input_size=10, hidden1=10, hidden2=10,
                 output_size=5, feedback_tau=2.0, init='he_normal'):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1)    # 10*10+10 = 110
        self.fc2 = nn.Linear(hidden1, hidden2)        # 10*10+10 = 110
        self.fc_out = nn.Linear(hidden2, output_size)  # 10*5+5 = 55
        self.W_rec = nn.Linear(output_size, hidden1, bias=False)  # 5*10 = 50
        self.feedback_tau = feedback_tau
        self.init_scheme = init
        if init != 'kaiming':
            self._init_weights(init)

    def _init_weights(self, scheme):
        """Custom initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if scheme == 'he_normal':
                    nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
                elif scheme == 'xavier':
                    nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x, T=3, feedback_mode='self', clone=None,
                scramble_rng=None, wrong_trial_outputs=None):
        """Forward pass with configurable feedback.

        Args:
            x: input tensor [batch, 10] (static) or [batch, T, 10] (VN)
            T: number of timesteps
            feedback_mode: 'self' | 'ablated' | 'scrambled' | 'clone' | 'wrong_trial'
            clone: another RecurrentMLP for C2 mode
            scramble_rng: torch.Generator for C1 mode
            wrong_trial_outputs: list of [batch, output_size] for wrong-trial mode
        """
        batch = x.shape[0] if x.dim() == 2 else x.shape[0]
        device = next(self.parameters()).device
        prev = torch.zeros(batch, self.fc_out.out_features, device=device)
        outputs = []

        # Clone runs its own self-feedback loop independently
        if feedback_mode == 'clone' and clone is not None:
            clone_prev = torch.zeros(batch, self.fc_out.out_features, device=device)

        for t in range(T):
            # Get input for this timestep
            x_t = x if x.dim() == 2 else x[:, t, :]

            # Determine feedback signal
            if feedback_mode == 'ablated':
                # Zero recurrent contribution (not zero input-sized vector)
                h1 = F.relu(self.fc1(x_t))
            elif feedback_mode == 'scrambled':
                # Per-trial coordinate permutation of the feedback BEFORE W_rec.
                # Each sample in the batch gets an INDEPENDENT permutation of the
                # feedback dimensions, matching the NumPy reference (src/network.py
                # shuffles per sample) and the paper's C1 definition ("permuted at
                # each trial"). A single batch-shared permutation would be an
                # unbiased but higher-variance estimator of the same control; we
                # use per-sample here for exact cross-implementation parity.
                fb = torch.tanh(prev / self.feedback_tau)
                rand = torch.rand(fb.shape[0], fb.shape[1],
                                  generator=scramble_rng, device=device)
                perm = rand.argsort(dim=1)
                fb_permuted = torch.gather(fb, 1, perm)
                h1 = F.relu(self.fc1(x_t) + self.W_rec(fb_permuted))
            elif feedback_mode == 'clone' and clone is not None:
                # 1. Target feedback from PAST clone state (t-1)
                if t == 0:
                    fb = torch.tanh(prev / self.feedback_tau)  # prev is zeros
                else:
                    fb = torch.tanh(clone_prev / self.feedback_tau)

                # 2. Advance target
                h1 = F.relu(self.fc1(x_t) + self.W_rec(fb))

                # 3. Advance clone for NEXT timestep (after target uses old state)
                clone_fb = torch.tanh(clone_prev / clone.feedback_tau)
                clone_h1 = F.relu(clone.fc1(x_t) + clone.W_rec(clone_fb))
                clone_h2 = F.relu(clone.fc2(clone_h1))
                clone_prev = clone.fc_out(clone_h2).detach()
            elif feedback_mode == 'wrong_trial' and wrong_trial_outputs is not None:
                if t > 0:
                    fb = torch.tanh(wrong_trial_outputs[t-1].detach() / self.feedback_tau)
                    h1 = F.relu(self.fc1(x_t) + self.W_rec(fb))
                else:
                    fb = torch.tanh(prev / self.feedback_tau)
                    h1 = F.relu(self.fc1(x_t) + self.W_rec(fb))
            else:  # 'self'
                fb = torch.tanh(prev / self.feedback_tau)
                h1 = F.relu(self.fc1(x_t) + self.W_rec(fb))

            h2 = F.relu(self.fc2(h1))
            out = self.fc_out(h2)
            outputs.append(out)
            prev = out

        return outputs


class DeepFeedforwardMLP(nn.Module):
    """D'': Compute-matched 6-layer feedforward (715 params)."""
    def __init__(self, input_size=10, hidden=10, output_size=5, n_layers=6):
        super().__init__()
        layers = [nn.Linear(input_size, hidden)]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden, hidden))
        layers.append(nn.Linear(hidden, output_size))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


class ParamMatchedFF(nn.Module):
    """D': Parameter-matched feedforward with skip connection (325 params)."""
    def __init__(self, input_size=10, hidden1=10, hidden2=10, output_size=5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc_out = nn.Linear(hidden2, output_size)
        self.skip = nn.Linear(input_size, output_size, bias=False)  # 50 params

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc_out(h2) + self.skip(x)
