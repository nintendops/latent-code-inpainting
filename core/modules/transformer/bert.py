import math
import warnings
import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    symbolic_assert = torch._assert # 1.80a0
except AttributeError:
    symbolic_assert = torch.Assert # 1.70

# Context manager to suppress known warnings in torch.jittrace()
class suppress_tracer_warnings(warnings.catch_warnings):
    def __enter__(self):
        super().__enter__()
        warnings.simplefilter("ignore", category=torch.jit.TracerWarning)
        return self

def assert_shape(tensor, ref_shape):
    if tensor.ndim != len(ref_shape):
        raise AssertionError(f"Wrong number of dimensions: got {tensor.ndim}, expected {len(ref_shape)}")
    for idx, (size, ref_size) in enumerate(zip(tensor.shape, ref_shape)):
        if ref_size is None:
            pass
        elif isinstance(ref_size, torch.Tensor):
            with suppress_tracer_warnings(): # as_tensor results are registered as constants
                symbolic_assert(torch.equal(torch.as_tensor(size), ref_size), f"Wrong size for dimension {idx}")
        elif isinstance(size, torch.Tensor):
            with suppress_tracer_warnings(): # as_tensor results are registered as constants
                symbolic_assert(torch.equal(size, torch.as_tensor(ref_size)), f"Wrong size for dimension {idx}: expected {ref_size}")
        elif size != ref_size:
            raise AssertionError(f"Wrong size for dimension {idx}: got {size}, expected {ref_size}")

# Flatten all dimensions of a tensor except the fist/last one
def to_2d(x, mode):
    if len(x.shape) == 2:
        return x
    if mode == "last":
        return x.flatten(end_dim = -2) # x.reshape(-1, x.shape[-1])
    else:
        return x.flatten(1) # x.reshape(x.shape[0], element_dim(x))

# Transpose tensor to scores
def transpose_for_scores(x, num_heads, elem_num, head_size):
    x = x.reshape(-1, elem_num, num_heads, head_size) # [B, N, H, S]
    x = x.permute(0, 2, 1, 3) # [B, H, N, S]
    return x

# Perform dropout
def dropout(x, dp_func, noise_shape = None):
    noise_shape = noise_shape or x.shape
    return dp_func(torch.ones(noise_shape, device = x.device)) * x

# Compute attention probabilities: perform softmax on att_scores and dropout
def compute_probs(scores, dp_func, temperature=0.1):
    # Compute attention probabilities
    probs = F.softmax(scores/temperature, dim = -1) # [B, N, F, T]
    shape = [int(d) for d in probs.shape]
    shape[-2] = 1
    # Dropout over random cells and over random full rows (randomly don't use a 'to' element)
    probs = dropout(probs, dp_func)
    probs = dropout(probs, dp_func, shape)
    return probs

def entropy_loss(probs):
    return torch.mean(-torch.sum(probs * torch.log(probs + 1e-7),-1))

# (Optional, only used when --ltnt-gate, --img-gate)
#
# Gate attention values either row-wise (from) or column-wise so that some of the elements
# in the from/to_tensor will not participate in sending/receiving information, when gate
# value is low for them.
class GateAttention(torch.nn.Module):
    def __init__(self, should_gate, dim, pos_dim, num_heads, from_len, to_len, gate_bias = 0):
        super().__init__()
        self.should_gate = should_gate
        self.from_len = from_len
        self.to_len = to_len
        self.num_heads = num_heads
        self.gate_bias = gate_bias

        if should_gate:
            self.gate = nn.Linear(dim, num_heads)
            self.gate_pos = nn.Linear(pos_dim, num_heads)

    def forward(self, att_probs, tensor, pos):
        if not self.should_gate:
            return att_probs
        gate = self.gate(tensor)
        if pos is not None:
            gate = gate + self.gate_pos(pos)
        gate = torch.sigmoid(gate + self.gate_bias)
        gate = gate.reshape(-1, self.from_len, self.to_len, self.num_heads).permute(0, 3, 1, 2)
        att_probs = att_probs * gate
        return att_probs

class Transformer(torch.nn.Module):
    def __init__(self,
            dim,                                    # The layer dimension
            pos_dim,                                # Positional encoding dimension e,g, 2
            from_len,           to_len,             # The from/to tensors length (must be specified if from/to has 2 dims) 
            from_dim,           to_dim,             # The from/to tensors dimensions
            from_gate = False,  to_gate = False,    # Add sigmoid gate on from/to, so that info may not be sent/received
                                                    # when gate is low (i.e. the attention probs may not sum to 1)
            # Additional options
            num_heads           = 1,                # Number of attention heads
            attention_dropout   = 0.12,             # Attention dropout rate
            temperature         = 1e-1,
            integration         = "none",            # Feature integration type: additive, multiplicative or both
            norm                = None,             # Feature normalization type (optional): instance, batch or layer
            use_topk            = False,
            finetune            = False,
            **_kwargs):                             # Ignore unrecognized keyword args

        super().__init__()
        self.dim = dim   # e.g. 256
        self.pos_dim = pos_dim # e.g. 2
        self.from_len = from_len # e.g. 16x16
        self.to_len = to_len # e.g. n_embed
        self.from_dim = from_dim # e.g. 256
        self.to_dim = to_dim # e.g. 256
        self.temperature = temperature
        self.use_topk = use_topk
        self.finetune = finetune

        self.num_heads = num_heads  # e.g. 1
        self.size_head = int(dim / num_heads) # e.g. 256

        # We divide by 2 since we apply the dropout layer twice, over elements and over columns
        self.att_dp = torch.nn.Dropout(p = attention_dropout / 2) 

        self.norm = norm
        self.integration = integration        
        
        # Query, Key and Value mappings
        self.to_queries = nn.Linear(from_dim, dim)
        self.to_keys    = nn.Linear(to_dim, dim)
        self.to_values  = nn.Linear(to_dim, dim)

        if self.integration == 'none':
            self.to_from_tensor = nn.Linear(dim, from_dim)

        # Positional encodings
        self.from_pos_map = nn.Linear(pos_dim, dim)
        self.to_pos_map   = nn.Linear(pos_dim, dim)

        # Attention gates
        self.to_gate_attention   = GateAttention(to_gate, dim, pos_dim, num_heads, from_len = 1, to_len = to_len)
        self.from_gate_attention = GateAttention(from_gate, dim, pos_dim, num_heads, from_len = from_len, to_len = 1, gate_bias = 1)

        # Features Integration (not used here)
        control_dim = (2 * self.dim) if self.integration == "both" else self.dim 
        self.modulation = nn.Linear(self.dim, control_dim)

    def freeze(self):
        if self.finetune:
            self.to_queries.eval()
            for param in self.to_queries.parameters():
                param.requires_grad = False
        else:
            return

    # Validate transformer input shape for from/to_tensor and reshape to 2d
    def process_input(self, t, t_pos, name):
        shape = t.shape
        t_len = getattr(self, f"{name}_len")
        t_dim = getattr(self, f"{name}_dim")

        # from/to_tensor should be either 2 or 3 dimensions. If it's 3, then t_len should be specified.

        assert len(shape) <= 3
        if len(shape) == 3:
            assert_shape(t, [None, t_len, t_dim])
            batch_size = shape[0]
        else:
            # Infer batch size for the 2-dims case
            assert_shape(t, [None, t_dim])
            batch_size = int(shape[0] / t_len)

        # Reshape tensors to 2d
        t = to_2d(t, "last")
        if t_pos is not None:
            t_pos = to_2d(t_pos, "last")
            assert_shape(t_pos, [t_len, self.pos_dim])
            t_pos = t_pos.tile([batch_size, 1])

        return t, t_pos, shape

    #  (Optional, not used by default) Normalizes the 'tensor' elements, and then integrate the new information from
    # 'control' with 'tensor', where 'control' controls the bias/gain of 'tensor'.
    # norm types: batch, instance, layers
    # integration types: add, mul, both
    def integrate(self, tensor, tensor_len, control): # integration, norm
        # Normalize tensor
        tensor = att_norm(tensor, tensor_len, self.integration, self.norm)

        # Compute gain/bias
        bias = gain = control = self.modulation(control)
        if self.integration == "both":
            gain, bias = torch.split(control, 2, dim = -1)

        # Modulate the bias/gain of 'tensor'
        if self.integration != "add":
            tensor = tensor * (gain + 1)
        if self.integration != "mul":
            tensor = tensor + bias

        return tensor

    # Transformer (multi-head attention) function originated from the Google-BERT repository.
    # https://github.com/google-research/bert/blob/master/modeling.py#L558
    #
    # We adopt their from/to notation:
    # from_tensor: [batch_size, from_len, dim] a list of 'from_len' elements
    # to_tensor: [batch_size, to_len, dim] a list of 'to_len' elements
    #
    # Each element in 'from_tensor' attends to elements from 'to_tensor',
    # Then we compute a weighted sum over the 'to_tensor' elements, and use it to update
    # the elements at 'from_tensor' (through additive/multiplicative integration).
    #
    # Overall it means that information flows in the direction to->from, or that the 'to'
    # modulates the 'from'. For instance, if from=image, and to=latents, then the latents
    # will control the image features. If from = to then this implements self-attention.
    #
    # We first project 'from_tensor' into a 'query', and 'to_tensor' into 'key' and 'value'.
    # Then, the query and key tensors are dot-producted and softmaxed to obtain
    # attention distribution over the to_tensor elements. The values are then
    # interpolated (weighted-summed) using this distribution, to get 'context'.
    # The context is used to modulate the bias/gain of the 'from_tensor' (depends on 'intervention').
    # Notation: B - batch_size, F - from_len, T - to_len, N - num_heads, H - head_size
    # Other arguments:
    # - att_vars: K-means variables carried over from layer to layer (only when --kmeans)
    # - att_mask: Attention mask to block from/to elements [batch_size, from_len, to_len]
    def forward(self, from_tensor, to_tensor, from_pos, to_pos, 
            att_vars = None, att_mask = None, hw_shape = None, use_topk = False):

        '''
        N: number of attention heads (default: 1)
        H: number of channels (default: 512)
        T: number of embeddings (codebook size, e.g. 1024)
        F: number of image codes (default: 16x16=256)
        '''

        self.freeze()

        # Validate input shapes and map them to 2d
        from_tensor, from_pos, from_shape = self.process_input(from_tensor, from_pos, "from") # from_tensor: image latents
        to_tensor,   to_pos,   to_shape   = self.process_input(to_tensor, to_pos, "to") # to_tensor: embeddings

        att_vars = att_vars or {}
        # to_from = att_vars.get("centroid_assignments") 

        # Compute queries, keys and values
        queries = self.to_queries(from_tensor)
        keys    = self.to_keys(to_tensor)
        values  = self.to_values(to_tensor)
        # _queries = queries

        # Add positional encodings to queries and keys
        if from_pos is not None:
            queries = queries + self.from_pos_map(from_pos)
        if to_pos is not None:
            keys = keys + self.to_pos_map(to_pos)

        # Reshape queries, keys and values, and then compute att_scores
        values = transpose_for_scores(values,  self.num_heads, self.to_len,   self.size_head)  # [B, N, T, H]
        queries = transpose_for_scores(queries, self.num_heads, self.from_len, self.size_head)  # [B, N, F, H]
        keys = transpose_for_scores(keys,    self.num_heads, self.to_len,   self.size_head)  # [B, N, T, H]

        # l2 normalization
        keys = nn.functional.normalize(keys, dim=-1)
        queries = nn.functional.normalize(queries, dim=-1)      

        # correlation tensor
        att_scores = queries.matmul(keys.permute(0, 1, 3, 2)) # [B, N, F, T]

        # q loss as cosine simularity
        qloss = torch.mean(1 - att_scores.max(-1)[0])
        # qloss = 0.0
      
        # print(qloss)
        # print(queries.squeeze() @ queries.squeeze().T)
        # import ipdb; ipdb.set_trace()

        att_probs = None

        # Scale attention scores given head size (see BERT)
        att_scores = att_scores / math.sqrt(float(self.size_head))

        # (optional, not used by default)
        # Mask attention logits using att_mask (to mask some components)
        if att_mask is not None:
            att_scores = logits_mask(att_scores, att_mask.unsqueeze(1))

        # Turn attention logits to probabilities (softmax + dropout)
        att_probs = compute_probs(att_scores, self.att_dp, temperature=self.temperature) # [B, N, F, T]

        # (optional) Gate attention values for the from/to elements
        att_probs = self.to_gate_attention(att_probs, to_tensor, to_pos)
        att_probs = self.from_gate_attention(att_probs, from_tensor, from_pos)

        # topk quantization
        if use_topk or self.use_topk:
            _, top = torch.topk(att_probs, k=1, dim=-1)
            B, N, F, _ = top.shape
            T, H = values.shape[-2:]
            top_ = top.reshape(-1)
            values_ = values.reshape(T,H)
            control = values_[top_].reshape(B, N, F, H)
        else:
            # Compute weighted-sum of the values using the attention distribution
            control = att_probs.matmul(values)      # [B, N, F, T] x [B, N, T, H] -> [B, N, F, H]

        if self.integration != 'none':
            control = control.permute(0, 2, 1, 3)   # [B, F, N, H]
            control = control.reshape(-1, self.dim) # [B*F, N*H]
            # This newly computed information will control the bias/gain of the new from_tensor
            from_tensor = self.integrate(from_tensor, self.from_len, control)
        else:
            from_tensor = control.mean(1) # [B, F, H]
            from_tensor = self.to_from_tensor(from_tensor) # [B, from_len, from_dim]

        # Reshape from_tensor to its original shape (if 3 dimensions)
        if len(from_shape) > 2:
            from_tensor = from_tensor.reshape(from_shape)

        if hw_shape is not None:
            att_probs = att_probs.reshape(-1, *hw_shape, self.to_len).permute(0, 3, 1, 2) # [NCHW]

        # qloss = entropy_loss(att_probs)

        if use_topk or self.use_topk:
            att_probs = top

        return from_tensor, att_probs, qloss

    @torch.no_grad()
    def query(self, x, att_probs):
        # for inference only, L = F denoting the length of input code sequences
        values  = self.to_values(x) 
        y = att_probs.matmul(values).mean(1) # [B, N, L, T] x [B, N, T, H] -> [B, N, L, H]
        y = self.to_from_tensor(y) # [B, L, H]
        return y