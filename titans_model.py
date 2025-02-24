import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EnhancedMemoryModule(nn.Module):
    def __init__(self, d_model, memory_size, num_memory_layers=2):
        super().__init__()
        self.d_model = d_model
        self.memory_size = memory_size
        
        # 더 깊은 메모리 네트워크 (논문의 제안대로)
        self.memory_layers = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_memory_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_memory_layers)
        ])
        
        # 메모리 버퍼
        self.register_buffer("memory", torch.zeros(memory_size, d_model))
        self.register_buffer("momentum", torch.zeros(memory_size, d_model))
        
        # Surprise-based 업데이트를 위한 projection layers
        self.surprise_proj = nn.Linear(d_model, d_model)
        self.memory_proj = nn.Linear(d_model, d_model)
        
        self.silu = nn.SiLU()  # 논문에서 사용된 activation
        
    def compute_surprise(self, x, memory):
        """
        Compute surprise metric based on gradient of memory projection
        """
        proj = self.surprise_proj(memory)
        grad = torch.autograd.grad(
            proj.sum(), memory, create_graph=True
        )[0]
        return grad.detach()
        
    def retrieve(self, query, mask=None):
        """
        Enhanced memory retrieval with multi-layer processing
        """
        # Scale dot-product attention
        scale = math.sqrt(self.d_model)
        batch_size = query.size(0)
        
        # 메모리를 배치 크기에 맞게 확장
        mem = self.memory.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, memory_size, d_model]
        query = query.unsqueeze(1)      # [batch, 1, d_model]
        
        # Compute attention scores
        attn_scores = torch.bmm(query, mem.transpose(1, 2)) / scale
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Weighted sum of memories
        retrieved = torch.bmm(attn_weights, mem)  # [batch, 1, d_model]
        
        # Process through memory layers
        x = retrieved.squeeze(1)
        for layer, norm in zip(self.memory_layers, self.layer_norms):
            x = self.silu(layer(norm(x)))
            
        return x
        
    def update(self, update_signal, alpha=0.1, theta=0.1, eta=0.5):
        """
        Enhanced memory update with momentum and surprise-based forgetting
        """
        # Compute surprise
        surprise = self.compute_surprise(update_signal, self.memory)
        
        # Update momentum (past surprise)
        self.momentum = eta * self.momentum - theta * surprise
        
        # Update memory with forget gate
        new_info = self.memory_proj(update_signal.mean(dim=0))
        forget_gate = torch.sigmoid(self.surprise_proj(new_info))
        
        self.memory = (1 - forget_gate) * self.memory + forget_gate * (new_info + self.momentum)
        return self.memory

class EnhancedTitansTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, vocab_size,
                 persistent_mem_len=10, segment_length=128, max_seq_len=10000):
        super().__init__()
        self.d_model = d_model
        self.segment_length = segment_length
        
        # 토큰 및 포지션 임베딩
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        # Enhanced persistent memory with learnable initialization
        self.persistent_memory = nn.Parameter(
            torch.randn(persistent_mem_len, d_model) / math.sqrt(d_model)
        )
        
        # Transformer layers with enhanced attention
        self.transformer_layers = nn.ModuleList([
            EnhancedTransformerLayer(d_model, nhead)
            for _ in range(num_layers)
        ])
        
        # Enhanced memory module
        self.memory_module = EnhancedMemoryModule(
            d_model, 
            memory_size=segment_length,
            num_memory_layers=2
        )
        
        # Output projection with tied weights
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.fc_out.weight = self.token_embedding.weight
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len = x.size()
        
        # Apply embeddings
        x = self.token_embedding(x) + self.pos_embedding[:, :seq_len]
        
        # Process segments
        segments = x.split(self.segment_length, dim=1)
        outputs = []
        
        for seg in segments:
            # Get query for memory retrieval
            query = self.norm(seg.mean(dim=1))
            
            # Retrieve from memory
            mem_out = self.memory_module.retrieve(query, mask)
            
            # Combine with persistent memory
            persistent = self.persistent_memory.unsqueeze(0).expand(batch_size, -1, -1)
            seg_input = torch.cat([
                persistent,
                mem_out.unsqueeze(1),
                seg
            ], dim=1)
            
            # Process through transformer layers
            for layer in self.transformer_layers:
                seg_input = layer(seg_input)
                
            # Extract segment output
            seg_out = seg_input[:, persistent.size(1) + 1:]
            outputs.append(seg_out)
            
            # Update memory
            self.memory_module.update(seg_out)
            
        # Combine outputs and project
        output = torch.cat(outputs, dim=1)
        logits = self.fc_out(output)
        
        return logits

class EnhancedTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # Self attention with pre-norm
        normed = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed)
        x = x + attn_out
        
        # Feed forward with pre-norm
        normed = self.norm2(x)
        ff_out = self.feed_forward(normed)
        x = x + ff_out
        
        return x

# Example usage
def test_enhanced_model():
    # Model configuration
    config = {
        'd_model': 512,
        'nhead': 8,
        'num_layers': 6,
        'vocab_size': 30522,
        'persistent_mem_len': 16,
        'segment_length': 128
    }
    
    # Initialize model
    model = EnhancedTitansTransformer(**config)
    
    # Generate sample input
    batch_size = 2
    seq_len = 256
    x = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    
    # Forward pass
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")

if __name__ == "__main__":
    test_enhanced_model()