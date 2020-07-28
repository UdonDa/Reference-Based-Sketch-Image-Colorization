class SCFT(nn.Module):
    def __init__(self, d_channel=992):
        super(SCFT, self).__init__()
        
        self.W_v = nn.Parameter(torch.randn(d_channel, d_channel)) # [992, 992]
        self.W_k = nn.Parameter(torch.randn(d_channel, d_channel)) # [992, 992]
        self.W_q = nn.Parameter(torch.randn(d_channel, d_channel)) # [992, 992]
        self.coef = d_channel ** .5
    
    def forward(self, V_r, V_s):
        
        wq_vs = torch.matmul(self.W_q, V_s) # [1, 992, 1024]
        wk_vr = torch.matmul(self.W_k, V_r).permute(0, 2, 1) # [1, 992, 1024]
        alpha = F.softmax(torch.matmul(wq_vs, wk_vr) / self.coef, dim=-1) # Eq.(2)
        
        wv_vr = torch.matmul(self.W_v, V_r)
        v_asta = torch.matmul(alpha, wv_vr) # [1, 992, 1024] # Eq.(3) 
        
        c_i = V_s + v_asta # [1, 992, 1024] # Eq.(4)
        
        bs,c,hw = c_i.size()
        spatial_c_i = torch.reshape(c_i.unsqueeze(-1), (bs,c,int(hw**0.5), int(hw**0.5))) #  [1, 992, 32, 32]
        
        return spatial_c_i