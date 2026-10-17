"""The six selected SSI model families with recurrent EQ descriptor encoders."""
import torch
from torch import nn
from torch.nn import functional as F

MODEL_NAMES=("mlp","tcn_lstm","gnn_tcn","meshgraphnet","graph_transformer","graph_operator")

def _mlp(i,h,o): return nn.Sequential(nn.Linear(i,h),nn.GELU(),nn.Linear(h,o))

class TemporalEncoder(nn.Module):
    def __init__(self,d,h,kind="gru",layers=2,dropout=.05):
        super().__init__();self.kind=kind
        if kind=="lstm": self.body=nn.LSTM(d,h,num_layers=layers,batch_first=True,dropout=dropout if layers>1 else 0.)
        else: self.body=nn.GRU(d,h,num_layers=layers,batch_first=True,dropout=dropout if layers>1 else 0.)
    def forward(self,x): return self.body(x)[0]

class GraphMessageLayer(nn.Module):
    def __init__(self,h,e):
        super().__init__();self.edge=_mlp(2*h+e,h,h);self.node=_mlp(2*h,h,h);self.norm=nn.LayerNorm(h)
    def forward(self,nodes,edge_index,edge_features):
        agg=torch.zeros_like(nodes)
        if edge_index.numel():
            src,dst=edge_index;msg=self.edge(torch.cat([nodes[src],nodes[dst],edge_features],-1));agg.index_add_(0,dst,msg)
            degree=torch.zeros(len(nodes),device=nodes.device,dtype=nodes.dtype);degree.index_add_(0,dst,torch.ones_like(dst,dtype=nodes.dtype))
            agg=agg/degree.clamp_min(1).unsqueeze(-1)
        return self.norm(nodes+self.node(torch.cat([nodes,agg],-1)))

class PeakBase(nn.Module):
    def __init__(self,name,global_dim,node_dim,edge_dim,motion_dim,response_dim=2,hidden=96,
                 recurrent_layers=2,dropout=.05,conv_kernel=3,attention_heads=4):
        super().__init__();self.name=name;self.hidden=hidden
        self.node_encoder=_mlp(node_dim,hidden,hidden);self.global_encoder=_mlp(global_dim,hidden,hidden)
        self.motion_in=nn.Linear(motion_dim,hidden)
        pad=conv_kernel//2
        self.temporal_conv=(nn.Sequential(nn.Conv1d(hidden,hidden,conv_kernel,padding=pad),nn.GELU(),
                            nn.Conv1d(hidden,hidden,conv_kernel,padding=pad),nn.GELU()) if name in ("tcn_lstm","gnn_tcn") else None)
        kind="lstm" if name=="tcn_lstm" else "gru"
        self.motion_encoder=TemporalEncoder(hidden,hidden,kind,recurrent_layers,dropout)
        self.dropout=nn.Dropout(dropout)
        self.graph_layers=nn.ModuleList()
        if name in ("gnn_tcn","meshgraphnet","graph_transformer","graph_operator"):
            depth={"gnn_tcn":3,"meshgraphnet":5,"graph_transformer":2,"graph_operator":3}[name]
            self.graph_layers=nn.ModuleList([GraphMessageLayer(hidden,edge_dim) for _ in range(depth)])
        self.node_attention=nn.MultiheadAttention(hidden,attention_heads,batch_first=True) if name=="graph_transformer" else None
        self.operator=_mlp(2*hidden,hidden,32) if name=="graph_operator" else None
        out_in=(32 if name=="graph_operator" else hidden)+5*hidden
        self.decoder=_mlp(out_in,hidden,response_dim)
    def _context(self,c):
        sequence=self.motion_in(c["ground_motion"])
        if self.temporal_conv is not None:sequence=sequence+self.temporal_conv(sequence.T.unsqueeze(0)).transpose(1,2).squeeze(0)
        motion=self.motion_encoder(sequence.unsqueeze(0)).squeeze(0)
        motion_summary=torch.cat([motion.mean(0),motion.amax(0),motion.amin(0),motion.square().mean(0).sqrt()],0)
        nodes=self.node_encoder(c["node_features"]);glob=self.global_encoder(c["global_features"]).expand(len(nodes),-1)
        for layer in self.graph_layers:nodes=layer(nodes,c["edge_index"],c["edge_features"])
        if self.node_attention is not None:
            attn,_=self.node_attention(nodes.unsqueeze(0),nodes.unsqueeze(0),nodes.unsqueeze(0),need_weights=False)
            nodes=nodes+attn.squeeze(0)
        index=int(c["target_index"]);node=nodes[index];global_node=glob[index]
        if self.operator is not None:node=self.operator(torch.cat([node,global_node],-1))
        return torch.cat([node,global_node,motion_summary],-1)
    def forward(self,c):
        return F.softplus(self.decoder(self.dropout(self._context(c))))

class MLP(PeakBase): pass
class TcnLstm(PeakBase): pass
class GnnTcn(PeakBase): pass
class MeshGraphNet(PeakBase): pass
class GraphTransformer(PeakBase): pass
class GraphOperator(PeakBase): pass

def build_model(name,global_dim,node_dim,edge_dim,motion_dim,response_dim=2,hidden=96,**hp):
    if name not in MODEL_NAMES:raise ValueError(f"Unknown model {name}; choose from {MODEL_NAMES}")
    return PeakBase(name,global_dim,node_dim,edge_dim,motion_dim,response_dim,hidden,**hp)

def model_hyperparameters(name):
    base=("recurrent_layers","dropout")
    if name=="graph_transformer":return base+("attention_heads",)
    if name in ("tcn_lstm","gnn_tcn"):return base+("conv_kernel",)
    return base
