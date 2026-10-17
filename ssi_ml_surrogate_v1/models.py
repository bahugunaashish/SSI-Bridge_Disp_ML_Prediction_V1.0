"""Six graph/temporal model families predicting first-node six-DOF peak values."""
import torch
from torch import nn
from torch.nn import functional as F

MODEL_NAMES = ("mlp", "tcn_lstm", "gnn_tcn", "meshgraphnet", "graph_transformer", "graph_operator")

def _mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))

class TemporalEncoder(nn.Module):
    def __init__(self, d, h, kind="tcn"):
        super().__init__()
        self.kind = kind
        if kind == "lstm": self.body = nn.LSTM(d, h, num_layers=2, batch_first=True, dropout=.05)
        else: self.body = nn.Sequential(nn.Conv1d(d,h,5,padding=2),nn.GELU(),nn.Conv1d(h,h,5,padding=4,dilation=2),nn.GELU(),nn.Conv1d(h,h,5,padding=8,dilation=4),nn.GELU())
    def forward(self, x):
        return self.body(x)[0] if self.kind == "lstm" else self.body(x.T.unsqueeze(0)).transpose(1,2).squeeze(0)

class GraphMessageLayer(nn.Module):
    def __init__(self,h,e):
        super().__init__(); self.edge=_mlp(2*h+e,h,h); self.node=_mlp(2*h,h,h); self.norm=nn.LayerNorm(h)
    def forward(self,nodes,edge_index,edge_features):
        agg=torch.zeros_like(nodes)
        if edge_index.numel():
            src,dst=edge_index; msg=self.edge(torch.cat([nodes[src],nodes[dst],edge_features],-1)); agg.index_add_(0,dst,msg)
            deg=torch.zeros(len(nodes),device=nodes.device,dtype=nodes.dtype); deg.index_add_(0,dst,torch.ones_like(dst,dtype=nodes.dtype)); agg=agg/deg.clamp_min(1).unsqueeze(-1)
        return self.norm(nodes+self.node(torch.cat([nodes,agg],-1)))

class PeakBase(nn.Module):
    def __init__(self,node_dim,edge_dim,global_dim,motion_dim,response_dim=6,hidden=96,kind="tcn"):
        super().__init__(); self.hidden=hidden
        self.node_encoder=_mlp(node_dim,hidden,hidden); self.global_encoder=_mlp(global_dim,hidden,hidden)
        self.motion_encoder=TemporalEncoder(motion_dim,hidden,kind)
        self.decoder=_mlp(hidden*6,hidden,response_dim)
    def context(self,c):
        m=self.motion_encoder(c["ground_motion"])
        # Summaries preserve signed directional content and motion amplitude across the full EQ.
        summary=torch.cat([m.mean(0),m.amax(0),m.amin(0),m.square().mean(0).sqrt()],0)
        n=self.node_encoder(c["node_features"]); g=self.global_encoder(c["global_features"]).expand(len(n),-1)
        return n,g,summary
    def graph(self,n,c): return n
    def forward(self,c):
        n,g,m=self.context(c); n=self.graph(n,c); m=m.expand(len(n),-1)
        return F.softplus(self.decoder(torch.cat([n,g,m],-1)))

class MLP(PeakBase):
    """MLP with whole-record summary features."""
    pass
class TcnLstm(PeakBase):
    def __init__(self,**kw): super().__init__(**kw,kind="lstm")
class GnnTcn(PeakBase):
    def __init__(self,layers=3,**kw):
        super().__init__(**kw); self.layers=nn.ModuleList([GraphMessageLayer(self.hidden,kw["edge_dim"]) for _ in range(layers)])
    def graph(self,n,c):
        for layer in self.layers: n=layer(n,c["edge_index"],c["edge_features"])
        return n
class MeshGraphNet(GnnTcn):
    def __init__(self,**kw): super().__init__(layers=5,**kw)
class GraphTransformer(GnnTcn):
    def __init__(self,heads=4,**kw): super().__init__(layers=2,**kw); self.attn=nn.MultiheadAttention(self.hidden,heads,batch_first=True)
    def graph(self,n,c):
        n=super().graph(n,c); a,_=self.attn(n.unsqueeze(0),n.unsqueeze(0),n.unsqueeze(0),need_weights=False); return n+a.squeeze(0)
class GraphOperator(GnnTcn):
    """Low-rank global-to-node operator-style peak decoder."""
    def __init__(self,**kw):
        super().__init__(layers=3,**kw); self.operator=_mlp(self.hidden*2,self.hidden,32); self.op_decoder=_mlp(32+self.hidden*4,self.hidden,kw.get("response_dim",6))
    def forward(self,c):
        n,g,m=self.context(c); n=self.graph(n,c); basis=self.operator(torch.cat([n,g],-1)); m=m.expand(len(n),-1)
        return F.softplus(self.op_decoder(torch.cat([basis,m],-1)))

def build_model(name,*,node_dim,edge_dim,global_dim,motion_dim,response_dim=6,hidden=96):
    kw=dict(node_dim=node_dim,edge_dim=edge_dim,global_dim=global_dim,motion_dim=motion_dim,response_dim=response_dim,hidden=hidden)
    classes={"mlp":MLP,"tcn_lstm":TcnLstm,"gnn_tcn":GnnTcn,"meshgraphnet":MeshGraphNet,"graph_transformer":GraphTransformer,"graph_operator":GraphOperator}
    if name not in classes: raise ValueError(f"Unknown model {name}; choose from {MODEL_NAMES}")
    return classes[name](**kw)
