import torch
import torch.nn as nn
import numpy as np

from keys import Cols
from Model.Dynamic_models import scaler

# 自定义自适应稀疏自注意力（ASSA）模块
class ASSA(nn.Module):
    def __init__(self, hidden_size, num_heads, sparsity_threshold=0.1):
        super(ASSA, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.sparsity_threshold = sparsity_threshold

        # 定义查询、键、值线性层
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        # 输出投影层
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # 计算查询、键、值
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 计算自适应稀疏掩码
        attn_probs = torch.softmax(attn_scores, dim=-1)
        mask = attn_probs > self.sparsity_threshold
        attn_probs = attn_probs * mask.float()

        # 归一化稀疏注意力分数
        attn_probs = attn_probs / (attn_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # 计算上下文向量
        context = torch.matmul(attn_probs, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        # 输出投影
        output = self.out_proj(context)
        return output

# 定义Transformer模型（使用ASSA进行稀疏注意力）
class SparseAttentionTransformer(nn.Module):
    def __init__(self, input_size, output_size, future_steps, hidden_size=256):
        super(SparseAttentionTransformer, self).__init__()

        # 新增线性层，将输入维度转换为hidden_size
        self.input_proj = nn.Linear(input_size, hidden_size)

        # 使用 ASSA 模块
        self.assa = ASSA(hidden_size=hidden_size, num_heads=8)

        # 后续全连接层
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        # x.shape = (batch_size, sequence_length, input_size)
        # 通过线性层将输入维度转换为hidden_size
        x = self.input_proj(x)

        # 使用 ASSA 进行编码
        transformer_out = self.assa(x)

        # 取最后一个时间步的输出
        last_hidden_state = transformer_out[:, -1, :]

        # 通过全连接层得到最终预测
        x = torch.relu(self.fc1(last_hidden_state))
        x = self.fc2(x)

        return x
    
class AWE_Electrolyzer_Dynamic():
    def __init__(self):
        pass
    
# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

feature_cols = [Cols.current_density, Cols.lye_flow, Cols.temp_environment, Cols.lye_temp, Cols.temp_out, Cols.pressure]
target_cols = [Cols.temp_out]

# 创建模型实例
input_size = len(feature_cols)
output_size = len(target_cols)
model = SparseAttentionTransformer(input_size, output_size, future_steps=10)

# 加载模型状态字典
model.load_state_dict(torch.load(r'D:\Devs\Single_Stack_MLJ\src\Model\Dynamic_models\Thermodynamic_model.pth', map_location=device))
model.eval()  # 将模型设置为评估模式
print("Model loaded successfully.")

model.to(device)

def temp_out_next_cal(awe_state_df):
    """_summary_

    Args:
        awe_state_df (_type_): _description_
        current_density (_type_): _description_
        lye_flow (_type_): _description_
        temp_environment (_type_): _description_
        lye_temp (_type_): _description_
        temp_out (_type_): _description_
        pressure (_type_): _description_
    """

    # 取电解槽含当前往前推10步的状态
    awe_state_before_df = awe_state_df.iloc[-10:]

    # 计算下一步电解槽的出口温度
    awe_state_before_scaler = np.array([scaler.transform(awe_state_before_df.values)])
    awe_state_before_tensor = torch.tensor(awe_state_before_scaler, dtype=torch.float32).to(device)
    with torch.no_grad():
        temp_out_pred = model(awe_state_before_tensor)
    
    return temp_out_pred