# ms-swift 训练框架

大语言模型训练框架，支持 deepspeed 和 Megatron 并行技术。[官方说明](https://swift.readthedocs.io/zh-cn/v3.7/index.html)


## 安装
参考wiki文档：【Megatron-swift 环境安装】（https://iwiki.woa.com/p/4016971017）
pip install -e .

## 参数说明
| 参数 | 是否必须 | 默认值 | 说明 |
|------|----------|--------|------|
| `--ips` | 是 | - | 分布式训练的节点IP地址列表，多个IP用逗号分隔 |
| `--ssh_password` | 是 | - | 分布式训练的节点的帐户密码 |
| `--save` | 是 | - | 模型checkpoint保存路径 |
| `--dataset` | 是 | - | 训练数据集路径，支持jsonl格式 |
| `--port` | 否 | `29500` | 分布式通信端口号 |
| `--micro_bs` | 否 | `8` | 微批次大小（Micro Batch Size），单个GPU单次前向的样本数 |
| `--global_bs` | 否 | `64` | 全局批次大小（Global Batch Size），所有GPU累计的总样本数 |
| `--tensor_model_parallel_size` | 否 | `8` | 张量并行度（TP），将模型参数在层内切分到多个GPU |
| `--context_parallel_size` | 否 | `4` | 上下文并行度（CP），将长序列切分到多个GPU处理 |
| `--max_length` | 否 | `64000` | 最大序列长度，单位为token |
| `--epochs` | 否 | `1` | 训练轮数 |
| `--save_interval` | 否 | `50` | 保存checkpoint的间隔步数 |
| `--save_strategy` | 否 | `steps` | 保存模型的策略，可选为'no'、'steps'、'epoch' |

## 🚀 快速开始
### 1. recommend_setting
| 模型大小 | 节点数 | micro_batch_size | global_batch_size | tp | cp | pp |
|---:|---:|---:|---:|---:|---:|---:|
| 32b | 4 | 4 | 32 | 8 | 4 | 1 |
| 32b | 8 | 16 | 128 | 8 | 8 | 1 |
| 32b | 12 | 4 | 96 | 8 | 4 | 1 |
| 32b | 16 | 32 | 512 | 8 | 8 | 1 |
| 72b | 8 | 4 | 32 | 8 | 8 | 1 |
| 72b | 12 | 4 | 96 | 8 | 4 | 1 |

### 2. qwen3-32B Agent SFT
```bash
# 8*8 GPUS 训练任务启动示例脚本
bash ./train/ms-swift/examples/echo_qwen_train_shell/Qwen3_32B_agent_sft.sh \
    --ips "10.0.8.2,10.0.8.3,10.0.8.4,10.0.8.5,10.0.8.6,10.0.8.7,10.0.8.8,10.0.8.9" \
    --ssh_password "<PASSWORD>" \
    --save /path/to/output \
    --dataset /path/to/jsonl \
    --port 29900 \
    --micro_bs 16 \
    --global_bs 128 \
    --tensor_model_parallel_size 8 \
    --context_parallel_size 8 \
    --max_length 64000 \
    --epochs 1 \
    --save_strategy "epoch"
```

### 3. qwen2.5-72B Agent SFT
```bash
# 8*8 GPUS 训练任务启动示例脚本
bash ./train/ms-swift/examples/echo_qwen_train_shell/Qwen2.5_72B_agent_sft.sh \
    --ips "10.0.8.2,10.0.8.3,10.0.8.4,10.0.8.5,10.0.8.6,10.0.8.7,10.0.8.8,10.0.8.9" \
    --ssh_password "<PASSWORD>" \
    --save /path/to/output \
    --dataset /path/to/jsonl \
    --port 29900 \
    --micro_bs 4 \
    --global_bs 32 \
    --tensor_model_parallel_size 8 \
    --context_parallel_size 8 \
    --max_length 64000 \
    --epochs 1 \
    --save_interval 100
```

## ⚙️更多Megatron并行参数配置说明

参考 [wiki](https://iwiki.woa.com/p/4016813135)
- 建议开启 flash_attn 和 SP；
- 建议 CP > 1，否则相当于同时关闭了 CP 和 SP；
- 优先开 TP，剩下的余量，如果剩余显存空间较多，用 DP 或使用 CP + 高 Batch_Size (长窗口下建议优先 DP，超大LLM 建议优先 CP)，如果剩余显存空间较少，可以尽量提高 CP，提高效率；
- QWEN3-32B 在 16 GPU 情况下，最佳参数组合：MBS=4，TP=8，CP=2，DP=1，训练效率为 2828 token/s，2w数据预计训练5天；
- QWEN2.5-72B-Instruct 在 32 GPU 情况下，最佳参数组合：MBS=1，TP=8，CP=4，DP=1，训练效率为 2884 token/s，2w数据预计训练5天；

## 📊 数据格式

### SFT Agent 数据格式
```json
{
    "id": str  "数据唯一ID",
    "tools": List[Dict[str, Any]] 工具列表,
    "messages": List[Dict[str, str]] 对话列表,
}
```

## 🛠️ 工具使用

### 1. 多节点训练中断，释放GPU
```bash
bash ./train/ms-swift/examples/echo_qwen_train_shell/kill_nodes.sh --ips "10.0.0.107" --ssh_password "xxx"
```

### 2. 查看训练日志
```bash
tensorboard --logdir /path/to/output/runs
```
