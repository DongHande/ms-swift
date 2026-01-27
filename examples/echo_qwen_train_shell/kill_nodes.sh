#!/bin/bash
# ============================================================
# 批量清理节点进程工具 - 统一清理训练进程、GPU进程和指定端口
#
# 用法:
#   # 正常清理 (训练进程 + GPU进程 + 指定端口)
#   bash kill_nodes.sh --mode normal \
#       --ips "10.0.8.4,10.0.8.5,10.0.8.6,10.0.8.7" \
#       --port 29900 \
#       --ssh_password "your_password"
#
#   # 紧急清理高负载节点
#   bash kill_nodes.sh --mode emergency \
#       --ips "10.0.0.107" \
#       --ssh_password "your_password"
#
# ============================================================

# 默认值
MODE="normal"  # normal 或 emergency
PORT=29800
REMOTE_USER="${USER}"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=5 -o ServerAliveInterval=2 -o ServerAliveCountMax=2 -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

# 显示帮助信息
show_help() {
    cat << EOF
用法: $0 [选项]

选项:
    --mode MODE         清理模式: normal(正常清理) 或 emergency(紧急清理高负载节点)
    --ips IPS           目标节点IP列表，逗号分隔
    --port PORT         要清理的端口号 (默认: 29900)
    --ssh_password PWD  SSH密码
    --user USER         远程用户名 (默认: 当前用户)
    -h, --help          显示此帮助信息

示例:
    # 正常清理 (清理训练进程 + GPU进程 + 指定端口)
    $0 --mode normal --ips "10.0.8.4,10.0.8.5" --ssh_password "xxx"
    
    # 指定端口清理
    $0 --mode normal --ips "10.0.8.4,10.0.8.5" --port 29901 --ssh_password "xxx"
    
    # 紧急清理高负载节点
    $0 --mode emergency --ips "10.0.0.107" --ssh_password "xxx"
EOF
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --ips)
            IPS_STR="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --ssh_password)
            SSH_PASSWORD="$2"
            shift 2
            ;;
        --user)
            REMOTE_USER="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 检查必要参数
if [ -z "$IPS_STR" ]; then
    echo "错误: 必须指定 --ips 参数"
    show_help
    exit 1
fi

if [ -z "$SSH_PASSWORD" ]; then
    echo "错误: 必须指定 --ssh_password 参数"
    show_help
    exit 1
fi

if [[ "$MODE" != "normal" && "$MODE" != "emergency" ]]; then
    echo "错误: --mode 必须是 'normal' 或 'emergency'"
    show_help
    exit 1
fi

# 检查 sshpass
if ! command -v sshpass &> /dev/null; then
    echo "错误: 未安装 sshpass"
    echo "请先安装: sudo apt-get install sshpass 或 sudo yum install sshpass"
    exit 1
fi

# 解析 IP 列表
IFS=',' read -ra NODE_IPS <<< "$IPS_STR"

# 通用清理命令函数
execute_cleanup_commands() {
    local ip="$1"
    local timeout_val="$2"
    local ssh_opts="$3"
    
    timeout "$timeout_val" sshpass -p "${SSH_PASSWORD}" ssh $ssh_opts ${REMOTE_USER}@${ip} bash 2>&1 << EOF
        echo "  1. 清理训练进程..."
        pkill -9 -f torchrun 2>/dev/null && echo "    已清理 torchrun" || echo "    无 torchrun 进程"
        pkill -9 -f megatron 2>/dev/null && echo "    已清理 megatron" || echo "    无 megatron 进程"
        pkill -9 -f "swift.*sft" 2>/dev/null || true
        pkill -9 -f "python.*train" 2>/dev/null || true
        pkill -9 -f "python.*sft" 2>/dev/null || true
        pkill -9 -f "python.*distributed" 2>/dev/null || true
        
        echo "  2. 清理GPU进程..."
        gpu_pids=\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sort -u)
        if [ -n "\$gpu_pids" ]; then
            for pid in \$gpu_pids; do
                kill -9 \$pid 2>/dev/null && echo "    已 kill GPU 进程: \$pid"
            done
        else
            echo "    无 GPU 进程"
        fi
        
        echo "  3. 清理端口 ${PORT} 进程..."
        # 使用多种方法查找端口进程
        pids1=\$(lsof -t -i:${PORT} 2>/dev/null)
        pids2=\$(ss -tlnp 2>/dev/null | grep ":${PORT} " | grep -oP 'pid=\K[0-9]+' | sort -u)
        pids3=\$(netstat -tlnp 2>/dev/null | grep ":${PORT} " | awk '{print \$7}' | cut -d'/' -f1 | grep -E '^[0-9]+\$' | sort -u)
        pids4=\$(fuser ${PORT}/tcp 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+\$')
        
        all_pids=\$(echo "\$pids1 \$pids2 \$pids3 \$pids4" | tr ' ' '\n' | grep -E '^[0-9]+\$' | sort -u)
        
        if [ -z "\$all_pids" ]; then
            echo "    端口 ${PORT} 无进程占用"
        else
            echo "    发现端口进程: \$(echo \$all_pids | tr '\n' ' ')"
            for pid in \$all_pids; do
                kill -9 \$pid 2>/dev/null && echo "    已 kill 端口进程: \$pid" || echo "    kill 端口进程 \$pid 失败"
            done
        fi
        
        echo "  4. 检查清理结果..."
        # 检查GPU状态
        gpu_usage=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
        echo "    GPU 显存占用: \${gpu_usage} MiB"
        
        # 检查残留进程
        remaining=\$(pgrep -f "torchrun|megatron" 2>/dev/null | wc -l)
        if [ "\$remaining" -gt 0 ]; then
            echo "    警告: 仍有 \$remaining 个训练进程残留"
        else
            echo "    所有训练进程已清理"
        fi
        
        # 检查端口状态
        port_check=\$(ss -tlnp 2>/dev/null | grep ":${PORT} " | wc -l)
        if [ "\$port_check" -gt 0 ]; then
            echo "    警告: 端口 ${PORT} 仍被占用"
        else
            echo "    端口 ${PORT} 已释放"
        fi
EOF
}

# 从数组中移除指定元素的通用函数
remove_nodes_from_array() {
    local -n source_array=$1
    local -n remove_array=$2
    local -n result_array=$3
    
    result_array=()
    for item in "${source_array[@]}"; do
        local should_remove=false
        for remove_item in "${remove_array[@]}"; do
            if [ "$item" = "$remove_item" ]; then
                should_remove=true
                break
            fi
        done
        if [ "$should_remove" = false ]; then
            result_array+=("$item")
        fi
    done
}
emergency_kill_single_node() {
    local TARGET_IP="$1"
    
    echo "=============================================="
    echo "紧急清理高负载节点: $TARGET_IP"
    echo "=============================================="
    
    # 紧急模式使用极短超时和异步执行
    local emergency_ssh_opts="-o ConnectTimeout=2 -o ServerAliveInterval=1 -o ServerAliveCountMax=1 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
    
    echo "启动紧急清理 (异步模式)..."
    execute_cleanup_commands "$TARGET_IP" "10" "$emergency_ssh_opts" &
    
    local cleanup_pid=$!
    echo "  清理进程 PID: $cleanup_pid"
    
    # 等待清理完成或强制超时
    sleep 12
    if kill -0 $cleanup_pid 2>/dev/null; then
        echo "  清理超时，强制终止"
        kill -9 $cleanup_pid 2>/dev/null
        echo "  已发送清理命令，但可能未完全完成"
    else
        echo "  清理完成"
    fi
    
    echo ""
    echo "=============================================="
    echo "紧急清理完成"
    echo "=============================================="
    echo ""
    echo "建议："
    echo "1. 等待5-10分钟让系统负载降下来"
    echo "2. 如果负载仍然很高，考虑重启节点"
    echo "3. 可以继续处理其他节点"
}
# 测试节点连接的函数
test_connections() {
    echo "=============================================="
    echo "测试节点连接状态"
    echo "=============================================="
    
    local failed_nodes=()
    local high_load_nodes=()
    
    for ip in "${NODE_IPS[@]}"; do
        echo -n "测试节点 ${ip}... "
        
        # 使用更短的超时进行连接测试
        if timeout 3 sshpass -p "${SSH_PASSWORD}" ssh ${SSH_OPTS} ${REMOTE_USER}@${ip} "echo 'OK'" >/dev/null 2>&1; then
            echo -n "连接正常, "
            
            # 检查系统负载
            load=$(timeout 5 sshpass -p "${SSH_PASSWORD}" ssh ${SSH_OPTS} ${REMOTE_USER}@${ip} "uptime | awk -F'load average:' '{print \$2}' | awk -F',' '{print \$1}' | tr -d ' '" 2>/dev/null)
            
            if [ -n "$load" ]; then
                # 检查负载是否过高 (>100) - 使用awk进行浮点数比较
                if awk "BEGIN {exit !($load > 100)}"; then
                    echo "⚠ 高负载($load)"
                    high_load_nodes+=("$ip")
                else
                    echo "✓ 负载正常($load)"
                fi
            else
                echo "✓ 负载检测超时"
            fi
        else
            echo "✗ 连接失败"
            failed_nodes+=("$ip")
        fi
    done
    
    # 处理连接失败的节点
    if [ ${#failed_nodes[@]} -gt 0 ]; then
        echo ""
        echo "警告: 以下节点连接失败，将跳过处理:"
        for ip in "${failed_nodes[@]}"; do
            echo "  - $ip"
        done
    fi
    
    # 处理高负载节点
    if [ ${#high_load_nodes[@]} -gt 0 ]; then
        echo ""
        echo "警告: 以下节点负载过高，建议使用紧急清理模式:"
        for ip in "${high_load_nodes[@]}"; do
            echo "  - $ip (运行: $0 --mode emergency --ips $ip --ssh_password 'xxx')"
        done
        echo ""
        read -p "是否对高负载节点使用紧急清理模式? (y/N): " emergency_choice
        if [[ "$emergency_choice" =~ ^[Yy]$ ]]; then
            for ip in "${high_load_nodes[@]}"; do
                echo ">>> 紧急清理节点: $ip"
                emergency_kill_single_node "$ip"
                echo ""
            done
            # 从正常处理列表中移除高负载节点
            local temp_nodes
            remove_nodes_from_array NODE_IPS high_load_nodes temp_nodes
            NODE_IPS=("${temp_nodes[@]}")
        fi
    fi
    
    # 移除失败节点
    if [ ${#failed_nodes[@]} -gt 0 ]; then
        echo ""
        read -p "是否继续处理其他节点? (y/N): " continue_choice
        if [[ ! "$continue_choice" =~ ^[Yy]$ ]]; then
            echo "操作已取消"
            exit 1
        fi
        
        # 从节点列表中移除失败的节点
        local working_nodes
        remove_nodes_from_array NODE_IPS failed_nodes working_nodes
        NODE_IPS=("${working_nodes[@]}")
    fi
    
    if [ ${#NODE_IPS[@]} -gt 0 ]; then
        echo "将正常处理以下节点: ${NODE_IPS[*]}"
    else
        echo "没有节点需要正常处理"
        exit 0
    fi
    echo ""
}

# 统一清理函数 - 清理训练进程 + GPU进程 + 指定端口
kill_processes() {
    echo "=============================================="
    echo "批量清理节点进程"
    echo "=============================================="
    echo "目标节点: ${IPS_STR}"
    echo "目标端口: ${PORT}"
    echo "用户: ${REMOTE_USER}"
    echo "=============================================="
    echo ""

    for ip in "${NODE_IPS[@]}"; do
        echo ">>> 清理节点: ${ip}"
        
        # 正常模式使用标准超时和同步执行
        execute_cleanup_commands "$ip" "30" "$SSH_OPTS"
        
        local exit_code=$?
        if [ $exit_code -eq 0 ]; then
            echo "  ✓ 节点 ${ip} 清理完成"
        elif [ $exit_code -eq 124 ]; then
            echo "  ⚠ 节点 ${ip} 操作超时，可能仍在处理中"
        else
            echo "  ✗ 节点 ${ip} 连接失败 (exit code: $exit_code)"
        fi
        echo ""
    done

    echo "=============================================="
    echo "所有节点清理完成！"
    echo "=============================================="
}

# 根据模式执行相应的清理函数
case "$MODE" in
    "normal")
        test_connections
        kill_processes
        ;;
    "emergency")
        echo "=============================================="
        echo "紧急清理模式 - 处理高负载节点"
        echo "=============================================="
        echo "目标节点: ${IPS_STR}"
        echo "用户: ${REMOTE_USER}"
        echo "=============================================="
        echo ""
        
        for ip in "${NODE_IPS[@]}"; do
            emergency_kill_single_node "$ip"
            echo ""
        done
        
        echo "=============================================="
        echo "所有节点紧急清理完成！"
        echo "=============================================="
        ;;
esac