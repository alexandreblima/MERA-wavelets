#!/bin/bash

# Monitor e gerenciador de experimentos paralelos
# Fornece visão em tempo real do progresso e recursos

MONITOR_INTERVAL=${MONITOR_INTERVAL:-10}  # Segundos entre atualizações
LOG_FILE=${LOG_FILE:-"experiment_monitor.log"}

echo "📊 MONITOR DE EXPERIMENTOS PARALELOS"
echo "===================================="
echo "Intervalo: ${MONITOR_INTERVAL}s | Log: $LOG_FILE"
echo ""

# Função para obter uso de CPU
get_cpu_usage() {
    top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}'
}

# Função para obter uso de memória
get_memory_usage() {
    free | awk 'NR==2{printf "%.1f", $3*100/$2}'
}

# Função para obter uso de GPU
get_gpu_usage() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
        awk -F',' '{printf "%.0f,%.0f,%.0f", $1, $2, $3}'
    else
        echo "N/A,N/A,N/A"
    fi
}

# Função para contar processos Julia ativos
count_julia_processes() {
    pgrep -f "julia.*run_mawi_experiments" | wc -l
}

# Função para obter progresso estimado
get_estimated_progress() {
    local total_windows=100  # Assumindo 100 janelas por experimento
    local active_experiments=$(count_julia_processes)

    # Tentar estimar progresso baseado em arquivos de saída
    local completed_files=$(ls results/optimized_retain_*.csv 2>/dev/null | wc -l)
    local total_experiments=5  # 5 taxas de retenção

    if [ $total_experiments -gt 0 ]; then
        local progress=$((completed_files * 100 / total_experiments))
        echo $progress
    else
        echo 0
    fi
}

# Cabeçalho do monitor
printf "%-8s %-8s %-12s %-15s %-10s %-12s\n" "TIME" "CPU%" "MEM%" "GPU%(MEM)" "PROCS" "PROGRESS%"
echo "--------------------------------------------------------------------------------"

# Loop de monitoramento
start_time=$(date +%s)
while true; do
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))

    cpu_usage=$(get_cpu_usage)
    mem_usage=$(get_memory_usage)
    gpu_info=$(get_gpu_usage)
    active_procs=$(count_julia_processes)
    progress=$(get_estimated_progress)

    # Formatar tempo decorrido
    hours=$((elapsed / 3600))
    minutes=$(( (elapsed % 3600) / 60 ))
    seconds=$((elapsed % 60))
    time_str=$(printf "%02d:%02d:%02d" $hours $minutes $seconds)

    # Extrair info da GPU
    gpu_util=$(echo $gpu_info | cut -d',' -f1)
    gpu_mem_used=$(echo $gpu_info | cut -d',' -f2)
    gpu_mem_total=$(echo $gpu_info | cut -d',' -f3)

    if [ "$gpu_util" = "N/A" ]; then
        gpu_str="N/A"
    else
        gpu_mem_percent=$((gpu_mem_used * 100 / gpu_mem_total))
        gpu_str="${gpu_util}%(${gpu_mem_percent}%)"
    fi

    # Exibir linha do monitor
    printf "%-8s %-8.1f %-12.1f %-15s %-10d %-12d%%\n" \
           "$time_str" "$cpu_usage" "$mem_usage" "$gpu_str" "$active_procs" "$progress"

    # Log detalhado
    echo "$(date '+%Y-%m-%d %H:%M:%S') - CPU: ${cpu_usage}%, MEM: ${mem_usage}%, GPU: $gpu_str, Procs: $active_procs, Progress: ${progress}%" >> "$LOG_FILE"

    # Verificar se ainda há experimentos rodando
    if [ $active_procs -eq 0 ] && [ $progress -ge 100 ]; then
        echo ""
        echo "🎉 Todos os experimentos concluídos!"
        break
    fi

    sleep $MONITOR_INTERVAL
done

echo ""
echo "📊 Monitor finalizado. Log salvo em: $LOG_FILE"