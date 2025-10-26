# Запуск сервиса
1. Создаём tmux окно для vLLM
```bash
tmux new -s vllm
```
2. Запускаем vLLM сервек
```bash
vllm serve models/qwen_llm   --host 0.0.0.0   --port 8001   --dtype float16   --tensor-parallel-size 1   --gpu-memory-utilization 0.2   --trust-remote-code   --max-model-len 4096   --swap-space 8
```
3. Создаём tmux окно для API
```bash
 tmux new -s api
```
4. Поднимаем API
```bash
python app.py
```