import torch

# 检查计算能力支持
print("支持的计算能力:", torch.cuda.get_arch_list())
# 应包含 sm_90 和 sm_12x

# 验证设备识别
print("设备名称:", torch.cuda.get_device_name(0))
print("计算能力:", torch.cuda.get_device_capability(0))
print("CUDA 可用:", torch.cuda.is_available())