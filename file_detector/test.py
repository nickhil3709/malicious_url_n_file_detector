import torch
print(torch.cuda.is_available())  # True means GPU is ready
print(torch.cuda.get_device_name(0))  # Prints GPU name
