import torch

if torch.cuda.is_available():
    print("CUDA está disponible. Usando GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA no está disponible. Asegúrate de tener una GPU compatible.")
