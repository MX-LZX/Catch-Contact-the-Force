from utils.dataset import HallTimeDataset

# 初始化一次 dataset，会自动保存 x/y 的 mean 和 scale
dataset = HallTimeDataset("data")
print("Scalers saved to checkpoints/")
