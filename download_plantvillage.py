import kagglehub

# Download latest version of the PlantVillage dataset
path = kagglehub.dataset_download("emmarex/plantdisease")

print("Path to dataset files:", path)
