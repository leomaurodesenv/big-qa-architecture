from src.model import ArchGuardModel
from src.dataset import DisasterTweetJailbreakingDataset


# Load all splits
dataset_loader = DisasterTweetJailbreakingDataset()
train_data = dataset_loader.get_split("train")
print(train_data["prompt_attack"][1])


# Initialize the model
model = ArchGuardModel()
# Batch predictions
predictions = model.predict(train_data["prompt_attack"][0:1])
print(predictions)
# Get probabilities for all classes
proba = model.predict_proba(train_data["prompt_attack"][0:1])
print(proba)
