import torch

# --- Dataset and Dataloader Configuration ---
DATASET_ROOT_PATH = r".\dataset\captures"  # IMPORTANT: Update this path
IMG_HEIGHT = 224
IMG_WIDTH = 224
MSI_CHANNELS = 6  # Number of multispectral channels (excluding RGB)
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
# Note: TEST_SPLIT is implicitly 1.0 - TRAIN_SPLIT - VAL_SPLIT

# --- Training Hyperparameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 100
BATCH_SIZE = 4
LR_G = 0.0002  # Learning rate for the generator
LR_D = 0.0002  # Learning rate for the discriminator
BETA1 = 0.5    # Adam optimizer beta1
N_CRITIC = 5   # Train generator every N_CRITIC discriminator iterations

# --- Loss Function Weights ---
LAMBDA_GP = 10         # Gradient penalty lambda
LAMBDA_RECON = 100     # L1 reconstruction loss lambda
LAMBDA_SSIM = 1.0      # SSIM loss lambda

# --- Model Configuration ---
SWIN_MODEL_NAME = 'swin_tiny_patch4_window7_224' # Swin Transformer model

# --- File Paths ---
BEST_GENERATOR_MODEL_PATH = r".\msi-generation-service\model_api\msi_generator_model.pth"