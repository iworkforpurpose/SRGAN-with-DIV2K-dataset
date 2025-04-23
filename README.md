📸 SRGAN: Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
This repository contains a PyTorch implementation of the SRGAN model, as described in the paper:

Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network

The model is trained on the DIV2K dataset to generate high-resolution images from their low-resolution counterparts.

    SRGAN-with-DIV2K-dataset/
    ├── src/
    │   ├── model.py           # Generator and Discriminator architectures
    │   ├── train_srgan.py     # Training script
    │   ├── data_loader.py     # Data loading utilities
    │   └── utils.py           # Utility functions (e.g., metrics)
    ├── .gitignore             # Specifies files to ignore in version control
    ├── requirements.txt       # Python dependencies
    ├── README.md              # Project overview and instructions

🧪 Results
After training the model for 50 epochs, the following performance metrics were achieved on the validation set:

PSNR: 8.7814 dB

SSIM: 0.0630

Note: These results are based on the current training setup and may vary with different configurations.

🛠️ Setup Instructions
1. Clone the Repository
git clone https://github.com/iworkforpurpose/SRGAN-with-DIV2K-dataset.git
cd SRGAN-with-DIV2K-dataset

2. Install Dependencies
It's recommended to use a virtual environment:
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

3. Prepare the DIV2K Dataset
Download the DIV2K dataset from the official DIV2K Dataset page. Organize the dataset as follows:

          data/
          ├── DIV2K_train/
          │   ├── HR/   # High-resolution training images
          │   └── LR/   # Corresponding low-resolution images
          ├── DIV2K_valid/
              ├── HR/   # High-resolution validation images
              └── LR/   # Corresponding low-resolution images
🚀 Training the Model
To train the SRGAN model, run:

python3 src/train_srgan.py \
  --train-lr data/DIV2K_train/LR \
  --train-hr data/DIV2K_train/HR \
  --val-lr   data/DIV2K_valid/LR \
  --val-hr   data/DIV2K_valid/HR \
  --epochs 50 \
  --batch-size 16 \
  --lr 1e-4 \
  --adv-weight 1e-3 \
  --hr-crop 96 \
  --num-workers 4 \
  --checkpoint-dir checkpoints/ \
  --sample-dir samples/
  
Adjust the parameters as needed for your specific setup.

📈 Evaluating the Model
After training, evaluate the model's performance using PSNR and SSIM metrics

📓 Colab Notebook
For an interactive demonstration and further experimentation, refer to the Colab notebook:
https://colab.research.google.com/drive/1t38OeKeG8_qglbJpNLDfqA2Y1PhKDFGz?usp=sharing

🙌 Acknowledgements

DIV2K Dataset - https://data.vision.ee.ethz.ch/cvl/DIV2K/
Original SRGAN Paper - https://arxiv.org/abs/1609.04802



