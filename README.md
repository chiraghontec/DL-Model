# PyTorch_Image_Classification_Pipeline

Goal: Train an image classification model on the crop pest and disease dataset using the ImageFolder format.

Project files:
- `requirements.txt` - Python dependencies
- `split_data.py` - Script to split a dataset into `train/`, `val/`, `test/` using `splitfolders`
- `data_load.py` - Example script that uses `torchvision.datasets.ImageFolder` and creates DataLoaders

Quick start

1) Install prerequisites (recommended in a virtualenv):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r "./requirements.txt"
# Or install manually as provided:
# pip3 install torch torchvision torchaudio
# pip3 install splitfolders
```

2) Prepare your original data

Place your images in a folder where each class has its own subfolder, for example:

```
/Users/vinayakprasad/Documents/Major Project/Dataset for Crop Pest and Disease Detection/Cashew_Healthy/...
/Users/vinayakprasad/Documents/Major Project/Dataset for Crop Pest and Disease Detection/Tomato_Blight/...
```

3) Run the splitter (80/10/10 default):

```bash
python3 "split_data.py" \
  --input "/Users/vinayakprasad/Documents/Major Project/Dataset for Crop Pest and Disease Detection" \
  --output "/Users/vinayakprasad/Documents/Major Project/YourCropDataset" \
  --ratio 0.8 0.1 0.1
```

4) Inspect / load with `data_load.py`:

```bash
python3 "data_load.py" --data_root "/Users/vinayakprasad/Documents/Major Project/YourCropDataset" --batch_size 32
```

Notes
- If your paths contain spaces (like `Major Project`), quote them on the command line as shown above.
- The `data_load.py` script demonstrates transforms (resize/center-crop/normalize) and prints dataset sizes and an example batch shape.
- For training models, adapt `data_load.py` to construct a model, optimizer, loss, and a training loop.
