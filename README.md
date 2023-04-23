### Implementation of ESRGAN using RRDB x4 architecture and VGG19
create a conda virtual environment using the command <br>
`conda create -n esrgan python3.11.3`

Add add the packages and dependencies using `pip install -r requirements.txt` <br>

Create a new directory in the project folder and name it `model` <br>

Download the pretrained model weights from the [Google Drive](https://drive.google.com/drive/folders/1ycDAl76gRDWxdFmpzNtzbTrTcXeDrpui?usp=sharing) Link and paste them in the `model` folder <br>

To run the pretrained model enter the command `python test_remote_finetune.py` <br>

To train the model without pretrained weights enter the command `python train_remote.py` <br>

To train the model with pretrained weights enter the command `python finetune_remote.py` <br>
