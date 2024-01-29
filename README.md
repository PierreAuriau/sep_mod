# separate modalities

## Experience CIFAR-MNIST
Train to disentanglement MNIST (or CIFAR-10) information from a dataset than combines CIFAR-10 and MNIST images.
Experiences :
* Freeze or not the weak encoder during strong encoder training
* Replace the weak and strong-common encoders by a unique encoder
* Try to compute jem loss between strong-specific & weak and/or strong-specific & strong-common
* Train a decoder to retrieve the weak modality (MNIST is this case)

## Experiences on neuro-imaging
Train to separate specific-VBM information to information common between VBM and sulcus skeletons