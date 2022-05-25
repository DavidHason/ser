# Leveraged Mel spectrograms using Harmonic and Percussive Components in Speech Emotion Recognition
Speech Emotion Recognition (SER) affective technology enables the intelligent embedded devices to interact with sensitivity. Similarly, call centre employees recognise customers’ emotions from their pitch, energy, and tone of voice so as to modify their speech for a highquality interaction with customers. This work explores, for the first time, the effects of the harmonic and percussive components of Mel spectrograms in SER. We attempt to leverage the Mel spectrogram by decomposing distinguishable acoustic features for exploitation in our proposed architecture, which includes a novel feature map generator algorithm,
a CNN-based network feature extractor and a multi-layer perceptron (MLP) classifier. This study specifically focuses on effective data augmentation techniques for building an enriched hybrid-based feature map.

This process results in a function that outputs a 2D image so that it can be used as input data for a pre-trained CNN-VGG16 feature extractor.
Furthermore, we also investigate other acoustic features such as MFCCs, chromagram, spectral contrast, and the tonnetz to assess our proposed
framework. A test accuracy of 92.79% on the Berlin EMO-DB database is achieved. Our result is higher than previous works using CNN-VGG16.

Model architecture and training: 
We use the CNN-VGG16 as a feature extractor to learn from high dimensional feature maps since the network can learn from small variations that occur in the extracted features maps. the architecture consists of an VGG16 and MLP network, which serve as an feature extractor and emotion classifier, respectively. First, the subsamples are extracted from a fixed window size and then feature maps are built using the proposed feature map function. Therefore, the input to the VGG16 feature extractor is a 2-D feature map in the dimension of (128 x 128 x 2). The input to the MLP classifier is a 2048 one-dimensional vector generated by VGG16.


Experimental analysis:
The sample voices are randomly partitioned and 80% are used for the training set and 10% for the validation and test set for the speaker-independent classification task. We apply an oversampling strategy to compensate the minority classes and increase the voice samples before feeding them to the feature extractor network during the pre-processing phase.

The MLP classifier includes four fully connected layers with the ReLU activation function and softmax in the output layer. Dense 1 and 2 have a 1024 input with a 0.5 dropout value, and dense 3 and 4 are set to 512 input with 0.3 dropouts. The ADAM optimiser with a learning rate of 0.0001 is selected for our architecture design.

 The classifier is trained on 128 epochs with a batch size of 128 and used an Nvidia GPU. The window size is set to 2048 with (128 x 128) bands and frames
to obtain each subsample length = 2.9 sec. Then, the subsamples are created in each defined data frame. Finally, 167426 signal subsamples and 9717 feature
maps are obtained from a sample rate of 88KHz. Based on the time-frequency trade-off, large frame size is chosen to obtain high-frequency resolution rather
than time resolution since analysing the frequency of speech signal enables us to decode emotion.

