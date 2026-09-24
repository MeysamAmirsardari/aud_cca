# Canonical Correlation Analysis (CCA) Toolkit
Meysam Amirsardari (Univ. of Maryland) and 
Malcolm Slaney (Stanford CCRMA and ICSI, Berkeley)

Our goal is to provide a production quality, reference implementation of
canonical correlation analysis (CCA). In our case, we are interested in finding
a linear model that best connects auditory stimulation to (EEG) brain waves. 
As shown in the figure below, this can be done in several different fashions:
including forward (predict EEG from audio) and backward (predict audio from EEG).
But the paper by de Cheveigne shows that we get the best model if we rotate
the audio and the EEG signals, each in their own canonical directions, to 
find a new subspace that maximizes the resulting correlation.

<img src="images/CCA Overview.png" alt="Overview if CCA and Alternatives" 
  width="300" height="200">

CCA is not limited to brain data, and can be used to find a match between 
any two related data [for example 
[FaceSync](https://papers.nips.cc/paper/2000/hash/9f6992966d4c363ea0162a056cb45fe5-Abstract.html)]

# Examples
Need to add code samples here....

See this Google colab for examples: 
[Telluride 2026 CCA Demo](https://colab.research.google.com/drive/1N1GSTs8QyAsurOZKiUbKH-MrDrMzxppE?usp=sharing) 

# References
Alain de Cheveigné, Daniel D.E. Wong, Giovanni M. Di Liberto, Jens Hjortkjær, Malcolm Slaney, Edmund Lalor,
Decoding the auditory brain with canonical component analysis,
NeuroImage,
Volume 172,
2018,
Pages 206-216,
ISSN 1053-8119,
[Link to Paper](https://doi.org/10.1016/j.neuroimage.2018.01.033).
