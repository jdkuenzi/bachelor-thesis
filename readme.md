# Connectionist models for chords and notes detection on guitar

This is my Bachelor’s thesis and work. The subject was to find a way to detect guitar notes and chords for mono sounds sampled at 44.1[kHz] in real time.

My thesis is in French and I made her with LaTex during my B.Sc. at the *Haute école du paysage, d’ingénierie et d’architecture de Genève (HEPIA)* at Geneva in Switzerland.

## Author

Jean-Daniel Küenzi - jeandanielkuenzi@gmail.com

## Resume

In recent years, Artificial Intelligence (AI) has only progressed and revolutionised many areas such as law, medicine, biology and even music. This work is a continuation of my previous work, the aim of which was to detect trumpet notes on an octave of C. The objective of this work is to use different connectionist model architectures (neural network) and see if it is possible to detect guitar notes and chords for mono sounds sampled at 44.1 [kHz]. In addition, it will be possible to consider the signal in the discrete world (sampled) and in the frequency world. Thus, the goal is to achieve real-time use with as little latency as possible (both visual and auditory). In the music world, latency greater than five milliseconds for signal processing (calculations) is undesirable and is felt strongly. The challenge is therefore that the proposed architectures predict as precisely as possible the chord or the note played in a time of less than five milliseconds. For this job, I also had to create a dataset from scratch. Considering the time and resources available to me, I decided to tackle the tempered scale of Western music (not microtonal music) and represent minor and major three-tone chords and notes in their different octaves for the standard tuning of a guitar (E, A, D, G, B, E). The architecture that gave me the best results, an average accuracy of 95.51%, uses a fast Fourier transform to pass through the frequency world and is made up of a bidirectional LSTM cell.

## Conda environement

For this work, I used a conda environement with the following packages:

Name | Version
:-:|:-:
pyaudio | 0.2.11
tk | 8.6.10
librosa | 0.8.1
pytorch | 1.8.1
cudatoolkit | 10.2.89
tqdm | 4.61.1

You can recreate my environments with the .yml file
