# Disfluency Detection Web Application

This repository contains the code for a web application that allows clinicians to infer possible diseases from audio files. It is inspired by the [disfluency_detection_from_audio](https://github.com/amritkromana/disfluency_detection_from_audio) project.

## Project Overview

This application uses machine learning algorithms to analyze audio files and detect disfluencies, which can be indicative of certain diseases. 

## Installation

### Dependencies
The following packages are needed:

pandas==1.5.0
torch==1.12.1
torchaudio==0.12.1
transformers==4.22.2
whisper_timestamped==1.14.4
gdown==5.1.0

## Usage

To launch the web application, run the following command:
```
python app.py
```