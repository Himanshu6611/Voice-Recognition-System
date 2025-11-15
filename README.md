# Voice-Recognition-System
This project implements a basic voice recognition system that records speech using a microphone, extracts audio features such as MFCCs and pitch, and authenticates users by comparing live voice samples with stored templates using cosine similarity. It demonstrates the core steps of voice-based biometric verification.
Overview

The application captures a user’s voice, processes it to extract identifying speech features, and stores them as a template. During authentication, the system records a new sample and compares it with the stored template using cosine similarity to determine a match. It is designed for practical lab demonstrations and beginner-level biometrics.

✨ Features

🔹 Voice Recording

-->Records 3-second audio samples using a microphone

-->Stores audio as .wav files

🔹 Feature Extraction

-->Extracts MFCCs (Mel-Frequency Cepstral Coefficients)

-->Computes average pitch

-->Generates a combined feature vector

🔹 User Enrollment

-->Saves extracted features as .npy template files

-->Creates unique templates per user

🔹 Authentication

-->Records a new voice sample (probe)


-->Extracts features and compares with saved template


-->Uses cosine similarity for matching


-->Displays ACCEPTED or REJECTED decision




⚙️ Installation

Install required Python packages:

pip install numpy sounddevice soundfile librosa scikit-learn

▶️ How to Run

Run the script:


python voice_recognition.py


Options in Program


1 → Enroll new user

2 → Authenticate existing user

🚀 Workflow

-->User chooses enroll or authenticate

-->System records 3-second microphone input

-->Extracts MFCC & pitch features

-->Stores template (enroll) or compares with template (authenticate)

Outputs similarity score and match result

🏁 Conclusion

This project demonstrates a simple voice recognition system focusing on recording audio, extracting distinguishing features, and authenticating users by comparing voice templates. It is ideal for biometric labs, practical assignments, and introductory speech-processing tasks.
