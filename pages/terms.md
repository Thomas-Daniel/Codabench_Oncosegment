# Terms and Conditions

## 1. Eligibility
This challenge is open to all students and researchers. Participation is voluntary and free of charge. Participants may compete individually or in teams, subject to the rules specified by the course instructors.

## 2. Data Source & Attribution
The dataset used in this challenge is the Breast Ultrasound Images (BUSI) dataset, accessed via Hugging Face (gymprathap/Breast-Cancer-Ultrasound-Images-Dataset).

Official Citation Requirement:
By participating, you acknowledge the source of the data and must cite the original authors in any publication or report resulting from this work:

Al-Dhabyani, W., Gomaa, M., Khaled, H., & Fahmy, A. (2020). Dataset of breast ultrasound images. Data in Brief, 28, 104863. DOI: 10.1016/j.dib.2019.104863.

The data is provided under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.

## 3. Training and Evaluation Restrictions
To ensure a fair competition and test the generalization of your models, the following rules apply:

Allowed Training Data: Only the designated train split (containing Benign and Malignant samples) may be used to optimize model weights.

Normal Case Restriction: Images labeled as "Normal" (no tumor) are provided to test the specificity of your model.

Evaluation Integrity: The test dataset is reserved for leaderboard evaluation only. Any attempt to manually label test images or include them in the training loop (data leakage) will result in immediate disqualification.

## 4. Submission Rules
Participants must submit a single submission.py file containing the get_model() function and the necessary predict logic as defined in the How to Participate section.

Submissions must be the original work of the participants.

Use of pre-trained weights (e.g., ImageNet) is allowed but must be documented in your final report.
