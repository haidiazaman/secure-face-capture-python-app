note to self: add in the eye and mouth blink stuff here


# Python demo app for Gojek Hackathon
![alt text](https://github.com/haidiazaman/secure-face-capture-app/blob/main/gifs/gojek_hackathon_gifs/gojek_hackathon_right_eye_blocked.gif)
![alt text](https://github.com/haidiazaman/secure-face-capture-app/blob/main/gifs/gojek_hackathon_gifs/gojek_hackathon_left_eye_blocked.gif)
![alt text](https://github.com/haidiazaman/secure-face-capture-app/blob/main/gifs/gojek_hackathon_gifs/gojek_hackathon_mouth_blocked.gif)

# Objective
The motivation behind this app is simple. Companies around the world have a selfie-capture as part of their onboarding process, face verification process, etc. Most of these selfie capture processes do not include checks to ensure certain potential issues like users wearing a mask, portion of face blocked, etc. This leads to issues like inappropriate images for usage in other models further down the pipeline, waste of storage resources of invalid images etc. 


The purpose of this app is to ensure that only real users are able to take a good quality selfie. The emphasis on real users is so that the app is able to reject spoof cases (e.g. fraudsters using a printed face to impersonate others) via an eye blink model. This eye blink model can be enhanced in robustness through ongoing training that incorporates new strategies employed by attackers attempting to overcome its defenses. When this has passed, the app will take a few seconds to ensure that the face is not blocked. This is done via several other models that will check for any face blocking items. The interesting part here is that the app will be able to tell the user which part of the face is blocked. This is an improvement over current face quality models which may include a "Block" label which only tells you that the face is blocked without specifying which part. This may be frustrating for users who may repeatedly be rejected without knowing the exact reason. This frustration multiplied by thousands of users could potentially lead to thousands of dollars lost by the company as users give up and decide not to onboard onto the companies' app. After this entire process, the selfie is taken.

# Potential flaws
The app is not flawless. The liveness check is a simple eye blink model. After the eye blink has passed, a fraudster could potentially just swap out a printed face to do the face checking before the selfie is taken. This is obviously an issue. The ideal way is to maybe maintain another model checking the liveness score of the selfie throughout the entire selfie-capture process in other to identify any face swaps in real-time, but this is beyond the scope of this project (or i might try this in the future). This can also be done without the use of a liveness check model. Instead a pixel similarity check can be deployed in the app logic, that checks the average similarity of the frames to be above a certain threshold.

# Motivation
This is the codebase for my project to summarise my learnings and work done during the first half of my internship in GoTo Financial (Gojek) as a KYC Data Scientist, from Sep - Dec '23. I led the liveness team's efforts in developing an eye blink model that was an essential part of our demo app submitted for Level 1 ISO Liveness Face Anti-spoofing application. Throughout this project, I navigated the entire data science lifecycle — from mining data in production to crafting scripts for model training. Data cleaning involved a combination of manual methods and soft labels, where a separate model was trained to make predictions on this noisy data to ‘soft’ label them, if confidence scores exceeded a set threshold. Additionally, during a 1 day Gojek hackathon I developed a Python app that is able to detect face blocking items by training specific face region models. I was pleasantly surprised at being able to develop the necessary models and the Python app all in 1 day and this inspired me to further develop this project, and into an Android app that can be deployed on mobile. 
[Note: The models deployed on the Android app were not trained on images from the Gojek database due to privacy issues. I mined data from separate sources from the web and trained on that.] 
