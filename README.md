# selfie-capture-app

## Objective
The motivation behind this app is simple. Companies around the world have a selfie-capture as part of their onboarding process, face verification process, etc. 


## Motivation
This is the codebase for my project to summarise my learnings and work done during the first half of my internship in GoTo Financial (Gojek) as a KYC Data Scientist, from Sep - Dec '23. I led the liveness team's efforts in developing an eye blink model that was an essential part of our demo app submitted for Level 1 ISO Liveness Face Anti-spoofing application. Throughout this project, I navigated the entire data science lifecycle — from mining data in production to crafting scripts for model training. Data cleaning involved a combination of manual methods and soft labels, where a separate model was trained to make predictions on this noisy data to ‘soft’ label them, if confidence scores exceeded a set threshold. Additionally, during a 1 day Gojek hackathon I developed a Python app that is able to detect face blocking items by training specific face region models. I was pleasantly surprised at being able to develop the necessary models and the Python app all in 1 day and this inspired me to further develop this project, and into an Android app that can be deployed on mobile. 
[Note: The models deployed on the Androied app were not trained on images from the Gojek database due to privacy issues. I mined data from separate sources from the web and trained on that.] 
