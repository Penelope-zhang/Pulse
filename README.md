# Pulse

A non-invasive and non-contact web-driven solution that calculates a person's heartbeat and mood through video has many beneficial factors for cross-industries, such as telemedicine. Video conferencing has fast become a primary communication medium of recent; since the Covid-19 outbreak, an uptake of more than a "500% increase in web conferencing software" (Sadler, 2021) has occurred. The trend of video conferencing coupled with the increased computational power of modern computers allows for new and emerging opportunities for data science products and services. This plan focuses on diversifying machine learning opportunities through video analysis of optic flow, colour enhancement and motion magnification with a primary focus on photoplethysmography (RPPG). Segmentation of facial pixels is of the highest interest; we will utilise GrabCut in OpenCV to partition the area from individual video frames for analysis. For heartbeats per minute (BPM), our region of interest (RoI) is centred on the forehead region based on the plethysmographic signal being "strongest on the face" (Bush, 2016). A central objective is obtaining a high heartbeat accuracy with a target of error of 2 ± 1.0 bpm through ensemble learning.

**Primary objectives**

To build a web application that can detect micromotion and colour changes from a frontal face view to calibrate an ML prediction of BPM, using the conjunction of technologies including front-end web engineering technologies HTML5, CSS, and JavaScript connected to either/both a JavaScript/Python server-side driven API.
Producing highly accurate health conditions, activity, stress level, and overall emotional state summarised in statistical insights from the video streams is a core output of data amalgamation analysis. Recent research indicates that human intelligence still exceeds that of AI, "human recognition accuracy of emotions was 72%" (DCU and Dupre, 2021), compared to a lower score ranging from 48% - 62%. We anticipate not to surpass the DCU analysis but, possibly, to provide unique insights in correlation with the heart BPM data.
Analyse the relationship between the heart rate and various activities while producing a consistently high accuracy beats per minute (BPM) extraction.

**Secondary objectives**

Integrate the app into a teleconference application such as Microsoft teams
Assess accuracy and audit the measurement efficiency of heartbeat detection through optic flow, colour enhancement and motion magnification. Accuracy will be measured with ECG wearable technologies such as the Fitbit or Apple watch, and a manual pulse check completed by counting the beats for a thirty-second interval.
Implement a continuous deployment and development (CI/CD) pipeline through Microsoft Azure. 
