# Xiaohao: Embodied Intelligence Project

## Project Overview
Welcome to the GitHub repository of Xiaohao's embodied intelligence system. Xiaohao is a humanoid robot designed by Hangzhou Shenhao Technology, serving as a guide in the exhibition hall. This is multi-agent system integrating a large language model and a vision language model at its core, supported by robotic arms, mobility wheels, and a camera. These agents communicate via ROS messages, allowing Xiaohao to respond to user prompts, make decisions, and interact with its environment.<br>

In this repository, you will find the open-sourced code for the large language model and the camera, as well as the ROS publisher nodes responsible for orchestrating communications among the different components.

## Repository Contents
1. **Language Processing Nodes:**
   - **[cn_chat.py](https://github.com/charliezcr/Xiaohao/blob/main/cn_chat.py):** This node operates as a ROS subscriber to the 'wake' topic. Upon activation, it records audio, processes it using VAD (Voice Activity Detection), and converts the audio to text via the [Paraformer](https://modelscope.cn/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)(Please download it yourself from Modelscope by Alibaba, also for other Modelscope model I used in the initialization part) speech recognition model. The text is then processed by the large language model (Qwen) to generate responses, which are converted back to speech by using SAMBERT TTS to interact with users. This node also handles movement commands for the robot.<br>
     - **[vad.py](https://github.com/charliezcr/Xiaohao/blob/main/vad.py):** Voice activity detection function utilized in cn_chat.py.<br>
     - **[silero_vad.onnx](https://github.com/charliezcr/Xiaohao/blob/main/silero_vad.onnx):** Open-source VAD model used throughout the project.<br>
     - **[tts_cloud.py](https://github.com/charliezcr/Xiaohao/blob/main/tts_cloud.py):** Text-to-speech function used in cn_chat.py.<br>
     - **[prompt.txt](https://github.com/charliezcr/Xiaohao/blob/main/prompt.txt):** System prompt for the large language model.<br>
     - **[vl_prompt.txt](https://github.com/charliezcr/Xiaohao/blob/main/vl_prompt.txt):** System prompt for the vision language model.

2. **Camera and Vision Processing:**
   - **[rs_cam.py](https://github.com/charliezcr/Xiaohao/blob/main/rs_cam.py):** This node manages the Intel® RealSense™ Stereo depth camera. It subscribes to the 'camera' topic and, upon receiving commands, captures color and depth images, identifies objects, and communicates with mobility components to navigate towards them.

## Flow Chart

   ![image](https://github.com/charliezcr/Xiaohao/assets/48685281/260d725f-23eb-4b71-b490-533a100ef9d7)

