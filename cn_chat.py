#!/usr/bin/env python
import collections
import sys
import signal
import threading
import pyaudio
from array import array
import time
import re
import ssl
import os
import json
import requests
from vad import VoiceActivityDetection
import subprocess
from http import HTTPStatus
from multiprocessing import Process, Manager
from random import randint
import dashscope
import rospy
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from dashscope import Generation, TextEmbedding, MultiModalConversation
from dashscope.api_entities.dashscope_response import Role
from dashscope.audio.tts import SpeechSynthesizer
from dashvector import Client
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from std_msgs.msg import Int32
from tts_cloud import Callback
import numpy as np
import torch

########################################################################
# initialization
########################################################################
# connect to dashscope
ssl._create_default_https_context = ssl._create_unverified_context
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')
print(dashscope.api_key)
callback = Callback()
# initialize speech recognition model with hot words
with open('hotword.txt', 'r', encoding='utf-8') as f:
    hotword = f.readline()
paraformer = pipeline(task=Tasks.auto_speech_recognition,
                      model='./iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
                      model_revision="v2.0.4")
# start dialog process
dialog_proc = None
# initialize face recognition model
retina = pipeline("face_recognition", model='./model/cv_retinafce_recognition', model_revision='v2.0.2')
# load the system prompts
with open('prompt.txt', 'r', encoding='utf-8') as f:
    sys_prompt = f.read()
with open('vl_prompt.txt', 'r', encoding='utf-8') as f:
    vl_sys_prompt = f.read()
# initialize the memory
memory = Manager().dict()
memory['messages'] = [{'role': Role.SYSTEM, 'content': sys_prompt}]  # LLM's memory
memory['vl'] = [{'role': Role.SYSTEM, 'content': [{'text': vl_sys_prompt}]}]  # VL's memory
memory['arm_control.py'] = []  # arm's memory
memory['move_ros.py'] = []  # wheel's memory
memory['position'] = 601
# initialize VAD
vad = VoiceActivityDetection()
# done with initialization play start voice
subprocess.run(["aplay", "recorded/start.wav"])


########################################################################
# recording audio
########################################################################
# use vad to record audio
def int2float(sound):
    # Find the absolute maximum value in the input sound array.
    abs_max = np.abs(sound).max()
    # Convert the sound array to 'float32' data type.
    sound = sound.astype('float32')
    # If the absolute maximum value is greater than 0, normalize the sound by dividing by 32768.
    if abs_max > 0:
        sound *= 1 / 32768
    # Squeeze the sound array if necessary (depends on the use case).
    return sound.squeeze()


# record audio
def listen():
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK_SIZE = 100  # ms
    PADDING_DURATION_MS = 400  # 400 ms for judgement
    RAW_CHUNK = int(RATE * CHUNK_SIZE / 1000)  # Chunk size to read
    NUM_WINDOW_CHUNKS = int(PADDING_DURATION_MS / CHUNK_SIZE)
    NUM_WINDOW_CHUNKS_END = NUM_WINDOW_CHUNKS * 2
    pa = pyaudio.PyAudio()
    stream = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, start=False, frames_per_buffer=RAW_CHUNK)
    got_a_sentence = False
    leave = False

    def handle_int(sig, chunk):
        global leave, got_a_sentence
        leave = True
        got_a_sentence = True

    signal.signal(signal.SIGINT, handle_int)

    while not leave:
        ring_buffer = collections.deque(maxlen=NUM_WINDOW_CHUNKS)
        triggered = False
        voiced_frames = []
        ring_buffer_flags = [0] * NUM_WINDOW_CHUNKS
        ring_buffer_index = 0

        ring_buffer_flags_end = [0] * NUM_WINDOW_CHUNKS_END
        ring_buffer_index_end = 0

        raw_data = array('h')
        index = 0
        StartTime = time.time()
        print("* recording: ")
        stream.start_stream()

        while not got_a_sentence and not leave:
            chunk = stream.read(RAW_CHUNK)
            raw_data.extend(array('h', chunk))
            index += RAW_CHUNK
            TimeUse = time.time() - StartTime

            frame_np = np.frombuffer(chunk, dtype=np.int16)
            frame_float = int2float(frame_np)

            # Use your VAD module to perform VAD and detect voice activity
            res = vad(torch.from_numpy(frame_float), sr=RATE).item()

            active = res > 0.85
            sys.stdout.write('1' if active else '_')
            ring_buffer_flags[ring_buffer_index] = 1 if active else 0
            ring_buffer_index += 1
            ring_buffer_index %= NUM_WINDOW_CHUNKS

            ring_buffer_flags_end[ring_buffer_index_end] = 1 if active else 0
            ring_buffer_index_end += 1
            ring_buffer_index_end %= NUM_WINDOW_CHUNKS_END

            # Start point detection
            if not triggered:
                if TimeUse > 20:
                    print('end of listening')
                    return False
                ring_buffer.append(chunk)
                num_voiced = sum(ring_buffer_flags)
                if num_voiced > 0.8 * NUM_WINDOW_CHUNKS:
                    sys.stdout.write(' Open ')
                    triggered = True
                    voiced_frames.extend(ring_buffer)
                    ring_buffer.clear()
            # End point detection
            else:
                voiced_frames.append(chunk)
                ring_buffer.append(chunk)
                num_unvoiced = NUM_WINDOW_CHUNKS_END - sum(ring_buffer_flags_end)
                if num_unvoiced > 0.90 * NUM_WINDOW_CHUNKS_END or TimeUse > 60:
                    sys.stdout.write(' Close ')
                    triggered = False
                    got_a_sentence = True

            sys.stdout.flush()

        sys.stdout.write('\n')
        if not voiced_frames:
            return False
        data = b''.join(voiced_frames)

        stream.stop_stream()
        print("* done recording")
        got_a_sentence = False
        leave = True
        subprocess.run(["aplay", "recorded/think.wav"])

    stream.close()
    return data


########################################################################
# function call
########################################################################
# face recognition
def face():
    # take a photo
    subprocess.run(['python', 'camera.py', 'jpg'])
    subprocess.run(["aplay", "recorded/snap.wav"])
    result = False
    try:
        # generate embedding for face detected
        embedding = retina(dict(user='../shoushi_detect/image/color_image.jpg'))[OutputKeys.IMG_EMBEDDING].tolist()[0]
        # search the embedding in vector database
        result = search_relevant_doc(embedding, 'photo', 1)
        # if the face is found, chat with this person with personalized topics
        if result:
            speak(f'你好，{result[4:].split("; 任职部门")[0]}')
            result = f'请根据我的岗位信息：{result}，找一个话题和我聊天。'
        else:
            subprocess.run(["aplay", "recorded/no_result.wav"])
    except:
        subprocess.run(["aplay", "recorded/no_face.wav"])
    return result


# search Hangzhou's weather
def weather():
    url = "http://www.nmc.cn/rest/weather?stationid=58457&_=1709888356915"
    # 请求响应
    reponse = requests.get(url)
    # 转换数据格式
    reponse_json = json.loads(reponse.text)
    # 获取需要的数据定位
    data = reponse_json["data"]["real"]
    weather_now = data["weather"]
    wind = data["wind"]
    result = f"杭州当前天气：{weather_now['info']}，气温：{int(weather_now['temperature'])}度，湿度：{int(weather_now['humidity'])}。{wind['direct']}：{wind['power']}"
    return result


########################################################################
# RAG
########################################################################
# generate text embedding
def generate_embeddings(doc):
    rsp = TextEmbedding.call(model=TextEmbedding.Models.text_embedding_v2, input=doc)
    embeddings = [record['embedding'] for record in rsp.output['embeddings']]
    return embeddings if isinstance(doc, list) else embeddings[0]


# search for relevant text in the vector database
def search_relevant_doc(question, collection_name, topk):
    client = Client(api_key=os.getenv('DASHVECTOR_API_KEY'), endpoint=os.getenv('DASHVECTOR_ENDPOINT'))
    collection = client.get(collection_name)
    rsp = collection.query(question, output_fields=['raw'], topk=topk)
    # for face recognition, select the closest result
    if topk == 1:
        result = rsp.output[0]
        raw = rsp.output[0].fields['raw']
        score = result.score
        print(score)
        if score > 1.1:
            print(raw)
            return False
        else:
            return raw
    # for RAG, select the closest k results
    else:
        result = [raw.fields['raw'] for raw in rsp.output]
        return ';'.join(result)


# RAG
def rag(question, collection_name, topk):
    # search relevant document in dashvector collection
    embedding = generate_embeddings(question)
    context = search_relevant_doc(embedding, collection_name, topk)
    print(context)
    prompt = f'请基于```内的内容回答问题。```{context}```我的问题是：{question}。'
    return prompt


########################################################################
# TTS
########################################################################
# tts for Mandarin
def speak(text):
    SpeechSynthesizer.call(model='sambert-zhistella-v1', text=text, sample_rate=16000, format='pcm', callback=callback)


# tts for streaming the reply
def stream_tts(responses, prompt):
    full_content = ''  # with incrementally we need to merge output.
    curr = ''
    for response in responses:
        if response.status_code == HTTPStatus.OK:
            chunk = response.output.choices[0]['message']['content']
            sep = re.split('[，：；。！？;!?\n]', chunk)
            curr += sep[0]
            if len(sep) > 1:
                execute_task_async(curr, prompt)
                if len(sep) > 2:
                    for phrase in sep[1:-1]:
                        if phrase != '':
                            execute_task_async(phrase, prompt)
                curr = sep[-1]
            full_content += chunk
        else:
            print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))
            subprocess.run(["aplay", "recorded/no_response.wav"])
    # speak the last sentence
    if curr.strip() != '':
        execute_task_async(curr, prompt)
    print('Full response:\n' + full_content)
    return full_content


########################################################################
# control movement
########################################################################
# turn left or right
def turn(prompt):
    if '前' in prompt or '后' in prompt:
        # If the command contains '前' (forward) or '后' (backward), turn left 180 degrees.
        movement_queue('move_ros.py', 'left_180_degree')
    else:
        # set the direction for left and right
        if '右' in prompt:
            direction = 'right_'
        elif '左' in prompt:
            direction = 'left_'
        else:
            # If none of the specified directions are found, play a sound to indicate no movement.
            subprocess.run(["aplay", "recorded/no_movement.wav"])

        if direction:
            # If a valid direction is set, attempt to extract the numeric value
            number = re.findall(r'[\d.]+', prompt)
            if len(number) == 0:
                # If no numeric value is found, turn the robot 90 degrees in the specified direction.
                argument = f'{direction}90_degree'
                movement_queue('move_ros.py', argument)
            elif len(number) == 1:
                # Execute a system command to turn the robot using 'move_ros.py' with the specified direction and numeric angle.
                argument = f'{direction}{str(number[0])}_degree'
                movement_queue('move_ros.py', argument)
            else:
                subprocess.run(["aplay", "recorded/no_movement.wav"])
        else:
            subprocess.run(["aplay", "recorded/no_movement.wav"])


# move forward or backward
def march(prompt):
    # set the direction
    if '前' in prompt:
        direction = 'forward_'
    elif '后' in prompt:
        direction = 'backward_'
    elif '左' in prompt:
        movement_queue('move_ros.py', 'left_90_degree')
        time.sleep(7)
        direction = 'forward_'
    elif '右' in prompt:
        movement_queue('move_ros.py', 'right_90_degree')
        time.sleep(7)
        direction = 'forward_'
    else:
        # If none of the specified directions are found, play a sound to indicate no movement.
        subprocess.run(["aplay", "recorded/no_movement.wav"])

    if direction:
        number = re.findall(r'[\d.]+', prompt)
        if len(number) == 0:
            # If no numeric value is found, move the robot 0.3 meters in the specified direction.
            argument = f'{direction}0.3_m'
            movement_queue('move_ros.py', argument)
        elif len(number) > 1:
            # If multiple numeric values are found, play a sound to indicate an issue.
            subprocess.run(["aplay", "recorded/no_movement.wav"])
        else:
            # Execute a system command to move the robot using 'move_ros.py' with the specified direction and distance.
            argument = f'{direction}{str(number[0])}_m'
            movement_queue('move_ros.py', argument)


# Use vision language model to locate the item
def locate(item, round):
    # take a photo
    subprocess.run(['python', 'camera.py', 'png'])
    subprocess.run(["aplay", "recorded/snap.wav"])
    msg = [{'role': Role.SYSTEM, 'content': [{
        'text': "你的任务是做目标检测，每次我将输入一个需要你识别的物体，请你返回box框和坐标。如果你不能找到该物体，请直接回复：《《未找到》》"}]},
           {'role': Role.USER, 'content': [{'image': 'file:///home/robot/shoushi_detect/image/color_image.png'},
                                           {'text': f"请在图中框出{item}"}]}]
    response = MultiModalConversation.call(model='qwen-vl-plus', messages=msg)
    print(response)
    try:
        input_str = f"<root>{response.output.choices[0]['message']['content'][0]['box']}</root>"
        # Parsing the string into an XML object
        root = ET.fromstring(input_str)
        # Finding the 'box' element and extracting its text content
        box_content = root.find('box').text
        coordinates = box_content.replace('(', '').replace(')', '').split(',')
        x1, y1, x2, y2 = map(int, coordinates)
        x = int((x1 + x2) * 640 / 2000)
        y = int((y1 + y2) * 480 / 2000)
        argument = f'locate:{x}:{y}'
        subprocess.run(['python', 'camera.py', argument])
    except:
        if round > 5:
            subprocess.run(["aplay", "recorded/not_found.wav"])
        else:
            subprocess.run(["aplay", "recorded/not_found_turn.wav"])
            movement_queue('move_ros.py', 'left_45_degree')
            time.sleep(1)
            locate(item, round + 1)


# save the movement in the memory and execute the task
def movement_queue(topic, code):
    queue = memory[topic]
    threshold = timedelta(seconds=0)
    if len(queue) > 0:
        task, start_time = queue[-1]
        print(f'last task{task}, executed at {start_time}')
        # arm
        if topic == 'arm_control.py':
            # taichi
            if task == '3':
                threshold = timedelta(seconds=68)
            # lift arm
            if task == '2':
                threshold = timedelta(seconds=3)
        # wheel
        elif topic == 'move_ros.py':
            code_list = task.split('_')
            move = task[2]
            amount = task[1]
            # turn
            if move.endswith('degree'):
                t = int(amount / 10)
                threshold = timedelta(seconds=t)
            # march
            if move.endswith('m'):
                t = int(amount / 0.1)
                threshold = timedelta(seconds=t)
        # check whether the last movement is done
        now = datetime.now()
        end_time = start_time + threshold
        # if the last task did not end, wait
        if end_time > now:
            wait_time = (now - end_time).seconds + 1
            print(f'wait for {wait_time}s')
            time.sleep(wait_time)
    # add to memory
    memory[topic].append((code, datetime.now()))
    # execute the task
    subprocess.run(['python', topic, code])
    if topic == 'arm_control.py' and code == '3':
        subprocess.Popen(['aplay', 'recorded/taiji.wav'])


########################################################################
# dialog process
########################################################################
# chat with LLM
def chat(prompt, personnel=False, electricity=False):
    web = True
    temperature = 0.6
    msg = memory['messages']
    if personnel:
        web = False
        temperature = 0.3
        # RAG for personnel search
        msg[0] = {'role': Role.SYSTEM, 'content': '你是小昊，现在你处于人员模式，你的功能是通过文本资料查找相关人员信息。'}
        prompt = rag(prompt, 'employee', 3)
    if electricity:
        web = True
        temperature = 0.3
        # RAG for electricity search
        msg[0] = {'role': Role.SYSTEM,
                  'content': '你是小昊，你是一位电力行业的专家。你可以上网搜索，并结合文本资料，回答专业的电力行业问题。'}
        prompt = rag(prompt, 'electricity', 1)
    msg.append({'role': Role.USER, 'content': prompt})
    # if the length of message exceeds 3k, pop the oldest round
    msg = check_len(msg)
    # Use language generation model to generate a response
    responses = Generation.call(
        Generation.Models.qwen_turbo,
        messages=msg,
        seed=randint(1, 100),
        enable_search=web,
        temperature=temperature,
        result_format='message',
        stream=True,
        incremental_output=True
    )
    # Process the generated response
    full_content = stream_tts(responses, prompt)
    # load the reply to the message
    msg.append({'role': Role.ASSISTANT, 'content': full_content})
    if personnel or electricity:
        msg[0] = {'role': Role.SYSTEM, 'content': sys_prompt}
    memory['messages'] = msg


# check if the length of messages exceeds 5k
def check_len(msg):
    while len(str(msg)) > 5000:
        msg.pop(1)
        msg.pop(1)
    return msg


# search if the prompt contains keywords
def contains_keywords(prompt, keywords):
    """
    Check if the prompt contains at least one keyword from each list of keywords.

    :param prompt: The string to search within.
    :param keywords: A list of lists, where each inner list contains interchangeable keywords.
    :return: True if at least one keyword from each list is found in the prompt, False otherwise.
    """
    # Iterate over each list of keywords
    for keyword_group in keywords:
        # Check if any keyword in the current group is in the prompt
        if not any(keyword in prompt for keyword in keyword_group):
            return False
    return True


# chat with vision language model
def vl_chat(prompt):
    # take a photo
    subprocess.run(['python', 'camera.py', 'png'])
    subprocess.run(["aplay", "recorded/snap.wav"])
    # load the prompt to the message
    msg = memory['vl']
    text_msg = memory['messages']
    text_msg.append({'role': Role.USER, 'content': prompt})
    msg.append({'role': Role.USER, 'content': [{'image': 'file:///home/robot/shoushi_detect/image/color_image.png'},
                                               {'text': prompt}]})
    # get the reply from tongyi vl
    responses = MultiModalConversation.call(model='qwen-vl-plus', messages=msg, stream=True, incremental_output=True)
    full_content = ''  # with incrementally we need to merge output.
    curr = ''
    for response in responses:
        if response.status_code == HTTPStatus.OK:
            try:
                chunk = response.output.choices[0]['message']['content'][0]['text']
                sep = re.split('[，：；。！？;!?\n]', chunk)
                curr += sep[0]
                if len(sep) > 1:
                    execute_task_async(curr, prompt)
                    if len(sep) > 2:
                        for phrase in sep[1:-1]:
                            if phrase != '':
                                execute_task_async(phrase, prompt)
                    curr = sep[-1]
                full_content += chunk
            except:
                pass
        else:
            print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))
            subprocess.run(["aplay", "recorded/no_response.wav"])
    # speak the last sentence
    if curr.strip() != '':
        execute_task_async(curr, prompt)
    print('Full response:\n' + full_content)
    # load the reply to the message
    msg.append({'role': Role.ASSISTANT, 'content': [{'text': full_content}]})
    # load the reply to text message
    text_msg.append({'role': Role.ASSISTANT, 'content': full_content})
    # check if the length of text messages exceeds 5k
    text_msg = check_len(text_msg)
    msg = check_len(msg)
    memory['vl'] = msg
    memory['messages'] = text_msg


# move to the point and introduce
def map_point(num):
    # update memory of position
    memory['position'] = num
    # send the message to navigation system
    subprocess.run(['python', 'navigation.py', str(num)])


# execute tasks in a thread
def execute_task_async(content, prompt):
    print(content)
    speak(content)
    # Create a new thread for the task
    task_thread = threading.Thread(target=task, args=(content, prompt))
    task_thread.start()
    task_thread.join()


# execute the task in the reply
def task(text, raw_prompt):
    prompts = text.split('《《')
    # execute all task in prompts
    if len(prompts) > 1:
        prompts = prompts[1:]
        for prompt in prompts:
            # play rock paper scissor
            if '猜拳||开始' in prompt:
                movement_queue('arm_control.py', '2')
            # lift right arm
            if '右臂||抬起' in prompt:
                movement_queue('arm_control.py', '2')
            # play music
            if '音乐||开始' in prompt:
                # stop all sound
                subprocess.run(['pkill', '-9', 'aplay'])
                # play music
                subprocess.run(['aplay', 'recorded/taiji.wav'])
            # play taichi
            if '太极||开始' in prompt:
                # do taichi
                movement_queue('arm_control.py', '3')
                # play music
            # move
            if '向' in prompt:
                # turn
                if '转' in prompt:
                    turn(prompt)
                # march
                elif contains_keywords(prompt, [['前', '后', '左', '右']]):
                    march(prompt)
            # face recognition
            if '人脸识别||开始' in prompt:
                face_search = face()
                if face_search:
                    chat(prompt=face_search, personnel=False, electricity=False)
                else:
                    subprocess.run(['aplay', 'recorded/no_face.wav'])
            # stop all the task
            if '任务||停止' in prompt:
                # stop arm movement
                subprocess.run(['python', 'arm_control.py', '6'])
                # stop the game
                subprocess.run(['python', 'game.py', '2'])
            if '正在定位' in prompt:
                # move to designated point in the exhibition hall
                target = prompt.split('|')[-1]
                print(target)
                if '原' in target:
                    map_point(601)
                elif '待客' in target:
                    map_point(0)
                elif contains_keywords(target, [['车底', '三六零', '360']]):
                    map_point(6)
                elif contains_keywords(target, [['遂', '轨交', '刚性', '接触', '网检']]):
                    map_point(8)
                elif contains_keywords(target, [['光伏', '清扫', '输煤', '栈桥']]):
                    map_point(12)
                elif '四足' in target:
                    map_point(16)
                elif contains_keywords(target, [['联合', '挂轨', '开关', '操作']]):
                    map_point(18)
                elif contains_keywords(target, [['变电站', '智能巡检']]):
                    map_point(20)
                elif contains_keywords(target, [['变压器', '气体', '油中', '声纹']]):
                    map_point(22)
                elif '除冰' in target:
                    map_point(24)
                elif contains_keywords(target, [['带电', '输电', '微拍', '台区']]):
                    map_point(26)
                elif '极寒' in target:
                    map_point(28)
                elif contains_keywords(target, [['管控', '监理', '监控']]):
                    map_point(30)
                elif '升降' in target:
                    map_point(32)
                elif contains_keywords(target, [['创新', '反馈', '电子皮肤', '无人机']]):
                    map_point(34)
                elif '防爆' in target:
                    map_point(38)
                elif '环保' in target:
                    map_point(40)
                elif contains_keywords(target, [['电梯', '厕所', '洗手间', '卫生间']]):
                    map_point(42)
                elif contains_keywords(target, [['电动楼梯', '扶梯']]):
                    map_point(44)
                else:
                    locate(target, 0)
            # if the task requires vision, use VL to chat
            if '拍照' in prompt or '环视' in prompt:
                vl_chat(raw_prompt)


def dialog() -> None:
    """
    Manage the dialog process
    """
    respond = True  # Initialize response flag

    # Listen to audio input and capture it
    audio = listen()

    if audio:
        # Record the start time of audio processing
        start = time.time()
        # Use a voice recognition model to transcribe audio. 如果是中文的语音识别结果，每个字都是带空格的。如果是英文的话是正常的
        raw_prompt = paraformer(audio, hotword=hotword)[0]['text']
        # Record the end time of audio processing
        end = time.time()

        # Calculate and print the time spent in speech-to-text conversion
        print(f'paraformer spent: {end - start} s')
        print(raw_prompt)

        # 如果是中文，去掉所有空格，仅用于关键词判断
        prompt = raw_prompt.replace(' ', '')

        # Disable RAG
        personnel = False
        electricity = False

        # Check if the input contains stop keywords and handle accordingly
        if contains_keywords(prompt, [['停止', '结束'], ['移动', '表演', '游戏', '任务']]):
            respond = False  # Shut the LLM up
            subprocess.run(['python', 'game.py', '2'])  # Stop a game
            subprocess.run(['python', 'arm_control.py', '6'])  # Stop arm control

        # Activate face recognition
        elif '脸识别' in prompt:
            raw_prompt = face()  # Get face recognition result

        # Activate camera and start visual-language interaction if '拍照' is in the prompt
        elif '拍照' in prompt:
            prompt = prompt.replace('拍照', '')  # Let VL chat without the intention of taking photo
            respond = False  # Shut the LLM up
            if prompt != '':
                vl_chat(prompt)  # chat after taking a photo
            else:
                subprocess.run(["aplay", f"recorded/no_response.wav"])

        # Provide context for weather in Hangzhou
        elif '天气' in prompt:
            raw_prompt = f'请根据以下信息：{weather()}，回答问题：{prompt}'

        # go back to the original position
        elif contains_keywords(prompt, ['回', '原']):
            subprocess.run(['python', 'navigation.py', '601'])
            memory['position'] = 601

        # Turn on personnel mode with RAG
        elif contains_keywords(prompt, [['人员', '员工'], '模式']):
            personnel = True  # Enable RAG for personnel search
            raw_prompt = prompt
            subprocess.run(["aplay", f"recorded/personnel_mode.wav"])  # play the sound of "search for personnel"

        # Turn on personnel mode with RAG
        elif contains_keywords(prompt, [['电力'], '模式']):
            electricity = True  # Enable RAG for personnel search
            raw_prompt = prompt

        # Introduce "Shenhao" immediately, bypassing LLM
        elif contains_keywords(prompt, ['申昊', '介绍']):
            respond = False
            subprocess.run(['aplay', 'recorded/shenhao_intro.wav'])  # play the introduction audio
            # add the introduction text to memory
            memory['messages'].append({'role': Role.USER, 'content': prompt})
            memory['messages'].append({'role': Role.ASSISTANT,
                                       'content': '杭州申昊科技股份有限公司（股票代码：300853）成立于2002年，是一家致力于设备检测及故障诊断的高新技术企业。通过充分利用传感器、机器人、人工智能及大数据分析技术，服务于工业大健康，为工业设备安全运行及智能化运维提供综合解决方案。目前，公司已开发了一系列具有自主知识产权的智能机器人及智能监测控制设备产品，可用于电力电网、轨道交通、油气化工等行业，解决客户的难点与痛点，为客户无人或少人值守和智能化管理提供有效的检测、监测手段。'})

        # Send preprocessed prompt to LLM
        if respond and prompt:
            chat(prompt=raw_prompt, personnel=personnel, electricity=electricity)


# stop the dialog process
def terminate():
    # kill all aplay sound
    subprocess.run(['killall', 'aplay'])
    # kil the dialog process
    if dialog_proc and dialog_proc.is_alive():
        print("Terminating the dialogue process")
        subprocess.run(['kill', '-9', str(dialog_proc.pid)])
        dialog_proc.join()


########################################################################
# ROS subscriber node
########################################################################
def wake_callback(ros_msg) -> None:
    """
    Callback function triggered upon a wake-up signal. Initiates or restarts the dialog process.

    Args:
    - msg (Any): The message containing wake-up signal information.
    """
    # start a new dialog
    global dialog_proc
    if ros_msg.data == 1:
        # stop the dialog
        terminate()
        # Start a new process for dialogue
        dialog_proc = Process(target=dialog)
        # play the greeting sound
        subprocess.run(["aplay", f"recorded/greet_{randint(1, 3)}.wav"])
        # start new conversation
        dialog_proc.start()
        rospy.loginfo("Starting a new dialog process")

    # arrived to the designated point
    if ros_msg.data == 2:
        # stop the dialog
        terminate()
        # load introduction text
        point = memory['position']
        # arrived at the designated point
        print(f'arriving at point {point}')
        # if the point is the waiting point or the home point, stand by
        if point in [0, 601]:
            pass
        # if the point is the elevator, play the introduction audio
        elif point in [42, 44]:
            subprocess.run(["aplay", f"position/{point}.wav"])
        # if the point is in the exhibition area, load the context and play the introduction audio
        else:
            # load the context at the designated point
            msg = memory['messages']
            with open(f'position/{point}.txt', 'r', encoding='utf-8') as f:
                intro = f.read()
            msg.append({'role': Role.USER, 'content': '请你介绍展品'})
            msg.append({'role': Role.ASSISTANT, 'content': intro})
            memory['messages'] = check_len(msg)
            # play introduction audio
            subprocess.run(["aplay", f"position/{point}.wav"])

    # perform taichi
    if ros_msg.data == 4:
        # stop all sound
        terminate()
        # do taichi
        movement_queue('arm_control.py', '3')


def subscriber() -> None:
    """
    Initializes a ROS node for wake-up signal subscription and starts the ROS event loop.
    """
    # Initialize a ROS node with the name 'wake_subscriber'
    rospy.init_node('wake_subscriber')
    # Subscribe to the 'wake' topic, expecting messages of type Int32, and specify the callback function
    rospy.Subscriber('wake', Int32, wake_callback)
    # The spin() function keeps Python from exiting until this ROS node is stopped
    rospy.spin()


if __name__ == '__main__':
    subscriber()
    dialog_proc = Process(target=dialog)
    dialog_proc.start()
