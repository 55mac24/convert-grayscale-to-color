from basicColoringAgent import BasicColoringAgent
from improvedColoringAgent import ImprovedColoringAgent
from improvedColoringAgent_KD import ImprovedColoringAgentKD
from calcuateMSE import CalculateMSE
import time

from definitions import DEFINITIONS

import os

import numpy
from matplotlib import pyplot


def getRunTime():
    end = time.time()
    totalTime = end - start
    totalTime %= 3600
    min = totalTime // 60
    totalTime %= 60
    sec = totalTime
    print("TOTAL TIME FOR RECOLORING: ", end='')
    print("%d min and %d sec" % (min, sec))


training_image_name = "3.jpg"
image_base_dir = "/training-data-imgs/"
image_dir = "IMAGE-" + os.path.splitext(training_image_name)[0] + "-PROCESSED-FILES"
k = 5
console_log = ""  # "consoleLog-centroid-calculations"
createGrayscaleRegardless = True

agent = None

agent_type = DEFINITIONS.IMPROVED_COLORING_AGENT
start = time.time()

if agent_type == DEFINITIONS.IMPROVED_COLORING_AGENT:
    improved_coloring_agent = ImprovedColoringAgent(k, image_base_dir, training_image_name, image_dir,
                                                    createGrayscaleRegardless, console_log, agent_type)
    agent = improved_coloring_agent
    improved_coloring_agent.recolor_image()
    getRunTime()
elif agent_type == DEFINITIONS.IMPROVED_COLORING_AGENT_KD:
    improved_coloring_agent_kd = ImprovedColoringAgentKD(k, image_base_dir, training_image_name, image_dir,
                                                         createGrayscaleRegardless, console_log, agent_type)
    agent = improved_coloring_agent_kd
    improved_coloring_agent_kd.recolor_image()
    getRunTime()
elif agent_type == DEFINITIONS.BASIC_AGAINST_IMPROVED_AGENT:
    basic_coloring_agent = BasicColoringAgent(k, image_base_dir, training_image_name, image_dir,
                                              createGrayscaleRegardless, console_log,
                                              DEFINITIONS.BASIC_COLORING_AGENT)
    agent = basic_coloring_agent
    basic_coloring_agent.recolor_image()
    getRunTime()

    start = time.time()
    improved_coloring_agent = ImprovedColoringAgent(k, image_base_dir, training_image_name, image_dir,
                                                    createGrayscaleRegardless, console_log,
                                                    DEFINITIONS.IMPROVED_COLORING_AGENT)
    agent_i = improved_coloring_agent
    improved_coloring_agent.recolor_image()
    getRunTime()
else:
    basic_coloring_agent = BasicColoringAgent(k, image_base_dir, training_image_name, image_dir,
                                              createGrayscaleRegardless, console_log, agent_type)
    agent = basic_coloring_agent
    basic_coloring_agent.recolor_image()
    getRunTime()

mse = CalculateMSE(agent.img_RGB_values, agent.predicted_img_RGB_values, agent.width, agent.height)
mse.MSE()
print("Left Side MSE: ", mse.left_side_MSE, "Right Side MSE: ", mse.right_side_MSE)
agent.close_image_files()
