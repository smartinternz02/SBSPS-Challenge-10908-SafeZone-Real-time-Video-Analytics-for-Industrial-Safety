# SBSPS-Challenge-10908-SafeZone-Real-time-Video-Analytics-for-Industrial-Safety
SafeZone: Real-time Video Analytics for Industrial Safety
# Hand Safety System

Welcome to the Hand Safety System project repository! This project aims to prevent hand and finger injuries in industrial settings by using computer vision and deep learning techniques to detect and alert workers when their hands are too close to hazardous machinery.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
- [Theoretical Analysis](#theoretical-analysis)
- [Experimental Investigations](#experimental-investigations)
- [Results](#results)
- [Advantages & Disadvantages](#advantages--disadvantages)
- [Applications](#applications)
- [Conclusion](#conclusion)
- [Future Scope](#future-scope)


## Introduction

The Hand Safety System is designed to address the prevalent issue of hand and finger injuries in industries, where workers interact with machinery that poses potential risks. Traditional safety measures often lack real-time monitoring capabilities and fail to prevent accidents effectively. This project proposes a novel solution by employing computer vision and deep learning technologies to detect and prevent such injuries.

## Features

- Real-time hand detection using the Mediapipe hand landmark model.
- Automatic adjustment of safety boundaries using circular object detection.
- Visual and auditory alarms to alert workers when their hands breach safety boundaries.
- Flexibility to use different cameras, including remote cameras through URL streaming.

## Getting Started

1. Clone this repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the main application script: `python hand_safety_system.py`.

## Theoretical Analysis

The Hand Safety System combines computer vision and deep learning techniques to enhance industrial safety. By leveraging the capabilities of the Mediapipe hand landmark model and circular object detection, the system accurately detects hands and adapts safety boundaries dynamically.

## Experimental Investigations

During the development of the Hand Safety System, several phases were undertaken to refine the solution:

1. **CNN Training**: Initial attempts to use a custom dataset and CNN for hand detection faced challenges due to limited data. This led to exploring more efficient alternatives.

2. **Model Selection**: After experimenting with various deep learning models, the Mediapipe hand landmark model was chosen for its high accuracy and real-time performance.

3. **Circular Object Detection**: Implementing circular object detection facilitated the automatic determination of safety boundaries, eliminating the need for fixed manual settings.

## Results

The Hand Safety System successfully detects hands in real-time using the Mediapipe model. It automatically adjusts safety boundaries based on circular object detection, providing an efficient and accurate safety solution. Visual and auditory alarms effectively alert workers when safety boundaries are breached.

## Advantages & Disadvantages

Advantages:
- Real-time hand detection.
- Automatic safety boundary adjustment.
- Compatible with various cameras.
- Immediate alerts for safety breaches.

Disadvantages:
- Dependency on camera quality and lighting conditions.
- Limited to visible areas within camera range.

## Applications

The Hand Safety System has the potential to revolutionize safety measures in industries such as manufacturing, textiles, and more, where hand and finger injuries are prevalent.

## Conclusion

The Hand Safety System demonstrates the effective utilization of computer vision and deep learning to enhance industrial safety. By dynamically adapting safety boundaries, the system mitigates the risk of hand and finger injuries associated with machinery interactions.

## Future Scope

Future enhancements could include:
- Multi-camera support for broader coverage.
- Integration with wearable devices for real-time worker tracking.
- Fine-tuning of safety boundary parameters based on user feedback.
