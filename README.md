# PA-EVIO: Polarity-aided Event-Visual-Inertial Odometry


## Introduction
Event cameras have received extensive research interest because of their advantages over conventional cameras in high-speed motion and high-dynamic-range (HDR) environments. **PA-EVIO** is a polarity-aided event-visual-inertial odometry system that leverages these advantages by integrating an adaptive time-surface generation module, a robust feature processing module, and a system state estimator. The system is designed to provide accurate and robust state estimation in challenging environments. 

### Related Work: 
This project is an implementation related to our research works:
1. **(T-IM 2026, Coming soon)** PA-EVIO: Polarity-aided Event-Visual-Inertial Odometry with Adaptive Event Representation 
2. **(IROS 2024)** [Monocular Event-Inertial Odometry with Adaptive decay-based Time Surface and Polarity-aware Tracking](https://ieeexplore.ieee.org/abstract/document/10802605)

## Features
- **Polarity-aided Tracking**: Effectively utilizes event polarity to enhance feature tracking robustness.
- **Adaptive Time Surface**: Employs an adaptive decay strategy for optimal event representation.
- **Robust State Estimation**: Supports optimization of event, visual, and inertial measurements in V-I, E-I, and E-V-I configurations.

## Installation

1. **Create a workspace (if not exists):**
   ```bash
   mkdir -p ~/pa_evio_ws/src
   cd ~/pa_evio_ws/src
   ```

2. **Clone the repository:**
   ```bash
   git clone https://github.com/APRIL-ZJU/PA-EVIO.git
   ```

3. **Install Dependencies:**
   ```bash
   sudo apt update
   sudo apt install python3-vcstool
   vcs import < dependencies.yaml
   ```

4. **Build the project:**
   ```bash
   cd ~/pa_evio_ws
   catkin_make
   source devel/setup.bash
   ```

## Usage

### 1. Configuration
Configuration files are organized as follows:
- **System Definitions**: Located in `msckf-evio/config/` and `msckf-evio/ov_msckf/launch/`. Please verify that topic names and calibration parameters correspond to your sensor setup.
- **Time Surface Parameters**: Located in `ts-ros/cfg`.

### 2. Quick Run
```bash
./scripts/1run_script_davis240c.sh
```



## Citation

```bib
@inproceedings{meio2024tang,
  author={Tang, Kai and Lang, Xiaolei and Ma, Yukai and Huang, Yuehao and Li, Laijian and Liu, Yong and Lv, Jiajun},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Monocular Event-Inertial Odometry with Adaptive decay-based Time Surface and Polarity-aware Tracking}, 
  year={2024},
  volume={},
  number={},
  pages={12544-12551},
  keywords={Power demand;Tracking;Dynamics;Cameras;Feature extraction;Robustness;Surface texture;High dynamic range;Odometry;Intelligent robots},
  doi={10.1109/IROS58592.2024.10802605}}
```

## Acknowledgement
The implementation of this project is based on the [OpenVINS](https://docs.openvins.com/) framework. We also thank everyone who has helped with this work.

## Contact
Thank you for your attention and support regarding this project. If you have any questions, please contact us via the following email: `kaitang [at] zju [dot] edu [dot] cn`
