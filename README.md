# FasTRack_Python_implementation

A recreated Python implementation of the FasTrack framework for safe, real-time motion planning and control (part of ECS618 project).

This project includes a recreation of the 5D–3D vehicle model example from the original FasTrack paper, with all constraints and dynamics faithfully implemented.

- Planning: Rapidly-exploring Random Tree (RRT) for global path planning

- Control: Carrot-Chase tracking controller for real-time trajectory tracking

- Reachability: HJ precomputation for computing tracking error bounds; Offline reachability computation module; Seamless integration of planner + tracker + safety constraints

- Simulation: 2D environment with circular obstacles; Start–goal navigation; Real-time visualization




## 📖 Citation

If you use this repository, please cite the original FasTrack paper:

https://doi.org/10.48550/arXiv.2102.07039



@inproceedings{herbert2017fastrack,
  title={FaSTrack: A modular framework for fast and guaranteed safe motion planning},
  author={Herbert, Sylvia L and Chen, Mo and Han, SooJean and Bansal, Somil and Fisac, Jaime F and Tomlin, Claire J},
  booktitle={2017 IEEE 56th Annual Conference on Decision and Control (CDC)},
  pages={1517--1522},
  year={2017},
  organization={IEEE}
}

🎥 Demo

![FasTrack Demo](Fastrack5D3D.gif)
