# About
I'm making a project about how to train DDPG model. Used 
Solidworks to make my 2 leg robot. Then, let it train from 
`.urdf` to `.proto`. Last, try to let it walk. Almost completed, 
but the biggest problem I encounter is I gave it a bad reward. 
I will use 3D printer to make a real robot if I find a batter reward.
 Then, I will use Raspberry Pi and Ros2 to make a real time control. 

# Getting started

[urdf_to_proto.py](code%2Furdf_to_proto.py) can let your robot into webots, 
but it sometimes has some problem. For example, your robot can't display in webots.
 If you want to fix it, open your `.proto`, and let all `\ ` to `/`. 

[DDPG_pytorch.py](controllers%2Fmain_controller_1%2FDDPG_pytorch.py) is main program.
If you want to run that. First, open webots then launch `Simulation_world.wbt`. 
After that, run [DDPG_pytorch.py](controllers%2Fmain_controller_1%2FDDPG_pytorch.py). 
And you will see it started training. But it still has many bugs. I will optimize it.
