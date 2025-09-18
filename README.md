# TrackManiac
Project for building RL Model trained on Track Mania Nations Forever

## Python Env:
-gymnasium
--stable_baselines3
## My Setup:
-TMNF via Steam (with latest Steam patch installed)
-TMInterface (https://donadigo.com/tminterface/)
-The AngelScripts (.as) in this proj are loaded into TMInterface
-The TMInterface scripts directory can be located by launching TMInterface, selecting settings in the consonle window (ctrl + `), clicking plugins, and then opening plugins location
-If you have trouble running .as, create a plugin through the TMInterface plugins GUI. Copy the code over and rename the Plugin.

## Starting RL Bridge
-In TMInterface, enable the plugin. It will listen on 127.0.0.1:5555 (change via console var rl_bridge_port).

-From your RL process, open one TCP connection to 127.0.0.1:5555.
    -Read fixed-size telemetry frames each tick, parse fields in the order shown in the comment.
    -Send control packets at any rate using the action protocol above.

-If you want to throttle telemetry rate, set rl_bridge_send_every_n_ticks (e.g., 2 ≈ half rate).

## Making first model
Quick runbook

1. Record your laps (you drive):
    python tmnf_record_telemetry.py --out laps_run1.npz --seconds 900
    --You can run multiple times on different maps; collect several .npz files
2. Derive Behavior Cloning (BC) Dataset
    python tmnf_derive_bc_dataset.py --out bc_dataset_run1.npz laps_run1.npz  
    --add more files as args if you have them
3. Train the BC Policy
    python train_bc.py --data bc_dataset_run1.npz --epochs 60 --out bc_policy.pt
4. RL fine-tine (one env)
    python ppo_finetune_from_bc.py --bc_data bc_dataset_run1.npz --warmup_epochs 5 --timesteps 500000


