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