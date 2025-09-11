// RLBridge.as — TMInterface plugin for RL telemetry + control (TMNF/TMUF)
// Default TCP port: 5555 (localhost). One bidirectional connection.
//
// Telemetry frame (little-endian, per physics tick) — 60 bytes total:
//   uint32 magic = 'TMFR' (0x544D4652)
//   uint32 seq
//   int32  race_time_ms
//   float  pos_x, pos_y, pos_z
//   float  yaw, pitch, roll                   // radians
//   float  vel_x, vel_y, vel_z                // m/s
//   float  speed_kmh
//   int32  cur_checkpoint
//   uint8  on_ground
//   uint8  just_crossed_cp
//   uint8  race_finished
//   uint8  lateral_contact
//
// Action protocol (client -> plugin):
//   [0x01][5 * float]  -> (steer[-1..1], throttle[0..1], brake[0..1], air_pitch[-1..1], air_roll[-1..1])
//   [0x02]             -> Respawn()
//   [0x03]             -> GiveUp()
//   [0x04]             -> Horn()
//
// Controls mapping (compat-friendly):
// - Analog: Gas/Steer set every tick.
// - Brake: uses Down arrow (binary) while ON GROUND (no InputType::Brake dependency).
// - Ground: also press Up arrow when throttle > 0 to satisfy setups that need a key.
// - Air: arrows are used ONLY while airborne for pitch/roll; cleared on ground.

PluginInfo@ GetPluginInfo()
{
    PluginInfo@ info = PluginInfo();
    info.Name        = "RL Bridge";
    info.Author      = "you + ChatGPT";
    info.Version     = "0.2.1";
    info.Description = "Streams car state to an external RL client and accepts actions over TCP.";
    return info;
}

// -------- Console variables (persisted) --------
const string VAR_PORT     = "rl_bridge_port";
const string VAR_TICKDIV  = "rl_bridge_send_every_n_ticks";
const string VAR_VERBOSE  = "rl_bridge_verbose";

void Main()
{
    RegisterVariable(VAR_PORT,    5555.0);    // uint16
    RegisterVariable(VAR_TICKDIV, 1.0);       // send every Nth physics tick
    RegisterVariable(VAR_VERBOSE, false);     // logs
    StartListening();
}

void OnDisabled() { CloseAll(); }

// -------- Networking state --------
Net::Socket@ gListen = null;
Net::Socket@ gClient = null;

uint gSeq = 0;
int  gLastCp = -1;
int  gTick = 0;

void StartListening()
{
    CloseAll();
    @gListen = Net::Socket();

    uint16 port = uint16(GetVariableDouble(VAR_PORT));
    if (!gListen.Listen("127.0.0.1", port)) {
        log("[RLBridge] Failed to listen on 127.0.0.1:" + port, Severity::Error);
        return;
    }
    log("[RLBridge] Listening on 127.0.0.1:" + port, Severity::Success);
}

void AcceptIfPending()
{
    if (gClient is null && gListen !is null) {
        Net::Socket@ c = gListen.Accept(0); // non-blocking
        if (c !is null) {
            @gClient = c;
            gClient.set_NoDelay(true);
            log("[RLBridge] Client connected from " + gClient.get_RemoteIP(), Severity::Success);
            gSeq = 0;
            gLastCp = -1;
        }
    }
}

void CloseAll()
{
    @gClient = null;
    @gListen = null;
}

void Render() { AcceptIfPending(); }

// -------- Helpers --------
bool HasAnyWheelContact(SimulationManager@ sim)
{
    auto@ w = sim.Wheels;
    return  w.FrontLeft.RTState.HasGroundContact
         || w.FrontRight.RTState.HasGroundContact
         || w.BackLeft.RTState.HasGroundContact
         || w.BackRight.RTState.HasGroundContact;
}

// -------- Per-physics-tick --------
void OnRunStep(SimulationManager@ sim)
{
    AcceptIfPending();
    if (gClient is null) return;

    // Throttle send rate
    int tickDiv = Math::Max(1, int(GetVariableDouble(VAR_TICKDIV)));
    gTick = (gTick + 1) % tickDiv;

    // Always try to read actions (non-blocking)
    ReadAndApplyActions(sim);

    if (!sim.InRace) return;
    if (gTick != 0)  return;

    // Build telemetry
    vec3 pos = sim.Dyna.CurrentState.Location.Position;
    vec3 vel = sim.Dyna.CurrentState.LinearSpeed;

    float yaw, pitch, roll;
    sim.Dyna.CurrentState.Location.Rotation.GetYawPitchRoll(yaw, pitch, roll);

    auto@ p   = sim.PlayerInfo;
    int   tms = sim.RaceTime;              // ms
    int   cp  = int(p.CurCheckpoint);
    bool  fin = p.RaceFinished;

    bool onGround = HasAnyWheelContact(sim);
    bool lateral  = sim.SceneVehicleCar.HasAnyLateralContact;
    bool justCp   = (cp != gLastCp); gLastCp = cp;

    float speedKmh = float(p.DisplaySpeed);

    if (!WriteTelemetryFrame(tms, pos, yaw, pitch, roll, vel, speedKmh, cp, onGround, justCp, fin, lateral)) {
        log("[RLBridge] Telemetry write failed; disconnecting client.", Severity::Warning);
        @gClient = null;
        return;
    }
}

// -------- Telemetry write (binary) --------
bool WriteTelemetryFrame(int raceTime, const vec3 &in pos,
                         float yaw, float pitch, float roll,
                         const vec3 &in vel, float speedKmh,
                         int cp, bool onGround, bool justCp, bool finished, bool lateral)
{
    if (gClient is null) return false;

    bool ok = true;
    ok = ok && gClient.Write(uint(0x544D4652));   // 'TMFR'
    ok = ok && gClient.Write(uint(gSeq++));
    ok = ok && gClient.Write(int(raceTime));

    ok = ok && gClient.Write(pos.x) && gClient.Write(pos.y) && gClient.Write(pos.z);
    ok = ok && gClient.Write(yaw)   && gClient.Write(pitch) && gClient.Write(roll);
    ok = ok && gClient.Write(vel.x) && gClient.Write(vel.y) && gClient.Write(vel.z);

    ok = ok && gClient.Write(speedKmh);
    ok = ok && gClient.Write(int(cp));

    ok = ok && gClient.Write(uint8(onGround ? 1 : 0));
    ok = ok && gClient.Write(uint8(justCp   ? 1 : 0));
    ok = ok && gClient.Write(uint8(finished ? 1 : 0));
    ok = ok && gClient.Write(uint8(lateral  ? 1 : 0));
    return ok;
}

// -------- Action handling --------
void ReadAndApplyActions(SimulationManager@ sim)
{
    if (gClient is null) return;

    while (gClient.get_Available() >= 1) {
        uint8 cmd = gClient.ReadUint8();

        if (cmd == 0x01) {
            if (gClient.get_Available() < 5 * 4) break; // wait for full payload

            float steer    = Math::Clamp(gClient.ReadFloat(),  -1.0f, 1.0f);
            float throttle = Math::Clamp(gClient.ReadFloat(),   0.0f, 1.0f);
            float brake    = Math::Clamp(gClient.ReadFloat(),   0.0f, 1.0f);
            float airPitch = Math::Clamp(gClient.ReadFloat(),  -1.0f, 1.0f);
            float airRoll  = Math::Clamp(gClient.ReadFloat(),  -1.0f, 1.0f);

            int steerVal = int(Math::Round(steer * 65536.0f));
            int gasVal   = int(Math::Round(throttle * 65536.0f));
            bool onGround = HasAnyWheelContact(sim);

            // Set analog every tick
            sim.SetInputState(InputType::Steer, steerVal); // [-65536..65536]
            sim.SetInputState(InputType::Gas,   gasVal);   //   0..+65536

            // Ground vs Air usage of arrow keys
            if (onGround) {
                // Binary accelerate assist
                sim.SetInputState(InputType::Up, throttle > 0.05f ? 1 : 0);
                // Brake on ground: Down arrow
                sim.SetInputState(InputType::Down, brake >= 0.5f ? 1 : 0);

                // Clear air-control arrows that could interfere
                sim.SetInputState(InputType::Left,  0);
                sim.SetInputState(InputType::Right, 0);
            } else {
                // Air control via arrows (no ground brake/assist here)
                sim.SetInputState(InputType::Up,    (airPitch >  0.2f) ? 1 : 0);
                sim.SetInputState(InputType::Down,  (airPitch < -0.2f) ? 1 : 0);
                sim.SetInputState(InputType::Right, (airRoll  >  0.2f) ? 1 : 0);
                sim.SetInputState(InputType::Left,  (airRoll  < -0.2f) ? 1 : 0);
            }

        } else if (cmd == 0x02) {
            sim.Respawn();
        } else if (cmd == 0x03) {
            sim.GiveUp();
        } else if (cmd == 0x04) {
            sim.Horn();
        } else {
            // Unknown command: stop to avoid desync
            break;
        }
    }
}
