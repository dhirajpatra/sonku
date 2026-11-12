from app.audio import devices as D

SOURCES = """50\talsa_output.usb-Jabra_EVOLVE_20_SE_MS_00001234-00.analog-stereo.monitor\tPipeWire\ts16le 2ch 48000Hz\tSUSPENDED
51\talsa_input.usb-Jabra_EVOLVE_20_SE_MS_00001234-00.mono-fallback\tPipeWire\ts16le 1ch 16000Hz\tRUNNING
60\tbluez_input.30:63:71:04:38:BB\tPipeWire\ts16le 1ch 16000Hz\tSUSPENDED
"""
SINKS = """49\talsa_output.usb-Jabra_EVOLVE_20_SE_MS_00001234-00.analog-stereo\tPipeWire\ts16le 2ch 48000Hz\tRUNNING
61\tbluez_output.30_63_71_04_38_BB.1\tPipeWire\ts16le 2ch 48000Hz\tSUSPENDED
"""
USB_ONLY_SOURCES = "\n".join(l for l in SOURCES.splitlines() if "bluez" not in l)
USB_ONLY_SINKS = "\n".join(l for l in SINKS.splitlines() if "bluez" not in l)


def fake_pactl(router, sources, sinks, defaults=None):
    calls = []

    def run(*args):
        calls.append(args)
        if args[:3] == ("list", "short", "sources"):
            return sources
        if args[:3] == ("list", "short", "sinks"):
            return sinks
        if args[0].startswith("get-default"):
            return (defaults or {}).get(args[0], "")
        return ""

    router._pactl = run
    return calls


def test_parse_and_classify():
    rows = D.parse_short(SOURCES)
    assert [r["kind"] for r in rows] == ["usb", "bluetooth"]  # the .monitor source is dropped
    assert D.classify("alsa_output.platform-107c701400.hdmi-stereo") == "other"


def test_bluetooth_preferred_by_default_and_made_default(rt):
    r = D.AudioRouter(rt.s)
    calls = fake_pactl(r, SOURCES, SINKS)
    r.scan()
    assert r.mode == "pulse" and r.source == "bluez_input.30:63:71:04:38:BB" and r.sink == "bluez_output.30_63_71_04_38_BB.1"
    assert ("set-default-source", "bluez_input.30:63:71:04:38:BB") in calls
    assert ("set-default-sink", "bluez_output.30_63_71_04_38_BB.1") in calls


def test_priority_can_prefer_usb_and_disconnect_falls_back(rt):
    rt.s.audio_priority = "usb,bluetooth,other"
    r = D.AudioRouter(rt.s)
    fake_pactl(r, SOURCES, SINKS)
    r.scan()
    assert "Jabra" in r.source and "Jabra" in r.sink
    rt.s.audio_priority = "bluetooth,usb,other"
    fake_pactl(r, USB_ONLY_SOURCES, USB_ONLY_SINKS)  # Bluetooth headset switched off
    r.scan()
    assert "Jabra" in r.source and "Jabra" in r.sink


def test_already_default_is_not_reset(rt):
    r = D.AudioRouter(rt.s)
    calls = fake_pactl(r, SOURCES, SINKS, {"get-default-source": "bluez_input.30:63:71:04:38:BB\n"})
    r.scan()
    assert not any(c[0] == "set-default-source" for c in calls)


def test_no_sound_server_falls_back_to_alsa_device(rt, monkeypatch):
    r = D.AudioRouter(rt.s)
    monkeypatch.setattr(D.shutil, "which", lambda name: None)
    r.scan()
    assert r.mode == "alsa"
    assert r.player_cmd("/tmp/x.wav") == ["aplay", "-q", "-D", rt.s.audio_output_device, "/tmp/x.wav"]


def test_pulse_mode_plays_through_default_sink(rt, monkeypatch):
    r = D.AudioRouter(rt.s)
    fake_pactl(r, SOURCES, SINKS)
    monkeypatch.setattr(D.shutil, "which", lambda name: "/usr/bin/" + name)
    assert r.player_cmd("/tmp/x.wav") == ["paplay", "/tmp/x.wav"]


def test_audio_priority_validation(rt):
    assert rt.s.set_value("audio_priority", "USB, bluetooth") == "usb,bluetooth"
    try:
        rt.s.set_value("audio_priority", "wifi")
        raise AssertionError("should be rejected")
    except ValueError:
        pass
