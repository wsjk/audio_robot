# Wiring Diagrams

Visual guides for connecting hardware components.

## Basic Setup (1 Servo)

```
┌─────────────────────────────────────────────────────────┐
│                   Raspberry Pi 3B                        │
│                                                          │
│   ┌────────────────────────────────────────┐            │
│   │  GPIO Header (Top View)                │            │
│   │                                        │            │
│   │  Pin 1  [3.3V]  [5V   ]  Pin 2         │            │
│   │  Pin 3  [GP 2]  [5V   ]  Pin 4         │            │
│   │  Pin 5  [GP 3]  [GND  ]  Pin 6 ────────┼──┐         │
│   │  Pin 7  [GP 4]  [GP 14]  Pin 8         │  │         │
│   │  Pin 9  [GND ]  [GP 15]  Pin 10        │  │         │
│   │  Pin 11 [GP 17] [GP 18]  Pin 12        │  │         │
│   │         └─────────┘                    │  │         │
│   │            │                           │  │         │
│   └────────────┼───────────────────────────┘  │         │
│                │                              │         │
└────────────────┼──────────────────────────────┼─────────┘
                 │                              │
                 │ GPIO 17                      │ GND
                 │ (Signal)                     │
                 │                              │
                 │         Servo Motor          │
                 │       ┌─────────────┐        │
                 │       │             │        │
                 └───────┤ Yellow/Org  │        │
                         │             │        │
                    ┌────┤ Red (VCC)   │        │
                    │    │             │        │
                    │  ┌─┤ Black/Brown │←───────┘
                    │  │ │  (GND)      │
                    │  │ └─────────────┘
                    │  │
                    │  └──────────────────────────┐
                    │                             │
            ┌───────┴─────────────────────────────┴────┐
            │   External 5V Power Supply (2-3A)        │
            │                                           │
            │   (+) ────── Red Wire                    │
            │   (-) ────── Black Wire + Pi GND         │
            └──────────────────────────────────────────┘
```

## Multiple Servos (4 Servos)

```
                        Raspberry Pi 3B
┌─────────────────────────────────────────────────────────┐
│                                                          │
│  Pin 6  [GND  ] ────────────────────────────────┐       │
│  Pin 11 [GP 17] ──────────────┐                 │       │
│  Pin 12 [GP 18] ────────────┐ │                 │       │
│  Pin 13 [GP 27] ──────────┐ │ │                 │       │
│  Pin 15 [GP 22] ────────┐ │ │ │                 │       │
│                         │ │ │ │                 │       │
└─────────────────────────┼─┼─┼─┼─────────────────┼───────┘
                          │ │ │ │                 │
                          │ │ │ │                 │
        ┌─────────────────┘ │ │ │                 │
        │   ┌─────────────────┘ │ │                 │
        │   │   ┌─────────────────┘ │                 │
        │   │   │   ┌─────────────────┘                 │
        │   │   │   │                                   │
        │   │   │   │   Servo 1   Servo 2   Servo 3   Servo 4
        │   │   │   │   ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
        │   │   │   │   │      │  │      │  │      │  │      │
        └───┼───┼───┼───┤Yellow│  │Yellow│  │Yellow│  │Yellow│
            │   │   │   │      │  │      │  │      │  │      │
            └───┼───┼───┼──────┤  ├──────┤  ├──────┤  ├──────┤
                │   │   │      │  │      │  │      │  │      │
                └───┼───┼──────┤  ├──────┤  ├──────┤  ├──────┤
                    │   │      │  │      │  │      │  │      │
                    └───┼──────┤  ├──────┤  ├──────┤  ├──────┤
                        │      │  │      │  │      │  │      │
                    ┌───┤ Red  │  │ Red  │  │ Red  │  │ Red  │
                    │   │      │  │      │  │      │  │      │
                    │ ┌─┤Black │  │Black │  │Black │  │Black │
                    │ │ │      │  │      │  │      │  │      │
                    │ │ └──────┘  └──────┘  └──────┘  └──────┘
                    │ │
                    │ └────┬────┬────┬───── All blacks connected
                    │      │    │    │
              ┌─────┴──────┴────┴────┴──────────────────┐
              │   External 5V Power Supply (2-3A)        │
              │                                           │
              │   (+) 5V  ──── All red wires             │
              │   (-) GND ──── All black wires + Pi GND  │
              └──────────────────────────────────────────┘
```

## Complete System Diagram

```
                    ┌──────────────────────┐
                    │   USB Microphone     │
                    └──────────┬───────────┘
                               │ USB Cable
                               │
    ┌──────────────────────────┴─────────────────────────┐
    │              Raspberry Pi 3B                        │
    │  ┌──────────────────────────────────────────┐      │
    │  │         Software Stack                   │      │
    │  │                                          │      │
    │  │  ┌────────────────────────────────────┐ │      │
    │  │  │      audio_reactive_robot          │ │      │
    │  │  │         (main program)             │ │      │
    │  │  └──────────┬──────────┬──────────────┘ │      │
    │  │             │          │                 │      │
    │  │  ┌──────────▼─┐  ┌────▼────────┐  ┌────▼────┐ │
    │  │  │AudioCapture│  │FFTAnalyzer  │  │  Servo  │ │
    │  │  │   Class    │  │   Class     │  │  Ctrl   │ │
    │  │  └──────┬─────┘  └─────────────┘  └────┬────┘ │
    │  │         │                                │     │ │
    │  │  ┌──────▼─────┐                   ┌─────▼────┐│
    │  │  │    ALSA    │                   │  pigpio  ││
    │  │  │  Library   │                   │  Library ││
    │  │  └────────────┘                   └─────┬────┘│
    │  └────────────────────────────────────────────┬──┘│
    │                                               │   │
    │  GPIO Pins: 17, 18, 27, 22 ──────────────────┘   │
    │                                                   │
    └───────────────────────────┬───────────────────────┘
                                │
                                │ PWM Signals
                                │
         ┌──────────────────────┴────────────────────┐
         │                                            │
    ┌────▼────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ Servo 1 │  │ Servo 2  │  │ Servo 3  │  │ Servo 4  │
    │ (Bass)  │  │  (Mid)   │  │ (Treble) │  │  (All)   │
    └────┬────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
         │            │              │             │
         └────────────┴──────────────┴─────────────┘
                            │
                 Connected to External Power
```

## GPIO Pinout Reference (BCM Numbering)

```
Raspberry Pi 3B GPIO Header (40 pins)

     3.3V  (1)   (2)  5V
    GPIO2  (3)   (4)  5V
    GPIO3  (5)   (6)  GND
    GPIO4  (7)   (8)  GPIO14
      GND  (9)  (10)  GPIO15
   GPIO17 (11)  (12)  GPIO18  ← Default Servo Pins
   GPIO27 (13)  (14)  GND     ← Default Servo Pins
   GPIO22 (15)  (16)  GPIO23  ← Default Servo Pins
     3.3V (17)  (18)  GPIO24
   GPIO10 (19)  (20)  GND
    GPIO9 (21)  (22)  GPIO25
   GPIO11 (23)  (24)  GPIO8
      GND (25)  (26)  GPIO7
    GPIO0 (27)  (28)  GPIO1
    GPIO5 (29)  (30)  GND
    GPIO6 (31)  (32)  GPIO12
   GPIO13 (33)  (34)  GND
   GPIO19 (35)  (36)  GPIO16
   GPIO26 (37)  (38)  GPIO20
      GND (39)  (40)  GPIO21

Note: Numbers in () are physical pin numbers
      Numbers without () are BCM GPIO numbers
      Use BCM numbers in code!
```

## Power Supply Options

### Option 1: AC Adapter (Best)
```
  ┌───────────────┐
  │  Wall Outlet  │
  └───────┬───────┘
          │
  ┌───────▼────────────┐
  │  AC Adapter        │
  │  5V 2-3A          │
  └───────┬────────────┘
          │
  ┌───────▼────────────┐
  │  (+) Red    (-)    │
  │  5V         GND    │
  └───┬────────────┬───┘
      │            │
      │    ┌───────┴──────┐
      │    │ To Pi GND    │
      │    │              │
      └────┴──────────────┴─── To All Servos
```

### Option 2: Battery Pack
```
  ┌──────────────────────┐
  │  4x AA Batteries     │
  │  (6V)                │
  └──────────┬───────────┘
             │
  ┌──────────▼───────────┐
  │  Voltage Regulator   │
  │  (LM7805 or Buck)    │
  │  6V → 5V             │
  └──────────┬───────────┘
             │
  ┌──────────▼───────────┐
  │  (+) 5V    (-) GND   │
  └──┬──────────────┬────┘
     │              │
     │    ┌─────────┴────┐
     │    │ To Pi GND    │
     └────┴──────────────┴─── To All Servos
```

## Connection Checklist

Before powering on:

☐ USB microphone connected to Pi  
☐ Servo signal wires to correct GPIO pins  
☐ All servo power (red) wires to (+) 5V  
☐ All servo ground (black) wires to (-) GND  
☐ Pi GND connected to power supply GND  
☐ Power supply voltage is 5V  
☐ Power supply can provide 2-3A  
☐ No shorts between power and ground  

## Testing Sequence

1. **Power on Pi** (without servos connected)
2. **Start pigpiod**: `sudo systemctl start pigpiod`
3. **Test one servo** with Python script
4. **If successful**, connect remaining servos
5. **Run test_servo**: `sudo ./test_servo`
6. **Run main program**: `sudo ./audio_reactive_robot`

## Safety Notes

⚠️ **NEVER connect servo power to Pi's 5V pins** - will damage Pi!  
⚠️ **ALWAYS connect grounds** - Pi GND + power supply GND  
⚠️ **CHECK polarity** before connecting power  
⚠️ **USE proper gauge wire** for servo power  
⚠️ **MONITOR temperature** - disconnect if hot  

## Common Wiring Mistakes

❌ **Mistake**: Servo red wire to Pi 5V pin  
✅ **Correct**: Servo red wire to external power supply  

❌ **Mistake**: Grounds not connected  
✅ **Correct**: Pi GND and power supply GND connected  

❌ **Mistake**: Signal wire to wrong pin  
✅ **Correct**: Check BCM vs physical pin numbering  

❌ **Mistake**: Reversed polarity on power supply  
✅ **Correct**: Red to (+), Black to (-)  

---

For detailed setup instructions, see **HARDWARE_SETUP.md**

For troubleshooting, see **QUICKSTART.md** Section "Common Issues"

