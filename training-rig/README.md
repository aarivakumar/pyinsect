# Offline training rig

Take insect photos on a phone and save them on the computer. **No internet or Wi-Fi needed.**
The phone is connected with a **USB cable**.

Only needs Python 3. It saves photos in `data/<name>/feature/` and `data/<name>/validation/`,
plus `labels.csv`. About 1 in 5 photos goes to **validation** (held back to test the AI). `data/` is
ignored by git, so photos never go online.

## Set up (Android phone, once)
1. On the computer, install Android's **adb** tool (Platform Tools). It's a free download. Install it once, while you still have internet.
2. On the phone: Settings → About phone → tap **Build number** 7 times → Developer options → turn on **USB debugging**.
3. Plug in the cable and tap **Allow** on the phone.

## Every time
1. `python3 training-rig/server.py` (Windows: `python`). It runs `adb reverse` for you.
2. Computer: open http://localhost:8000 (the Pyinsect site). The training rig is at http://localhost:8000/rig.
3. Phone (Chrome): open http://localhost:8000/capture, type the insect's name, tap **Take Photo**.
4. If the cable falls out, photos wait on the phone. Plug in and tap **Send waiting photos**.

No adb? Turn on **USB tethering** on the phone, run `python3 server.py --lan`, and open
`http://<computer's address>:8000/capture` on the phone. (`--lan` lets anyone on that network open the page, so use it only on your own cable or hotspot.)

**Link not working?** The server must be running (the terminal says "Computer page"). On the phone,
`localhost` only works after `adb reverse` ran (the terminal says "USB phone connected").

## Test it
`python3 server.py`, then in another terminal:
`curl -X POST --data-binary @photo.jpg "http://localhost:8000/api/upload?label=Ladybug"`
