import importlib, sys, inspect, pprint
try:
    mod = importlib.import_module("tws_simulator")
    print("tws_simulator found at:", getattr(mod, "__file__", "<built-in>"))
    print("module attributes:")
    pprint.pprint([name for name in dir(mod) if not name.startswith("_")])
    TW = getattr(mod, "TWSSimulator", None)
    print("TWSSimulator object:", TW)
    print("callable?:", callable(TW))
    if callable(TW):
        print("Type info:", type(TW))
        # show constructor signature if possible
        try:
            sig = inspect.signature(TW)
            print("signature:", sig)
        except Exception as e:
            print("Could not get signature:", e)
except Exception as e:
    print("Import failed:", repr(e))
    raise