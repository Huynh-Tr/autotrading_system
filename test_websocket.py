try:
    from websocket import create_connection
    print("SUCCESS: websocket import works")
except ImportError as e:
    print(f"ERROR: {e}")
    
try:
    import websocket
    print(f"websocket module location: {websocket.__file__}")
except ImportError as e:
    print(f"ERROR importing websocket module: {e}") 